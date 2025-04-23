import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from PIL import Image

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 定义 U - Net 模型
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]
    ):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 下采样部分
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # 瓶颈层
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # 上采样部分
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(
                    x, size=skip_connection.shape[2:], mode='bilinear',
                    align_corners=True
                )

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


# 定义扩散过程
def diffusion_process(x, t, beta):
    # 确保 beta 是张量并且在正确的设备上
    if not isinstance(beta, torch.Tensor):
        beta = torch.tensor(beta, dtype=torch.float32, device=x.device)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    sqrt_alpha_bar = torch.sqrt(alpha_bar[t])
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar[t])
    # 调整形状以匹配输入数据
    sqrt_alpha_bar = sqrt_alpha_bar.view(-1, 1, 1, 1).to(x.device)
    sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.view(-1, 1, 1, 1).to(x.device)
    noise = torch.randn_like(x)
    xt = sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise
    return xt, noise


# 反向去噪过程
def reverse_process(model, xT, num_steps, beta):
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    for t in reversed(range(num_steps)):
        z = torch.randn_like(xT) if t > 0 else torch.zeros_like(xT)
        alpha_t = alpha[t]
        alpha_bar_t = alpha_bar[t]
        predicted_noise = model(xT)
        xT = (1 / torch.sqrt(alpha_t)) * (
                xT - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise
        ) + torch.sqrt(1 - alpha_t) * z
    return xT


# 训练扩散模型
def train_diffusion_model(model, dataloader, num_steps, beta, num_epochs, lr, test_dir):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    total_steps = len(dataloader) * num_epochs
    current_step = 0

    for epoch in range(num_epochs):
        total_loss = 0
        print(f"Starting epoch {epoch + 1}...")
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            t = torch.randint(0, num_steps, (data.shape[0],), device=device)
            xt, noise = diffusion_process(data, t, beta)
            predicted_noise = model(xt)
            loss = criterion(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            current_step += 1
            print(
                f"Step {current_step}/{total_steps} in epoch {epoch + 1}, Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}')

        # 每个 epoch 结束后进行预测测试
        perform_prediction(model, num_steps, beta, test_dir, epoch)

    return model


# 进行预测测试并保存图片
def perform_prediction(model, num_steps, beta, test_dir, epoch):
    # 随机选择一个样本进行去噪
    data_iter = iter(train_dataloader)
    images, _ = next(data_iter)
    random_index = np.random.randint(0, images.shape[0])
    x = images[random_index].unsqueeze(0).to(device)
    xT, _ = diffusion_process(x, num_steps - 1, beta)

    # 进行去噪
    denoised_x = reverse_process(model, xT, num_steps, beta)

    # 可视化去噪前后的结果
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(np.transpose(x.cpu().squeeze().numpy(), (1, 2, 0)))
    axes[0].set_title('Original Image')
    axes[1].imshow(np.transpose(denoised_x.cpu().squeeze().detach().numpy(), (1, 2, 0)))
    axes[1].set_title('Denoised Image')

    # 保存图片
    os.makedirs(test_dir, exist_ok=True)
    fig.savefig(os.path.join(test_dir, f'epoch_{epoch + 1}_test.png'))
    plt.close(fig)

    # 随机生成噪声
    noise = torch.randn(1, input_channels, 64, 64).to(device)

    # 进行去噪
    generated_image = reverse_process(model, noise, num_steps, beta)

    # 可视化生成的图片
    plt.figure(figsize=(5, 5))
    plt.imshow(np.transpose(generated_image.cpu().squeeze().detach().numpy(), (1, 2, 0)))
    plt.title('Generated Image')

    # 保存生成的图片
    plt.savefig(os.path.join(test_dir, f'epoch_{epoch + 1}_generated.png'))
    plt.close()


# 数据加载
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 将图像大小调整为 64x64
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset = datasets.ImageFolder(root=r'./data/anime-faces',  # 读取 data 文件夹下的 aime-face 目录
                                     transform=transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128,
                                               shuffle=True)

# 超参数设置
input_channels = 3
output_channels = 3
num_steps = 100
beta = torch.linspace(0.0001, 0.02, num_steps).to(device)
num_epochs = 10
lr = 0.001

# 打印 num_steps 信息
print(f"The value of num_steps is: {num_steps}")

# 创建模型
model = UNet(in_channels=input_channels, out_channels=output_channels).to(device)

os.makedirs('models', exist_ok=True)
# 检查是否存在保存的模型
model_path = "models/diffusion_model.pth"
test_dir = os.path.join(r'./data/anime-faces', 'test')
if os.path.exists(model_path):
    print("Loading saved model...")
    model.load_state_dict(torch.load(model_path))

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Diffusion Model')
    parser.add_argument('--mode', type=str, default='predict',
                        choices=['predict', 'train'],
                        help='Choose mode: predict or train')
    args = parser.parse_args()

    if args.mode == 'predict':
        print("Performing prediction...")
        perform_prediction(model, num_steps, beta, test_dir, 0)
    elif args.mode == 'train':
        print("Continuing training...")
        model = train_diffusion_model(model, train_dataloader, num_steps, beta, num_epochs, lr, test_dir)
        torch.save(model.state_dict(), model_path)
        print("Model trained and saved successfully.")
else:
    print("No saved model found, starting training...")
    model = train_diffusion_model(model, train_dataloader, num_steps, beta, num_epochs, lr, test_dir)
    torch.save(model.state_dict(), model_path)
    print("Model trained and saved successfully.")
