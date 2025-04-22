import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import clip
from torch.utils.data import DataLoader
from torchvision.datasets import Flickr30k

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载 CLIP 模型
clip_model, preprocess = clip.load("ViT-B/32", device=device)


# 定义条件 U - Net 模型
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


class ConditionalUNet(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=3, features=[64, 128, 256, 512], text_embedding_dim=512
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

        # 文本特征融合层
        self.text_proj = nn.Linear(text_embedding_dim, in_channels)

    def forward(self, x, text_features):
        text_emb = self.text_proj(text_features).unsqueeze(-1).unsqueeze(-1)
        x = x + text_emb

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
def reverse_process(model, xT, num_steps, beta, text_features):
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    for t in reversed(range(num_steps)):
        z = torch.randn_like(xT) if t > 0 else torch.zeros_like(xT)
        alpha_t = alpha[t]
        alpha_bar_t = alpha_bar[t]
        predicted_noise = model(xT, text_features)
        xT = (1 / torch.sqrt(alpha_t)) * (
                xT - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise
        ) + torch.sqrt(1 - alpha_t) * z
    return xT


# 训练扩散模型
def train_diffusion_model(model, dataloader, num_steps, beta, num_epochs, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    total_steps = len(dataloader) * num_epochs
    current_step = 0

    for epoch in range(num_epochs):
        total_loss = 0
        print(f"Starting epoch {epoch + 1}...")
        for batch_idx, (data, captions) in enumerate(dataloader):
            data = data.to(device)
            # 这里可以选择一个描述，也可以进行处理合并多个描述
            text_descriptions = [captions[i][0] for i in range(len(data))]
            text_tokens = clip.tokenize(text_descriptions).to(device)
            with torch.no_grad():
                text_features = clip_model.encode_text(text_tokens).float()
            t = torch.randint(0, num_steps, (data.shape[0],), device=device)
            xt, noise = diffusion_process(data, t, beta)
            predicted_noise = model(xt, text_features)
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

    return model


# 数据加载
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载 Flickr30K 数据集
train_dataset = Flickr30k(root='path/to/flickr30k/images',
                          ann_file='path/to/flickr30k/annotations.json',
                          transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 超参数设置
input_channels = 3
output_channels = 3
num_steps = 100
beta = torch.linspace(0.0001, 0.02, num_steps).to(device)
num_epochs = 10
lr = 0.001
text_embedding_dim = 512

# 打印 num_steps 信息
print(f"The value of num_steps is: {num_steps}")

# 创建模型
model = ConditionalUNet(in_channels=input_channels, out_channels=output_channels,
                        text_embedding_dim=text_embedding_dim).to(device)

os.makedirs('models', exist_ok=True)
# 检查是否存在保存的模型
model_path = "models/diffusion_model.pth"
if os.path.exists(model_path):
    print("Loading saved model...")
    model.load_state_dict(torch.load(model_path))

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Text-to-Image Conditional Diffusion Model')
    parser.add_argument('--mode', type=str, default='predict',
                        choices=['predict', 'train'],
                        help='Choose mode: predict or train')
    parser.add_argument('--text_description', type=str, default='A dog',
                        help='Text description for prediction')
    args = parser.parse_args()

    if args.mode == 'predict':
        print("Performing prediction...")
        text_tokens = clip.tokenize([args.text_description]).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(text_tokens).float()
        # 随机生成噪声
        noise = torch.randn(1, input_channels, 224, 224).to(device)  # 调整为 224x224

        # 进行去噪
        generated_image = reverse_process(model, noise, num_steps, beta, text_features)

        # 可视化生成的图片
        plt.figure(figsize=(5, 5))
        plt.imshow(np.transpose(generated_image.cpu().squeeze().detach().numpy(), (1, 2, 0)))
        plt.title(f'Generated Image: {args.text_description}')
        plt.show()
    elif args.mode == 'train':
        print("Continuing training...")
        model = train_diffusion_model(model, train_dataloader, num_steps, beta, num_epochs, lr)
        torch.save(model.state_dict(), model_path)
        print("Model trained and saved successfully.")
else:
    print("No saved model found, starting training...")
    model = train_diffusion_model(model, train_dataloader, num_steps, beta, num_epochs, lr)
    torch.save(model.state_dict(), model_path)
    print("Model trained and saved successfully.")
