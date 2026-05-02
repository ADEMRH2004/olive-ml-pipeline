import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class OliveResNet(nn.Module):
    """
    ResNet-style CNN for olive classification using tabular features
    Adapted for satellite imagery features (NDVI, EVI, spectral bands, etc.)
    """

    def __init__(self, input_dim=20, num_classes=3, hidden_dims=[64, 128, 256]):
        super(OliveResNet, self).__init__()

        # Initial projection
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])

        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            self.res_blocks.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    nn.BatchNorm1d(hidden_dims[i+1]),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dims[i+1], hidden_dims[i+1]),
                    nn.BatchNorm1d(hidden_dims[i+1])
                )
            )

        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dims[-1]//2, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_proj(x)
        x = F.relu(x)

        # Residual connections
        for block in self.res_blocks:
            residual = x
            x = block(x)
            x = x + residual  # Skip connection
            x = F.relu(x)

        x = self.classifier(x)
        return x


class OliveUNet(nn.Module):
    """
    U-Net style architecture for olive grove segmentation
    Designed for spatial feature maps from satellite imagery
    """

    def __init__(self, in_channels=10, out_channels=3, features=[64, 128, 256]):
        super(OliveUNet, self).__init__()

        # Encoder (Contracting Path)
        self.encoder = nn.ModuleList()
        for feature in features:
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, feature, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
            )
            in_channels = feature

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1]*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features[-1]*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[-1]*2, features[-1]*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features[-1]*2),
            nn.ReLU(inplace=True)
        )

        # Decoder (Expanding Path)
        self.decoder = nn.ModuleList()
        for feature in reversed(features):
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2),
                    nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True)
                )
            )

        # Final convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        skip_connections = []
        for encoder_block in self.encoder:
            x = encoder_block(x)
            skip_connections.append(x)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # Reverse for decoder

        # Decoder
        for i, decoder_block in enumerate(self.decoder):
            x = decoder_block(x)
            if i < len(skip_connections):
                skip = skip_connections[i]
                # Resize skip connection if needed
                if x.shape != skip.shape:
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
                x = torch.cat((skip, x), dim=1)

        return self.final_conv(x)


class OliveDataset(Dataset):
    """
    PyTorch Dataset for olive classification/segmentation
    """

    def __init__(self, features, labels=None, transform=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels) if labels is not None else None
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        if self.labels is not None:
            y = self.labels[idx]
            return x, y
        return x


def create_olive_resnet_model(input_dim=20, num_classes=3):
    """
    Factory function to create ResNet model for olive classification
    """
    model = OliveResNet(input_dim=input_dim, num_classes=num_classes)
    return model


def create_olive_unet_model(in_channels=10, out_channels=3):
    """
    Factory function to create U-Net model for olive segmentation
    """
    model = OliveUNet(in_channels=in_channels, out_channels=out_channels)
    return model


def prepare_tabular_data_for_resnet(csv_path, target_col='intensity', feature_cols=None):
    """
    Prepare tabular data for ResNet training
    """
    df = pd.read_csv(csv_path)

    if feature_cols is None:
        # Use all numeric columns except target and geometry
        feature_cols = [col for col in df.columns
                       if col not in [target_col, 'geometry', 'zone', 'longitude', 'latitude']
                       and df[col].dtype in ['int64', 'float64']]

    X = df[feature_cols].values
    y = df[target_col].values if target_col in df.columns else None

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, feature_cols


def train_resnet_model(model, train_loader, val_loader, num_epochs=50, device='cpu'):
    """
    Training loop for ResNet model
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    model.to(device)
    best_acc = 0.0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = 100. * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_olive_resnet.pth')

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}: '
                  f'Train Loss: {train_loss/len(train_loader):.4f}, '
                  f'Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss/len(val_loader):.4f}, '
                  f'Val Acc: {val_acc:.2f}%')

    return model


def visualize_model_performance(history, save_path='./models/dl_performance.png'):
    """
    Visualize training history for deep learning models
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Loss plot
    ax1.plot(history['train_loss'], label='Training Loss', marker='o')
    ax1.plot(history['val_loss'], label='Validation Loss', marker='s')
    ax1.set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(history['train_acc'], label='Training Accuracy', marker='o')
    ax2.plot(history['val_acc'], label='Validation Accuracy', marker='s')
    ax2.set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f'✅ Deep learning performance visualization saved to {save_path}')


if __name__ == "__main__":
    # Example usage
    print("Olive Deep Learning Models")
    print("=" * 50)

    # Test ResNet creation
    resnet_model = create_olive_resnet_model(input_dim=20, num_classes=3)
    print(f"✅ ResNet model created with {sum(p.numel() for p in resnet_model.parameters())} parameters")

    # Test U-Net creation
    unet_model = create_olive_unet_model(in_channels=10, out_channels=3)
    print(f"✅ U-Net model created with {sum(p.numel() for p in unet_model.parameters())} parameters")

    print("\n🚀 Ready for training with satellite imagery data!")
    print("💡 Use prepare_tabular_data_for_resnet() for tabular features")
    print("💡 Use train_resnet_model() for training")
    print("💡 Use visualize_model_performance() for results")