import torch
import torch.nn as nn
import torch.nn.functional as F

class Unit3D(nn.Module):
    """Basic 3D convolutional unit"""
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1, 1),
                 stride=(1, 1, 1), padding=0, use_batch_norm=True, use_bias=False):
        super().__init__()
        
        self.conv3d = nn.Conv3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=use_bias
        )
        
        if use_batch_norm:
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            self.bn = None
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv3d(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.relu(x)
        return x

class InceptionModule(nn.Module):
    """Inception module for I3D"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # 1x1x1 conv
        self.branch1 = Unit3D(in_channels, out_channels[0], kernel_size=(1, 1, 1))
        
        # 1x1x1 -> 3x3x3 conv
        self.branch2 = nn.Sequential(
            Unit3D(in_channels, out_channels[1], kernel_size=(1, 1, 1)),
            Unit3D(out_channels[1], out_channels[2], kernel_size=(3, 3, 3), padding=1)
        )
        
        # 1x1x1 -> 3x3x3 conv (deeper)
        self.branch3 = nn.Sequential(
            Unit3D(in_channels, out_channels[3], kernel_size=(1, 1, 1)),
            Unit3D(out_channels[3], out_channels[4], kernel_size=(3, 3, 3), padding=1)
        )
        
        # MaxPool -> 1x1x1 conv
        self.branch4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1),
            Unit3D(in_channels, out_channels[5], kernel_size=(1, 1, 1))
        )
    
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)

class I3D(nn.Module):
    """
    Inflated 3D ConvNet for video deepfake detection
    Based on: "Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset"
    """
    def __init__(self, num_classes=2, dropout_prob=0.5):
        super().__init__()
        
        # Initial conv
        self.conv1 = Unit3D(3, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=3)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        # Conv2
        self.conv2 = Unit3D(64, 64, kernel_size=(1, 1, 1))
        self.conv3 = Unit3D(64, 192, kernel_size=(3, 3, 3), padding=1)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        # Inception 3a, 3b
        self.inception3a = InceptionModule(192, [64, 96, 128, 16, 32, 32])
        self.inception3b = InceptionModule(256, [128, 128, 192, 32, 96, 64])
        self.maxpool3 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)
        
        # Inception 4a, 4b, 4c, 4d, 4e
        self.inception4a = InceptionModule(480, [192, 96, 208, 16, 48, 64])
        self.inception4b = InceptionModule(512, [160, 112, 224, 24, 64, 64])
        self.inception4c = InceptionModule(512, [128, 128, 256, 24, 64, 64])
        self.inception4d = InceptionModule(512, [112, 144, 288, 32, 64, 64])
        self.inception4e = InceptionModule(528, [256, 160, 320, 32, 128, 128])
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        # Inception 5a, 5b
        self.inception5a = InceptionModule(832, [256, 160, 320, 32, 128, 128])
        self.inception5b = InceptionModule(832, [384, 192, 384, 48, 128, 128])
        
        # Average pooling
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Dropout
        self.dropout = nn.Dropout(dropout_prob)
        
        # Classifier
        self.fc = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        # Input: (batch, 3, frames, height, width)
        
        # Initial layers
        x = self.conv1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        
        # Inception blocks
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        
        x = self.inception5a(x)
        x = self.inception5b(x)
        
        # Average pooling
        x = self.avgpool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        features = x  # Save for analysis
        
        # Dropout + FC
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
    def predict(self, x):
        """Returns probability of fake"""
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        return probs[:, 1]  # Probability of class 1 (FAKE)