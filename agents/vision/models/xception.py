import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x

class XceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reps, stride=1, start_with_relu=True):
        super().__init__()
        
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
        
        rep = []
        for i in range(reps):
            if i == 0:
                if start_with_relu:
                    rep.append(self.relu)
                rep.append(SeparableConv2d(in_channels, out_channels, 3, 1, 1))
            else:
                rep.append(self.relu)
                rep.append(SeparableConv2d(out_channels, out_channels, 3, 1, 1))
        
        if stride != 1:
            rep.append(nn.MaxPool2d(3, stride, 1))
        
        self.rep = nn.Sequential(*rep)
        
    def forward(self, x):
        output = self.rep(x)
        skip = self.skip(x)
        return output + skip

class XceptionNet(nn.Module):
    """
    Xception model for deepfake detection
    Based on: "Xception: Deep Learning with Depthwise Separable Convolutions"
    """
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.block1 = XceptionBlock(64, 128, 2, 2, start_with_relu=False)
        self.block2 = XceptionBlock(128, 256, 2, 2)
        self.block3 = XceptionBlock(256, 728, 2, 2)
        
        # Middle flow
        self.block4 = XceptionBlock(728, 728, 3, 1)
        self.block5 = XceptionBlock(728, 728, 3, 1)
        self.block6 = XceptionBlock(728, 728, 3, 1)
        self.block7 = XceptionBlock(728, 728, 3, 1)
        self.block8 = XceptionBlock(728, 728, 3, 1)
        self.block9 = XceptionBlock(728, 728, 3, 1)
        self.block10 = XceptionBlock(728, 728, 3, 1)
        self.block11 = XceptionBlock(728, 728, 3, 1)
        
        # Exit flow
        self.block12 = XceptionBlock(728, 1024, 2, 2)
        
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.relu4 = nn.ReLU(inplace=True)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        
        # Exit flow
        x = self.block12(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        features = x  # Save for Grad-CAM
        x = self.fc(x)
        
        return x, features
    
    def predict(self, x):
        """Returns probability of fake"""
        logits, _ = self.forward(x)
        probs = F.softmax(logits, dim=1)
        return probs[:, 1]  # Probability of class 1 (FAKE)