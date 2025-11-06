import torch
import torch.nn as nn
from torchvision import models

class LegoCNN(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(LegoCNN, self).__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        setattr(self.backbone, 'fc', nn.Identity())
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)