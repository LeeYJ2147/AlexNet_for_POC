import torch
import torch.nn as nn
from torchvision import models

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        attention_weights = self.sigmoid(avg_out + max_out)
        return x * attention_weights

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_concat = torch.cat([avg_out, max_out], dim=1)
        attention_weights = self.sigmoid(self.conv(x_concat))
        return x * attention_weights

class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channel, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

# =========================================================================
# AlexNet Model Definitions
# =========================================================================

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), # 0
            nn.ReLU(inplace=True), # 1
            nn.MaxPool2d(kernel_size=3, stride=2), # 2
            nn.Conv2d(64, 192, kernel_size=5, padding=2), # 3
            nn.ReLU(inplace=True), # 4
            nn.MaxPool2d(kernel_size=3, stride=2), # 5
            nn.Conv2d(192, 384, kernel_size=3, padding=1), # 6
            nn.ReLU(inplace=True), # 7
            nn.Conv2d(384, 256, kernel_size=3, padding=1), # 8
            nn.ReLU(inplace=True), # 9
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # 10
            nn.ReLU(inplace=True), # 11
            nn.MaxPool2d(kernel_size=3, stride=2), # 12
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x, mode='classify'):
        conv_features = self.features(x)
        pooled_features = self.avgpool(conv_features)
        flattened_features = torch.flatten(pooled_features, 1)
        
        fc7_feat = self.classifier[:-1](flattened_features)

        if mode == 'classify':
            output = self.classifier[-1](fc7_feat)
            return output
        elif mode == 'extract_fc7':
            return fc7_feat
        elif mode == 'extract_fused':
            conv5_feat = nn.functional.adaptive_avg_pool2d(conv_features, (1, 1)).flatten(1)
            fused_feat = torch.cat([conv5_feat, fc7_feat], dim=1)
            return fused_feat
        else:
            raise ValueError(f"Unknown mode: {mode}")

class AlexNetBN(AlexNet):
    def __init__(self, num_classes=1000):
        super().__init__(num_classes)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2), nn.BatchNorm2d(192), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.BatchNorm2d(384), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(), nn.Linear(256 * 6 * 6, 4096), nn.BatchNorm1d(4096), nn.ReLU(inplace=True),
            nn.Dropout(), nn.Linear(4096, 4096), nn.BatchNorm1d(4096), nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

class AlexNetCBAM(AlexNet):
    def __init__(self, num_classes=1000):
        super().__init__(num_classes)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), nn.ReLU(inplace=True), CBAM(64),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2), nn.ReLU(inplace=True), CBAM(192),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), CBAM(256),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
class AlexNetBN_CBAM(AlexNet):
    def __init__(self, num_classes=1000):
        super().__init__(num_classes)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True), CBAM(64),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2), nn.BatchNorm2d(192), nn.ReLU(inplace=True), CBAM(192),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.BatchNorm2d(384), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), CBAM(256),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(), nn.Linear(256 * 6 * 6, 4096), nn.BatchNorm1d(4096), nn.ReLU(inplace=True),
            nn.Dropout(), nn.Linear(4096, 4096), nn.BatchNorm1d(4096), nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

# =========================================================================
# Model Factory
# =========================================================================
def _load_pretrained_weights(model):
    pretrained_alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model_dict = model.state_dict()
    pretrained_dict = pretrained_alexnet.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)

def create_model(model_name: str, num_classes: int, pretrained: bool = True, **kwargs):
    model_name = model_name.lower()
    model_map = {
        'alexnet': AlexNet, 
        'alexnet_bn': AlexNetBN, 
        'alexnet_cbam': AlexNetCBAM,
        'alexnet_bn_cbam': AlexNetBN_CBAM,
    }
    if model_name not in model_map:
        raise ValueError(f"Model '{model_name}' not recognized.")
    
    model = model_map[model_name](num_classes=num_classes)

    if pretrained:
        _load_pretrained_weights(model)

    final_layer_in_features = model.classifier[-1].in_features
    if model.classifier[-1].out_features != num_classes:
        model.classifier[-1] = nn.Linear(final_layer_in_features, num_classes)
    
    print(f"Model '{model_name}' created. Pretrained: {pretrained}, Num classes: {num_classes}")
    return model
