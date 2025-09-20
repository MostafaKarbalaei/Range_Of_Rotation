import torch
import torch.nn as nn
import torch.nn.functional as F

def _init_last_fc_bias_as_identity(fc: nn.Linear, K: int):
    nn.init.zeros_(fc.weight)
    nn.init.constant_(fc.bias, 0.0)
    with torch.no_grad():
        fc.bias.view(K, K).copy_(torch.eye(K))

class _BN1d(nn.BatchNorm1d):
    def set_momentum(self, m: float):
        self.momentum = m

class _BN2d(nn.BatchNorm2d):
    def set_momentum(self, m: float):
        self.momentum = m

class InputTransformNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,   64, kernel_size=(1,3), bias=False)
        self.bn1   = _BN2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(1,1), bias=False)
        self.bn2   = _BN2d(128)
        self.conv3 = nn.Conv2d(128,1024, kernel_size=(1,1), bias=False)
        self.bn3   = _BN2d(1024)

        self.fc1   = nn.Linear(1024, 512, bias=False)
        self.bn_fc1= _BN1d(512)
        self.fc2   = nn.Linear(512, 256, bias=False)
        self.bn_fc2= _BN1d(256)
        self.fc3   = nn.Linear(256, 9, bias=True)
        _init_last_fc_bias_as_identity(self.fc3, 3)

    def forward(self, point_cloud_bnc):
        B, N, _ = point_cloud_bnc.shape
        x = point_cloud_bnc.unsqueeze(1)             # (B,1,N,3)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=(N,1))       # (B,1024,1,1)
        x = x.view(B, 1024)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.fc3(x).view(B, 3, 3)
        return x

class FeatureTransformNet(nn.Module):

    def __init__(self, K=64):
        super().__init__()
        self.K = K
        self.conv1 = nn.Conv2d(K,  64,  kernel_size=(1,1), bias=False)
        self.bn1   = _BN2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(1,1), bias=False)
        self.bn2   = _BN2d(128)
        self.conv3 = nn.Conv2d(128,1024, kernel_size=(1,1), bias=False)
        self.bn3   = _BN2d(1024)

        self.fc1   = nn.Linear(1024, 512, bias=False)
        self.bn_fc1= _BN1d(512)
        self.fc2   = nn.Linear(512, 256, bias=False)
        self.bn_fc2= _BN1d(256)
        self.fc3   = nn.Linear(256, K*K, bias=True)
        _init_last_fc_bias_as_identity(self.fc3, K)

    def forward(self, inputs_bnk):
        B, N, K = inputs_bnk.shape
        assert K == self.K
        x = inputs_bnk.permute(0, 2, 1).unsqueeze(-1).contiguous()  # (B,K,N,1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=(N,1))       # (B,1024,1,1)
        x = x.view(B, 1024)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.fc3(x).view(B, K, K)
        return x
