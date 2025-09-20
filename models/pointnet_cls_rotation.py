import torch
import torch.nn as nn
import torch.nn.functional as F
from .transform_nets_torch import InputTransformNet, FeatureTransformNet, _BN1d, _BN2d

def transform_regularizer(end_points, reg_weight=0.001):

    A = end_points.get('transform', None)
    if A is None:
        return torch.zeros((), device=next(iter(end_points.values())).device if end_points else 'cpu')
    B, K, _ = A.shape
    I = torch.eye(K, device=A.device).unsqueeze(0).expand(B, -1, -1)
    diff = torch.bmm(A, A.transpose(1, 2)) - I
    return reg_weight * (diff.pow(2).sum(dim=(1,2)).mean())

def set_bn_momentum(model: nn.Module, momentum: float):
    for m in model.modules():
        if isinstance(m, (_BN1d, _BN2d, nn.BatchNorm1d, nn.BatchNorm2d)):
            m.momentum = momentum

class PointNetRotation(nn.Module):
    def __init__(self, num_angles: int,
                 use_input_trans: bool = True,
                 use_feature_trans: bool = True,
                 dropout_keep_prob: float = 0.7):  # TF keep_prob
        super().__init__()
        self.num_angles = num_angles
        self.use_input_trans = use_input_trans
        self.use_feature_trans = use_feature_trans

        self.input_tnet   = InputTransformNet() if use_input_trans else None
        # stem
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1,3), bias=False)
        self.bn1   = _BN2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(1,1), bias=False)
        self.bn2   = _BN2d(64)

        self.feature_tnet = FeatureTransformNet(64) if use_feature_trans else None

        self.conv3 = nn.Conv2d(64,  64,  kernel_size=(1,1), bias=False)
        self.bn3   = _BN2d(64)
        self.conv4 = nn.Conv2d(64, 128,  kernel_size=(1,1), bias=False)
        self.bn4   = _BN2d(128)
        self.conv5 = nn.Conv2d(128,1024, kernel_size=(1,1), bias=False)
        self.bn5   = _BN2d(1024)

        self.fc1   = nn.Linear(1024, 512, bias=False)
        self.bn_fc1= _BN1d(512)
        self.fc2   = nn.Linear(512, 256, bias=False)
        self.bn_fc2= _BN1d(256)
        self.dp1   = nn.Dropout(p=1.0 - dropout_keep_prob)  # p=0.3
        self.dp2   = nn.Dropout(p=1.0 - dropout_keep_prob)

        self.fc3   = nn.Linear(256, num_angles, bias=True)

    def forward(self, point_cloud_bnc, is_training: bool = True):

        B, N, _ = point_cloud_bnc.shape
        end_points = {}

        if self.use_input_trans:
            A_in = self.input_tnet(point_cloud_bnc)          # (B,3,3)
            pc   = torch.bmm(point_cloud_bnc, A_in)          # (B,N,3)
        else:
            pc   = point_cloud_bnc

        x = pc.unsqueeze(1)                                   # (B,1,N,3)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))                   # → (B,64,N,1)

        if self.use_feature_trans:
            feat = x.squeeze(3).permute(0,2,1).contiguous()   # (B,N,64)
            A_feat = self.feature_tnet(feat)                  # (B,64,64)
            end_points['transform'] = A_feat
            feat = torch.bmm(feat, A_feat)                    # (B,N,64)
            x = feat.permute(0,2,1).unsqueeze(3).contiguous() # back to (B,64,N,1)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        x = F.max_pool2d(x, kernel_size=(N,1))                # (B,1024,1,1)
        x = x.view(B, 1024)

        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dp1(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dp2(x)
        logits = self.fc3(x)
        return logits, end_points
