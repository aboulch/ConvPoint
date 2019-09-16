from convpoint.nn import PtConv
from convpoint.nn.utils import apply_bn
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetS3DISFusion(nn.Module):
    def __init__(self, input_channels, output_channels, dimension=3):
        super(NetS3DISFusion, self).__init__()

        n_centers = 16

        self.cv1 = PtConv(input_channels, 96, n_centers, dimension, use_bias=False)
        self.cv2 = PtConv(96, 48, n_centers, dimension, use_bias=False)

        self.fcout = nn.Linear(48+2*output_channels, output_channels)

        self.bn1 = nn.BatchNorm1d(96)
        self.bn2 = nn.BatchNorm1d(48)

        self.drop = nn.Dropout(0.5)

    def forward(self, out1, out2, features1, features2, input_pts):

        x = torch.cat([features1, features2], dim=2)
        x1, _ = self.cv1(x, input_pts, 16, input_pts.size(1))
        x1 = F.relu(apply_bn(x1, self.bn1))

        x2, pts2 = self.cv2(x1, input_pts, 16, input_pts.size(1))
        x2 = F.relu(apply_bn(x2, self.bn2))

        xout = x2
        xout = torch.cat([xout, out1, out2], dim=2)
        xout = xout.view(-1, xout.size(2))
        xou = self.drop(xout)
        xout = self.fcout(xout)
        xout = xout.view(x.size(0), -1, xout.size(1))

        # return xout + out1 + out2
        return xout #+ out1 + out2


