from layers import PtConv
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.utils import apply_bn
import random

class NetShapeNet(nn.Module):
    def __init__(self, input_channels, output_channels, dimension=3):
        super(NetShapeNet, self).__init__()

        self.config = []

        n_centers = 16

        pl = 48
        # self.cv1 = PtConv(input_channels, pl, n_centers, dimension, use_bias=False)
        self.cv2 = PtConv(input_channels, pl, n_centers, dimension, use_bias=False)
        self.cv3 = PtConv(pl, pl, n_centers, dimension, use_bias=False)
        self.cv4 = PtConv(pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv5 = PtConv(2*pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv6 = PtConv(2*pl, 2*pl, n_centers, dimension, use_bias=False)

        self.cv5d = PtConv(2*pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv4d = PtConv(4*pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv3d = PtConv(4*pl, pl, n_centers, dimension, use_bias=False)
        self.cv2d = PtConv(2*pl, pl, n_centers, dimension, use_bias=False)
        self.cv1d = PtConv(2*pl, pl, n_centers, dimension, use_bias=False)
        # self.cv0d = PtConv(2*pl, pl, n_centers, dimension, use_bias=False)

        self.fcout = nn.Linear(pl, output_channels)

        # self.bn1 = nn.BatchNorm1d(pl)
        self.bn2 = nn.BatchNorm1d(pl)
        self.bn3 = nn.BatchNorm1d(pl)
        self.bn4 = nn.BatchNorm1d(2*pl)
        self.bn5 = nn.BatchNorm1d(2*pl)
        self.bn6 = nn.BatchNorm1d(2*pl)

        self.bn5d = nn.BatchNorm1d(2*pl)
        self.bn4d = nn.BatchNorm1d(2*pl)
        self.bn3d = nn.BatchNorm1d(pl)
        self.bn2d = nn.BatchNorm1d(pl)
        self.bn1d = nn.BatchNorm1d(pl)
        # self.bn0d = nn.BatchNorm1d(pl)


        self.drop = nn.Dropout(0.2)

    def forward(self, x, input_pts, return_features=False):


        # x1, pts1 = self.cv1(x, input_pts, 16, 2048)
        # x1 = F.relu(apply_bn(x1, self.bn1))

        x2, pts2 = self.cv2(x, input_pts, 16, 1024)
        x2 = F.relu(apply_bn(x2, self.bn2))

        x3, pts3 = self.cv3(self.drop(x2), pts2, 16, 256)
        x3 = F.relu(apply_bn(x3, self.bn3))

        x4, pts4 = self.cv4(self.drop(x3), pts3, 8, 64)
        x4 = F.relu(apply_bn(x4, self.bn4))

        x5, pts5 = self.cv5(self.drop(x4), pts4, 8, 16)
        x5 = F.relu(apply_bn(x5, self.bn5))

        x6, pts6 = self.cv6(self.drop(x5), pts5, 4, 8)
        x6 = F.relu(apply_bn(x6, self.bn6))

        x5d, _ = self.cv5d(self.drop(x6), pts6, 4, pts5)
        x5d = F.relu(apply_bn(x5d, self.bn5d))
        x5d = torch.cat([x5d, x5], dim=2)

        x4d, _ = self.cv4d(self.drop(x5d), pts5, 4, pts4)
        x4d = F.relu(apply_bn(x4d, self.bn4d))
        x4d = torch.cat([x4d, x4], dim=2)

        x3d, _ = self.cv3d(self.drop(x4d), pts4, 4, pts3)
        x3d = F.relu(apply_bn(x3d, self.bn3d))
        x3d = torch.cat([x3d, x3], dim=2)

        x2d, _ = self.cv2d(self.drop(x3d), pts3, 8, pts2)
        x2d = F.relu(apply_bn(x2d, self.bn2d))
        x2d = torch.cat([x2d, x2], dim=2)
        
        x1d, _ = self.cv1d(self.drop(x2d), pts2, 8, input_pts)
        x1d = F.relu(apply_bn(x1d, self.bn1d))
        # x1d = torch.cat([x1d, x1], dim=2)

        # x0d, _ = self.cv0d(x1d, pts1, 8, input_pts)
        # x0d = F.relu(apply_bn(x0d, self.bn0d))

        xout = x1d
        xout = xout.view(-1, xout.size(2))
        xout = self.fcout(xout)
        xout = xout.view(x.size(0), -1, xout.size(1))

        if return_features:
            return xout, x0d
        else:
            return xout



class NetS3DIS(nn.Module):
    def __init__(self, input_channels, output_channels, dimension=3):
        super(NetS3DIS, self).__init__()

        self.config = []

        n_centers = 16

        pl = 48 
        self.cv1 = PtConv(input_channels, pl, n_centers, dimension, use_bias=False)
        self.cv2 = PtConv(pl, pl, n_centers, dimension, use_bias=False)
        self.cv3 = PtConv(pl, pl, n_centers, dimension, use_bias=False)
        self.cv4 = PtConv(pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv5 = PtConv(2*pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv6 = PtConv(2*pl, 2*pl, n_centers, dimension, use_bias=False)

        self.cv5d = PtConv(2*pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv4d = PtConv(4*pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv3d = PtConv(4*pl, pl, n_centers, dimension, use_bias=False)
        self.cv2d = PtConv(2*pl, pl, n_centers, dimension, use_bias=False)
        self.cv1d = PtConv(2*pl, pl, n_centers, dimension, use_bias=False)
        self.cv0d = PtConv(2*pl, pl, n_centers, dimension, use_bias=False)

        self.fcout = nn.Linear(pl, output_channels)

        self.bn1 = nn.BatchNorm1d(pl)
        self.bn2 = nn.BatchNorm1d(pl)
        self.bn3 = nn.BatchNorm1d(pl)
        self.bn4 = nn.BatchNorm1d(2*pl)
        self.bn5 = nn.BatchNorm1d(2*pl)
        self.bn6 = nn.BatchNorm1d(2*pl)

        self.bn5d = nn.BatchNorm1d(2*pl)
        self.bn4d = nn.BatchNorm1d(2*pl)
        self.bn3d = nn.BatchNorm1d(pl)
        self.bn2d = nn.BatchNorm1d(pl)
        self.bn1d = nn.BatchNorm1d(pl)
        self.bn0d = nn.BatchNorm1d(pl)


        self.drop = nn.Dropout(0.2)

    def forward(self, x, input_pts, return_features=False):


        x1, pts1 = self.cv1(x, input_pts, 16, 2048)
        x1 = F.relu(apply_bn(x1, self.bn1))

        x2, pts2 = self.cv2(x1, pts1, 16, 1024)
        x2 = F.relu(apply_bn(x2, self.bn2))

        x3, pts3 = self.cv3(self.drop(x2), pts2, 16, 256)
        x3 = F.relu(apply_bn(x3, self.bn3))

        x4, pts4 = self.cv4(self.drop(x3), pts3, 8, 64)
        x4 = F.relu(apply_bn(x4, self.bn4))

        x5, pts5 = self.cv5(self.drop(x4), pts4, 8, 16)
        x5 = F.relu(apply_bn(x5, self.bn5))

        x6, pts6 = self.cv6(self.drop(x5), pts5, 4, 8)
        x6 = F.relu(apply_bn(x6, self.bn6))

        x5d, _ = self.cv5d(self.drop(x6), pts6, 4, pts5)
        x5d = F.relu(apply_bn(x5d, self.bn5d))
        x5d = torch.cat([x5d, x5], dim=2)

        x4d, _ = self.cv4d(self.drop(x5d), pts5, 4, pts4)
        x4d = F.relu(apply_bn(x4d, self.bn4d))
        x4d = torch.cat([x4d, x4], dim=2)

        x3d, _ = self.cv3d(self.drop(x4d), pts4, 4, pts3)
        x3d = F.relu(apply_bn(x3d, self.bn3d))
        x3d = torch.cat([x3d, x3], dim=2)

        x2d, _ = self.cv2d(self.drop(x3d), pts3, 8, pts2)
        x2d = F.relu(apply_bn(x2d, self.bn2d))
        x2d = torch.cat([x2d, x2], dim=2)
        
        x1d, _ = self.cv1d(self.drop(x2d), pts2, 8, pts1)
        x1d = F.relu(apply_bn(x1d, self.bn1d))
        x1d = torch.cat([x1d, x1], dim=2)

        x0d, _ = self.cv0d(x1d, pts1, 8, input_pts)
        x0d = F.relu(apply_bn(x0d, self.bn0d))

        xout = x0d
        xout = xout.view(-1, xout.size(2))
        xout = self.fcout(xout)
        xout = xout.view(x.size(0), -1, xout.size(1))

        if return_features:
            return xout, x0d
        else:
            return xout

