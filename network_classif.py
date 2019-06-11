from layers import PtConv
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.utils import apply_bn


class Net(nn.Module):
    
    def __init__(self, input_channels, output_channels, dimension=3):
        super(Net, self).__init__()
        
        n_centers = 27
        pl = 64

        # convolutions
        self.cv1 = PtConv(input_channels, pl, n_centers, dimension)
        self.cv2 = PtConv(pl, pl, n_centers, dimension)
        self.cv3 = PtConv(pl, pl, n_centers, dimension)
        self.cv4 = PtConv(pl, 2*pl, n_centers, dimension)
        self.cv5 = PtConv(2*pl, 2*pl, n_centers, dimension)

        # last layer
        self.fcout = nn.Linear(8*2*pl, output_channels)

        # batchnorms
        self.bn1 = nn.BatchNorm1d(pl)
        self.bn2 = nn.BatchNorm1d(pl)
        self.bn3 = nn.BatchNorm1d(pl)
        self.bn4 = nn.BatchNorm1d(2*pl)
        self.bn5 = nn.BatchNorm1d(2*pl)

    def forward(self, x, input_pts):

        x1, pts1 = self.cv1(x, input_pts, 16, 1024)
        x1 = F.relu(apply_bn(x1, self.bn1))

        x2, pts2 = self.cv2(x1, pts1, 16, 256)
        x2 = F.relu(apply_bn(x2, self.bn2))

        x3, pts3 = self.cv3(x2, pts2, 8, 64)
        x3 = F.relu(apply_bn(x3, self.bn3))

        x4, pts4 = self.cv4(x3, pts3, 8, 16)
        x4 = F.relu(apply_bn(x4, self.bn4))

        x5, _ = self.cv5(x4, pts4, 4, 8)
        x5 = F.relu(apply_bn(x5, self.bn5))

        xout = x5.view(x5.size(0), -1)
        xout = self.fcout(xout)

        return xout