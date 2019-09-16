import torch
import torch.nn as nn
import torch.nn.functional as F

from convpoint.nn import PtConv
from convpoint.nn.utils import apply_bn



class ModelNet40(nn.Module):
    
    def __init__(self, input_channels, output_channels, dimension=3):
        super(ModelNet40, self).__init__()
        
        n_centers = 16
        pl = 64

        # convolutions
        self.cv1 = PtConv(input_channels, pl, n_centers, dimension)
        self.cv2 = PtConv(pl, 2*pl, n_centers, dimension)
        self.cv3 = PtConv(2*pl, 4*pl, n_centers, dimension)
        self.cv4 = PtConv(4*pl, 4*pl, n_centers, dimension)
        self.cv5 = PtConv(4*pl, 8*pl, n_centers, dimension)

        # last layer
        self.fcout = nn.Linear(8*pl, output_channels)

        # batchnorms
        self.bn1 = nn.BatchNorm1d(pl)
        self.bn2 = nn.BatchNorm1d(2*pl)
        self.bn3 = nn.BatchNorm1d(4*pl)
        self.bn4 = nn.BatchNorm1d(4*pl)
        self.bn5 = nn.BatchNorm1d(8*pl)

        self.dropout = nn.Dropout(0.5)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, input_pts):

        x1, pts1 = self.cv1(x, input_pts, 32, 1024)
        x1 = self.relu(apply_bn(x1, self.bn1))

        x2, pts2 = self.cv2(x1, pts1, 32, 256)
        x2 = self.relu(apply_bn(x2, self.bn2))

        x3, pts3 = self.cv3(x2, pts2, 16, 64)
        x3 = self.relu(apply_bn(x3, self.bn3))

        x4, pts4 = self.cv4(x3, pts3, 16, 16)
        x4 = self.relu(apply_bn(x4, self.bn4))

        x5, _ = self.cv5(x4, pts4, 16, 1)
        x5 = self.relu(apply_bn(x5, self.bn5))

        xout = x5.view(x5.size(0), -1)
        xout = self.dropout(xout)
        xout = self.fcout(xout)

        return xout
