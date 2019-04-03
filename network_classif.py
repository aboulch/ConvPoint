from layers import PtConv
import torch
import torch.nn as nn
import torch.nn.functional as F

# a function for computing the convolution
def conv(x, layer, tree, layer_id, bn=None, input_pts=None):
    if layer_id == -1:
        x = layer(x, input_pts, tree[layer_id+1]["indices"])
    else:
        x = layer(x, tree[layer_id]["points"], tree[layer_id+1]["indices"])
    if bn is not None:
        x = x.transpose(1,2)
        x = bn(x).transpose(1,2).contiguous()
    return F.relu(x)

class Net(nn.Module):
    
    def __init__(self, input_channels, output_channels, dimension=3):
        super(Net, self).__init__()

        self.config = [
                    [1024, 16, "conv_reduction"],
                    [256, 16, "conv_reduction"],
                    [64,  8, "conv_reduction"],
                    [16,   8, "conv_reduction"],
                    [8,   4, "conv_reduction"],
                    ]

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

    def forward(self, x, input_pts, tree):

        layer_id = -1
        x1 = conv(x, self.cv1, tree, layer_id, self.bn1, input_pts)
        layer_id += 1

        x2 = conv(x1, self.cv2, tree, layer_id, self.bn2)
        layer_id += 1

        x3 = conv(x2, self.cv3, tree, layer_id, self.bn3)
        layer_id += 1

        x4 = conv(x3, self.cv4, tree, layer_id, self.bn4)
        layer_id += 1

        x5 = conv(x4, self.cv5, tree, layer_id, self.bn5)
        layer_id += 1

        xout = x5.view(x5.size(0), -1)
        xout = self.fcout(xout)
        return xout