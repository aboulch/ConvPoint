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

                    [-1,   4, "deconv"],
                    [-1,   4, "deconv"],
                    [-1,   4, "deconv"],
                    [-1,   8, "deconv"],
                    [-1,   8, "deconv"]]

        n_centers = 27
        pl = 48

        self.cv1 = PtConv(input_channels, pl, n_centers, dimension)
        self.cv2 = PtConv(pl, pl, n_centers, dimension)
        self.cv3 = PtConv(pl, pl, n_centers, dimension)
        self.cv4 = PtConv(pl, 2*pl, n_centers, dimension)
        self.cv5 = PtConv(2*pl, 2*pl, n_centers, dimension)

        self.cv4d = PtConv(2*pl, 2*pl, n_centers, dimension)
        self.cv3d = PtConv(4*pl, pl, n_centers, dimension)
        self.cv2d = PtConv(2*pl, pl, n_centers, dimension)
        self.cv1d = PtConv(2*pl, pl, n_centers, dimension)

        self.cv0d = PtConv(2*pl, pl, n_centers, dimension)
        self.fcout = nn.Linear(pl, output_channels)

        self.bn1 = nn.BatchNorm1d(pl)
        self.bn2 = nn.BatchNorm1d(pl)
        self.bn3 = nn.BatchNorm1d(pl)
        self.bn4 = nn.BatchNorm1d(2*pl)
        self.bn5 = nn.BatchNorm1d(2*pl)

        self.bn4d = nn.BatchNorm1d(2*pl)
        self.bn3d = nn.BatchNorm1d(pl)
        self.bn2d = nn.BatchNorm1d(pl)
        self.bn1d = nn.BatchNorm1d(pl)
        self.bn0d = nn.BatchNorm1d(pl)

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

        x4d = conv(x5, self.cv4d, tree, layer_id, self.bn4d)
        x4d = torch.cat([x4d, x4], dim=2)
        layer_id += 1

        x3d = conv(x4d, self.cv3d, tree, layer_id, self.bn3d)
        x3d = torch.cat([x3d, x3], dim=2)
        layer_id += 1

        x2d = conv(x3d, self.cv2d, tree, layer_id, self.bn2d)
        x2d = torch.cat([x2d, x2], dim=2)
        layer_id += 1

        x1d = conv(x2d, self.cv1d, tree, layer_id, self.bn1d)
        x1d = torch.cat([x1d, x1], dim=2)
        layer_id += 1

        x0d = conv(x1d, self.cv0d, tree, layer_id, self.bn0d)
        xout = x0d
        xout = xout.view(-1, xout.size(2))
        xout = self.fcout(xout)
        xout = xout.view(x.size(0), -1, xout.size(1))
        return xout
