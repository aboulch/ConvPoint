import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from global_tags import GlobalTags
if GlobalTags.legacy_layer_base():
    from .legacy.layer_base import LayerBase
else:
    from .layer_base import LayerBase

class PtConv(LayerBase):
    def __init__(self, input_features, output_features, n_centers, dim, use_bias=True):
        super(PtConv, self).__init__()

        # Weight
        self.weight = nn.Parameter(
                        torch.Tensor(input_features, n_centers, output_features), requires_grad=True)
        bound = math.sqrt(3.0) * math.sqrt(2.0 / (input_features + output_features))
        self.weight.data.uniform_(-bound, bound)

        # bias
        self.use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_features), requires_grad=True)
            self.bias.data.uniform_(0,0)

        # centers
        center_data = np.zeros((dim, n_centers))
        for i in range(n_centers):
            coord = np.random.rand(dim)*2 - 1
            while (coord**2).sum() > 1:
                coord = np.random.rand(dim)*2 - 1
            center_data[:,i] = coord
        self.centers = nn.Parameter(torch.from_numpy(center_data).float(), 
                                    requires_grad=True)

        # MLP
        self.l1 = nn.Linear(dim*n_centers, 2*n_centers)
        self.l2 = nn.Linear(2*n_centers, n_centers)
        self.l3 = nn.Linear(n_centers, n_centers)


    def forward(self, input, points, K, next_pts=None, normalize=True):

        # 
        if isinstance(next_pts, int) and points.size(1) > next_pts:
            # convolution with reduction
            indices, next_pts = self.indices_conv_reduction(points, K, next_pts)
        elif (next_pts is None) or (isinstance(next_pts, int) and points.size(1) == next_pts):
            # convolution without reduction
            indices, next_pts = self.indices_conv(points, K)
        elif points.size(1) < next_pts.size(1):
            # convolution with up sampling
            indices, next_pts = self.indices_deconv(points, next_pts, K)
        else:
            raise("ERROR: error while computing indices")

        batch_size = input.size(0)
        n_pts = input.size(1)

        # compute indices for indexing points
        add_indices = torch.arange(batch_size).type(indices.type()) * n_pts
        indices = indices + add_indices.view(-1,1,1)

        # get the features and point cooridnates associated with the indices
        features = input.view(-1, input.size(2))[indices]
        pts = points.view(-1, points.size(2))[indices]

        # center the neighborhoods
        pts = pts - next_pts.unsqueeze(2)

        # normalize to unit ball, or not
        if normalize:
            maxi = torch.sqrt((pts**2).sum(3).max(2)[0])
            maxi[maxi==0] = 1
            pts = pts / maxi.view(maxi.size()+(1,1,))

        # compute the distances
        dists = pts.view(pts.size()+(1,)) - self.centers
        dists = dists.view(dists.size(0), dists.size(1), dists.size(2), -1)
        dists = F.relu(self.l1(dists))
        dists = F.relu(self.l2(dists))
        dists = F.relu(self.l3(dists))

         # compute features
        fs = features.size()
        features = features.transpose(2,3)
        features = features.view(-1, features.size(2), features.size(3))
        dists= dists.view(-1, dists.size(2), dists.size(3))

        features = torch.bmm(features, dists)

        features = features.view(fs[0], fs[1], -1)

        features = torch.matmul(features, self.weight.view(-1, self.weight.size(2)))
        features = features/ fs[2]

        # add a bias
        if self.use_bias:
            features = features + self.bias

        return features, next_pts
