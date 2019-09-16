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

class MaxPool(LayerBase):
    def __init__(self):
        super(MaxPool, self).__init__()

    def forward(self, input, points, K, next_pts=None, normalize=True):

        if isinstance(next_pts, int) and points.size(1) != next_pts:
            # convolution with reduction
            indices, next_pts_ = self.indices_conv_reduction(points, K, next_pts)
        elif (next_pts is None) or (isinstance(next_pts, int) and points.size(1) == next_pts):
            # convolution without reduction
            indices, next_pts_ = self.indices_conv(points, K)
        else:
            # convolution with up sampling or projection on given points
            indices, next_pts_ = self.indices_deconv(points, next_pts, K)

        if next_pts is None or isinstance(next_pts, int):
            next_pts = next_pts_

        batch_size = input.size(0)
        n_pts = input.size(1)

        # compute indices for indexing points
        add_indices = torch.arange(batch_size).type(indices.type()) * n_pts
        indices = indices + add_indices.view(-1,1,1)

        # get the features and point cooridnates associated with the indices
        features = input.view(-1, input.size(2))[indices]

        features, _ = features.max(dim=2)

        return features, next_pts
