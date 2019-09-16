import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import convpoint.knn.lib.python.nearest_neighbors as nearest_neighbors


class LayerBase(nn.Module):

    def __init__(self):
        super(LayerBase, self).__init__()

    def indices_conv_reduction(self, input_pts, K, npts):

        indices, queries = nearest_neighbors.knn_batch_distance_pick(input_pts.cpu().detach().numpy(), npts, K, omp=True)
        indices = torch.from_numpy(indices).long()
        queries = torch.from_numpy(queries).float()
        if input_pts.is_cuda:
            indices = indices.cuda()
            queries = queries.cuda()

        return indices, queries

    def indices_conv(self, input_pts, K):
        indices = nearest_neighbors.knn_batch(input_pts.cpu().detach().numpy(), input_pts.cpu().detach().numpy(), K, omp=True)
        indices = torch.from_numpy(indices).long()
        if input_pts.is_cuda:
            indices = indices.cuda()
        return indices, input_pts

    def indices_deconv(self, input_pts, next_pts, K):
        indices = nearest_neighbors.knn_batch(input_pts.cpu().detach().numpy(), next_pts.cpu().detach().numpy(), K, omp=True)
        indices = torch.from_numpy(indices).long()
        if input_pts.is_cuda:
            indices = indices.cuda()
        return indices, next_pts