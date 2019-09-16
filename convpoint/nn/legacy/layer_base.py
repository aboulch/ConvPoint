import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import multiprocessing as mp
from sklearn.neighbors import BallTree

def mp_indices_conv(pts, K=16):
    tree = BallTree(pts, leaf_size=2)
    _, indices = tree.query(pts, k=K)
    return torch.LongTensor(indices).unsqueeze(0)

def mp_indices_deconv(pts, pts_next, K):
    tree = BallTree(pts, leaf_size=2)
    _, indices = tree.query(pts_next, k=K)
    return torch.LongTensor(indices).unsqueeze(0)

def mp_indices_conv_reduction(pts, K, npts):
    tree = BallTree(pts, leaf_size=2)
    used = np.zeros(pts.shape[0])
    current_id = 0
    indices = []
    pts_n = []
    for ptid in range(npts):

        # index = np.random.randint(pts.shape[0])
        possible_ids = np.argwhere(used==current_id).ravel().tolist()
        while(len(possible_ids)==0):
            current_id = used.min()
            possible_ids = np.argwhere(used==current_id).ravel().tolist()

        index = possible_ids[np.random.randint(len(possible_ids))]

        # pick a point
        pt = pts[index]

        # perform the search
        dist, ids = tree.query([pt], k=K)
        ids = ids[0]

        used[ids] +=1
        used[index] += 1e7

        indices.append(ids.tolist())
        pts_n.append(pt)

    pts_n = np.array(pts_n)

    return torch.LongTensor(indices).unsqueeze(0), torch.from_numpy(pts_n).float().unsqueeze(0)


class LayerBase(nn.Module):

    # class attribute, shared accross PtConv objects
    pool = mp.Pool(16)

    def __init__(self):
        super(LayerBase, self).__init__()

    def indices_conv_reduction(self, input_pts, K, npts):
        pts = [(input_pts[i].cpu().numpy(), K, npts) for i in range(input_pts.size(0))]
        indices = self.pool.starmap(mp_indices_conv_reduction, pts)
        indices, pts = zip(*indices)
        indices = torch.cat(indices, dim=0).long().cuda()
        pts = torch.cat(pts, dim=0).float().cuda()
        return indices, pts

    def indices_conv(self, input_pts, K):
        pts = [(input_pts[i].cpu().numpy(), K) for i in range(input_pts.size(0))]
        indices = self.pool.starmap(mp_indices_conv, pts)
        indices = torch.cat(indices, dim=0).long().cuda()
        return indices, input_pts

    def indices_deconv(self, pts, next_pts, K):
        pts_ = [(pts[i].cpu().numpy(), next_pts[i].cpu().numpy(), K) for i in range(pts.size(0))]
        indices = self.pool.starmap(mp_indices_deconv, pts_)
        indices = torch.cat(indices, dim=0).long().cuda()
        return indices, next_pts