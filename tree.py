from sklearn.neighbors import BallTree
import numpy as np
import torch

def computeTree(input_pts, config, targets=None, forcePt=None):

    data = []
    pts = input_pts.copy()
    if targets is not None:
        tgs = targets.copy()

    conv_reduction_ids = []

    for config_id, conf in enumerate(config):

        npts = conf[0]
        K = conf[1]
        layertype=conf[2]

        # build the ball tree at current level
        tree = BallTree(pts, leaf_size=2)

        indices = []
        pts_n = []
        tgs_n = []

        if layertype == "conv": # nosize reduction

            _, indices = tree.query(pts, k=K)

            if targets is not None:
                data.append({"indices":torch.LongTensor(indices).unsqueeze(0),#"indices":indices,
                    "points":torch.from_numpy(pts).float().unsqueeze(0),
                    "targets":torch.from_numpy(tgs).long().unsqueeze(0)})
            else:
                data.append({"indices":torch.LongTensor(indices).unsqueeze(0),#"indices":indices,
                    "points":torch.from_numpy(pts).float().unsqueeze(0)})

        elif layertype == "convForce": # predefined points
            _, indices = tree.query(forcePt, k=K)

            if targets is not None:
                data.append({"indices":torch.LongTensor(indices).unsqueeze(0),#"indices":indices,
                    "points":torch.from_numpy(forcePt).float().unsqueeze(0),
                    "targets":torch.from_numpy(tgs).long().unsqueeze(0)})
            else:
                data.append({"indices":torch.LongTensor(indices).unsqueeze(0),#"indices":indices,
                    "points":torch.from_numpy(forcePt).float().unsqueeze(0)})

        elif layertype == "conv_reduction": # size reduction

            used = np.zeros(pts.shape[0])
            current_id = 0
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
                if targets is not None:
                    tgs_n.append(tgs[index])

            pts_n = np.array(pts_n)
            tgs_n = np.array(tgs_n)

            if targets is not None:
                data.append({
                    "indices":torch.LongTensor(indices).unsqueeze(0),
                    "points":torch.from_numpy(pts_n).float().unsqueeze(0),
                    "targets":torch.from_numpy(tgs_n).long().unsqueeze(0)})
            else:
                data.append({
                    "indices":torch.LongTensor(indices).unsqueeze(0),#"indices":indices,
                    "points":torch.from_numpy(pts_n).float().unsqueeze(0)})

            pts = pts_n
            tgs = tgs_n

            conv_reduction_ids.append(config_id)
        
        elif layertype == "deconv": # increasing size
            
            #corresponding points are in the previous layers
            corresponding_layers_points = conv_reduction_ids[-1] -1 

            conv_reduction_ids =  conv_reduction_ids[:-1] 


            if corresponding_layers_points < 0:
                pts_n = input_pts
                if targets is not None:
                    tgs_n = targets
            else:
                pts_n = data[corresponding_layers_points]["points"].numpy()[0]
                if targets is not None:
                    tgs_n = data[corresponding_layers_points]["targets"].numpy()[0]

            _, indices = tree.query(pts_n, k=K)

            if targets is not None:
                data.append({"indices":torch.LongTensor(indices).unsqueeze(0),#"indices":indices,
                    "points":torch.from_numpy(pts_n).float().unsqueeze(0),
                    "targets":torch.from_numpy(tgs_n).long().unsqueeze(0)})
            else:
                data.append({"indices":torch.LongTensor(indices).unsqueeze(0),#"indices":indices,
                    "points":torch.from_numpy(pts_n).float().unsqueeze(0)})
            
            pts = pts_n
            tgs = tgs_n

        else:
            raise(Exception("Error Tree construction: layer type"))
    return data


# function for merging elements in dataloader
def tree_collate(batch):

    merged_points = []
    merged_features = []
    merged_seg = []
    if len(batch[0])==4 or len(batch[0])==5:
        merged_tree = [{"indices":[],"points":[],"targets":[]} for i in range(len(batch[0][3]))]
    else:
        merged_tree = [{"indices":[],"points":[],"targets":[]} for i in range(len(batch[0][2]))]
    merged_labels = []

    for b_id, b in enumerate(batch):

        merged_points.append(b[0])
        merged_features.append(b[1])

        if len(b)==4 or len(b)==5:
            merged_seg.append(b[2])
            tree = b[3]
        else:
            tree = b[2]

        for l_id in range(len(tree)):
            merged_tree[l_id]["points"].append(tree[l_id]["points"])
            merged_tree[l_id]["indices"].append(tree[l_id]["indices"])
            if "targets" in tree[l_id]:
                merged_tree[l_id]["targets"].append(tree[l_id]["targets"])

        if len(b)==5:
            merged_labels.append(b[4])

    merged_points = torch.cat(merged_points, dim=0)
    merged_features = torch.cat(merged_features, dim=0)

    for l_id in range(len(merged_tree)):
        merged_tree[l_id]["points"] = torch.cat(merged_tree[l_id]["points"], dim=0)
        merged_tree[l_id]["indices"] = torch.cat(merged_tree[l_id]["indices"], dim=0)
        if len(merged_tree[l_id]["targets"])>0:
            merged_tree[l_id]["targets"] = torch.cat(merged_tree[l_id]["targets"], dim=0)

    if len(merged_seg)>0:
        merged_seg = torch.cat(merged_seg, dim=0)

        if len(merged_labels)>0:
            merged_labels = torch.LongTensor(merged_labels)
            return merged_points, merged_features, merged_seg, merged_tree, merged_labels
        else:
            return merged_points, merged_features, merged_seg, merged_tree

    else:
        if len(merged_labels)>0:
            merged_labels = torch.LongTensor(merged_labels)
            return merged_points, merged_features, merged_tree, merged_labels
        else:
            return merged_points, merged_features, merged_tree
