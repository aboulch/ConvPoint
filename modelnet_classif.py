
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import random
import os
from tqdm import tqdm
import argparse
from datetime import datetime
import trimesh
from sklearn.metrics import confusion_matrix

import metrics
from tree import computeTree, tree_collate
from network_classif import Net



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    # function inspired by PointNet, translated to Pytorch
    # https://github.com/charlesq34/pointnet
    N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data


class PointCloudFileLists(torch.utils.data.Dataset):
    """Main Class for Image Folder loader."""

    def __init__(self, filenames, config, pt_nbr=1000, training=True, labels=None):
        """Init function."""

        self.filenames = filenames
        self.training = training
        self.pt_nbr = pt_nbr

        self.labels = labels
        if labels is None:
            raise Exception("Missing Labels Exception")

        self.config = config

    def __getitem__(self, index):
        """Get item."""

        # get the filename
        filename = self.filenames[index]

        # load the mesh and sample the points
        mesh = trimesh.load(filename)
        pts = trimesh.sample.sample_surface(mesh, self.pt_nbr)[0]

        # create features
        features = np.ones((pts.shape[0], 1))

        pts = pc_normalize(pts)
        if self.training:
            pts = jitter_point_cloud(pts)

        # create the tree
        tree = computeTree(pts, self.config)

        # get the label
        target = -1
        for label_id, label in enumerate(self.labels):
            if label in filename:
                target = label_id
        assert len(self.labels)==0 or target != -1

        # convert to torch
        pts = torch.from_numpy(pts).float().unsqueeze(0)
        target = torch.Tensor([target]).long().unsqueeze(0)
        features = torch.from_numpy(features).float().unsqueeze(0)

        # return data
        return pts, features, target, tree

    def __len__(self):
        """Length."""
        return len(self.filenames)

    def get_filename(self, id):
        """Get the filename."""
        return self.filenames[id]


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize", "-b", type=int, default=16)
    parser.add_argument("--nocuda", action="store_true")
    parser.add_argument("--npoints", type=int, default=1024)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--ntree", type=int, default=1)
    parser.add_argument("--savedir", type=str, default="./results")
    parser.add_argument("--rootdir", type=str, required=True)
    args = parser.parse_args()

    args.cuda = not args.nocuda

    if args.cuda:
        torch.backends.cudnn.benchmark=True

    # modelnet40
    labels = [
                "airplane", "bowl",     "desk",         "keyboard",     "person",
                "sofa",     "tv_stand", "bathtub",      "car",          "door",
                "lamp",     "piano",    "stairs",       "vase",         "bed",
                "chair",    "dresser",  "laptop",       "plant",        "stool",
                "wardrobe", "bench",    "cone",         "flower_pot",   "mantel",
                "radio",    "table",    "xbox",         "bookshelf",    "cup",
                "glass_box","monitor",  "range_hood",   "tent",         "bottle",
                "curtain",  "guitar",   "night_stand",  "sink",         "toilet",
                ]

    # parameters for training
    THREADS = 4
    N_LABELS = len(labels)
    epoch_nbr = 100
    input_channels = 1

    # define the save directory
    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    name = "modelnet40_bs{:02d}_pts{}_{}".format(args.batchsize, args.npoints, time_string)
    save_dir = os.path.join(args.savedir, name)

    
    print("Creating network")
    net = Net(input_channels=input_channels, output_channels=N_LABELS)
    if args.test:
        save_dir = args.savedir
        net.load_state_dict(torch.load(os.path.join(save_dir, "state_dict.pth")))
    if args.cuda:
        net.cuda()
    print("Number of parameters", count_parameters(net))


    print("Getting filenames...", end='')
    rootdir_train = os.path.join(args.rootdir, "train")
    rootdir_test = os.path.join(args.rootdir, "test")
    train_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(args.rootdir) for f in filenames if os.path.splitext(f)[1] == '.off' and "train" in dp]
    test_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(args.rootdir) for f in filenames if os.path.splitext(f)[1] == '.off' and "test" in dp]
    train_files.sort()
    test_files.sort()
    print("done")
    print("train files:", len(train_files))
    print("test files:", len(test_files))


    print("Preprocessing off files")
    # some files may have first line "OFF122.." instead of "OFF\n122..."
    for filename in tqdm(train_files, ncols=100):
        f = open(filename, 'r')
        line = f.readline()
        if line == "OFF\n":
            continue
        lines = ["OFF\n", line.split("OFF")[1]]+ list(f)
        f.close()
        f = open(filename, "w")
        f.write("".join(lines))
        f.close()
    for filename in tqdm(test_files, ncols=100):
        f = open(filename, 'r')
        line = f.readline()
        if line == "OFF\n":
            continue
        lines = ["OFF\n", line.split("OFF")[1]]+ list(f)
        f.close()
        f = open(filename, "w")
        f.write("".join(lines))
        f.close()

    print("Creating dataloaders...", end="")
    ds = PointCloudFileLists(train_files, config=net.config, pt_nbr=args.npoints, labels=labels)
    train_loader = torch.utils.data.DataLoader(ds, batch_size=args.batchsize, shuffle=True, num_workers=THREADS, collate_fn=tree_collate)
    ds_test = PointCloudFileLists(test_files, config=net.config, pt_nbr=args.npoints, training=False, labels=labels)
    test_loader = torch.utils.data.DataLoader(ds_test, batch_size=args.batchsize, shuffle=False, num_workers=THREADS, collate_fn=tree_collate)
    print("done")

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 60, 90])

    def apply(epoch, training=False):
        if training:
            t = tqdm(train_loader, desc="Epoch "+str(epoch), ncols=130)
        else:
            t = tqdm(test_loader, desc="  Test "+str(epoch), ncols=130)

        error = 0
        cm = np.zeros((N_LABELS, N_LABELS))

        for pts, features, targets, tree in t:

            if args.cuda:
                features = features.cuda()
                pts = pts.cuda()
                targets = targets.cuda()
                for l_id in range(len(tree)):
                    tree[l_id]["points"] = tree[l_id]["points"].cuda()

            if training:
                optimizer.zero_grad()

            outputs = net(features, pts, tree)
            targets = targets.view(-1)
            loss = F.cross_entropy(outputs, targets)

            if training:
                loss.backward()
                optimizer.step()

            output_np = np.argmax(outputs.cpu().detach().numpy(), axis=1)
            target_np = targets.cpu().numpy()

            cm_ = confusion_matrix(target_np.ravel(), output_np.ravel(), labels=list(range(N_LABELS)))
            cm += cm_
            error += loss.item()

            # scores
            oa = "{:.3f}".format(metrics.stats_overall_accuracy(cm))
            aa = "{:.3f}".format(metrics.stats_accuracy_per_class(cm)[0])
            aiou = "{:.3f}".format(metrics.stats_iou_per_class(cm)[0])
            aloss = "{:.3e}".format(error / cm.sum())

            t.set_postfix(OA=oa, AA=aa, AIOU=aiou, ALoss=aloss)
        return aloss, oa, aa, aiou


    if not args.test: # training mode
        os.makedirs(save_dir, exist_ok=True)
        f = open(os.path.join(save_dir, "logs.txt"), "w")
        for epoch in range(epoch_nbr):

            scheduler.step()

            net.train()
            train_aloss, train_oa, train_aa, train_aiou = apply(epoch, training=True)

            net.eval()
            with torch.no_grad():
                test_aloss, test_oa, test_aa, test_aiou = apply(epoch, training=False)


            # save network
            torch.save(net.state_dict(), os.path.join(save_dir, "state_dict.pth"))

            # write the logs
            f.write(str(epoch)+" ")
            f.write(train_aloss+" ")
            f.write(train_oa+" ")
            f.write(train_aa+" ")
            f.write(train_aiou+" ")
            f.write(test_aloss+" ")
            f.write(test_oa+" ")
            f.write(test_aa+" ")
            f.write(test_aiou+"\n")
            f.flush()

        f.close()
    else:
        net.eval()
        if args.ntree==1:
            with torch.no_grad():
                test_aloss, test_oa, test_aa, test_aiou = apply(0, training=False)
                # print("Loss", test_aloss)
                print("OA:", test_oa)
                print("AA:", test_aa)
                print("AIOU:", test_aiou)
        else:

            with torch.no_grad():

                cm = np.zeros((N_LABELS, N_LABELS))

                t = tqdm(range(len(ds_test)), ncols=120)
                for shape_id in t:
                    scores = np.zeros(N_LABELS)
                    batch = []
                    for tree_id in range(args.ntree):
                        batch.append(ds_test.__getitem__(shape_id))

                    pts, features, targets, tree = tree_collate(batch)

                    if args.cuda:
                        features = features.cuda()
                        pts = pts.cuda()
                        targets = targets.cuda()
                        for l_id in range(len(tree)):
                            tree[l_id]["points"] = tree[l_id]["points"].cuda()

                    outputs = net(features, pts, tree)
                    outputs = outputs.sum(dim=0, keepdim=True)
                    targets = targets.view(-1)[0]

                    output_np = np.argmax(outputs.cpu().detach().numpy(), axis=1)
                    target_np = targets.cpu().numpy()

                    cm_ = confusion_matrix(target_np.ravel(), output_np.ravel(), labels=list(range(N_LABELS)))
                    cm += cm_

                    # scores
                    oa = "{:.3f}".format(metrics.stats_overall_accuracy(cm))
                    aa = "{:.3f}".format(metrics.stats_accuracy_per_class(cm)[0])
                    aiou = "{:.3f}".format(metrics.stats_iou_per_class(cm)[0])

                    t.set_postfix(OA=oa, AA=aa, AIOU=aiou)
            
                print("OA:", oa)
                print("AA:", aa)
                print("AIOU:", aiou)



if __name__ == '__main__':
    main()