# MODELNET40 Example with ConvPoint

# add the parent folder to the python path to access convpoint library
import sys
sys.path.append('../../')

# other imports
import numpy as np
import random
import os
from tqdm import tqdm
import argparse
from datetime import datetime
from sklearn.metrics import confusion_matrix
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import utils.metrics as metrics


def get_data(rootdir, files):
#read all file in rootdir
    train_filenames = []
    for line in open(os.path.join(rootdir, files), "r"):
        line = line.split("\n")[0]
        line = os.path.basename(line)
        train_filenames.append(os.path.join(rootdir, line))
#
    data = []
    labels = []
    for filename in train_filenames:
        f = h5py.File(filename, 'r')
        data.append(f["data"])
        labels.append(f["label"])

    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)

    return data, labels

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


class PointCloudFileLists(torch.utils.data.Dataset):
    """Main Class for Image Folder loader."""

    def __init__(self, data, labels, pt_nbr=1024, training=True, num_iter_per_shape=1):
        """Init function."""

        self.data = data
        self.labels = labels
        self.training = training
        self.pt_nbr = pt_nbr
        self.num_iter_per_shape = num_iter_per_shape

    def __getitem__(self, index):
        """Get item."""


        index_ = index//self.num_iter_per_shape

        # get the filename
        pts = self.data[index_]
        target = self.labels[index_]


        indices = np.random.choice(pts.shape[0], self.pt_nbr)
        pts = pts[indices]

        # create features
        features = np.ones((pts.shape[0], 1))

        pts = pc_normalize(pts)

        return pts.astype(np.float32), features.astype(np.float32), int(target), index_

    def __len__(self):
        """Length."""
        return self.data.shape[0] * self.num_iter_per_shape



def get_model(model_name, input_channels, output_channels):
    if model_name == "ModelNet40":
        from networks.network_classif import ModelNet40 as Net
    return Net(input_channels, output_channels)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize", "-b", type=int, default=16)
    parser.add_argument("--nocuda", action="store_true")
    parser.add_argument("--npoints", type=int, default=1024)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--ntree", type=int, default=1)
    parser.add_argument("--savedir", type=str, default="./results")
    parser.add_argument("--rootdir", type=str, required=True)
    parser.add_argument("--model", type=str, default="ModelNet40")
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
    name = "{}_bs{:02d}_pts{}_{}".format(args.model, args.batchsize, args.npoints, time_string)
    save_dir = os.path.join(args.savedir, name)

    
    print("Creating network")
    net = get_model(args.model, input_channels=input_channels, output_channels=N_LABELS)
    if args.test:
        save_dir = args.savedir
        net.load_state_dict(torch.load(os.path.join(save_dir, "state_dict.pth")))
    if args.cuda:
        net.cuda()
    print("Number of parameters", count_parameters(net))


    print("Getting train files...")
    train_data, train_labels = get_data(args.rootdir, "train_files.txt")
    print(train_data.shape, train_labels.shape)
    print("Getting test files...")
    test_data, test_labels = get_data(args.rootdir, "test_files.txt")
    print(test_data.shape, test_labels.shape)
    print("done")


    print("Creating dataloaders...", end="")
    ds = PointCloudFileLists(train_data, train_labels, pt_nbr=args.npoints)
    train_loader = torch.utils.data.DataLoader(ds, batch_size=args.batchsize, shuffle=True, num_workers=THREADS)
    ds_test = PointCloudFileLists(test_data, test_labels, pt_nbr=args.npoints, training=False, num_iter_per_shape=args.ntree)
    test_loader = torch.utils.data.DataLoader(ds_test, batch_size=args.batchsize, shuffle=False, num_workers=THREADS)
    print("done")

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30,60,90])

    def apply(epoch, training=False):
        error = 0
        cm = np.zeros((N_LABELS, N_LABELS))


        if training:
            t = tqdm(train_loader, desc="Epoch "+str(epoch), ncols=130)
            for pts, features, targets, indices in t:
                if args.cuda:
                    features = features.cuda()
                    pts = pts.cuda()
                    targets = targets.cuda()

                optimizer.zero_grad()

                outputs = net(features, pts)
                targets = targets.view(-1)
                loss = F.cross_entropy(outputs, targets)

                loss.backward()
                optimizer.step()

                output_np = np.argmax(outputs.cpu().detach().numpy(), axis=1)
                target_np = targets.cpu().numpy()

                cm_ = confusion_matrix(target_np.ravel(), output_np.ravel(), labels=list(range(N_LABELS)))
                cm += cm_
                error += loss.item()

                # scores
                oa = "{:.5f}".format(metrics.stats_overall_accuracy(cm))
                aa = "{:.5f}".format(metrics.stats_accuracy_per_class(cm)[0])
                aiou = "{:.5f}".format(metrics.stats_iou_per_class(cm)[0])
                aloss = "{:.5e}".format(error / cm.sum())

                t.set_postfix(OA=oa, AA=aa, AIOU=aiou, ALoss=aloss)

        else:
            predictions = np.zeros((test_data.shape[0], N_LABELS), dtype=float)
            t = tqdm(test_loader, desc="  Test "+str(epoch), ncols=100)
            for pts, features, targets, indices in t:
                if args.cuda:
                    features = features.cuda()
                    pts = pts.cuda()
                    targets = targets.cuda()

                outputs = net(features, pts)
                targets = targets.view(-1)
                loss = F.cross_entropy(outputs, targets)

                outputs_np = outputs.cpu().detach().numpy()
                for i in range(indices.size(0)):
                    predictions[indices[i]] += outputs_np[i]
                    # l_ = np.argmax(outputs_np[i])
                    # predictions[indices[i],l_] += 1


                error += loss.item()

                if args.ntree == 1:
                    pred_labels = np.argmax(outputs_np, axis=1)
                    cm_ = confusion_matrix(targets.cpu().numpy(), pred_labels, labels=list(range(N_LABELS)))
                    cm += cm_

                    # scores
                    oa = "{:.5f}".format(metrics.stats_overall_accuracy(cm))
                    aa = "{:.5f}".format(metrics.stats_accuracy_per_class(cm)[0])
                    aiou = "{:.5f}".format(metrics.stats_iou_per_class(cm)[0])
                    aloss = "{:.5e}".format(error / cm.sum())

                    t.set_postfix(OA=oa, AA=aa, AIOU=aiou, ALoss=aloss)


            predictions = np.argmax(predictions, axis=1)
            cm = confusion_matrix(test_labels, predictions, labels=list(range(N_LABELS)))

            oa = "{:.5f}".format(metrics.stats_overall_accuracy(cm))
            aa = "{:.5f}".format(metrics.stats_accuracy_per_class(cm)[0])
            aiou = "{:.5f}".format(metrics.stats_iou_per_class(cm)[0])
            aloss = "{:.5e}".format(error / cm.sum())

            print("Predictions","loss",aloss, "OA",oa, "AA", aa, "IOU", aiou)

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

        with torch.no_grad():
            test_aloss, test_oa, test_aa, test_aiou = apply(0, training=False)
            # print("Loss", test_aloss)
            print("OA:", test_oa)
            print("AA:", test_aa)
            print("AIOU:", test_aiou)



if __name__ == '__main__':
    main()
