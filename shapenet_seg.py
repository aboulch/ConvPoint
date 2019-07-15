import os
import argparse
from datetime import datetime
from tqdm import tqdm
import numpy as np
import math

import torch
import torch.utils.data
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix
from sklearn.neighbors import BallTree

import metrics
from network_seg import NetShapeNet as Net

import utils.data_utils as data_utils

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# in pointNet2 ==> 2500 points
class PartNormalDataset():

    def __init__ (self, data, data_num, label, npoints, shape_labels=None):

        self.data = data
        self.data_num = data_num
        self.label = label
        self.npoints = npoints
        self.shape_labels= shape_labels

    def __getitem__(self, index):

        npts = self.data_num[index]
        pts = self.data[index, :npts]
        choice = np.random.choice(npts, self.npoints, replace=True)
        
        pts = pts[choice]
        lbs = self.label[index][choice]
        features = torch.ones(pts.shape[0], 1).float()

        pts = torch.from_numpy(pts).float()
        lbs = torch.from_numpy(lbs).long()

        if self.shape_labels is not None:
            sh_lb = torch.FloatTensor([self.shape_labels[index]])
            return pts, features, lbs, sh_lb
        else:
            return pts, features, lbs

    def __len__(self):
        return self.data.shape[0]


def nearest_correspondance(pts_src, pts_dest, data_src):
    tree = BallTree(pts_src, leaf_size=2)
    _, indices = tree.query(pts_dest, k=1)
    indices = indices.ravel()
    data_dest = data_src[indices]
    return data_dest


def train(args):

    # Prepare inputs
    print('{}-Preparing datasets...'.format(datetime.now()))
    is_list_of_h5_list = not data_utils.is_h5_list(args.filelist)
    if is_list_of_h5_list:
        seg_list = data_utils.load_seg_list(args.filelist)
        seg_list_idx = 0
        filelist_train = seg_list[seg_list_idx]
        seg_list_idx = seg_list_idx + 1
    else:
        filelist_train = args.filelist
    data_train, labels, data_num_train, label_train, _ = data_utils.load_seg(filelist_train)
    print("Done", data_train.shape)

    THREADS = 4
    BATCH_SIZE = args.batchsize
    USE_CUDA = True
    N_CLASSES = 50
    EPOCHS = 150
    MILESTONES = [60,120]

    print("Creating network...")
    net = Net(input_channels=1, output_channels=N_CLASSES)
    net.cuda()
    print("parameters", count_parameters(net))


    ds = PartNormalDataset(data_train, data_num_train, label_train, npoints=args.npoints, shape_labels=labels)
    train_loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                                            num_workers=THREADS
                                            )
    
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, MILESTONES)

    # create the model folder
    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    root_folder = os.path.join(args.savedir, "Net_b{}_pts{}_{}".format(args.batchsize, args.npoints, time_string))
    os.makedirs(root_folder, exist_ok=True)

    # create the log file
    logs = open(os.path.join(root_folder, "log.txt"), "w")
    for epoch in range(EPOCHS):
        scheduler.step()
        cm = np.zeros((N_CLASSES, N_CLASSES))
        t = tqdm(train_loader, ncols=120, desc="Epoch {}".format(epoch))
        for pts, features, seg, labels in t:

            if USE_CUDA:
                features = features.cuda()
                pts = pts.cuda()
                seg = seg.cuda()

            optimizer.zero_grad()
            outputs = net(features, pts)
            loss =  F.cross_entropy(outputs.view(-1, N_CLASSES), seg.view(-1))
            loss.backward()
            optimizer.step()

            output_np = np.argmax(outputs.cpu().detach().numpy(), axis=2).copy()
            target_np = seg.cpu().numpy().copy()

            cm_ = confusion_matrix(target_np.ravel(), output_np.ravel(), labels=list(range(N_CLASSES)))
            cm += cm_

            oa = "{:.3f}".format(metrics.stats_overall_accuracy(cm))
            aa = "{:.3f}".format(metrics.stats_accuracy_per_class(cm)[0])

            t.set_postfix(OA=oa, AA=aa)

        # save the model
        torch.save(net.state_dict(), os.path.join(root_folder, "state_dict.pth"))

        # write the logs
        logs.write("{} {} {} \n".format(epoch, oa, aa))
        logs.flush()


    logs.close()

def test(args):
    THREADS = 4
    BATCH_SIZE = args.batchsize
    USE_CUDA = True
    N_CLASSES = 50

    args.data_folder = os.path.join(args.rootdir, "test_data")

    # create the output folders
    output_folder = os.path.join(args.savedir,'_predictions')
    category_list = [(category, int(label_num)) for (category, label_num) in
                     [line.split() for line in open(args.category, 'r')]]
    offset = 0
    category_range = dict()
    for category, category_label_seg_max in category_list:
        category_range[category] = (offset, offset + category_label_seg_max)
        offset = offset + category_label_seg_max
        folder = os.path.join(output_folder, category)
        if not os.path.exists(folder):
            os.makedirs(folder)

    
    input_filelist = []
    output_filelist = []
    output_ply_filelist = []
    for category in sorted(os.listdir(args.data_folder)):
        data_category_folder = os.path.join(args.data_folder, category)
        for filename in sorted(os.listdir(data_category_folder)):
            input_filelist.append(os.path.join(args.data_folder, category, filename))
            output_filelist.append(os.path.join(output_folder, category, filename[0:-3] + 'seg'))
            output_ply_filelist.append(os.path.join(output_folder + '_ply', category, filename[0:-3] + 'ply'))

    # Prepare inputs
    print('{}-Preparing datasets...'.format(datetime.now()))
    data, label, data_num, label_test, _ = data_utils.load_seg(args.filelist_val) # no segmentation labels

    net = Net(input_channels=1, output_channels=N_CLASSES)
    net.load_state_dict(torch.load(os.path.join(args.savedir, "state_dict.pth")))
    net.cuda()
    net.eval()

    ds = PartNormalDataset(data, data_num, label_test, npoints=args.npoints)
    test_loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                                            num_workers=THREADS
                                            )


    cm = np.zeros((N_CLASSES, N_CLASSES))
    t = tqdm(test_loader, ncols=120)
    with torch.no_grad():
        count = 0
        for pts, features, seg in t:

            if USE_CUDA:
                features = features.cuda()
                pts = pts.cuda()

            outputs = net(features, pts)

            # save results
            for i in range(pts.size(0)):
                # pts_src
                pts_src = pts[i].cpu().numpy()

                # pts_dest
                point_num = data_num[count+i]
                pts_dest = data[count+i]
                pts_dest = pts_dest[:point_num]

                object_label = label[count+i]
                category = category_list[object_label][0]
                label_start, label_end = category_range[category]

                seg_ = outputs[i][:,label_start:label_end].cpu().numpy()
                seg_ = np.argmax(seg_, axis=1)
                seg_ = nearest_correspondance(pts_src, pts_dest, seg_)

                # save labels
                np.savetxt(output_filelist[count+i], seg_, fmt="%i")

                if args.ply:
                    data_utils.save_ply_property(pts_dest, seg_, 6, output_ply_filelist[count+i])
            count += pts.size(0)

            output_np = np.argmax(outputs.cpu().detach().numpy(), axis=2).copy()
            target_np = seg.cpu().numpy().copy()

            cm_ = confusion_matrix(target_np.ravel(), output_np.ravel(), labels=list(range(N_CLASSES)))
            cm += cm_

            oa = "{:.3f}".format(metrics.stats_overall_accuracy(cm))
            aa = "{:.3f}".format(metrics.stats_accuracy_per_class(cm)[0])

            t.set_postfix(OA=oa, AA=aa)


def test_multiple(args):
    THREADS = 4
    BATCH_SIZE = args.batchsize
    USE_CUDA = True
    N_CLASSES = 50


    args.data_folder = os.path.join(args.rootdir, "test_data")

    # create the output folders
    output_folder = os.path.join(args.savedir, '_predictions_multi_{}'.format(args.ntree))
    category_list = [(category, int(label_num)) for (category, label_num) in
                     [line.split() for line in open(args.category, 'r')]]
    offset = 0
    category_range = dict()
    for category, category_label_seg_max in category_list:
        category_range[category] = (offset, offset + category_label_seg_max)
        offset = offset + category_label_seg_max
        folder = os.path.join(output_folder, category)
        if not os.path.exists(folder):
            os.makedirs(folder)

    input_filelist = []
    output_filelist = []
    output_ply_filelist = []
    for category in sorted(os.listdir(args.data_folder)):
        data_category_folder = os.path.join(args.data_folder, category)
        for filename in sorted(os.listdir(data_category_folder)):
            input_filelist.append(os.path.join(args.data_folder, category, filename))
            output_filelist.append(os.path.join(output_folder, category, filename[0:-3] + 'seg'))
            output_ply_filelist.append(os.path.join(output_folder + '_ply', category, filename[0:-3] + 'ply'))

    # Prepare inputs
    print('{}-Preparing datasets...'.format(datetime.now()))
    data, label, data_num, label_test, _ = data_utils.load_seg(args.filelist_val) # no segmentation labels

    net = Net(input_channels=1, output_channels=N_CLASSES)
    net.load_state_dict(torch.load(os.path.join(args.savedir, "state_dict.pth")))
    net.cuda()
    net.eval()

    ds = PartNormalDataset(data, data_num, label_test, npoints=args.npoints)
    test_loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                                            num_workers=THREADS
                                            )


    cm = np.zeros((N_CLASSES, N_CLASSES))
    t = tqdm(test_loader, ncols=120)
    with torch.no_grad():
        count = 0

        for shape_id in tqdm(range(len(ds)), ncols=120):

            segmentation_ = None

            batches = []

            if args.ntree <= args.batchsize:

                batch = []
                for tree_id in range(args.ntree):
                    batch.append(ds.__getitem__(shape_id))
                batches.append(batch)
            else:
                for i in range(math.ceil(args.ntree / args.batchsize)):
                    bs = min(args.batchsize, args.ntree - i*args.batchsize)
                    batch = []
                    for tree_id in range(bs):
                        batch.append(ds.__getitem__(shape_id))
                    batches.append(batch)

            for batch in batches:

                pts, features, seg = torch.utils.data._utils.collate.default_collate(batch)
                if USE_CUDA:
                    features = features.cuda()
                    pts = pts.cuda()

                outputs = net(features, pts)

                for i in range(pts.size(0)):
                    pts_src = pts[i].cpu().numpy()

                    # pts_dest
                    point_num = data_num[count]
                    pts_dest = data[count]
                    pts_dest = pts_dest[:point_num]

                    object_label = label[count]
                    category = category_list[object_label][0]
                    label_start, label_end = category_range[category]

                    seg_ = outputs[i][:,label_start:label_end].cpu().numpy()
                    seg_ = nearest_correspondance(pts_src, pts_dest, seg_)

                    if segmentation_ is None:
                        segmentation_ = seg_
                    else:
                        segmentation_ += seg_

            segmentation_ = np.argmax(segmentation_, axis=1)

            # save labels
            np.savetxt(output_filelist[count], segmentation_, fmt="%i")

            if args.ply:
                    data_utils.save_ply_property(pts_dest, segmentation_, 6, output_ply_filelist[count])

            count += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--ply", action="store_true", help="save ply files (test mode)")
    parser.add_argument("--savedir", default="results/", type=str)
    parser.add_argument("--rootdir", type=str, required=True)
    parser.add_argument("--batchsize", "-b", default=16, type=int)
    parser.add_argument("--ntree", default=1, type=int)
    parser.add_argument("--npoints", default=2500, type=int)
    args = parser.parse_args()

    args.filelist = os.path.join(args.rootdir, "train_files.txt")
    args.filelist_val = os.path.join(args.rootdir,"test_files.txt")
    args.category = os.path.join(args.rootdir, "categories.txt")


    if args.test:
        if args.ntree == 1:
            test(args)
        else:
            test_multiple(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
