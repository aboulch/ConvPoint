# MODELNET40 Example with ConvPoint

# add the parent folder to the python path to access convpoint library
import sys
sys.path.append('../../')

import os
import argparse
from datetime import datetime
from tqdm import tqdm
import numpy as np
import math
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import BallTree

import torch
import torch.utils.data
import torch.nn.functional as F

import utils.metrics as metrics
import utils.data_utils as data_utils

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class PartNormalDataset():

    def __init__ (self, data, data_num, label, npoints, num_iter_per_shape=1, training=False):

        self.data = data
        self.data_num = data_num
        self.label = label
        self.npoints = npoints
        self.num_iter_per_shape = num_iter_per_shape

    def __getitem__(self, index):

        index = index//self.num_iter_per_shape

        npts = self.data_num[index]
        pts = self.data[index, :npts]
        choice = np.random.choice(npts, self.npoints, replace=True)
        
        pts = pts[choice]
        lbs = self.label[index][choice]
        features = torch.ones(pts.shape[0], 1).float()

        pts = torch.from_numpy(pts).float()
        lbs = torch.from_numpy(lbs).long()
        
        return pts, features, lbs, index

    def __len__(self):
        return self.data.shape[0] * self.num_iter_per_shape

def nearest_correspondance(pts_src, pts_dest, data_src):
    tree = BallTree(pts_src, leaf_size=2)
    _, indices = tree.query(pts_dest, k=1)
    indices = indices.ravel()
    data_dest = data_src[indices]
    return data_dest


def get_model(model_name,input_channels, output_channels):
    if model_name == "SegSmall":
        from networks.network_seg import SegSmall as Net
    return Net(input_channels, output_channels)


def train(args):

    THREADS = 4
    USE_CUDA = True
    N_CLASSES = 50
    EPOCHS = 200
    MILESTONES = [60,120]

    shapenet_labels = [['Airplane',4],
        ['Bag',2],
        ['Cap',2],
        ['Car',4],
        ['Chair',4],
        ['Earphone',3],
        ['Guitar',3],
        ['Knife',2],
        ['Lamp',4],
        ['Laptop',2],
        ['Motorbike',6],
        ['Mug',2],
        ['Pistol',3],
        ['Rocket',3],
        ['Skateboard',3],
        ['Table',3],]
    category_range = []
    count = 0
    for element in shapenet_labels:
        part_start = count
        count += element[1]
        part_end = count
        category_range.append([part_start, part_end])

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

    print("Computing class weights (if needed, 1 otherwise)...")
    if args.weighted:
        frequences = []
        for i in range(len(shapenet_labels)):
            frequences.append((labels == i).sum())
        frequences = np.array(frequences)
        frequences = frequences.mean() / frequences
    else:
        frequences = [1 for _ in range(len(shapenet_labels))]
    weights = torch.FloatTensor(frequences)
    if USE_CUDA:
        weights = weights.cuda()
    print("Done")


    print("Creating network...")
    net = get_model(args.model, input_channels=1, output_channels=N_CLASSES)
    net.cuda()
    print("parameters", count_parameters(net))


    ds = PartNormalDataset(data_train, data_num_train, label_train, npoints=args.npoints)
    train_loader = torch.utils.data.DataLoader(ds, batch_size=args.batchsize, shuffle=True,
                                            num_workers=THREADS
                                            )
    
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, MILESTONES)

    # create the model folder
    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    root_folder = os.path.join(args.savedir, "{}_b{}_pts{}_weighted{}_{}".format(args.model,args.batchsize, args.npoints, args.weighted, time_string))
    os.makedirs(root_folder, exist_ok=True)

    # create the log file
    logs = open(os.path.join(root_folder, "log.txt"), "w")
    for epoch in range(EPOCHS):
        scheduler.step()
        cm = np.zeros((N_CLASSES, N_CLASSES))
        t = tqdm(train_loader, ncols=120, desc="Epoch {}".format(epoch))
        for pts, features, seg, indices in t:

            if USE_CUDA:
                features = features.cuda()
                pts = pts.cuda()
                seg = seg.cuda()

            optimizer.zero_grad()
            outputs = net(features, pts)

            # loss =  F.cross_entropy(outputs.view(-1, N_CLASSES), seg.view(-1))

            loss = 0
            for i in range(pts.size(0)):
                # get the number of part for the shape
                object_label = labels[indices[i]]
                part_start, part_end = category_range[object_label]
                part_nbr = part_end - part_start
                loss = loss + weights[object_label] * F.cross_entropy(outputs[i,:,part_start:part_end].view(-1, part_nbr), seg[i].view(-1)-part_start)

            loss.backward()
            optimizer.step()
            
            outputs_np = outputs.cpu().detach().numpy()
            for i in range(pts.size(0)):
                # get the number of part for the shape
                object_label = labels[indices[i]]
                part_start, part_end = category_range[object_label]
                part_nbr = part_end - part_start
                outputs_np[i,:,:part_start] = -1e7
                outputs_np[i,:,part_end:] = -1e7
                
            output_np = np.argmax(outputs_np, axis=2).copy()
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


def IoU_from_confusions(confusions):
    """
    Computes IoU from confusion matrices.
    :param confusions: ([..., n_c, n_c] np.int32). Can be any dimension, the confusion matrices should be described by
    the last axes. n_c = number of classes
    :param ignore_unclassified: (bool). True if the the first class should be ignored in the results
    :return: ([..., n_c] np.float32) IoU score
    """

    # Compute TP, FP, FN. This assume that the second to last axis counts the truths (like the first axis of a
    # confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
    TP = np.diagonal(confusions, axis1=-2, axis2=-1)
    TP_plus_FN = np.sum(confusions, axis=-1)
    TP_plus_FP = np.sum(confusions, axis=-2)

    # Compute IoU
    IoU = TP / (TP_plus_FP + TP_plus_FN - TP + 1e-6)

    # Compute mIoU with only the actual classes
    mask = TP_plus_FN < 1e-3
    counts = np.sum(1 - mask, axis=-1, keepdims=True)
    mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)

    # If class is absent, place mIoU in place of 0 IoU to get the actual mean later
    IoU += mask * mIoU

    return IoU


def test(args):
    THREADS = 4
    USE_CUDA = True
    N_CLASSES = 50

    args.data_folder = os.path.join(args.rootdir, "test_data")

    # create the output folders
    output_folder = os.path.join(args.savedir,'_predictions2')
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

    # net = Net(input_channels=1, output_channels=N_CLASSES)
    net = get_model(args.model, input_channels=1, output_channels=N_CLASSES)
    net.load_state_dict(torch.load(os.path.join(args.savedir, "state_dict.pth")))
    net.cuda()
    net.eval()

    ds = PartNormalDataset(data, data_num, label_test, npoints=args.npoints, num_iter_per_shape=args.ntree)
    test_loader = torch.utils.data.DataLoader(ds, batch_size=args.batchsize, shuffle=False,
                                            num_workers=THREADS
                                            )
    shapenet_labels = [['Airplane',4],
        ['Bag',2],
        ['Cap',2],
        ['Car',4],
        ['Chair',4],
        ['Earphone',3],
        ['Guitar',3],
        ['Knife',2],
        ['Lamp',4],
        ['Laptop',2],
        ['Motorbike',6],
        ['Mug',2],
        ['Pistol',3],
        ['Rocket',3],
        ['Skateboard',3],
        ['Table',3],]





    cm = np.zeros((N_CLASSES, N_CLASSES))
    t = tqdm(test_loader, ncols=120)
    Confs = []

    predictions = [None for _ in range(data.shape[0])]
    predictions_max = [[] for _ in range(data.shape[0])]
    with torch.no_grad():

        for pts, features, seg, indices in t:

            if USE_CUDA:
                features = features.cuda()
                pts = pts.cuda()

            outputs = net(features, pts)

            indices = np.int32(indices.numpy())
            outputs = np.float32(outputs.cpu().numpy())

            # save results
            for i in range(pts.size(0)):

                # shape id
                shape_id = indices[i]

                # pts_src
                pts_src = pts[i].cpu().numpy()

                # pts_dest
                point_num = data_num[shape_id]
                pts_dest = data[shape_id]
                pts_dest = pts_dest[:point_num]

                # get the number of part for the shape
                object_label = label[indices[i]]
                category = category_list[object_label][0]
                part_start, part_end = category_range[category]
                part_nbr = part_end - part_start

                # get the segmentation correspongin to part range
                seg_ = outputs[i][:,part_start:part_end]

                # interpolate to original points
                seg_ = nearest_correspondance(pts_src, pts_dest, seg_)

                if predictions[shape_id] is None:
                    predictions[shape_id] = seg_
                else:
                    predictions[shape_id] += seg_

                predictions_max[shape_id].append(seg_)
            
    for i in range(len(predictions)):
        a = np.stack(predictions_max[i], axis=1)
        a = np.argmax(a, axis=2)
        a = np.apply_along_axis(np.bincount,1,a, minlength=6)
        predictions_max[i] = np.argmax(a, axis=1)

    # compute labels
    for i in range(len(predictions)):
        predictions[i] = np.argmax(predictions[i], axis=1)
    

    def scores_from_predictions(predictions):
        
        shape_ious = {cat[0]:[] for cat in category_list}
        for shape_id, prediction in enumerate(predictions):

            segp = prediction
            cat = label[shape_id]
            category = category_list[cat][0]
            part_start, part_end = category_range[category]
            part_nbr = part_end - part_start
            point_num = data_num[shape_id]
            segl = label_test[shape_id][:point_num] - part_start

            part_ious = [0.0 for _ in range(part_nbr)]
            for l in range(part_nbr):
                if (np.sum(segl==l) == 0) and (np.sum(segp==l) == 0): # part is not present, no prediction as well
                    part_ious[l] = 1.0
                else:
                    part_ious[l] = np.sum((segl==l) & (segp==l)) / float(np.sum((segl==l) | (segp==l)))
            shape_ious[category].append(np.mean(part_ious))
        
        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        print(len(all_shape_ious))
        mean_shape_ious = np.mean(list(shape_ious.values()))
        for cat in sorted(shape_ious.keys()):
            print('eval mIoU of %s:\t %f'%(cat, shape_ious[cat]))
        print('eval mean mIoU: %f' % (mean_shape_ious))
        print('eval mean mIoU (all shapes): %f' % (np.mean(all_shape_ious)))

    scores_from_predictions(predictions)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--ply", action="store_true", help="save ply files (test mode)")
    parser.add_argument("--savedir", default="results/", type=str)
    parser.add_argument("--rootdir", type=str, required=True)
    parser.add_argument("--batchsize", "-b", default=16, type=int)
    parser.add_argument("--ntree", default=1, type=int)
    parser.add_argument("--npoints", default=2500, type=int)
    parser.add_argument("--weighted", action="store_true")
    parser.add_argument("--model", default="SegSmall", type=str)
    args = parser.parse_args()

    args.filelist = os.path.join(args.rootdir, "train_files.txt")
    args.filelist_val = os.path.join(args.rootdir,"test_files.txt")
    args.category = os.path.join(args.rootdir, "categories.txt")


    if args.test:
        # if args.ntree == 1:
        #     test(args)
        # else:
        #     test_multiple(args)
        test(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
