import argparse
import os
from datetime import datetime
from network_seg import NetS3DIS as Net
import numpy as np

import torch
import torch.utils.data
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image

from tqdm import tqdm
import random
import metrics
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import BallTree

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# wrap blue / green
def wblue(str):
    return bcolors.OKBLUE+str+bcolors.ENDC
def wgreen(str):
    return bcolors.OKGREEN+str+bcolors.ENDC


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def nearest_correspondance(pts_src, pts_dest, data_src, K=1):
    tree = BallTree(pts_src, leaf_size=2)
    _, indices = tree.query(pts_dest, k=K)
    if K==1:
        indices = indices.ravel()
        data_dest = data_src[indices]
    else:
        data_dest = data_src[indices].mean(1)
    return data_dest

def rotate_point_cloud_z(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, sinval, 0],
                                [-sinval, cosval, 0],
                                [0, 0, 1],])
    return np.dot(batch_data, rotation_matrix)



# Part dataset only for training / validation
class PartDatasetTrainVal():

    def __init__ (self, filelist, folder,
                    training=False, 
                    block_size=2,
                    npoints = 4096,
                    iteration_number = None,):

        self.training = training
        self.filelist = filelist
        self.folder = folder
        self.bs = block_size
        
        self.npoints = npoints
        self.iterations = iteration_number
        self.verbose = False
        self.number_of_run = 10


        self.transform = transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4)

    def __getitem__(self, index):

        folder = self.folder
        if self.training:
            index = random.randint(0, len(self.filelist)-1)
            dataset = self.filelist[index]
        else:
            dataset = self.filelist[index//self.number_of_run]

        # load data
        filename_data = os.path.join(folder, dataset, 'xyzrgb.npy')
        if self.verbose:
            print('{}-Loading {}...'.format(datetime.now(), filename_data))
        xyzrgb = np.load(filename_data)

        # load labels
        filename_labels = os.path.join(folder, dataset, 'label.npy')
        if self.verbose:
            print('{}-Loading {}...'.format(datetime.now(), filename_labels))
        labels = np.load(filename_labels).astype(int).flatten()

        # pick a random point
        pt_id = random.randint(0, xyzrgb.shape[0]-1)
        pt = xyzrgb[pt_id, :3]

        mask_x = np.logical_and(xyzrgb[:,0]<pt[0]+self.bs/2, xyzrgb[:,0]>pt[0]-self.bs/2)
        mask_y = np.logical_and(xyzrgb[:,1]<pt[1]+self.bs/2, xyzrgb[:,1]>pt[1]-self.bs/2)
        mask = np.logical_and(mask_x, mask_y)
        pts = xyzrgb[mask]
        lbs = labels[mask]


        choice = np.random.choice(pts.shape[0], self.npoints, replace=True)
        pts = pts[choice]
        lbs = lbs[choice]

        features = pts[:,3:]
        if self.training:
            features = features.astype(np.uint8)
            features = np.array(self.transform( Image.fromarray(np.expand_dims(features, 0)) ))
            features = np.squeeze(features, 0)
        
        features = features.astype(np.float32)
        features = features / 255 - 0.5

        pts = pts[:,:3]
        
        if self.training:
            pts = rotate_point_cloud_z(pts)
    
        pts = torch.from_numpy(pts).float()
        fts = torch.from_numpy(features).float()
        lbs = torch.from_numpy(lbs).long()

        return pts, fts, lbs

    def __len__(self):
        if self.iterations is None:
            return len(self.filelist) * self.number_of_run
        else:
            return self.iterations

# Part dataset only for testing
class PartDatasetTest():


    def compute_mask(self, pt, bs):
        # build the mask
        mask_x = np.logical_and(self.xyzrgb[:,0]<pt[0]+bs/2, self.xyzrgb[:,0]>pt[0]-bs/2)
        mask_y = np.logical_and(self.xyzrgb[:,1]<pt[1]+bs/2, self.xyzrgb[:,1]>pt[1]-bs/2)
        mask = np.logical_and(mask_x, mask_y)
        return mask

    def __init__ (self, filename, folder,
                    block_size=2,
                    npoints = 4096,
                    min_pick_per_point = 1):

        self.folder = folder
        self.bs = block_size
        self.npoints = npoints
        self.verbose = False
        self.min_pick_per_point = min_pick_per_point

        # load data
        self.filename = filename
        filename_data = os.path.join(folder, self.filename, 'xyzrgb.npy')
        if self.verbose:
            print('{}-Loading {}...'.format(datetime.now(), filename_data))
        self.xyzrgb = np.load(filename_data)
        filename_labels = os.path.join(folder, self.filename, 'label.npy')
        if self.verbose:
            print('{}-Loading {}...'.format(datetime.now(), filename_labels))
        self.labels = np.load(filename_labels).astype(int).flatten()

        # create the list of points
        # ensures that each point is seen at least min_pick_per_points
        current_pick_min = 0
        count = np.zeros(self.xyzrgb.shape[0], dtype=int)
        self.pts = []
        while(True):

            # check for points that can be used as a seed
            possible_points_ids = np.where(count==current_pick_min)[0]
            if self.verbose:
                print(current_pick_min, self.min_pick_per_point, possible_points_ids.shape[0],"/",self.xyzrgb.shape[0])
            
            # if no points: update the current indice
            if(possible_points_ids.shape[0]==0):
                current_pick_min += 1
                if current_pick_min == self.min_pick_per_point:
                    break
                continue
            
            # pick a point
            pt_id = random.randint(0, possible_points_ids.shape[0]-1)
            pt = self.xyzrgb[possible_points_ids[pt_id]]
            mask = self.compute_mask(pt, self.bs)
            count[mask] += 1   

            # save the point 
            self.pts.append(pt)
        

    def __getitem__(self, index):

        # get the data
        mask = self.compute_mask(self.pts[index], self.bs)
        pts = self.xyzrgb[mask]
        lbs = self.labels[mask]

        # choose right number of points
        choice = np.random.choice(pts.shape[0], self.npoints, replace=True)
        pts = pts[choice]

        # labels will indices in the original point cloud
        lbs = np.where(mask)[0][choice]

        features = pts[:,3:] / 255 - 0.5
        pts = pts[:,:3].copy()

        # convert to torch
        pts = torch.from_numpy(pts).float()
        fts = torch.from_numpy(features).float()
        lbs = torch.from_numpy(lbs).long()

        return pts, fts, lbs

    def __len__(self):
        return len(self.pts)


def train(args, flist_train, flist_test):
    N_CLASSES = 13


    # create the network
    print("Creating network...")
    net = Net(input_channels=3, output_channels=N_CLASSES)
    net.cuda()
    print("parameters", count_parameters(net))


    print("Creating dataloader and optimizer...")
    ds = PartDatasetTrainVal(flist_train, args.rootdir,
                             training=True, block_size=args.blocksize,
                             npoints=args.npoints,iteration_number=args.batchsize*args.iter)
    train_loader = torch.utils.data.DataLoader(ds, batch_size=args.batchsize, shuffle=True,
                                        num_workers=args.threads
                                        )

    ds_val = PartDatasetTrainVal(flist_test, args.rootdir,
                             training=False, block_size=args.blocksize,
                             npoints=args.npoints)
    test_loader = torch.utils.data.DataLoader(ds_val, batch_size=args.batchsize, shuffle=False,
                                        num_workers=args.threads
                                        )

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    print("done")

    # create the root folder
    print("Creating results folder")
    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    root_folder = os.path.join(args.savedir, "s3dis_area{}_{}_{}".format(args.area, args.npoints, time_string))
    os.makedirs(root_folder, exist_ok=True)
    print("done at", root_folder)
    
    # create the log file
    logs = open(os.path.join(root_folder, "log.txt"), "w")


    weights = torch.ones(N_CLASSES).float().cuda()
    # iterate over epochs
    for epoch in range(100):

        #######
        # training
        net.train()

        # if epoch==0:
        #     weights = torch.ones(N_CLASSES).float().cuda()
        # else:
        #     # use the previous epoch to define next weights
        #     w_tmp = cm.sum(axis=1)
        #     w_med = np.median(w_tmp[w_tmp>0])
        #     w_tmp[w_tmp==0] = 1
        #     w_tmp = w_med/w_tmp
        #     if epoch==1: # define from previous
        #         weights = torch.from_numpy(w_tmp).float().cuda()
        #     else:
        #         weights = 0.9*weights + 0.1*torch.from_numpy(w_tmp).float().cuda()


        train_loss = 0
        cm = np.zeros((N_CLASSES, N_CLASSES))
        t = tqdm(train_loader, ncols=100, desc="Epoch {}".format(epoch))
        for pts, features, seg in t:

            features = features.cuda()
            pts = pts.cuda()
            seg = seg.cuda()
            
            optimizer.zero_grad()
            outputs = net(features, pts)
            loss =  F.cross_entropy(outputs.view(-1, N_CLASSES), seg.view(-1), weight=weights)
            loss.backward()
            optimizer.step()

            output_np = np.argmax(outputs.cpu().detach().numpy(), axis=2).copy()
            target_np = seg.cpu().numpy().copy()

            cm_ = confusion_matrix(target_np.ravel(), output_np.ravel(), labels=list(range(N_CLASSES)))
            cm += cm_

            oa = f"{metrics.stats_overall_accuracy(cm):.5f}"
            aa = f"{metrics.stats_accuracy_per_class(cm)[0]:.5f}"
            iou = f"{metrics.stats_iou_per_class(cm)[0]:.5f}"

            train_loss += loss.detach().cpu().item()

            t.set_postfix(OA=wblue(oa), AA=wblue(aa), IOU=wblue(iou), LOSS=wblue(f"{train_loss/cm.sum():.4e}"))


            average_iou, w_tmp = metrics.stats_iou_per_class(cm)
            w_tmp[w_tmp==0] = 1
            w_tmp = average_iou / w_tmp
            alpha = 1e-4
            weights = (1-alpha)*weights + alpha*torch.from_numpy(w_tmp).float().cuda()

        ######
        ## validation
        net.eval()
        cm_test = np.zeros((N_CLASSES, N_CLASSES))
        test_loss = 0
        t = tqdm(test_loader, ncols=80, desc="  Test epoch {}".format(epoch))
        with torch.no_grad():
            for pts, features, seg in t:
                
                features = features.cuda()
                pts = pts.cuda()
                seg = seg.cuda()
                
                
                outputs = net(features, pts)
                loss =  F.cross_entropy(outputs.view(-1, N_CLASSES), seg.view(-1))

                output_np = np.argmax(outputs.cpu().detach().numpy(), axis=2).copy()
                target_np = seg.cpu().numpy().copy()

                cm_ = confusion_matrix(target_np.ravel(), output_np.ravel(), labels=list(range(N_CLASSES)))
                cm_test += cm_

                oa_val = f"{metrics.stats_overall_accuracy(cm_test):.5f}"
                aa_val = f"{metrics.stats_accuracy_per_class(cm_test)[0]:.5f}"
                iou_val = f"{metrics.stats_iou_per_class(cm_test)[0]:.5f}"

                test_loss += loss.detach().cpu().item()

                t.set_postfix(OA=wgreen(oa_val), AA=wgreen(aa_val), IOU=wgreen(iou_val), LOSS=wgreen(f"{test_loss/cm_test.sum():.4e}"))

        # save the model
        torch.save(net.state_dict(), os.path.join(root_folder, "state_dict.pth"))

        # write the logs
        logs.write(f"{epoch} {oa} {aa} {iou} {oa_val} {aa_val} {iou_val}\n")
        logs.flush()

    logs.close()

def test(args, flist_test):
    

    N_CLASSES = 13


    # create the network
    print("Creating network...")
    net = Net(input_channels=3, output_channels=N_CLASSES)
    net.load_state_dict(torch.load(os.path.join(args.savedir, "state_dict.pth")))
    net.cuda()
    net.eval()
    print("parameters", count_parameters(net))

    for filename in flist_test:
        print(filename)
        ds = PartDatasetTest(filename, args.rootdir,
                            block_size=args.blocksize,
                            min_pick_per_point= args.npick,
                            npoints= args.npoints
                            )
        loader = torch.utils.data.DataLoader(ds, batch_size=args.batchsize, shuffle=False,
                                        num_workers=args.threads
                                        )

        xyzrgb = ds.xyzrgb[:,:3]
        scores = np.zeros((xyzrgb.shape[0], N_CLASSES))
        with torch.no_grad():
            t = tqdm(loader, ncols=80)
            for pts, features, indices in t:
                
                features = features.cuda()
                pts = pts.cuda()
                outputs = net(features, pts)

                outputs_np = outputs.cpu().numpy().reshape((-1, N_CLASSES))
                scores[indices.cpu().numpy().ravel()] += outputs_np

        mask = np.logical_not(scores.sum(1)==0)
        scores = scores[mask]
        pts_src = xyzrgb[mask]

        # create the scores for all points
        scores = nearest_correspondance(pts_src, xyzrgb, scores, K=1)

        # compute softmax
        scores = scores - scores.max(axis=1)[:,None]
        scores = np.exp(scores) / np.exp(scores).sum(1)[:,None]
        scores = np.nan_to_num(scores)

        os.makedirs(os.path.join(args.savedir, filename), exist_ok=True)

        # saving labels
        save_fname = os.path.join(args.savedir, filename, "pred.txt")
        scores = scores.argmax(1)
        np.savetxt(save_fname,scores,fmt='%d')

        if args.savepts:
            save_fname = os.path.join(args.savedir, filename, "pts.txt")
            xyzrgb = np.concatenate([xyzrgb, np.expand_dims(scores,1)], axis=1)
            np.savetxt(save_fname,xyzrgb,fmt=['%.4f','%.4f','%.4f','%d'])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--ply", action="store_true", help="save ply files (test mode)")
    parser.add_argument("--savedir", default="results/", type=str)
    parser.add_argument("--rootdir", type=str, required=True)
    parser.add_argument("--batchsize", "-b", default=16, type=int)
    parser.add_argument("--npoints", default=8192, type=int)
    parser.add_argument("--area", default=1, type=int)
    parser.add_argument("--blocksize", default=2, type=int)
    parser.add_argument("--iter", default=1000, type=int)
    parser.add_argument("--threads", default=2, type=int)
    parser.add_argument("--npick", default=16, type=int)
    parser.add_argument("--savepts", action="store_true")
    args = parser.parse_args()


    # create the filelits (train / val) according to area
    print("Create filelist...", end="")
    filelist_train = []
    filelist_test = []
    for area_idx in range(1 ,7):
        folder = os.path.join(args.rootdir, f"Area_{area_idx}")
        datasets = [os.path.join(f"Area_{area_idx}", dataset) for dataset in os.listdir(folder)]
        if area_idx == args.area:
            filelist_test = filelist_test + datasets
        else:
            filelist_train = filelist_train + datasets
    filelist_train.sort()
    filelist_test.sort()
    print(f"done, {len(filelist_train)} train files, {len(filelist_test)} test files")

    if args.test:
        test(args, filelist_test)
    else:
        train(args, filelist_train, filelist_test)


if __name__ == '__main__':
    main()
