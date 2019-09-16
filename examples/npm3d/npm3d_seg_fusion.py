import numpy as np
import argparse
from datetime import datetime
import os
import random


import torch
import torch.utils.data
import torch.nn.functional as F

from network_seg import NetS3DIS as Net
from network_seg_fusion import NetS3DISFusion as NetFusion

from torchvision import transforms
from tqdm import tqdm
import metrics
from sklearn.metrics import confusion_matrix
import time

from sklearn.neighbors import BallTree
from PIL import Image

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

import nearest_neighbors.lib.python.nearest_neighbors as nearest_neighbors

def nearest_correspondance(pts_src, pts_dest, data_src, K=1):
    print(pts_dest.shape)
    indices = nearest_neighbors.knn(pts_src, pts_dest, K, omp=True)
    # tree = BallTree(pts_src, leaf_size=2)
    # _, indices = tree.query(pts_dest, k=K)
    print(indices.shape)
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
class PartDataset():

    def __init__ (self, filelist, folder, config,
                    training=False, 
                    iteration_number = None,
                    block_size=8,
                    npoints = 8192):

        self.folder = folder
        self.training = training
        self.filelist = filelist
        self.bs = block_size

        self.npoints = npoints
        self.iterations = iteration_number
        self.verbose = False


        self.transform = transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4)

    def __getitem__(self, index):
        
        # load the data
        index = random.randint(0, len(self.filelist)-1)
        pts = np.load(os.path.join(self.folder, self.filelist[index]))

        # get the features
        fts = np.expand_dims(pts[:,3], 1).astype(np.float32)

        # get the labels
        lbs = pts[:,4].astype(int)

        # get the point coordinates
        pts = pts[:, :3]

        # pick a random point
        pt_id = random.randint(0, pts.shape[0]-1)
        pt = pts[pt_id]

        # create the mask
        mask_x = np.logical_and(pts[:,0]<pt[0]+self.bs/2, pts[:,0]>pt[0]-self.bs/2)
        mask_y = np.logical_and(pts[:,1]<pt[1]+self.bs/2, pts[:,1]>pt[1]-self.bs/2)
        mask = np.logical_and(mask_x, mask_y)
        pts = pts[mask]
        lbs = lbs[mask]
        fts = fts[mask]
        
        # random selection
        choice = np.random.choice(pts.shape[0], self.npoints, replace=True)
        pts = pts[choice]
        lbs = lbs[choice]
        fts = fts[choice]

        # data augmentation
        if self.training:
            # random rotation
            pts = rotate_point_cloud_z(pts)
        
        fts = fts.astype(np.float32)
        fts = fts / 255 - 0.5

        fts2 = np.ones((pts.shape[0], 1))

        pts = torch.from_numpy(pts).float()
        fts = torch.from_numpy(fts).float()
        fts2 =  torch.from_numpy(fts2).float()
        lbs = torch.from_numpy(lbs).long()

        return pts, fts, fts2, lbs

    def __len__(self):
        return self.iterations

class PartDatasetTest():

    def compute_mask(self, pt, bs):
        # build the mask
        mask_x = np.logical_and(self.xyzrgb[:,0]<pt[0]+bs/2, self.xyzrgb[:,0]>pt[0]-bs/2)
        mask_y = np.logical_and(self.xyzrgb[:,1]<pt[1]+bs/2, self.xyzrgb[:,1]>pt[1]-bs/2)
        mask = np.logical_and(mask_x, mask_y)
        return mask

    def __init__ (self, filename, folder,
                    block_size=8,
                    npoints = 8192,
                    test_step=0.8):

        self.folder = folder
        self.bs = block_size
        self.npoints = npoints
        self.verbose = False
        self.filename = filename

        # load the points
        self.xyzrgb = np.load(os.path.join(self.folder, self.filename))

        # # create the point
        # step = test_step
        # self.pts = []
        # discretized_x = ((self.xyzrgb[:,0])/step).astype(int)
        # discretized_y = ((self.xyzrgb[:,1])/step).astype(int)
        # print((discretized_x.max() - discretized_x.min()) * (discretized_y.max()- discretized_y.min()))
        # for xid in range(discretized_x.min(), discretized_x.max()):
        #     for yid in range(discretized_y.min(), discretized_y.max()):
        #         mask = np.logical_and(discretized_x==xid, discretized_y==yid)
        #         if mask.sum() == 0:
        #             continue
        #         else:
        #             tmp_pts = self.xyzrgb[mask]
        #             pt = tmp_pts[np.random.randint(0, tmp_pts.shape[0])]
        #             self.pts.append(pt)

        ############ OPTION 3
        step = test_step
        discretized = ((self.xyzrgb[:,:2]).astype(float)/step).astype(int)
        self.pts = np.unique(discretized, axis=0)
        self.pts = self.pts.astype(np.float)*step

    def __getitem__(self, index):
        
        # get the data
        mask = self.compute_mask(self.pts[index], self.bs)
        pts = self.xyzrgb[mask]

        # choose right number of points
        choice = np.random.choice(pts.shape[0], self.npoints, replace=True)
        pts = pts[choice]

        # labels will contain indices in the original point cloud
        lbs = np.where(mask)[0][choice]

        # separate between features and points

        # separate between features and points
        fts2 = np.ones((pts.shape[0], 1))
        fts = np.expand_dims(pts[:,3], 1).astype(np.float32)
        fts = fts/255 - 0.5

        pts = pts[:, :3].copy()

        pts = torch.from_numpy(pts).float()
        fts = torch.from_numpy(fts).float()
        fts2 = torch.from_numpy(fts2).float()
        lbs = torch.from_numpy(lbs).long()

        return pts, fts, fts2, lbs

    def __len__(self):
        return len(self.pts)

def get_model(model_name, input_channels, output_channels, args):
    if model_name == "NetS3DIS":
        from network_seg import NetS3DIS as Net
    elif model_name == "NetS3DIS2":
        from network_seg import NetS3DIS2 as Net
    elif model_name == "NetS3DIS3":
        from network_seg import NetS3DIS3 as Net
    elif model_name == "NetS3DIS4":
        from network_seg import NetS3DIS4 as Net
    elif model_name == "NetS3DIS5":
        from network_seg import NetS3DIS5 as Net
    return Net(input_channels, output_channels, args=args)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootdir', '-s', help='Path to data folder')
    parser.add_argument("--savedir", type=str, default="./results")
    parser.add_argument('--block_size', help='Block size', type=float, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", "-b", type=int, default=16)
    parser.add_argument("--iter", "-i", type=int, default=1000)
    parser.add_argument("--npoints", "-n", type=int, default=8192)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--model_rgb", type=str, default="./")
    parser.add_argument("--model_noc", type=str, default="./")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--savepts", action="store_true")
    parser.add_argument("--test_step", default=0.8, type=float)
    parser.add_argument("--model", default="NetS3DIS", type=str)
    parser.add_argument("--drop", default=0.5, type=float)
    args = parser.parse_args()

    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    root_folder = os.path.join(args.savedir, "{}_{}_fusion_{}".format(args.model,
            args.npoints, time_string))


    # create the filelits (train / val) according to area
    print("Create filelist...", end="")
    filelist_train = []
    train_dir = os.path.join(args.rootdir, "train_pointclouds")
    filelist_train = [dataset for dataset in os.listdir(train_dir)]
    filename_test = []
    test_dir = os.path.join(args.rootdir, "test_pointclouds")
    filelist_test = [dataset for dataset in os.listdir(test_dir)]
    print(f"done, {len(filelist_train)} train files, {len(filelist_test)} test files")


    N_CLASSES = 10


    # create model
    print("Creating the network...", end="", flush=True)
    net_rgb = get_model(args.model, input_channels=1, output_channels=N_CLASSES, args=args)
    net_noc = get_model(args.model, input_channels=1, output_channels=N_CLASSES, args=args)
    net_rgb.load_state_dict(torch.load(os.path.join(args.model_rgb, "state_dict.pth")))
    net_noc.load_state_dict(torch.load(os.path.join(args.model_noc, "state_dict.pth")))
    net_rgb.cuda()
    net_noc.cuda()
    net_rgb.eval()
    net_noc.eval()
    
    net_fusion = NetFusion(input_channels=2*128, output_channels=N_CLASSES)
    if args.test:
        net_fusion.load_state_dict(torch.load(os.path.join(args.savedir, "state_dict.pth")))
        net_fusion.eval()
    net_fusion.cuda()
    print("Done")


    if not args.test:

        print("Create the datasets...", end="", flush=True)

        ds = PartDataset(filelist_train, train_dir, None,
                                training=True, block_size=args.block_size,
                                iteration_number=args.batch_size*args.iter,
                                npoints=args.npoints)
        train_loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                            num_workers=args.threads
                                            )
        print("Done")


        print("Create optimizer...", end="", flush=True)
        optimizer = torch.optim.Adam(net_fusion.parameters(), lr=1e-3)
        print("Done")
        
        # create the root folder
        os.makedirs(root_folder, exist_ok=True)
        
        # create the log file
        logs = open(os.path.join(root_folder, "log.txt"), "w")

        # iterate over epochs
        for epoch in range(args.epochs):

            #######
            # training
            net_fusion.train()

            train_loss = 0
            cm = np.zeros((N_CLASSES, N_CLASSES))
            t = tqdm(train_loader, ncols=100, desc="Epoch {}".format(epoch))
            for pts, features, features_nc, seg in t:

                features = features.cuda()
                features_nc = features_nc.cuda()
                pts = pts.cuda()
                seg = seg.cuda()

                with torch.no_grad():
                    rgb_out, rgb_features = net_rgb(features, pts, return_features=True)
                    noc_out, noc_features = net_noc(features_nc, pts, return_features=True)

                
                optimizer.zero_grad()
                outputs = net_fusion(rgb_out, noc_out, rgb_features, noc_features, pts)
                loss =  F.cross_entropy(outputs.view(-1, N_CLASSES), seg.view(-1))
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

            # save the model
            torch.save(net_fusion.state_dict(), os.path.join(root_folder, "state_dict.pth"))

            # write the logs
            logs.write(f"{epoch} {oa} {aa} {iou}\n")
            logs.flush()

        logs.close()

    else:

        net_fusion.eval()
        for filename in filelist_test:
            print(filename)
            ds = PartDatasetTest(filename, test_dir,
                            block_size=args.block_size,
                            npoints= args.npoints,
                            test_step=args.test_step
                            )
            loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                            num_workers=args.threads
                                            )

            xyzrgb = ds.xyzrgb[:,:3]
            scores = np.zeros((xyzrgb.shape[0], N_CLASSES))
            with torch.no_grad():
                t = tqdm(loader, ncols=80)
                for pts, features, features_nc, indices in t:
                    
                    features = features.cuda()
                    features_nc = features_nc.cuda()
                    pts = pts.cuda()

                    rgb_out, rgb_features = net_rgb(features, pts, return_features=True)
                    noc_out, noc_features = net_noc(features_nc, pts, return_features=True)
                    outputs = net_fusion(rgb_out, noc_out, rgb_features, noc_features, pts)

                    outputs_np = outputs.cpu().numpy().reshape((-1, N_CLASSES))
                    scores[indices.cpu().numpy().ravel()] += outputs_np
            
            mask = np.logical_not(scores.sum(1)==0)
            scores = scores[mask]
            pts_src = xyzrgb[mask]


            # create the scores for all points
            scores = nearest_correspondance(pts_src.astype(np.float32), xyzrgb.astype(np.float32), scores, K=1)

            # compute softmax
            # scores = scores - scores.max(axis=1)[:,None]
            # scores = np.exp(scores) / np.exp(scores).sum(1)[:,None]
            # scores = np.nan_to_num(scores)

            os.makedirs(os.path.join(args.savedir, "results"), exist_ok=True)

            # saving labels
            save_fname = os.path.join(args.savedir, "results", filename)
            scores = scores[:,1:]
            scores = scores.argmax(1)+1
            print(scores.min(), scores.max())
            np.savetxt(save_fname,scores,fmt='%d')

            if args.savepts:
                save_fname = os.path.join(args.savedir, "results", f"{filename}_pts.txt")
                xyzrgb = np.concatenate([xyzrgb, np.expand_dims(scores,1)], axis=1)
                np.savetxt(save_fname,xyzrgb,fmt=['%.4f','%.4f','%.4f','%d'])


if __name__ == '__main__':
    main()
    print('{}-Done.'.format(datetime.now()))
