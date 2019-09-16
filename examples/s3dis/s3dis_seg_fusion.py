# S3DIS Fusion - Example with ConvPoint

# add the parent folder to the python path to access convpoint library
import sys
sys.path.append('../../')

import argparse
import os
from datetime import datetime
import numpy as np
from PIL import Image
import time
from tqdm import tqdm
import random
from sklearn.metrics import confusion_matrix

import torch
import torch.utils.data
import torch.nn.functional as F
import torchvision
from torchvision import transforms

import utils.metrics as metrics
import convpoint.knn.lib.python.nearest_neighbors as nearest_neighbors
from networks.network_seg_fusion import NetS3DISFusion as NetFusion

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
    print(pts_dest.shape)
    indices = nearest_neighbors.knn(pts_src.astype(np.float32), pts_dest.astype(np.float32), K, omp=True)
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
class PartDatasetTrainVal():

    def __init__ (self, filelist, folder,
                    training=False, 
                    block_size=2,
                    npoints = 4096,
                    iteration_number = None, nocolor=False):

        self.training = training
        self.filelist = filelist
        self.folder = folder
        self.bs = block_size
        self.nocolor = nocolor
        
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

        # if self.normals:
        #     filename_data = os.path.join(folder, dataset, 'xyzrgb_normals.npy')
        #     xyzrgb = np.load(filename_data).astype(np.float32)
        #     xyzrgb = xyzrgb[:,:6]
        # else:
        filename_data = os.path.join(folder, dataset, 'xyzrgb.npy')
        xyzrgb = np.load(filename_data).astype(np.float32)

        # load data
        # filename_data = os.path.join(folder, dataset, 'xyzrgb.npy')
        # if self.verbose:
        #     print('{}-Loading {}...'.format(datetime.now(), filename_data))
        # xyzrgb = np.load(filename_data)

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

        if self.nocolor:
            features = np.ones((pts.shape[0], 1))
        else:
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
        fts2 = torch.ones(pts.shape[0], 1).float()

        lbs = torch.from_numpy(lbs).long()

        return pts, fts, fts2, lbs

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
                    min_pick_per_point = 1, test_step=0.5):

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

        
        # ########### Option 2
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


        step = test_step
        discretized = ((self.xyzrgb[:,:2]).astype(float)/step).astype(int)
        self.pts = np.unique(discretized, axis=0)
        self.pts = self.pts.astype(np.float)*step




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

        features = pts[:,3:6] / 255 - 0.5
        pts = pts[:,:3].copy()

        # convert to torch
        pts = torch.from_numpy(pts).float()
        fts = torch.from_numpy(features).float()
        lbs = torch.from_numpy(lbs).long()
        fts2 = torch.ones(pts.shape[0], 1).float()


        return pts, fts, fts2, lbs

    def __len__(self):
        return len(self.pts)


def get_model(model_name, input_channels, output_channels, args):
    if model_name == "SegBig":
        from networks.network_seg import SegBig as Net
    return Net(input_channels, output_channels, args=args)


def train(args, flist_train, flist_test):
    N_CLASSES = 13


    # create the network
    print("Creating network...")
    net_rgb = get_model(args.model, input_channels=3, output_channels=N_CLASSES, args=args)
    net_noc = get_model(args.model, input_channels=1, output_channels=N_CLASSES, args=args)
    net_rgb.load_state_dict(torch.load(os.path.join(args.model_rgb, "state_dict.pth")))
    net_noc.load_state_dict(torch.load(os.path.join(args.model_noc, "state_dict.pth")))
    net_fusion = NetFusion(input_channels=2*128, output_channels=N_CLASSES)
    net_rgb.cuda()
    net_noc.cuda()
    net_fusion.cuda()
    net_rgb.eval()
    net_noc.eval()


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

    optimizer = torch.optim.Adam(net_fusion.parameters(), lr=1e-3)
    print("done")

    # create the root folder
    print("Creating results folder")
    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    root_folder = os.path.join(args.savedir, "{}_area{}_{}_fusion_{}".format(args.model, args.area, args.npoints, time_string))
    os.makedirs(root_folder, exist_ok=True)
    print("done at", root_folder)
    
    # create the log file
    logs = open(os.path.join(root_folder, "log.txt"), "w")


    # weights = torch.ones(N_CLASSES).float().cuda()
    # iterate over epochs
    for epoch in range(20):

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

        ######
        ## validation
        net_fusion.eval()
        cm_test = np.zeros((N_CLASSES, N_CLASSES))
        test_loss = 0
        t = tqdm(test_loader, ncols=80, desc="  Test epoch {}".format(epoch))
        with torch.no_grad():
            for pts, features, features_nc, seg in t:
                
                features = features.cuda()
                features_nc = features_nc.cuda()
                pts = pts.cuda()
                seg = seg.cuda()
                
                rgb_out, rgb_features = net_rgb(features, pts, return_features=True)
                noc_out, noc_features = net_noc(features_nc, pts, return_features=True)
                outputs = net_fusion(rgb_out, noc_out, rgb_features, noc_features, pts)

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
        torch.save(net_fusion.state_dict(), os.path.join(root_folder, "state_dict.pth"))

        # write the logs
        logs.write(f"{epoch} {oa} {aa} {iou} {oa_val} {aa_val} {iou_val}\n")
        logs.flush()

    logs.close()

def test(args, flist_test):
    

    N_CLASSES = 13


    # create the network
    print("Creating network...")

    net_rgb = get_model(args.model, input_channels=3, output_channels=N_CLASSES, args=args)
    net_noc = get_model(args.model, input_channels=1, output_channels=N_CLASSES, args=args)
    net_rgb.load_state_dict(torch.load(os.path.join(args.model_rgb, "state_dict.pth")))
    net_noc.load_state_dict(torch.load(os.path.join(args.model_noc, "state_dict.pth")))
    net_rgb.cuda()
    net_noc.cuda()
    net_rgb.eval()
    net_noc.eval()

    if not args.sum:    
        net_fusion = NetFusion(input_channels=2*128, output_channels=N_CLASSES)
        net_fusion.load_state_dict(torch.load(os.path.join(args.savedir, "state_dict.pth")))
        net_fusion.cuda()
        net_fusion.eval()
        print("parameters", count_parameters(net_fusion))

    for filename in flist_test:
        print(filename)
        ds = PartDatasetTest(filename, args.rootdir,
                            block_size=args.blocksize,
                            min_pick_per_point= args.npick,
                            npoints= args.npoints,
                            test_step=args.test_step
                            )
        loader = torch.utils.data.DataLoader(ds, batch_size=args.batchsize, shuffle=False,
                                        num_workers=args.threads
                                        )

        xyzrgb = ds.xyzrgb[:,:3]
        scores = np.zeros((xyzrgb.shape[0], N_CLASSES))
        total_time = 0
        iter_nb = 0
        with torch.no_grad():
            t = tqdm(loader, ncols=80)

            for pts, features, features_nc, indices in t:
                
                t1 = time.time()
                features = features.cuda()
                features_nc = features_nc.cuda()
                pts = pts.cuda()
                indices = indices.cuda()
                
                rgb_out, rgb_features = net_rgb(features, pts, return_features=True)
                noc_out, noc_features = net_noc(features_nc, pts, return_features=True)

                if args.sum:
                    outputs = rgb_out + noc_out
                else:
                    outputs = net_fusion(rgb_out, noc_out, rgb_features, noc_features, pts)
                t2 = time.time()
                outputs_np = outputs.cpu().numpy().reshape((-1, N_CLASSES))
                scores[indices.cpu().numpy().ravel()] += outputs_np
                iter_nb +=1
                total_time += (t2-t1)
                t.set_postfix(time=f"{total_time/(iter_nb*args.batchsize):05e}")

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
    parser.add_argument("--test_step", default=0.5, type=float)
    parser.add_argument("--model_rgb", type=str, default="./")
    parser.add_argument("--model_noc", type=str, default="./")
    parser.add_argument("--sum", action="store_true")
    parser.add_argument("--model", type=str, default="SegBig")
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
