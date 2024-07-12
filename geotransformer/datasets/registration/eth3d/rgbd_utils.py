import numpy as np
import os.path as osp

import torch
#from lietorch import SE3
#from pypose import SE3
import ipdb
#from . import projective_ops as pops
from scipy.spatial.transform import Rotation


def parse_list(filepath, skiprows=0):
    """ read list data """
    data = np.loadtxt(filepath, delimiter=' ', dtype=np.unicode_, skiprows=skiprows)
    return data

def associate_frames(tstamp_image, tstamp_depth, tstamp_pose, max_dt=1.0):
    """ pair images, depths, and poses """
    associations = []
    for i, t in enumerate(tstamp_image):
        if tstamp_pose is None:
            j = np.argmin(np.abs(tstamp_depth - t))
            if (np.abs(tstamp_depth[j] - t) < max_dt):
                associations.append((i, j))

        else:
            j = np.argmin(np.abs(tstamp_depth - t))
            k = np.argmin(np.abs(tstamp_pose - t))
        
            if (np.abs(tstamp_depth[j] - t) < max_dt) and \
                    (np.abs(tstamp_pose[k] - t) < max_dt):
                associations.append((i, j, k))
            
    return associations

def loadtum(datapath, frame_rate=1):
    """ read video data in tum-rgbd format """

    if osp.isfile(osp.join(datapath, 'groundtruth.txt')):
        pose_list = osp.join(datapath, 'groundtruth.txt')
    
    elif osp.isfile(osp.join(datapath, 'pose.txt')):
        pose_list = osp.join(datapath, 'pose.txt')

    else:
        return None, None, None, None, None

    image_list = osp.join(datapath, 'rgb.txt')
    depth_list = osp.join(datapath, 'depth.txt')

    calib_path = osp.join(datapath, 'calibration.txt')
    intrinsic = None
    if osp.isfile(calib_path):
        intrinsic = np.loadtxt(calib_path, delimiter=' ')
        intrinsic = intrinsic.astype(np.float64)

    image_data = parse_list(image_list)
    depth_data = parse_list(depth_list)
    pose_data = parse_list(pose_list, skiprows=1)
    pose_vecs = pose_data[:,1:].astype(np.float64)

    tstamp_image = image_data[:,0].astype(np.float64)
    tstamp_depth = depth_data[:,0].astype(np.float64)
    tstamp_pose = pose_data[:,0].astype(np.float64)
    associations = associate_frames(tstamp_image, tstamp_depth, tstamp_pose)

    # print(len(tstamp_image))
    # print(len(associations))

    indicies = range(len(associations))[::frame_rate]

    # indicies = [ 0 ]
    # for i in range(1, len(associations)):
    #     t0 = tstamp_image[associations[indicies[-1]][0]]
    #     t1 = tstamp_image[associations[i][0]]
    #     if t1 - t0 > 1.0 / frame_rate:
    #         indicies += [ i ]

    counter = 0
    init_pose_inv = np.eye(4)
    images, poses, depths, intrinsics, tstamps = [], [], [], [], []
    for ix in indicies:
        (i, j, k) = associations[ix]
        images += [ osp.join(datapath, image_data[i,1]) ]
        depths += [ osp.join(datapath, depth_data[j,1]) ]

        if counter == 0:
            init_pose_inv = np.linalg.inv(xyzquat_to_pose_mat(pose_vecs[k]))
            poses += [ np.array([0,0,0,0,0,0,1.0]) ]
        else:
            curr_pose = init_pose_inv @ xyzquat_to_pose_mat(pose_vecs[k])
            poses += [ pose_matrix_to_quaternion(curr_pose) ]
            
        tstamps += [ tstamp_image[i] ]
        
        if intrinsic is not None:
            intrinsics += [ intrinsic ]
        counter += 1


    return images, depths, poses, intrinsics, tstamps, pose_data


#def all_pairs_distance_matrix(poses, beta=2.5):
#    """ compute distance matrix between all pairs of poses """
#    poses = np.array(poses, dtype=np.float32)
#    poses[:,:3] *= beta # scale to balence rot + trans
#    poses = SE3(torch.from_numpy(poses))

#    r = (poses[:,None].inv() * poses[None,:]).log()
#    return r.norm(dim=-1).cpu().numpy()

def pose_matrix_to_quaternion(pose):
    """ convert 4x4 pose matrix to (t, q) """
    q = Rotation.from_matrix(pose[:3, :3]).as_quat()
    return np.concatenate([pose[:3, 3], q], axis=0)

def xyzquat_to_pose_mat(xyzquat_arr):
    pose_mat = np.eye(4)
    pose_mat[:3,:3] = Rotation.from_quat(xyzquat_arr[-4:]).as_matrix()
    pose_mat[:3, 3] = xyzquat_arr[:3]
    return pose_mat

