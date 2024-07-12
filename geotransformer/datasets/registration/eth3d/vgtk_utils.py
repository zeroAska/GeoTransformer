'''

    Credit: all the functions of this file are borrowed from the Vision-Graphics deep learning ToolKit,
    author: Shichen Liu*, Haiwei Chen*,
    author_email: liushichen95@gmail.com,
    license: MIT License

'''
import os
#import trimesh
import random
import numpy as np
import torch
import math
from scipy.spatial.transform import Rotation as sciR
import open3d as o3d



'''
Point cloud augmentation
Only numpy function is included for now
'''


def crop_2d_array(pc_in, crop_ratio):

    ind_src = np.random.randint(pc_in.shape[1])
    selected_col = pc_in[:, ind_src]
    ind_order = np.argsort(selected_col)
    sorted_pc = pc_in[ind_order, :]

    head_or_tail = np.random.randint(2)
    
    if head_or_tail:
        crop_ratio = max(crop_ratio, 1-crop_ratio)
        sorted_pc = np.split(sorted_pc, [int(crop_ratio*sorted_pc.shape[0])], axis=0)[0]
    else:
        crop_ratio = min(crop_ratio, 1-crop_ratio)
        sorted_pc = np.split(sorted_pc, [int(crop_ratio*sorted_pc.shape[0])], axis=0)[1]

    return sorted_pc
    


def R_from_euler_np(angles):
    '''
    angles: [(b, )3]
    '''
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(angles[0]), -math.sin(angles[0]) ],
                    [0,         math.sin(angles[0]), math.cos(angles[0])  ]
                    ])
    R_y = np.array([[math.cos(angles[1]),    0,      math.sin(angles[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(angles[1]),   0,      math.cos(angles[1])  ]
                    ])
                 
    R_z = np.array([[math.cos(angles[2]),    -math.sin(angles[2]),    0],
                    [math.sin(angles[2]),    math.cos(angles[2]),     0],
                    [0,                     0,                      1]
                    ])
    return np.dot(R_z, np.dot( R_y, R_x ))


def translate_point_cloud(data: np.array, max_translation_norm: float):
    """
    Input: Nx3 array
    
    """
    
    T = np.random.rand(1,3) * max_translation_norm

    return data+T, T.transpose().squeeze()

def rotate_point_cloud(data, R = None, max_degree = None):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original point clouds
        R: 
          3x3 array, optional Rotation matrix used to rotate the input
        max_degree:
          float, optional maximum DEGREE to randomly generate rotation 
        Return:
          Nx3 array, rotated point clouds
    """
    # rotated_data = np.zeros(data.shape, dtype=np.float32)

    if R is not None:
      rotation_angle = R
    elif max_degree is not None and abs(max_degree) > 1e-1:
      #if (max_degree == 0):
      #  rotation_angle = np.zeros_like(data.shape)
      #else:
      rotation_angle = np.random.randint(0, max_degree, 3) * np.pi / 180.0
    else:
      rotation_angle = sciR.random().as_matrix() if R is None else R

    if isinstance(rotation_angle, list) or  rotation_angle.ndim == 1:
      rotation_matrix = R_from_euler_np(rotation_angle)
    else:
      assert rotation_angle.shape[0] >= 3 and rotation_angle.shape[1] >= 3
      rotation_matrix = rotation_angle[:3, :3]
    
    if data is None:
      return None, rotation_matrix
    rotated_data = np.dot(rotation_matrix, data.reshape((-1, 3)).T)

    return rotated_data.T, rotation_matrix



def label_relative_rotation_simple(anchors, T, rot_ref_tgt=False):
    """Find the anchor rotation that is closest to the queried rotation. 
    return: 
    R_target: [3,3], T = R_target * anchors[label]
    label: int"""
    if rot_ref_tgt:
        # anchors.T * T = R_target -> T = anchors * R_target
        T_then_anchors = np.einsum('aji,jk->aik', anchors, T)
    else:
        # T * anchors.T = R_target -> T = R_target * anchors
        T_then_anchors = np.einsum('ij,akj->aik', T, anchors)
    label = np.argmax(np.einsum('aii->a', T_then_anchors),axis=0)
    R_target = T_then_anchors[label.item()]
    # return: [3,3], int
    return R_target, label


def label_relative_rotation_np(anchors, T):
    """For all anchor rotations, it finds their corresponding anchor rotation such that the difference between two rotations is closest to the queried rotation.
    They are used as the rotation classification label. 
    return: 
    R_target: [60,3,3]
    label: [60]"""
    T_from_anchors = np.einsum('abc,bj,ijk -> aick', anchors, T, anchors)
    # R_res = Ra^T R Ra (Ra R_res = R Ra)
    label = np.argmax(np.einsum('abii->ab', T_from_anchors),axis=1)
    idxs = np.vstack([np.arange(label.shape[0]), label]).T
    R_target = T_from_anchors[idxs[:,0], idxs[:,1]]
    return R_target, label


'''
    (B)x3x3, Nx3x3 -> dist, idx
'''
def rotation_distance_np(r0, r1):
    '''
    tip: r1 is usally the anchors
    '''
    if r0.ndim == 3:
        bidx = np.zeros(r0.shape[0]).astype(np.int32)
        traces = np.zeros([r0.shape[0], r1.shape[0]]).astype(np.int32)
        for bi in range(r0.shape[0]):
            diff_r = np.matmul(r1, r0[bi].T)
            traces[bi] = np.einsum('bii->b', diff_r)
            bidx[bi] = np.argmax(traces[bi])
        return traces, bidx
    else:
        # diff_r = np.matmul(r0, r1.T)
        # return np.einsum('ii', diff_r)

        diff_r = np.matmul(np.transpose(r1,(0,2,1)), r0)
        traces = np.einsum('bii->b', diff_r)

        return traces, np.argmax(traces), diff_r



'''
Point cloud transform:
    pc: 
        torch: [b, 3, p]
        np: [(b, )3, p]
'''

# translation normalization
def centralize(pc):
    return pc - pc.mean(dim=2, keepdim=True)

def centralize_np(pc, batch=False):
    axis = 2 if batch else 1
    return pc - pc.mean(axis=axis, keepdims=True)


def normalize(pc):
    """Centralize and normalize to a unit ball. Take a batched pytorch tensor. """
    pc = centralize(pc)
    var = pc.pow(2).sum(dim=1, keepdim=True).sqrt()
    return pc / var.max(dim=2, keepdim=True)

def normalize_np(pc, batch=False):
    """Centralize and normalize to a unit ball. Take a numpy array. """
    pc = centralize_np(pc, batch)
    axis = 1 if batch else 0
    var = np.sqrt((pc**2).sum(axis=axis, keepdims=True))
    return pc / var.max(axis=axis+1, keepdims=True)

def uniform_resample_index_np(pc, n_sample, batch=False):
    if batch == True:
        raise NotImplementedError('resample in batch is not implemented')
    n_point = pc.shape[0]
    if n_point >= n_sample:
        # downsample
        idx = np.random.choice(n_point, n_sample, replace=False)
    else:
        # upsample
        idx = np.random.choice(n_point, n_sample-n_point, replace=True)
        idx = np.concatenate((np.arange(n_point), idx), axis=0)
    return idx

def uniform_resample_np(pc, n_sample, label=None, batch=False):
    if batch == True:
        raise NotImplementedError('resample in batch is not implemented')
    idx = uniform_resample_index_np(pc, n_sample, batch)
    if label is None:
        return idx, pc[idx]
    else:
        return idx, pc[idx], label[idx]



def get_so3_from_anchors_np(face_normals, gsize=3):
    # alpha, beta
    na = face_normals.shape[0]
    sbeta = face_normals[...,-1]
    cbeta = (1 - sbeta**2)**0.5
    calpha = face_normals[...,0] / cbeta
    salpha = face_normals[...,1] / cbeta

    # gamma
    gamma = np.linspace(0, 2 * np.pi, gsize, endpoint=False, dtype=np.float32)
    gamma = -gamma[None].repeat(na, axis=0)

    # Compute na rotation matrices Rx, Ry, Rz
    Rz = np.zeros([na, 9], dtype=np.float32)
    Ry = np.zeros([na, 9], dtype=np.float32)
    Rx = np.zeros([na, gsize, 9], dtype=np.float32)
    Rx2 = np.zeros([na, gsize, 9], dtype=np.float32)

    # see xyz convention in http://mathworld.wolfram.com/EulerAngles.html
    # D matrix
    Rz[:,0] = calpha
    Rz[:,1] = salpha
    Rz[:,2] = 0
    Rz[:,3] = -salpha
    Rz[:,4] = calpha
    Rz[:,5] = 0
    Rz[:,6] = 0
    Rz[:,7] = 0
    Rz[:,8] = 1

    # C matrix
    Ry[:,0] = cbeta
    Ry[:,1] = 0
    Ry[:,2] = sbeta
    Ry[:,3] = 0
    Ry[:,4] = 1
    Ry[:,5] = 0
    Ry[:,6] = -sbeta
    Ry[:,7] = 0
    Ry[:,8] = cbeta

    # B Matrix
    Rx[:,:,0] = 1
    Rx[:,:,1] = 0
    Rx[:,:,2] = 0
    Rx[:,:,3] = 0
    Rx[:,:,4] = np.cos(gamma)
    Rx[:,:,5] = np.sin(gamma)
    Rx[:,:,6] = 0
    Rx[:,:,7] = -np.sin(gamma)
    Rx[:,:,8] = np.cos(gamma)

    padding = 60
    Rx2[:,:,0] = 1
    Rx2[:,:,1] = 0
    Rx2[:,:,2] = 0
    Rx2[:,:,3] = 0
    Rx2[:,:,4] = np.cos(gamma+padding/180*np.pi)
    Rx2[:,:,5] = np.sin(gamma+padding/180*np.pi)
    Rx2[:,:,6] = 0
    Rx2[:,:,7] = -np.sin(gamma+padding/180*np.pi)
    Rx2[:,:,8] = np.cos(gamma+padding/180*np.pi)

    Rz = Rz[:,None].repeat(gsize,axis=1).reshape(na*gsize, 3,3)
    Ry = Ry[:,None].repeat(gsize,axis=1).reshape(na*gsize, 3,3)
    Rx = Rx.reshape(na*gsize,3,3)
    Rx2 = Rx2.reshape(na*gsize,3,3)

    # R = BCD
    Rxy = np.einsum('bij,bjh->bih', Rx, Ry)
    Rxy2 = np.einsum('bij,bjh->bih', Rx2, Ry)
    Rs1 = np.einsum('bij,bjh->bih', Rxy, Rz)
    Rs2 = np.einsum('bij,bjh->bih', Rxy2, Rz)

    z_val = (face_normals[:, -1])[:, None].repeat(gsize, axis=1).reshape(na*gsize, 1, 1)
    # import ipdb; ipdb.set_trace()
    Rs = Rs1*(np.abs(z_val+0.79)<0.01)+Rs2*(np.abs(z_val+0.19)<0.01)+\
         Rs1*(np.abs(z_val-0.19)<0.01)+Rs2*(np.abs(z_val-0.79)<0.01)
    return Rs


def get_so3_from_anchors_np_zyz(face_normals, gsize=3):
    # alpha, beta
    na = face_normals.shape[0]
    cbeta = face_normals[...,-1]
    sbeta = (1 - cbeta**2)**0.5
    calpha = face_normals[...,0] / sbeta
    salpha = face_normals[...,1] / sbeta

    if gsize==5:
        calpha = np.where(np.isnan(calpha) & (cbeta>0), np.ones_like(calpha), calpha)
        calpha = np.where(np.isnan(calpha) & (cbeta<0), -np.ones_like(calpha), calpha)
        salpha = np.where(np.isnan(salpha), np.zeros_like(salpha), salpha)

    # gamma
    gamma = np.linspace(0, 2 * np.pi, gsize, endpoint=False, dtype=np.float32)
    gamma = gamma[None].repeat(na, axis=0)

    # Compute na rotation matrices Rx, Ry, Rz
    Rz = np.zeros([na, 9], dtype=np.float32)
    Ry = np.zeros([na, 9], dtype=np.float32)
    Rx = np.zeros([na, gsize, 9], dtype=np.float32)
    Rx2 = np.zeros([na, gsize, 9], dtype=np.float32)
    # Rx3 = np.zeros([na, gsize, 9], dtype=np.float32)
    # Rx4 = np.zeros([na, gsize, 9], dtype=np.float32)

    # see xyz convention in http://mathworld.wolfram.com/EulerAngles.html
    # D matrix
    Rz[:,0] = calpha
    Rz[:,1] = -salpha
    Rz[:,2] = 0
    Rz[:,3] = salpha
    Rz[:,4] = calpha
    Rz[:,5] = 0
    Rz[:,6] = 0
    Rz[:,7] = 0
    Rz[:,8] = 1

    # C matrix
    Ry[:,0] = cbeta
    Ry[:,1] = 0
    Ry[:,2] = sbeta
    Ry[:,3] = 0
    Ry[:,4] = 1
    Ry[:,5] = 0
    Ry[:,6] = -sbeta
    Ry[:,7] = 0
    Ry[:,8] = cbeta

    # B Matrix
    Rx[:,:,0] = np.cos(gamma)
    Rx[:,:,1] = -np.sin(gamma)
    Rx[:,:,2] = 0
    Rx[:,:,3] = np.sin(gamma)
    Rx[:,:,4] = np.cos(gamma)
    Rx[:,:,5] = 0
    Rx[:,:,6] = 0
    Rx[:,:,7] = 0
    Rx[:,:,8] = 1

    # padding = 60  # hardcoded for gsize=3
    padding = 2 * np.pi / gsize / 2 # adaptive to gsize
    Rx2[:,:,0] = np.cos(gamma+padding) #/180*np.pi
    Rx2[:,:,1] = -np.sin(gamma+padding) #/180*np.pi
    Rx2[:,:,2] = 0
    Rx2[:,:,3] = np.sin(gamma+padding) #/180*np.pi
    Rx2[:,:,4] = np.cos(gamma+padding) #/180*np.pi
    Rx2[:,:,5] = 0
    Rx2[:,:,6] = 0
    Rx2[:,:,7] = 0
    Rx2[:,:,8] = 1

    # Rx3[:,:,0] = np.cos(gamma+2*padding) #/180*np.pi
    # Rx3[:,:,1] = -np.sin(gamma+2*padding) #/180*np.pi
    # Rx3[:,:,2] = 0
    # Rx3[:,:,3] = np.sin(gamma+2*padding) #/180*np.pi
    # Rx3[:,:,4] = np.cos(gamma+2*padding) #/180*np.pi
    # Rx3[:,:,5] = 0
    # Rx3[:,:,6] = 0
    # Rx3[:,:,7] = 0
    # Rx3[:,:,8] = 1

    # Rx4[:,:,0] = np.cos(gamma+3*padding) #/180*np.pi
    # Rx4[:,:,1] = -np.sin(gamma+3*padding) #/180*np.pi
    # Rx4[:,:,2] = 0
    # Rx4[:,:,3] = np.sin(gamma+3*padding) #/180*np.pi
    # Rx4[:,:,4] = np.cos(gamma+3*padding) #/180*np.pi
    # Rx4[:,:,5] = 0
    # Rx4[:,:,6] = 0
    # Rx4[:,:,7] = 0
    # Rx4[:,:,8] = 1

    Rz = Rz[:,None].repeat(gsize,axis=1).reshape(na*gsize, 3,3)
    Ry = Ry[:,None].repeat(gsize,axis=1).reshape(na*gsize, 3,3)
    Rx = Rx.reshape(na*gsize,3,3)
    Rx2 = Rx2.reshape(na*gsize,3,3)
    # Rx3 = Rx3.reshape(na*gsize,3,3)
    # Rx4 = Rx4.reshape(na*gsize,3,3)

    Ryx = np.einsum('bij,bjh->bih', Ry, Rx)
    Ryx2 = np.einsum('bij,bjh->bih', Ry, Rx2)
    # Ryx3 = np.einsum('bij,bjh->bih', Ry, Rx3)
    # Ryx4 = np.einsum('bij,bjh->bih', Ry, Rx4)
    Rs1 = np.einsum('bij,bjh->bih', Rz, Ryx)
    Rs2 = np.einsum('bij,bjh->bih', Rz, Ryx2)
    # Rs3 = np.einsum('bij,bjh->bih', Rz, Ryx3)
    # Rs4 = np.einsum('bij,bjh->bih', Rz, Ryx4)

    z_val = (face_normals[:, -1])[:, None].repeat(gsize, axis=1).reshape(na*gsize, 1, 1)
    # import ipdb; ipdb.set_trace()
    if gsize == 3:
        Rs = Rs1*(np.abs(z_val+0.79)<0.01)+Rs2*(np.abs(z_val+0.19)<0.01)+\
            Rs1*(np.abs(z_val-0.19)<0.01)+Rs2*(np.abs(z_val-0.79)<0.01)
        # -0.7947, -0.1876, 0.1876, 0.7967
        # each will make only one of the four conditions true
    elif gsize == 5:
        Rs = Rs2*(np.abs(z_val+1)<0.01)+Rs1*(np.abs(z_val+0.447)<0.01)+\
            Rs2*(np.abs(z_val-0.447)<0.01)+Rs1*(np.abs(z_val-1)<0.01)
        # Rs = Rs1
    else:
        raise NotImplementedError('gsizee other than 3 (for faces) or 5 (for vertices) are not supported: %d'%gsize)
    return Rs
