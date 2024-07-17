import numpy as np
import torch
import ipdb
from geotransformer.modules.ops import apply_transform, pairwise_distance, get_rotation_translation_from_transform
from geotransformer.utils.registration import compute_transform_mse_and_mae


def modified_chamfer_distance(raw_points, ref_points, src_points, gt_transform, transform, reduction='mean'):
    r"""Compute the modified chamfer distance.

    Args:
        raw_points (Tensor): (B, N_raw, 3)
        ref_points (Tensor): (B, N_ref, 3)
        src_points (Tensor): (B, N_src, 3)
        gt_transform (Tensor): (B, 4, 4)
        transform (Tensor): (B, 4, 4)
        reduction (str='mean'): reduction method, 'mean', 'sum' or 'none'

    Returns:
        chamfer_distance
    """
    assert reduction in ['mean', 'sum', 'none']

    # P_t -> Q_raw
    aligned_src_points = apply_transform(src_points, transform)  # (B, N_src, 3)
    sq_dist_mat_p_q = pairwise_distance(aligned_src_points, raw_points)  # (B, N_src, N_raw)
    nn_sq_distances_p_q = sq_dist_mat_p_q.min(dim=-1)[0]  # (B, N_src)
    chamfer_distance_p_q = torch.sqrt(nn_sq_distances_p_q).mean(dim=-1)  # (B)

    # Q -> P_raw
    composed_transform = torch.matmul(transform, torch.inverse(gt_transform))  # (B, 4, 4)
    aligned_raw_points = apply_transform(raw_points, composed_transform)  # (B, N_raw, 3)
    sq_dist_mat_q_p = pairwise_distance(ref_points, aligned_raw_points)  # (B, N_ref, N_raw)
    nn_sq_distances_q_p = sq_dist_mat_q_p.min(dim=-1)[0]  # (B, N_ref)
    chamfer_distance_q_p = torch.sqrt(nn_sq_distances_q_p).mean(dim=-1)  # (B)

    # sum up
    chamfer_distance = chamfer_distance_p_q + chamfer_distance_q_p  # (B)

    if reduction == 'mean':
        chamfer_distance = chamfer_distance.mean()
    elif reduction == 'sum':
        chamfer_distance = chamfer_distance.sum()
    return chamfer_distance


def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")


def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
) -> torch.Tensor:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def matrix_to_euler_angles(matrix: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.
    Borrowd from pytorch3d

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)

def mean_and_std_error(gt_poses, poses):

    r"""Mean Angle  Error and std.

    AAE = || Log(R^T \cdot \bar{R}) || 

    Args:
        gt_rotations (Tensor): ground truth rotation matrix (*, 3, 3)
        rotations (Tensor): estimated rotation matrix (*, 3, 3)

    Returns:
        mae (Tensor): mean angle errors (*)
    """

    gt_rotations = gt_poses[..., :3, :3]
    rotations = poses[..., :3, :3]
    mat = torch.matmul(rotations.transpose(-1, -2), gt_rotations)
    pose_diff = torch.matmul(torch.inverse(gt_poses), poses)
    
    angles = matrix_to_euler_angles(pose_diff[..., :3, :3], "XYZ")
    #trace = mat[..., 0, 0] + mat[..., 1, 1] + mat[..., 2, 2]
    #x = 0.5 * (trace - 1.0)
    #x = x.clamp(min=-1.0, max=1.0)
    #x = torch.arccos(x)
    aae = angles.norm(dim=-1)/ np.pi * 180
    std_angle = torch.std(angles, dim=-1)

    ate = (pose_diff[..., :3, 3]).norm(dim=-1)
    std_trans = torch.std(pose_diff[..., :3, 3], dim=-1)    
    return aae, ate, std_angle, std_trans
    


def mean_angle_error(gt_poses, poses):
    r"""Mean Angle  Error.

    AAE = || Log(R^T \cdot \bar{R}) ||^2   

    Args:
        gt_rotations (Tensor): ground truth rotation matrix (*, 3, 3)
        rotations (Tensor): estimated rotation matrix (*, 3, 3)

    Returns:
        mae (Tensor): mean angle errors (*)
    """

    gt_rotations = gt_poses[..., :3, :3]
    rotations = poses[..., :3, :3]
    mat = torch.matmul(rotations.transpose(-1, -2), gt_rotations)
    angles = matrix_to_euler_angles(mat, "XYZ")
    #trace = mat[..., 0, 0] + mat[..., 1, 1] + mat[..., 2, 2]
    #x = 0.5 * (trace - 1.0)
    #x = x.clamp(min=-1.0, max=1.0)
    #x = torch.arccos(x)
    aae = angles.norm(dim=-1)/ np.pi * 180

    gt_trans = gt_poses[..., :3, 3]
    trans = poses[..., :3, 3]
    ate = (gt_trans - trans).norm(dim=-1) 
    return aae, ate


def relative_rotation_error(gt_rotations, rotations):
    r"""Isotropic Relative Rotation Error.

    RRE = acos((trace(R^T \cdot \bar{R}) - 1) / 2)

    Args:
        gt_rotations (Tensor): ground truth rotation matrix (*, 3, 3)
        rotations (Tensor): estimated rotation matrix (*, 3, 3)

    Returns:
        rre (Tensor): relative rotation errors (*)
    """
    mat = torch.matmul(rotations.transpose(-1, -2), gt_rotations)
    trace = mat[..., 0, 0] + mat[..., 1, 1] + mat[..., 2, 2]
    x = 0.5 * (trace - 1.0)
    x = x.clamp(min=-1.0, max=1.0)
    x = torch.arccos(x)
    rre = 180.0 * x / np.pi
    return rre


def relative_translation_error(gt_translations, translations):
    r"""Isotropic Relative Rotation Error.

    RTE = \lVert t - \bar{t} \rVert_2

    Args:
        gt_translations (Tensor): ground truth translation vector (*, 3)
        translations (Tensor): estimated translation vector (*, 3)

    Returns:
        rre (Tensor): relative rotation errors (*)
    """
    rte = torch.linalg.norm(gt_translations - translations, dim=-1)
    return rte


def isotropic_transform_error(gt_transforms, transforms, reduction='mean'):
    r"""Compute the isotropic Relative Rotation Error and Relative Translation Error.

    Args:
        gt_transforms (Tensor): ground truth transformation matrix (*, 4, 4)
        transforms (Tensor): estimated transformation matrix (*, 4, 4)
        reduction (str='mean'): reduction method, 'mean', 'sum' or 'none'

    Returns:
        rre (Tensor): relative rotation error.
        rte (Tensor): relative translation error.
    """
    assert reduction in ['mean', 'sum', 'none']

    gt_rotations, gt_translations = get_rotation_translation_from_transform(gt_transforms)
    rotations, translations = get_rotation_translation_from_transform(transforms)

    rre = relative_rotation_error(gt_rotations, rotations)  # (*)
    rte = relative_translation_error(gt_translations, translations)  # (*)

    if reduction == 'mean':
        rre = rre.mean()
        rte = rte.mean()
    elif reduction == 'sum':
        rre = rre.sum()
        rte = rte.sum()

    return rre, rte


def anisotropic_transform_error(gt_transforms, transforms, reduction='mean'):
    r"""Compute the anisotropic Relative Rotation Error and Relative Translation Error.

    This function calls numpy-based implementation to achieve batch-wise computation and thus is non-differentiable.

    Args:
        gt_transforms (Tensor): ground truth transformation matrix (B, 4, 4)
        transforms (Tensor): estimated transformation matrix (B, 4, 4)
        reduction (str='mean'): reduction method, 'mean', 'sum' or 'none'

    Returns:
        r_mse (Tensor): rotation mse.
        r_mae (Tensor): rotation mae.
        t_mse (Tensor): translation mse.
        t_mae (Tensor): translation mae.
    """
    assert reduction in ['mean', 'sum', 'none']

    batch_size = gt_transforms.shape[0]
    gt_transforms_array = gt_transforms.detach().cpu().numpy()
    transforms_array = transforms.detach().cpu().numpy()

    all_r_mse = []
    all_r_mae = []
    all_t_mse = []
    all_t_mae = []
    for i in range(batch_size):
        r_mse, r_mae, t_mse, t_mae = compute_transform_mse_and_mae(gt_transforms_array[i], transforms_array[i])
        all_r_mse.append(r_mse)
        all_r_mae.append(r_mae)
        all_t_mse.append(t_mse)
        all_t_mae.append(t_mae)
    r_mse = torch.as_tensor(all_r_mse).to(gt_transforms)
    r_mae = torch.as_tensor(all_r_mae).to(gt_transforms)
    t_mse = torch.as_tensor(all_t_mse).to(gt_transforms)
    t_mae = torch.as_tensor(all_t_mae).to(gt_transforms)

    if reduction == 'mean':
        r_mse = r_mse.mean()
        r_mae = r_mae.mean()
        t_mse = t_mse.mean()
        t_mae = t_mae.mean()
    elif reduction == 'sum':
        r_mse = r_mse.sum()
        r_mae = r_mae.sum()
        t_mse = t_mse.sum()
        t_mae = t_mae.sum()

    return r_mse, r_mae, t_mse, t_mae
