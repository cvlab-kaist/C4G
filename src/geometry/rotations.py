import torch

def mat_to_quat_batch(rot_mat):
    """Convert a batch of rotation matrices to quaternions.

    Args:
        rot_mat: Tensor of shape (N, 3, 3) representing rotation matrices.

    Returns:
        Tensor of shape (N, 4) representing quaternions in (w, x, y, z) format.
    """
    N = rot_mat.shape[0]
    r11 = rot_mat[:, 0, 0]
    r12 = rot_mat[:, 0, 1]
    r13 = rot_mat[:, 0, 2]
    r21 = rot_mat[:, 1, 0]
    r22 = rot_mat[:, 1, 1]
    r23 = rot_mat[:, 1, 2]
    r31 = rot_mat[:, 2, 0]
    r32 = rot_mat[:, 2, 1]
    r33 = rot_mat[:, 2, 2]

    trace = r11 + r22 + r33
    qw = torch.sqrt(1 + trace) / 2
    qx = (r32 - r23) / (4 * qw)
    qy = (r13 - r31) / (4 * qw)
    qz = (r21 - r12) / (4 * qw)

    quats = torch.stack([qw, qx, qy, qz], dim=1)
    return quats

def quat_to_mat_batch(quat):
    """Convert a batch of quaternions to rotation matrices.

    Args:
        quat: Tensor of shape (N, 4) representing quaternions in (w, x, y, z) format.

    Returns:
        Tensor of shape (N, 3, 3) representing rotation matrices.
    """
    N = quat.shape[0]
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    r11 = 1 - 2 * (y ** 2 + z ** 2)
    r12 = 2 * (x * y - z * w)
    r13 = 2 * (x * z + y * w)

    r21 = 2 * (x * y + z * w)
    r22 = 1 - 2 * (x ** 2 + z ** 2)
    r23 = 2 * (y * z - x * w)

    r31 = 2 * (x * z - y * w)
    r32 = 2 * (y * z + x * w)
    r33 = 1 - 2 * (x ** 2 + y ** 2)

    rot_mats = torch.stack([
        torch.stack([r11, r12, r13], dim=1),
        torch.stack([r21, r22, r23], dim=1),
        torch.stack([r31, r32, r33], dim=1)
    ], dim=1)

    return rot_mats


def slerp_batch(r_a, r_b, alpha, quaternion=False):
    """Spherical linear interpolation (slerp) between two batches of rotations.

    Args:
        r_a: Tensor of shape (N, 3, 3) or (N, 4) representing the first batch of rotations.
        r_b: Tensor of shape (N, 3, 3) or (N, 4) representing the second batch of rotations.
        alpha: Float in [0, 1] representing the interpolation factor.
        quaternion: Boolean indicating whether the inputs are quaternions. If False, inputs are rotation matrices.

    Returns:
        Tensor of shape (N, 3, 3) or (N, 4) representing the interpolated rotations.
    """
    if not quaternion:
        q_a = mat_to_quat_batch(r_a)
        q_b = mat_to_quat_batch(r_b)
    else:
        q_a = r_a
        q_b = r_b
    
    from pyquaternion import Quaternion
    r_list = []
    for b in range(q_a.shape[0]):
        rs = []
        for i in range(q_a.shape[1]):
            qa = Quaternion(q_a[b, i].cpu().numpy())
            qb = Quaternion(q_b[b, i].cpu().numpy())
            q_interp = Quaternion.slerp(qa, qb, alpha)
            rs.append(torch.tensor([q_interp.w, q_interp.x, q_interp.y, q_interp.z], device=q_a.device).to(torch.float32))
        r_list.append(torch.stack(rs, dim=0))
    r_list = torch.stack(r_list, dim=0)
    return r_list

    dot_product = torch.sum(q_a * q_b, dim=1)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)

    theta = torch.acos(dot_product)  # angle between quaternions
    sin_theta = torch.sin(theta)

    s1 = torch.sin((1 - alpha) * theta) / sin_theta
    s2 = torch.sin(alpha * theta) / sin_theta

    s1 = s1.unsqueeze(1)
    s2 = s2.unsqueeze(1)

    q_interp = s1 * q_a + s2 * q_b

    if not quaternion:
        r_interp = quat_to_mat_batch(q_interp)
        return r_interp
    else:
        return q_interp