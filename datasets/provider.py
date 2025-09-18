# pure_torch_rotations.py
# Pure PyTorch rotations for point clouds — no TensorFlow, no NumPy.

import math
import torch

# ----------------------------
# Core rotation
# ----------------------------

def rotate_tensor_by_angle_xyz(input_data, angle_x=0.0, angle_y=0.0, angle_z=0.0):
    """
    Rotate an Nx3 point cloud by angles (apply X, then Y, then Z).
      input_data: torch.Tensor [N, 3]
      angle_x/y/z: float or 0-dim torch.Tensor (radians)
    Returns: torch.Tensor [N, 3]
    """
    pc = input_data if isinstance(input_data, torch.Tensor) else torch.as_tensor(input_data, dtype=torch.float32)
    if pc.ndim != 2 or pc.shape[-1] != 3:
        raise ValueError("input_data must be a [N,3] tensor")
    device, dtype = pc.device, pc.dtype

    ax = torch.as_tensor(angle_x, device=device, dtype=dtype)
    ay = torch.as_tensor(angle_y, device=device, dtype=dtype)
    az = torch.as_tensor(angle_z, device=device, dtype=dtype)

    one  = torch.tensor(1.0, device=device, dtype=dtype)
    zero = torch.tensor(0.0, device=device, dtype=dtype)

    # Rx
    cx, sx = torch.cos(ax), torch.sin(ax)
    Rx = torch.stack([
        torch.stack([one,  zero, zero]),
        torch.stack([zero,  cx,  -sx]),
        torch.stack([zero,  sx,   cx]),
    ], dim=0)

    # Ry
    cy, sy = torch.cos(ay), torch.sin(ay)
    Ry = torch.stack([
        torch.stack([ cy, zero,  sy]),
        torch.stack([zero,  one, zero]),
        torch.stack([-sy, zero,  cy]),
    ], dim=0)

    # Rz
    cz, sz = torch.cos(az), torch.sin(az)
    Rz = torch.stack([
        torch.stack([ cz, -sz, zero]),
        torch.stack([ sz,  cz, zero]),
        torch.stack([zero, zero,  one]),
    ], dim=0)

    return pc @ Rx @ Ry @ Rz


def rotate_tensor_point_cloud(point_cloud):
    """
    Random rotation around x,y,z for a single point cloud.
      point_cloud: torch.Tensor [N,3]
    Returns: torch.Tensor [N,3]
    """
    pc = point_cloud if isinstance(point_cloud, torch.Tensor) else torch.as_tensor(point_cloud, dtype=torch.float32)
    rand = torch.rand(3, device=pc.device, dtype=pc.dtype) * (2 * math.pi)
    return rotate_tensor_by_angle_xyz(pc, rand[0], rand[1], rand[2])

# ----------------------------
# Label → angles (discrete schemes)
# ----------------------------

def map_label_to_angle_18(label):
    """
    Map labels (0..17) to (ax, ay, az) angles.
      label: [B]-like int tensor/list
    Returns: list of B tensors (each [3], float32)
    """
    lab = torch.as_tensor(label, dtype=torch.long).view(-1).cpu().tolist()
    angles = []
    for l in lab:
        if l == 0:
            angles.append(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32))
        elif 1 <= l <= 3:
            angles.append(torch.tensor([l * math.pi / 2, 0.0, 0.0], dtype=torch.float32))
        elif 4 <= l <= 5:
            angles.append(torch.tensor([0.0, 0.0, (l * 2 - 7) * math.pi / 2], dtype=torch.float32))
        elif 6 <= l <= 9:
            angles.append(torch.tensor([(l * 2 - 11) * math.pi / 4, 0.0, 0.0], dtype=torch.float32))
        elif 10 <= l <= 13:
            angles.append(torch.tensor([0.0, 0.0, (l * 2 - 19) * math.pi / 4], dtype=torch.float32))
        else:  # 14..17
            angles.append(torch.tensor([math.pi / 2, 0.0, (l * 2 - 27) * math.pi / 4], dtype=torch.float32))
    return angles


def map_label_to_angle_32(label):
    """
    Map labels (0..31) to (ax, ay, az) angles.
      label: [B]-like
    Returns: list of B tensors (each [3], float32)
    """
    lab = torch.as_tensor(label, dtype=torch.long).view(-1).cpu().tolist()
    angles = []
    for l in lab:
        if l == 0:
            angles.append(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32))
        elif 1 <= l <= 5:
            angles.append(torch.tensor([0.646, (l - 1) * math.pi / 2.5, 0.0], dtype=torch.float32))
        elif 6 <= l <= 10:
            angles.append(torch.tensor([1.108, (l * 2 - 11) * math.pi / 5, 0.0], dtype=torch.float32))
        elif 11 <= l <= 20:
            angles.append(torch.tensor([math.pi / 2, (l - 11) * math.pi / 5, 0.0], dtype=torch.float32))
        elif 21 <= l <= 25:
            angles.append(torch.tensor([2.033, (l * 2 - 11) * math.pi / 5, 0.0], dtype=torch.float32))
        elif 26 <= l <= 30:
            angles.append(torch.tensor([2.496, (l - 26) * math.pi / 2.5, 0.0], dtype=torch.float32))
        else:  # 31
            angles.append(torch.tensor([math.pi, 0.0, 0.0], dtype=torch.float32))
    return angles

# ----------------------------
# Batch rotations by discrete labels
# ----------------------------

def rotate_tensor_by_label(batch_data, label):
    """
    18-direction scheme (0..17).
      batch_data: torch.Tensor [B,N,3]
      label: [B]-like ints
    Returns: torch.Tensor [B,N,3]
    """
    X = batch_data if isinstance(batch_data, torch.Tensor) else torch.as_tensor(batch_data, dtype=torch.float32)
    if X.ndim != 3 or X.shape[-1] != 3:
        raise ValueError("batch_data must be [B,N,3]")
    B = X.shape[0]
    lab = torch.as_tensor(label, dtype=torch.long, device='cpu').view(-1).tolist()
    out = []
    for k in range(B):
        l = int(lab[k])
        pc = X[k]
        if l == 0:
            pass
        elif 1 <= l <= 3:
            pc = rotate_tensor_by_angle_xyz(pc, angle_x=l * math.pi / 2)
        elif 4 <= l <= 5:
            pc = rotate_tensor_by_angle_xyz(pc, angle_z=(l * 2 - 7) * math.pi / 2)
        elif 6 <= l <= 9:
            pc = rotate_tensor_by_angle_xyz(pc, angle_x=(l * 2 - 11) * math.pi / 4)
        elif 10 <= l <= 13:
            pc = rotate_tensor_by_angle_xyz(pc, angle_z=(l * 2 - 19) * math.pi / 4)
        else:  # 14..17
            pc = rotate_tensor_by_angle_xyz(pc, angle_x=math.pi / 2, angle_z=(l * 2 - 27) * math.pi / 4)
        out.append(pc)
    return torch.stack(out, dim=0)


def rotate_tensor_by_label_32(batch_data, label):
    """
    32-direction scheme (0..31).
    """
    X = batch_data if isinstance(batch_data, torch.Tensor) else torch.as_tensor(batch_data, dtype=torch.float32)
    B = X.shape[0]
    lab = torch.as_tensor(label, dtype=torch.long, device='cpu').view(-1).tolist()
    out = []
    for k in range(B):
        l = int(lab[k])
        pc = X[k]
        if l == 0:
            pass
        elif 1 <= l <= 5:
            pc = rotate_tensor_by_angle_xyz(pc, angle_x=0.646, angle_y=(l - 1) * math.pi / 2.5)
        elif 6 <= l <= 10:
            pc = rotate_tensor_by_angle_xyz(pc, angle_x=1.108, angle_y=(l * 2 - 11) * math.pi / 5)
        elif 11 <= l <= 20:
            pc = rotate_tensor_by_angle_xyz(pc, angle_x=math.pi / 2, angle_y=(l - 11) * math.pi / 5)
        elif 21 <= l <= 25:
            pc = rotate_tensor_by_angle_xyz(pc, angle_x=2.033, angle_y=(l * 2 - 11) * math.pi / 5)
        elif 26 <= l <= 30:
            pc = rotate_tensor_by_angle_xyz(pc, angle_x=2.496, angle_y=(l - 26) * math.pi / 2.5)
        else:  # 31
            pc = rotate_tensor_by_angle_xyz(pc, angle_x=math.pi)
        out.append(pc)
    return torch.stack(out, dim=0)


def rotate_tensor_by_label_54(batch_data, label):
    """
    54-direction scheme (0..53): 2 poles, 14 @30°, 24 @60°, 14 equator.
    """
    X = batch_data if isinstance(batch_data, torch.Tensor) else torch.as_tensor(batch_data, dtype=torch.float32)
    B = X.shape[0]
    lab = torch.as_tensor(label, dtype=torch.long, device='cpu').view(-1).tolist()
    out = []
    for k in range(B):
        l = int(lab[k])
        pc = X[k]
        if l in (0, 1):
            pc = rotate_tensor_by_angle_xyz(pc, angle_x=math.pi * l)
        elif 2 <= l <= 15:
            pc = rotate_tensor_by_angle_xyz(pc, angle_z=2 * (l - 2) * math.pi / 7)
            pc = rotate_tensor_by_angle_xyz(pc, angle_x=(math.pi / 6) if (l % 2 == 0) else (5 * math.pi / 6))
        elif 16 <= l <= 39:
            pc = rotate_tensor_by_angle_xyz(pc, angle_z=2 * (l - 16) * math.pi / 12)
            pc = rotate_tensor_by_angle_xyz(pc, angle_x=(math.pi / 3) if (l % 2 == 0) else (2 * math.pi / 3))
        else:  # 40..53 (equator)
            pc = rotate_tensor_by_angle_xyz(pc, angle_x=math.pi / 2, angle_z=2 * (l - 40) * math.pi / 14)
        out.append(pc)
    return torch.stack(out, dim=0)

# ----------------------------
# Spherical directions
# ----------------------------

def sunflower_distri(num_pts, *, device=None, dtype=torch.float32):
    """
    Sunflower distribution on the unit sphere.
    Returns: torch.Tensor [num_pts, 3]
    """
    idx = torch.arange(0, num_pts, dtype=dtype, device=device) + 0.5
    phi = torch.arccos(1 - 2.0 * idx / float(num_pts))
    theta = math.pi * (1 + 5 ** 0.5) * idx
    x = torch.cos(theta) * torch.sin(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(phi)
    return torch.stack([x, y, z], dim=1)


def normal_distri(num_pts, *, device=None, dtype=torch.float32):
    """
    Sample from N(0,1)^3 and L2-normalize to the unit sphere.
    Returns: torch.Tensor [num_pts, 3]
    """
    pts = torch.randn((num_pts, 3), device=device, dtype=dtype)
    norms = torch.linalg.norm(pts, dim=1, keepdim=True).clamp_min(1e-12)
    return pts / norms

# ----------------------------
# Angle between vectors / label diffs
# ----------------------------

def compute_angle_tensor(pts1, pts2):
    """
    Angle between corresponding 3D vectors (relative to origin).
      pts1, pts2: list of [3] tensors OR [B,3] tensors
    Returns: torch.Tensor [B] in [0, pi]
    """
    def to_B3(x):
        if isinstance(x, (list, tuple)):
            t = torch.stack([torch.as_tensor(v, dtype=torch.float32) for v in x], dim=0)
        else:
            t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32)
            if t.ndim == 1:
                t = t.unsqueeze(0)
        return t

    A = to_B3(pts1)
    C = to_B3(pts2)
    dot = (A * C).sum(dim=-1)
    na = torch.linalg.norm(A, dim=-1)
    nc = torch.linalg.norm(C, dim=-1)
    cos = torch.clamp(dot / (na * nc + 1e-12), -1.0, 1.0)
    return torch.arccos(cos).to(dtype=torch.float32)


def get_rotated_angle_diff(num_angles, label1, label2):
    """
    Angle between two rotation directions represented by labels.
      num_angles: int (<=18) or 32
      label1, label2: [B]-like
    Returns: torch.Tensor [B] (radians)
    """
    l1 = torch.as_tensor(label1, dtype=torch.long).view(-1)
    l2 = torch.as_tensor(label2, dtype=torch.long).view(-1)
    if num_angles <= 18:
        a1 = map_label_to_angle_18(l1)
        a2 = map_label_to_angle_18(l2)
    elif num_angles == 32:
        a1 = map_label_to_angle_32(l1)
        a2 = map_label_to_angle_32(l2)
    else:
        raise NotImplementedError("Only <=18 and 32-angle schemes are implemented.")
    return compute_angle_tensor(a1, a2)

# ----------------------------
# Batch rotations from spherical directions
# ----------------------------

def rotate_tensor_by_batch(batch_data, label):
    """
    Rotate an entire batch using a single sunflower-based direction
    derived from `label` (index into B sunflower points).
      batch_data: torch.Tensor [B,N,3]
      label: int-like in [0, B-1]
    Returns: torch.Tensor [B,N,3]
    """
    X = batch_data if isinstance(batch_data, torch.Tensor) else torch.as_tensor(batch_data, dtype=torch.float32)
    B = X.shape[0]
    pts = sunflower_distri(B, device=X.device, dtype=X.dtype)
    l = int(torch.as_tensor(label).item())
    pt = pts[l]  # [3]

    # phi: angle between [x,y,0] and +x
    xy = torch.stack([pt[0], pt[1], torch.tensor(0.0, device=X.device, dtype=X.dtype)])
    ex = torch.tensor([1.0, 0.0, 0.0], device=X.device, dtype=X.dtype)
    phi = compute_angle_tensor(xy.unsqueeze(0), ex.unsqueeze(0))[0]

    # theta: angle between pt and +z
    ez = torch.tensor([0.0, 0.0, 1.0], device=X.device, dtype=X.dtype)
    theta = compute_angle_tensor(pt.unsqueeze(0), ez.unsqueeze(0))[0]

    out = []
    for k in range(B):
        out.append(rotate_tensor_by_angle_xyz(X[k], angle_x=theta, angle_z=phi))
    return torch.stack(out, dim=0)


def rotate_point_by_label_n(batch_data, label, n, distri='sunflower'):
    """
    Rotate a batch by picking n directions on the sphere and mapping each sample's label to one.
      batch_data: torch.Tensor [B,N,3]
      label: [B]-like (ints in [0, n-1])
      n: number of directions
      distri: 'sunflower' or 'normal'
    Returns: torch.Tensor [B,N,3]
    """
    X = batch_data if isinstance(batch_data, torch.Tensor) else torch.as_tensor(batch_data, dtype=torch.float32)
    B = X.shape[0]
    lab = torch.as_tensor(label, dtype=torch.long).view(-1)

    if distri == 'sunflower':
        pts = sunflower_distri(n, device=X.device, dtype=X.dtype)  # [n,3]
    elif distri == 'normal':
        pts = normal_distri(n, device=X.device, dtype=X.dtype)
    else:
        raise ValueError("distri must be 'sunflower' or 'normal'")

    # Optional shuffle to mimic original variability
    pts = pts[torch.randperm(n, device=X.device)]

    out = []
    ex = torch.tensor([1.0, 0.0, 0.0], device=X.device, dtype=X.dtype)
    ez = torch.tensor([0.0, 0.0, 1.0], device=X.device, dtype=X.dtype)
    z0 = torch.tensor(0.0, device=X.device, dtype=X.dtype)

    for k in range(B):
        pt = pts[int(lab[k].item())]
        xy = torch.stack([pt[0], pt[1], z0])
        phi = compute_angle_tensor(xy.unsqueeze(0), ex.unsqueeze(0))[0]
        theta = compute_angle_tensor(pt.unsqueeze(0), ez.unsqueeze(0))[0]
        out.append(rotate_tensor_by_angle_xyz(X[k], angle_x=theta, angle_z=phi))
    return torch.stack(out, dim=0)
