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
    Returns: torch.Tensor [N, 3]  (on same device/dtype as input)
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


def rotate_point_cloud_by_angle_list(batch_data, rotation_angles):
    """
    Rotate each point cloud around Y by its given angle (radians).
      batch_data: torch.Tensor [B,N,3]
      rotation_angles: array-like [B]
    Returns: torch.Tensor [B,N,3]
    """
    X = batch_data if isinstance(batch_data, torch.Tensor) else torch.as_tensor(batch_data, dtype=torch.float32)
    device, dtype = X.device, X.dtype
    angles = torch.as_tensor(rotation_angles, device=device, dtype=dtype).view(-1)
    out = [rotate_tensor_by_angle_xyz(X[i], angle_y=float(angles[i])) for i in range(X.shape[0])]
    return torch.stack(out, 0)


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """
    Per-point Gaussian jitter (like TF version).
      batch_data: torch.Tensor or array [B,N,3]
    Returns: torch.Tensor [B,N,3]
    """
    X = batch_data if isinstance(batch_data, torch.Tensor) else torch.as_tensor(batch_data, dtype=torch.float32)
    noise = torch.clamp(torch.randn_like(X) * sigma, -clip, clip)
    return X + noise

# ----------------------------
# Label → angles (discrete schemes)
# ----------------------------

def map_label_to_angle_18(label, *, device=None, dtype=torch.float32):
    """
    Map labels (0..17) to (ax, ay, az).
      label: [B]-like
    Returns: list of B tensors (each [3], on device/dtype)
    """
    if isinstance(label, torch.Tensor):
        device = label.device if device is None else device
    lab = torch.as_tensor(label, dtype=torch.long, device=device).view(-1).tolist()
    def t(a,b,c): return torch.tensor([a,b,c], device=device, dtype=dtype)
    angles = []
    for l in lab:
        if l == 0:            angles.append(t(0.0, 0.0, 0.0))
        elif 1 <= l <= 3:     angles.append(t(l * math.pi / 2, 0.0, 0.0))
        elif 4 <= l <= 5:     angles.append(t(0.0, 0.0, (l * 2 - 7) * math.pi / 2))
        elif 6 <= l <= 9:     angles.append(t((l * 2 - 11) * math.pi / 4, 0.0, 0.0))
        elif 10 <= l <= 13:   angles.append(t(0.0, 0.0, (l * 2 - 19) * math.pi / 4))
        else:                 angles.append(t(math.pi / 2, 0.0, (l * 2 - 27) * math.pi / 4))
    return angles


def map_label_to_angle_32(label, *, device=None, dtype=torch.float32):
    """
    Map labels (0..31) to (ax, ay, az).
      label: [B]-like
    Returns: list of B tensors (each [3], on device/dtype)
    """
    if isinstance(label, torch.Tensor):
        device = label.device if device is None else device
    lab = torch.as_tensor(label, dtype=torch.long, device=device).view(-1).tolist()
    def t(a,b,c): return torch.tensor([a,b,c], device=device, dtype=dtype)
    angles = []
    for l in lab:
        if l == 0:                angles.append(t(0.0, 0.0, 0.0))
        elif 1 <= l <= 5:         angles.append(t(0.646, (l - 1) * math.pi / 2.5, 0.0))
        elif 6 <= l <= 10:        angles.append(t(1.108, (l * 2 - 11) * math.pi / 5, 0.0))
        elif 11 <= l <= 20:       angles.append(t(math.pi / 2, (l - 11) * math.pi / 5, 0.0))
        elif 21 <= l <= 25:       angles.append(t(2.033, (l * 2 - 11) * math.pi / 5, 0.0))
        elif 26 <= l <= 30:       angles.append(t(2.496, (l - 26) * math.pi / 2.5, 0.0))
        else:                     angles.append(t(math.pi, 0.0, 0.0))
    return angles

# ----------------------------
# Batch rotations by discrete labels
# ----------------------------

def rotate_tensor_by_label(batch_data, label):
    """ 18-direction scheme (0..17). """
    X = batch_data if isinstance(batch_data, torch.Tensor) else torch.as_tensor(batch_data, dtype=torch.float32)
    if X.ndim != 3 or X.shape[-1] != 3:
        raise ValueError("batch_data must be [B,N,3]")
    B, device = X.shape[0], X.device
    lab = torch.as_tensor(label, dtype=torch.long, device=device).view(-1)
    out = []
    for k in range(B):
        l = int(lab[k].item())
        pc = X[k]
        if   l == 0:         pass
        elif 1 <= l <= 3:    pc = rotate_tensor_by_angle_xyz(pc, angle_x=l * math.pi / 2)
        elif 4 <= l <= 5:    pc = rotate_tensor_by_angle_xyz(pc, angle_z=(l * 2 - 7) * math.pi / 2)
        elif 6 <= l <= 9:    pc = rotate_tensor_by_angle_xyz(pc, angle_x=(l * 2 - 11) * math.pi / 4)
        elif 10 <= l <= 13:  pc = rotate_tensor_by_angle_xyz(pc, angle_z=(l * 2 - 19) * math.pi / 4)
        else:                pc = rotate_tensor_by_angle_xyz(pc, angle_x=math.pi / 2, angle_z=(l * 2 - 27) * math.pi / 4)
        out.append(pc)
    return torch.stack(out, dim=0)


def rotate_tensor_by_label_32(batch_data, label):
    """ 32-direction scheme (0..31). """
    X = batch_data if isinstance(batch_data, torch.Tensor) else torch.as_tensor(batch_data, dtype=torch.float32)
    B, device = X.shape[0], X.device
    lab = torch.as_tensor(label, dtype=torch.long, device=device).view(-1)
    out = []
    for k in range(B):
        l = int(lab[k].item())
        pc = X[k]
        if   l == 0:         pass
        elif 1 <= l <= 5:    pc = rotate_tensor_by_angle_xyz(pc, angle_x=0.646, angle_y=(l - 1) * math.pi / 2.5)
        elif 6 <= l <= 10:   pc = rotate_tensor_by_angle_xyz(pc, angle_x=1.108, angle_y=(l * 2 - 11) * math.pi / 5)
        elif 11 <= l <= 20:  pc = rotate_tensor_by_angle_xyz(pc, angle_x=math.pi / 2, angle_y=(l - 11) * math.pi / 5)
        elif 21 <= l <= 25:  pc = rotate_tensor_by_angle_xyz(pc, angle_x=2.033, angle_y=(l * 2 - 11) * math.pi / 5)
        elif 26 <= l <= 30:  pc = rotate_tensor_by_angle_xyz(pc, angle_x=2.496, angle_y=(l - 26) * math.pi / 2.5)
        else:                pc = rotate_tensor_by_angle_xyz(pc, angle_x=math.pi)
        out.append(pc)
    return torch.stack(out, dim=0)


def rotate_tensor_by_label_54(batch_data, label):
    """ 54-direction scheme (0..53): 2 poles, 14 @30°, 24 @60°, 14 equator. """
    X = batch_data if isinstance(batch_data, torch.Tensor) else torch.as_tensor(batch_data, dtype=torch.float32)
    B, device = X.shape[0], X.device
    lab = torch.as_tensor(label, dtype=torch.long, device=device).view(-1)
    out = []
    for k in range(B):
        l = int(lab[k].item())
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
# Angle between vectors / label diffs
# ----------------------------

def compute_angle_tensor(pts1, pts2):
    """
    Angle between corresponding 3D vectors (relative to origin).
      pts1, pts2: list of [3] tensors OR [B,3] tensors
    Returns: torch.Tensor [B] (radians, in [0, pi])
    """
    def to_B3(x):
        if isinstance(x, (list, tuple)):
            t = torch.stack([torch.as_tensor(v, dtype=torch.float32, device=pts2.device if isinstance(pts2, torch.Tensor) else None) for v in x], dim=0)
        else:
            t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32)
            if t.ndim == 1:
                t = t.unsqueeze(0)
        return t.to(device=pts2.device if isinstance(pts2, torch.Tensor) else t.device)

    A = to_B3(pts1)
    C = to_B3(pts2)
    # ensure same device/dtype
    if A.device != C.device: C = C.to(A.device)
    if A.dtype  != C.dtype:  C = C.to(A.dtype)

    dot = (A * C).sum(dim=-1)
    na = torch.linalg.norm(A, dim=-1)
    nc = torch.linalg.norm(C, dim=-1)
    cos = torch.clamp(dot / (na * nc + 1e-12), -1.0, 1.0)
    return torch.arccos(cos).to(dtype=torch.float32, device=A.device)


def get_rotated_angle_diff(num_angles, label1, label2):
    """
    Angle between two rotation directions represented by labels.
      num_angles: int (<=18) or 32
      label1, label2: [B]-like
    Returns: torch.Tensor [B] (radians)
    """
    l1 = torch.as_tensor(label1, dtype=torch.long)
    l2 = torch.as_tensor(label2, dtype=torch.long)
    device = l1.device
    if num_angles <= 18:
        a1 = map_label_to_angle_18(l1, device=device)
        a2 = map_label_to_angle_18(l2, device=device)
    elif num_angles == 32:
        a1 = map_label_to_angle_32(l1, device=device)
        a2 = map_label_to_angle_32(l2, device=device)
    else:
        raise NotImplementedError("Only <=18 and 32-angle schemes are implemented.")
    return compute_angle_tensor(a1, a2)
