"""World/root heading conversion shared by joint-reference data and inference."""

import numpy as np


def quaternion_matrix(values):
    q = np.asarray(values, dtype=np.float64)
    if q.shape[-1:] != (4,) or not np.isfinite(q).all():
        raise ValueError("Finite world wxyz quaternions required")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Zero quaternion cannot define heading")
    w, x, y, z = np.moveaxis(q / norm, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def heading_from_quaternion_wxyz(current):
    q = np.asarray(current)
    if q.shape != (4,):
        raise ValueError("One CURRENT world-root quaternion anchor required")
    r = quaternion_matrix(q)
    if np.hypot(r[0, 0], r[1, 0]) < 1e-6:
        raise ValueError("Current heading is degenerate near vertical pitch")
    return float(np.arctan2(r[1, 0], r[0, 0]))


def _heading_rotation(current):
    yaw = heading_from_quaternion_wxyz(current)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def world_quaternion_to_heading_relative_rotation6d(values, current):
    r = np.matmul(_heading_rotation(current).T, quaternion_matrix(values))
    return np.concatenate((r[..., :, 0], r[..., :, 1]), axis=-1).astype(np.float32)


def heading_relative_rotation6d_to_world_quaternion(values, current):
    a = np.asarray(values, dtype=np.float64)
    if a.shape[-1:] != (6,) or not np.isfinite(a).all():
        raise ValueError("Finite rotation6D columns required")
    u, v = a[..., :3], a[..., 3:]
    n = np.linalg.norm(u, axis=-1, keepdims=True)
    if np.any(n < 1e-8):
        raise ValueError("Degenerate rotation6D first column")
    u = u / n
    v = v - (u * v).sum(axis=-1, keepdims=True) * u
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(n < 1e-8):
        raise ValueError("Degenerate rotation6D second column")
    v = v / n
    r = np.matmul(_heading_rotation(current), np.stack((u, v, np.cross(u, v)), axis=-1))
    out = []
    # Symmetric eigenproblem is stable around rotations of pi; output is wxyz.
    for m in r.reshape(-1, 3, 3):
        k = (
            np.asarray(
                [
                    [
                        m[0, 0] - m[1, 1] - m[2, 2],
                        m[1, 0] + m[0, 1],
                        m[2, 0] + m[0, 2],
                        m[2, 1] - m[1, 2],
                    ],
                    [
                        m[1, 0] + m[0, 1],
                        m[1, 1] - m[0, 0] - m[2, 2],
                        m[2, 1] + m[1, 2],
                        m[0, 2] - m[2, 0],
                    ],
                    [
                        m[2, 0] + m[0, 2],
                        m[2, 1] + m[1, 2],
                        m[2, 2] - m[0, 0] - m[1, 1],
                        m[1, 0] - m[0, 1],
                    ],
                    [
                        m[2, 1] - m[1, 2],
                        m[0, 2] - m[2, 0],
                        m[1, 0] - m[0, 1],
                        m.trace(),
                    ],
                ]
            )
            / 3
        )
        _, vectors = np.linalg.eigh(k)
        q = vectors[:, -1][[3, 0, 1, 2]]
        if q[0] < 0:
            q = -q
        out.append(q)
    return np.asarray(out, dtype=np.float32).reshape(a.shape[:-1] + (4,))
