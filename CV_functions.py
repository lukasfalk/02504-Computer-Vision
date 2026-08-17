import itertools as it
import numpy as np
import cv2

def box3d(n=16):
    points = []
    N = tuple(np.linspace(-1, 1, n))
    for i, j in [(-1, -1), (-1, 1), (1, 1), (0, 0)]:
        points.extend(set(it.permutations([(i,)*n, (j,)*n, N])))
    return np.hstack(points)/2

def Pi(p: np.ndarray) -> np.ndarray:
    '''
    Drop the last coordinate and divide the other coords with it.
    Turns homogeneous coordinates into non-homogeneous coordinates.
    '''
    return p[:-1] / p[-1]

def PiInv(p: np.ndarray) -> np.ndarray:
    '''
    Inverse of Pi, add a 1 at the end of the vector.
    Turns non-homogeneous coordinates into homogeneous coordinates.
    '''
    return np.vstack([p, np.ones((1, p.shape[1]))])

def projectpoints(K: np.ndarray, R: np.ndarray, t: np.ndarray, Q: np.ndarray) -> np.ndarray:
    '''
    Camera matrix K
    Pose of the camera (R)otation, (t)ranslation
    3xn matrix Q, which is n points in 3D to be projected
    '''
    t = t.reshape(3, 1)
    return Pi(K @ (R @ Q + t))

def projectpoints_distortion(K: np.ndarray, R: np.ndarray, t: np.ndarray, Q: np.ndarray, distCoeffs: np.ndarray) -> np.ndarray:
    '''
    Camera matrix K
    Pose of the camera (R)otation, (t)ranslation
    3xn matrix Q, which is n points in 3D to be projected
    '''
    t = t.reshape(3, 1)
    Qc = R @ Q + t
    xy = Pi(Qc)
    r = np.sqrt(xy[0]**2 + xy[1]**2)
    delta_r = sum([k * r**((i+1)*2) for i, k in enumerate(distCoeffs)])
    xy_d = xy * (1 + delta_r)
    pd = K @ PiInv(xy_d)
    return Pi(pd)

def undistortImage(image, K, distCoeffs):
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    p = np.stack((x, y, np.ones(x.shape))).reshape(3, -1)

    q = np.linalg.inv(K) @ p
    xy = q[:2]
    r = np.sqrt(xy[0]**2 + xy[1]**2)
    delta_r = sum([k * r**((i+1)*2) for i, k in enumerate(distCoeffs)])
    xy_d = xy * (1 + delta_r)
    q_d = PiInv(xy_d)
    p_d = K @ q_d

    x_d = p_d[0].reshape(x.shape).astype(np.float32)
    y_d = p_d[1].reshape(y.shape).astype(np.float32)
    assert (p_d[2]==1).all(), 'You did a mistake somewhere'
    im_undistorted = cv2.remap(image, x_d, y_d, cv2.INTER_LINEAR)
    return im_undistorted

def CrossOp(p):
    p = p.flatten()
    return np.cross(np.eye(3), p)

def triangulate(qs, Ps):
    """
    qs: list of n pixel coords, each shape (2,) or (2,1) — inhomogeneous
    Ps: list of n projection matrices, each shape (3,4)
    Returns: Q, the triangulated 3D point (homogeneous, shape (4,1))
    """
    B = []
    for q, P in zip(qs, Ps):
        q = np.array(q).flatten()
        x, y = q[0], q[1]
        B.append(x * P[2] - P[0])
        B.append(y * P[2] - P[1])
    B = np.array(B)

    U, S, Vt = np.linalg.svd(B)
    Q = Vt[-1]          # smallest singular vector = null-space solution
    Q = Q / Q[-1]       # normalize so last coord is 1
    return Q.reshape(-1, 1)

def normalize2d(q):
    """q: (2, n) inhomogeneous. Returns normalized points and T (acts on homogeneous)."""
    mu = np.mean(q, axis=1, keepdims=True)
    sigma = np.std(q - mu, axis=1, keepdims=True)

    T = np.array([[1/sigma[0, 0], 0, -mu[0, 0]/sigma[0, 0]],
                  [0, 1/sigma[1, 0], -mu[1, 0]/sigma[1, 0]],
                  [0, 0, 1]])

    return Pi(T @ PiInv(q)), T


def hest(q1, q2, normalize=False):
    """Estimate H such that q1 ~ H q2. q1, q2: (2, n) inhomogeneous."""
    if normalize:
        q1, T1 = normalize2d(q1)
        q2, T2 = normalize2d(q2)

    q1h, q2h = PiInv(q1), PiInv(q2)
    n = q1h.shape[1]

    B = np.zeros((3 * n, 9))
    for i in range(n):
        B[3*i:3*i+3, :] = np.kron(q2h[:, i].T, CrossOp(q1h[:, i]))

    _, _, Vt = np.linalg.svd(B)
    H = Vt[-1].reshape(3, 3).T

    if normalize:
        H = np.linalg.inv(T1) @ H @ T2

    return H