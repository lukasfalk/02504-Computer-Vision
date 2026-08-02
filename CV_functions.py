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