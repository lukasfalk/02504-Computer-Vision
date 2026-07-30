import itertools as it
import numpy as np

def box3d(n=16):
    points = []
    N = tuple(np.linspace(-1, 1, n))
    for i, j in [(-1, -1), (-1, 1), (1, 1), (0, 0)]:
        points.extend(set(it.permutations([(i,)*n, (j,)*n, N])))
    return np.hstack(points)/2

def Pi(p: np.ndarray) -> np.ndarray:
    '''
    Drop the last coordinate and divide the other coords with it
    '''
    return p[:-1] / p[-1]

def PiInv(p: np.ndarray) -> np.ndarray:
    '''
    Inverse of Pi, add a 1 at the end of the vector
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