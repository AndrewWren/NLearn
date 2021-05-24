"""Adapted from https://stackoverflow.com/questions/33976911/generate-a-random-sample-of-points-distributed-on-the-surface-of-a-unit-sphere33977530
"""


import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
from src.lib.ml_utilities import c, h


def sample_spherical(n_points, sphere_dim=2):
    """
    :param n_points: int > 0
    :param sphere_dim: int >= 1
    :return: array.shape = (n_points, sphere_dim + 1)
    """
    vec = h.ne_rng.normal(size=(sphere_dim + 1, n_points))
    norms = np.linalg.norm(vec, axis=0)
    if not np.any(norms == 0):
        vec /= norms
        return vec.transpose()
    # Probability zero edge case:
    non_zero_norms = (norms != 0)
    vec = vec[:, non_zero_norms]
    n_points_left = n_points - vec.shape[1]
    new_points = sample_spherical(n_points_left, sphere_dim).transpose()
    return np.concatenate((vec, new_points), axis=1).transpose()


def plot_sphere_points(points, highlighted_points=np.array([]),
                       comment = '', save_name=None):
    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones_like(phi))
    xi, yi, zi = points.transpose()

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d'})  #, 'aspect':'auto'})
    ax.plot_wireframe(x, y, z, color='k', rstride=1, cstride=1, alpha=0.1)
    ax.scatter(xi, yi, zi, s=10, c='r', zorder=10)
    if highlighted_points.shape[0] > 0:
        xh, yh, zh = highlighted_points.transpose()
        ax.scatter(xh, yh, zh, s=40, c='b', zorder=10)
    if comment != '':
        comment = ':\n' + comment
    plt.title(f'{points.shape[0]} points on the sphere' + comment)
    if save_name:
        plt.savefig(os.path.join(c.LOGS_FOLDER, save_name + '.pdf'))
    else:
        plt.show()
