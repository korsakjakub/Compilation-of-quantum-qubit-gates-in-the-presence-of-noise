# Matrix transforming the Bloch vector
from enum import Enum
from scipy.spatial.transform import Rotation as R

import numpy as np


class Axis(Enum):
    X = 1
    Y = 2
    Z = 3


def get_rotation_matrix(axis: Axis, angle):
    x = angle if axis == Axis.X else 0
    y = angle if axis == Axis.Y else 0
    z = angle if axis == Axis.Z else 0
    return R.from_rotvec([x, y, z]).as_matrix()


class BlochMatrix:
    def __init__(self, matrix: np.matrix):
        self.matrix = matrix
