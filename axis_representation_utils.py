import numpy as np
from typing import List, Union, Any

from numpy import ndarray

table = {0: [(-1, 2, 1), (-1, 0, 2), (-1, 1, 0)],
         1: [(-1, 2, 1), (1, 0, 1), (1, 0, 2)],
         2: [(-1, 0, 2), (1, 0, 1), (1, 1, 2)],
         3: [(-1, 1, 0), (1, 0, 2), (1, 1, 2)],
         }


def axis_angle_to_quaternion(axis: List) -> List:
    vector = [np.sin(axis[0]) * ax for ax in axis[1:]]
    real = np.cos(axis[0] / 2)
    vector.insert(0, real)
    return vector


def quaternion_to_axis_angle(axis: list) -> List:
    if axis[0] == 1:
        return [0, 1, 0, 0]
    else:
        theta = 2 * np.arcsin(axis[0])
        x, y, z = (q / np.sin(theta) for q in axis[1:])
    return [theta, x, y, z]


def quaternion_to_rot(q: list) -> np.ndarray:
    rot: ndarray = np.zeros((3, 3), dtype=np.float32)
    rot[0][0] = 1 - 2 * (q[2] ** 2 + q[3] ** 2)
    rot[1][1] = 1 - 2 * (q[1] ** 2 + q[3] ** 2)
    rot[2][2] = 1 - 2 * (q[1] ** 2 + q[2] ** 2)
    rot[0][1] = 2 * (q[1] * q[2] - q[0] * q[3])
    rot[1][0] = 2 * (q[1] * q[2] + q[0] * q[3])
    rot[0][2] = 2 * (q[1] * q[3] + q[0] * q[2])
    rot[2][0] = 2 * (q[1] * q[3] - q[0] * q[2])
    rot[1][2] = 2 * (q[2] * q[3] - q[0] * q[1])
    rot[2][1] = 2 * (q[2] * q[3] + q[0] * q[1])
    return rot


def get_max_index(rot: ndarray) -> tuple[float, int]:
    q0 = np.sqrt((1 + rot[0][0] + rot[1][1] + rot[2][2]) / 4)
    q1 = np.sqrt((1 + rot[0][0] - rot[1][1] - rot[2][2]) / 4)
    q2 = np.sqrt((1 - rot[0][0] + rot[1][1] - rot[2][2]) / 4)
    q3 = np.sqrt((1 - rot[0][0] - rot[1][1] + rot[2][2]) / 4)
    max_val = np.max([q0, q1, q2, q3])
    index_max = np.argmax([q0, q1, q2, q3])
    return max_val, index_max


def switch(a, b):
    c = a
    a = b
    b = c
    return a, b


def get_quaternion(a: float, b: float, c: float, sign: int = 1, swap: bool = False) -> float:
    assert sign in (1, -1), "sign should be either 1 for summation or -1 for subtraction"
    if swap:
        a, b = switch(a, b)
    return (a + sign * b) / (4 * c)


def rot_to_quaternion(rot: ndarray) -> list:
    quaternion: list[Union[int, float]] = [0, 0, 0, 0]
    sign = 1
    swap = False
    index_set = {0, 1, 2, 3}
    max_q, index_max_q = get_max_index(rot)
    residual_index = index_set - {index_max_q}

    quaternion[index_max_q] = max_q
    for i, q in enumerate(sorted(residual_index)):
        sign, j, k = table[index_max_q][i]
        # if q == 2:
        #     swap = True
        quaternion[q] = get_quaternion(rot[j][k], rot[k][j], max_q, sign, swap)
    return quaternion


def quaternion_to_euler(q: list[float]) -> list[float]:
    x_roll, y_roll = 2 * (q[0] * q[1] + q[2] * q[3]), q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2
    x_pitch = 2 * (q[0] * q[2] - q[1] * q[3])
    x_yaw, y_yaw = 2 * (q[0] * q[3] + q[1] * q[2]), q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2
    pitch = np.arcsin(x_pitch)
    # Avoid Gimbal lock
    if pitch == np.pi/2:
        roll = 0
        yaw = -2*np.arctan2(q[1],q[0])
    elif pitch == -np.pi/2:
        roll = 0
        yaw = 2*np.arctan2(q[1],q[0])
    else:
        roll = np.arctan2(x_roll, y_roll)
        yaw = np.arctan2(x_yaw, y_yaw)
    euler: list[float] = [roll, pitch, yaw]
    return euler


def euler_to_quaternion(e: list[float]) -> list[float]:
    """
    :e parameter: euler angles vector
    :q return: quaternion representation
    """

    def c(theta):
        return np.cos(theta)

    def s(theta):
        return np.sin(theta)

    e = [i / 2 for i in e]
    q: list[float] = [0., 0., 0., 0.]
    q[0] = c(e[0])*c(e[1])*c(e[2]) + s(e[0])*s(e[1])*s(e[2])
    q[1] = s(e[0])*c(e[1])*c(e[2]) - c(e[0])*s(e[1])*s(e[2])
    q[2] = c(e[0])*s(e[1])*c(e[2]) + s(e[0])*c(e[1])*s(e[2])
    q[3] = c(e[0])*c(e[1])*s(e[2]) - s(e[0])*s(e[1])*c(e[2])
    return q
