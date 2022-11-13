# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from axis_representation_utils import *
from quaternion_utils.q_utils import *


if __name__ == '__main__':
    sum_sqr = 0
    axis = [np.pi / 4, 1, 0, 0]
    quaternion = axis_angle_to_quaternion(axis)
    axis_q = quaternion_to_axis_angle(quaternion)
    rot = quaternion_to_rot(axis)
    quaternion_r = rot_to_quaternion(rot)
    rot_q = quaternion_to_rot(quaternion_r)
    euler_q = quaternion_to_euler(quaternion)
    quaternion_e = euler_to_quaternion(euler_q)
    euler_q_e = quaternion_to_euler(quaternion_e)
    quaternion_2 = Q([0.8, 0, 1, 0])
    quaternion_3 = Q(quaternion)
    res_q = rotate(quaternion_2,quaternion_3,mode='active')
    quaternion_3.rotate(quaternion_2)
    res = quaternion_2.prod(quaternion)
    x = quaternion_2.x
    for q in quaternion_r[1:]:
        sum_sqr += q**2
    print(f'initial axis-angle:{axis}')
    print(f'quaternion from axis-angle: {quaternion}')
    print(f'axis-angle from quaternion: {axis_q}')
    print(f'Rotation from quaternion: {rot}')
    print(f'quaternion from rotation: {quaternion_r}')
    print(f'rot from quaternion:{rot_q}')
    print(f'Euler angles: {euler_q}')
    print(f'quaternion from euler: {quaternion_e}')
    print(f'euler from quaternion_e: {euler_q_e}')
    print(f'quaternion product: {res}')
    print(quaternion_2.inverse.x)
    print(f'quaternion rotation: {quaternion_3.res}')

    print(f'square_sum: {sum_sqr}')


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
