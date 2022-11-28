import c3d
import numpy as np


def read_c3d(path):
    reader = c3d.Reader(open(path, 'rb'))
    points_list = []
    analog_list = []
    for i, points, analog in reader.read_frames():
        points_list.append(points)
        analog_list.append(analog)
        # print('frame {}: point {}, analog {}'.format(i, points.shape, analog.shape))
    points_arr = np.array(points_list)
    analog_arr = np.array(analog_list)
    return points_arr, analog_arr


path = 'Data/HDM_bd_rotateArmsBothBackward1Reps_001_120.C3D'
points, analogs = read_c3d(path)
print(points.shape)
print(analogs.shape)
print(points[0][0])
print(points[0][1])
print(points[0][2])
