import c3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
PATH_TO_DATA = 'Data'


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


def get_trajectory(points: np.array):
    num_frames = len(points)
    x, y, z = (0, 1, 2)
    x_t, y_t, z_t = {}, {}, {}
    n_frames, n_DOF, n_coords = points.shape
    # print(n_frames)
    for joint in range(n_DOF):
        x_t[joint] = []
        y_t[joint] = []
        z_t[joint] = []
    for f in range(n_frames):
        # print(f)
        for d in range(n_DOF):
            # print(i, points[i][0])
            # print(f)
            x_t[d].append(points[f][d][x])
            y_t[d].append(points[f][d][y])
            z_t[d].append(points[f][d][z])
    return x_t, y_t, z_t


def plot_motion(X, Y, Z):
    t = [i for i in range(len(X))]
    f = plt.figure()
    ax = f.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.plot(X, Y, Z)
    plt.show()

def plot_scatter(X, Y, Z):
    pass



motion_class = sorted(os.listdir(PATH_TO_DATA))
print(motion_class)
path_to_motion_class_data = PATH_TO_DATA + f'/{motion_class[6]}'
data_list = os.listdir(path_to_motion_class_data)
path = path_to_motion_class_data+f'/{data_list[1]}'
points, analogs = read_c3d(path)
print(points.shape)
X, Y, Z = get_trajectory(points)
joint = 0
plot_motion(X[joint], Y[joint], Z[joint])

