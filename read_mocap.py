import c3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import display
import os
PATH_TO_DATA = 'MoCapTrajectory/Data'
PATH_TO_RECORDED_DATA = os.path.join(PATH_TO_DATA, 'human_arm_tracks')
print(f'path to recorded data = {PATH_TO_RECORDED_DATA}')
# Get the csv records only
records = [f for f in sorted(os.listdir(PATH_TO_RECORDED_DATA)) if f.endswith('.csv')]
print(f'recorded list = {records}')

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

def read_and_preprocess_csv_data(path, start_col = 0, end_col = None):
    '''
    This Function reads the data from csv file,
    convert strings to numeric and does forward
    and backward linear interpolation to fill the NaNs
        :param path: the path to csv file
        :param start_col: idx of the data's 1st col needed (to take a slice)
        :param end_col: idx of the data's final col needed (to take a slice)
        :return data: the csv dataframe after doing the abovementioned preprocessing
        '''
    data = pd.read_csv(path)
    if end_col is not None:
        Clean_data = data.iloc[:, start_col:end_col] # rotation around (X, Y, Z) + Postition (X, Y, Z)
    else:
        Clean_data = data.iloc[:, start_col:]
    # Convert strings to numeric
    for col in Clean_data:
        Clean_data[col] = pd.to_numeric(Clean_data[col], errors='coerce')
    
    # Used forward and backward linear interpolation for filling nan values
    Clean_data.interpolate(method='linear', limit_direction='both', inplace=True)
    return Clean_data

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

def calc_dist_feature(df):
    '''
    This function takes the data frame as input where we have coordinates
    for target (pih_data[x,y,z]) and wrist (human arm end point) wirst[x,y,z]
    returns the distance between the human arm and the target tool.
    '''
    wrst_vec = df.loc[:,['arm:wrist_X_Position','arm:wrist_Y_Position', 'arm:wrist_Z_Position']].to_numpy()
    pih_vec = df.loc[:,['PIH_X_Position', 'PIH_Y_Position','PIH_Z_Position']].to_numpy()
    distance = np.array([])
    for wrst_pt, pih_pt in zip(wrst_vec, pih_vec):
        # if wrst_pt is not None and pih_pt is not None:
        dist =  np.linalg.norm(wrst_pt - pih_pt)
        distance = np.append(distance,dist)
    return distance

