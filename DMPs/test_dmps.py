import sys
import os
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import plotly.express as px
import c3d
from IPython.display import display
from mpl_toolkits.mplot3d import Axes3D

from scipy.signal import savgol_filter

# sys.path.append("../")
# sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from read_mocap import PATH_TO_DATA, PATH_TO_RECORDED_DATA, read_and_preprocess_csv_data, calc_dist_feature

from Dmpling.dmpling.dmp import DMP
pd.options.plotting.backend = "plotly"


# read Motion Capture data
print(f'path to recorded data = {PATH_TO_RECORDED_DATA}')
# Get the csv records only
records = [f for f in sorted(os.listdir(PATH_TO_RECORDED_DATA)) if f.endswith('.csv')]
print(f'recorded list = {records}')
all_data = read_and_preprocess_csv_data(os.path.join(PATH_TO_RECORDED_DATA,records[-1]))
pih_data = all_data.iloc[:, :9]
arm_data = all_data.iloc[:, 10:]
print(f' all data records:\t {all_data.columns}')
print(f'Rigid body records:\t {pih_data.columns}')
print(f'Human arm records: \t {arm_data.columns}')
display(arm_data)
wrst_vec = arm_data.loc[:,['arm:wrist_X_Position','arm:wrist_Y_Position', 'arm:wrist_Z_Position']].to_numpy()
print("Wrist data : \n")
display(wrst_vec)

data = wrst_vec[1000:2000, :] # take a sample when the wrist reached the target and inserted the tool
n_frames = data.shape[0]

# indices of links
head = 0
back = np.arange(3, 7)
left_hand = np.arange(11, 19)
right_hand = [19, 20, 21, 22, 24]
left_leg = [27, 28, 30, 31, 32, 33, 34]
right_leg = [35, 37, 38, 39, 40, 41, 42]

# Trajectory of human's hand
# traj_x = data[:, right_hand[-1], 0]
# traj_y = data[:, right_hand[-1], 1]
traj_x = data[:,0]
traj_y = data[:,1]

# Adjust initial coordinates
traj_x = traj_x - traj_x[0]
traj_y = traj_y - traj_y[0]

# illustrate demonstrated right hand trajectory
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(traj_x.shape[0]), traj_x, label="X coordinate")
ax2 = ax.twinx()
ax2.spines["right"].set_edgecolor("orange")
ax2.tick_params(axis='y', colors="orange")
ax2.plot(range(traj_y.shape[0]), traj_y, label="Y coordinate", c="orange")

ax.legend(loc=(0.05, 0.9))
ax2.legend(loc=(0.7, 0.9))
ax.set_xlabel("Samples")
plt.title("Demonstrated Trajetory")
plt.show()

plt.figure(figsize=(4, 4))
plt.scatter([traj_x[0]], [traj_y[0]], c="k")
plt.scatter([traj_x[-1]], [traj_y[-1]], c="r")
plt.plot(traj_x, traj_y, c="k", linestyle="--")
plt.show()
# plt.savefig("mygraph.png")

# define DMPs
dt = 1e-3
T = n_frames * dt
a = 10
b = a / 4
n_bfs = 100

# DMP for X coordinate
dmp_x = DMP(T, dt, n_bfs=n_bfs, a=a, b=b)
dmp_x.fit(traj_x)
learned_x = np.zeros(dmp_x.cs.N)
for i in range(dmp_x.cs.N):
    learned_x[i], _, _, _ = dmp_x.step(k=1.0, goal=10)
    
    
# DMP for Y coordinate
dmp_y = DMP(T, dt, n_bfs=n_bfs, a=a, b=b)
dmp_y.fit(traj_y - traj_y[0])
learned_y = np.zeros(dmp_y.cs.N)
for i in range(dmp_y.cs.N):
    learned_y[i], _, _, _ = dmp_y.step(k=1.0, goal=10)

# illuatrating the learned trajectories
fig, ax = plt.subplots(2, 2, figsize=(12, 4))

ax[0, 0].plot(np.linspace(0, T, n_frames), traj_x)
ax[0, 0].set_ylabel("Demonstration")
ax[0, 1].plot(np.linspace(0, T, n_frames), traj_y)

ax[1, 0].plot(np.arange(0, T, dt), learned_x, c="orange")
ax[1, 0].set_ylabel("DMP")
ax[1, 1].plot(np.arange(0, T, dt), learned_y, c="orange")

plt.show()


plt.figure(figsize=(4, 4))
plt.scatter([learned_x[0]], [learned_y[0]], c="k")
plt.scatter([learned_x[-1]], [learned_y[-1]], c="r")
plt.plot(learned_x, learned_y, c="k", linestyle="--")
plt.show()

dist = calc_dist_feature(all_data)
print(min(dist))
all_data['Distance'] = dist
# display(all_data)
print(all_data.columns)

all_data_sample = all_data.loc[:, ['Frame', 'arm:wrist_X_Position', 'arm:wrist_Y_Position', 'arm:wrist_Z_Position', 'Distance']]#.plot()
df_long=pd.melt(all_data_sample , id_vars=['Frame'])
fig = px.line(df_long, x='Frame', y='value', color='variable')
fig.show()
print('Done')