import pandas as pd
import ipdb
import h5py

hdf5_path = '/home/galen/SLAI_data_ws/control_your_robot/example/teleop/save/dual_piper_data/teleop_task/2.hdf5'
with h5py.File(hdf5_path, 'r') as f:
    image = f['slave_cam_left']['color'][:]
    ipdb.set_trace()