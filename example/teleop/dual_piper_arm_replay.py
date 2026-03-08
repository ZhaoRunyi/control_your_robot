import os
import time
import h5py
import cv2
import numpy as np

from robot.robot.base_robot import Robot
from robot.controller.Piper_controller import PiperController
from robot.sensor.Realsense_sensor import RealsenseSensor

# ================= 注意事项 =================
# 回放时，由于你的主从臂物理上并联在同一根 CAN 线上，并且 ID 默认可能都是 7。
# 如果你向该 CAN 发送控制指令，主臂和从臂都会收到。
# 解决方案：
# 回放阶段请【物理断开主臂的电源或CAN线】，确保只有从臂连接在总线上接收动作指令。
# ==========================================

class PiperDualSlaveReplay(Robot):
    def __init__(self, move_check=False, start_episode=0):
        super().__init__(move_check=move_check, start_episode=start_episode)
        self.controllers = {
            "arm": {
                "left_arm": PiperController("slave_left_arm"),
                "right_arm": PiperController("slave_right_arm"),
            }
        }
        self.sensors = {
            "image": {
                "cam_high": RealsenseSensor("cam_high"),
            }
        }

    def set_up(self):
        super().set_up()
        self.controllers["arm"]["left_arm"].set_up("can0") 
        self.controllers["arm"]["right_arm"].set_up("can1")
        # 【请填入D435的真实序列号】
        self.sensors["image"]["cam_high"].set_up("323422071854")

        self.set_collect_type({
            "arm": ["joint", "ee_pose", "gripper"],
            "image": ["color"]
        })

def decode_image(img_data):
    if isinstance(img_data, (bytes, np.bytes_)):
        return cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
    elif isinstance(img_data, np.ndarray) and len(img_data.shape) == 1:
        return cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
    elif isinstance(img_data, np.ndarray) and len(img_data.shape) == 3:
        return cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
    return img_data

def replay_and_record(hdf5_path):
    print(f"Loading data from {hdf5_path}...")
    
    robot = PiperDualSlaveReplay()
    robot.set_up()
    
    with h5py.File(hdf5_path, 'r') as f:
        orig_images = f['slave_cam_high']['color'][:]
        left_joints = f['slave_left_arm']['joint'][:]
        right_joints = f['slave_right_arm']['joint'][:]
        left_grippers = f['slave_left_arm']['gripper'][:]
        right_grippers = f['slave_right_arm']['gripper'][:]
        
        episode_length = len(orig_images)
        fps = 30.0
        
        # 探测第一张图的分辨率以初始化 VideoWriter
        img0 = decode_image(orig_images[0])
            
        h, w, c = img0.shape
        out_w = w * 2  # 水平拼接
        
        out_video_path = os.path.join(os.path.dirname(hdf5_path), f"replay_compare_{os.path.basename(hdf5_path)}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(out_video_path, fourcc, fps, (out_w, h))
        
        print("Start replaying and recording...")
        time_interval = 1.0 / fps
        
        for i in range(episode_length):
            loop_start = time.time()
            
            # --- a. 控制从臂执行 ---
            move_data = {
                "arm": {
                    "left_arm": {"joint": left_joints[i], "gripper": left_grippers[i]},
                    "right_arm": {"joint": right_joints[i], "gripper": right_grippers[i]}
                }
            }
            robot.move(move_data)
            
            # --- b. 获取当前相机画面 ---
            current_data = robot.get()
            live_img = current_data[1]["cam_high"]["color"]
            live_img = decode_image(live_img)
            
            # --- c. 解码历史画面 ---
            orig_img = decode_image(orig_images[i])
                
            # --- d. 拼接并写入视频 ---
            live_img = cv2.resize(live_img, (w, h))
            orig_img = cv2.resize(orig_img, (w, h))
            
            cv2.putText(orig_img, "Original HDF5", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(live_img, "Live Execution", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            combined_img = cv2.hconcat([orig_img, live_img])
            video_writer.write(combined_img)
            
            # --- e. 频率对齐 ---
            elapsed = time.time() - loop_start
            if elapsed < time_interval:
                time.sleep(time_interval - elapsed)
                
        video_writer.release()
        print(f"Replay finished! Video saved to {out_video_path}")

if __name__ == "__main__":
    # 替换为你实际采集生成的HDF5路径
    test_hdf5_path = "./save/dual_piper_data/teleop_task/0.hdf5"
    if os.path.exists(test_hdf5_path):
        replay_and_record(test_hdf5_path)
    else:
        print(f"File not found: {test_hdf5_path}")