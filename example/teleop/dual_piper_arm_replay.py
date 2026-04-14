import os
import time
import ipdb
import h5py
import argparse
import cv2
import numpy as np

from robot.robot.base_robot_node import build_robot_node
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
    def __init__(self, move_check=False, start_episode=0, condition={}):
        super().__init__(move_check=move_check, start_episode=start_episode)
        self.left_cam = condition.get("left_cam_serial", None)
        self.high_cam = condition.get("high_cam_serial", None)
        self.right_cam = condition.get("right_cam_serial", None)
        # 动态初始化控制器
        self.controllers = {"arm": {}}
        self.sensors = {"image": {}}
        if condition.get("enable_which_arm", "both") in ["left", "both"]:
            self.controllers["arm"]["left_arm"] = PiperController("slave_left_arm")
        if condition.get("enable_which_arm", "both") in ["right", "both"]:
            self.controllers["arm"]["right_arm"] = PiperController("slave_right_arm")

        # 动态初始化相机
        if condition.get("enable_which_camera", "all") in ["high", "all"]:
             self.sensors["image"]["cam_high"] = RealsenseSensor("cam_high")
        if condition.get("enable_which_camera", "all") in ["right", "all"]:
            self.sensors["image"]["cam_right"] = RealsenseSensor("cam_right")
        if condition.get("enable_which_camera", "all") in ["left", "all"]:
            self.sensors["image"]["cam_left"] = RealsenseSensor("cam_left")

    def set_up(self):
        super().set_up()
        # 左侧主从臂并联在can0，右侧主从臂并联在can1
        if "left_arm" in self.controllers["arm"]:
            self.controllers["arm"]["left_arm"].set_up("can0")
        if "right_arm" in self.controllers["arm"]:
            self.controllers["arm"]["right_arm"].set_up("can1")

        # 【请填入D435的真实序列号】可以通过终端输入 rs-enumerate-devices 查看
        if "cam_high" in self.sensors["image"]:
            self.sensors["image"]["cam_high"].set_up(self.high_cam)
        if "cam_left" in self.sensors["image"]:
            self.sensors["image"]["cam_left"].set_up(self.left_cam)
        if "cam_right" in self.sensors["image"]:
            self.sensors["image"]["cam_right"].set_up(self.right_cam)
        
        # 记录关节角度和夹爪（注意不要录制具有歧义的qpos(现在已修正为ee_pose)）
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

def process_recorded_data(current_data, camera:str="left", fps:float=30.0, orig_images=None, w:int=None,
                out_w:int=None, h:int=None, hdf5_path:str="./save/dual_piper_data/teleop_task/0.hdf5"):
    attr = f"cam_{camera}"
    out_video_path = os.path.join(os.path.dirname(hdf5_path), f"replay_compare_{os.path.basename(hdf5_path)}_{camera}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(out_video_path, fourcc, fps, (out_w, h))
    live_img = current_data[1][attr]["color"]
    live_img = decode_image(live_img)

    # 读取原始图像数据
    orig_img = decode_image(orig_images)
    live_img = cv2.resize(live_img, (w, h))
    orig_img = cv2.resize(orig_img, (w, h))

    cv2.putText(orig_img, "Original HDF5", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(live_img, "Live Execution", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    combined_img = cv2.hconcat([orig_img, live_img])
    video_writer.write(combined_img)
    video_writer.release()
    return 

def replay_and_record(condition):
    hdf5_path = condition.get("test_hdf5_path", None)
    print(f"Loading data from {hdf5_path}...")
    
    robot = PiperDualSlaveReplay(condition=condition)
    robot.set_up()
    
    with h5py.File(hdf5_path, 'r') as f:
        # 根据启用的相机动态加载数据
        orig_images_high, orig_images_left, orig_images_right = None, None, None
        # ipdb.set_trace()
        if condition.get("enable_which_camera", "all") in ["high", "all"]:
            orig_images_high = f['slave_cam_high']['color'][:]
        if condition.get("enable_which_camera", "all") in ["left", "all"]:
            orig_images_left = f['slave_cam_left']['color'][:]
        if condition.get("enable_which_camera", "all") in ["right", "all"]:
            orig_images_right = f['slave_cam_right']['color'][:]

        # 根据启用的机械臂动态加载数据
        if condition.get("enable_which_arm", "both") in ["left", "both"]:
            left_joints = f['slave_left_arm']['joint'][:]
            left_grippers = f['slave_left_arm']['gripper'][:]
        if condition.get("enable_which_arm", "both") in ["right", "both"]:
            right_joints = f['slave_right_arm']['joint'][:]
            right_grippers = f['slave_right_arm']['gripper'][:]
        
        # 动态--
        for orig_images in [orig_images_high, orig_images_left, orig_images_right]:
            if orig_images is not None:
                episode_length = len(orig_images)
                img0 = decode_image(orig_images[0])
                break

        fps = 30.0
        
        # 探测第一张图的分辨率以初始化 VideoWriter
        h, w, c = img0.shape
        out_w = w * 2  # 水平拼接
        
        print("Start replaying and recording...")
        time_interval = 1.0 / fps
        
        for i in range(episode_length):
            loop_start = time.time()
            
            # --- a. 控制从臂执行 ---
            move_data = {"arm": {}}
            if condition.get("enable_which_arm", "both") in ["left", "both"]:
                move_data["arm"]["left_arm"] = {"joint": left_joints[i], "gripper": left_grippers[i]}
            if condition.get("enable_which_arm", "both") in ["right", "both"]:
                move_data["arm"]["right_arm"] = {"joint": right_joints[i], "gripper": right_grippers[i]}
            
            robot.move(move_data)
            
            # --- b. 获取当前相机画面 ---
            # current_data[0] 是机械臂的信息，current_data[1] 是摄像机返回的信息
            # current_data[1].keys() ==> ['cam_high', 'cam_left', 'cam_right']
            current_data = robot.get()
            
            if condition.get("enable_which_camera", "all") in ["high", "all"]:  
                process_recorded_data(current_data, camera="high", fps=fps, orig_images=orig_images_high[i], w=w, h=h, out_w=out_w, hdf5_path=hdf5_path)
            if condition.get("enable_which_camera", "all") in ["left", "all"]:
                process_recorded_data(current_data, camera="left", fps=fps, orig_images=orig_images_left[i], w=w, h=h, out_w=out_w, hdf5_path=hdf5_path)
            if condition.get("enable_which_camera", "all") in ["right", "all"]:
                process_recorded_data(current_data, camera="right", fps=fps, orig_images=orig_images_right[i], w=w, h=h, out_w=out_w, hdf5_path=hdf5_path)
            # --- e. 频率对齐 ---
            elapsed = time.time() - loop_start
            if elapsed < time_interval:
                time.sleep(time_interval - elapsed)
                
        print(f"Replay finished! Videos saved to {os.path.dirname(hdf5_path)}")

if __name__ == "__main__":
    # 替换为你实际采集生成的HDF5路径
    parser = argparse.ArgumentParser(description='parameter for teleop data collection replay')
    parser.add_argument('--hdf5_path', type=str, default="./save/dual_piper_data/teleop_task/0.hdf5")
    parser.add_argument('--enable_which_arm', type=str, default="both", choices=["left", "right", "both"], help="which arm to enable for replay")
    parser.add_argument('--enable_which_camera', type=str, default="all", choices=["left", "right", "high", "all"], help="which camera to enable for replay")
    parser.add_argument('--left_cam_serial', type=str, default="344322073012")
    parser.add_argument('--high_cam_serial', type=str, default="323422071854")
    parser.add_argument('--right_cam_serial', type=str, default="335522070790")
    args = parser.parse_args()
    test_hdf5_path = args.hdf5_path
    condition = { 'test_hdf5_path': test_hdf5_path, 'left_cam_serial': args.left_cam_serial,
                 'enable_which_arm': args.enable_which_arm, 'enable_which_camera': args.enable_which_camera,
                  'high_cam_serial': args.high_cam_serial, 'right_cam_serial': args.right_cam_serial }
    if os.path.exists(test_hdf5_path):
        replay_and_record(condition=condition)
    else:
        print(f"File not found: {test_hdf5_path}")