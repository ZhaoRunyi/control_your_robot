import os
import time
import ipdb
import argparse
from typing import Dict, Any
from multiprocessing import Manager, Event

from robot.robot.base_robot import Robot
from robot.controller.Piper_controller import PiperController
from robot.sensor.Realsense_sensor import RealsenseSensor

from robot.robot.base_robot_node import build_robot_node
from robot.utils.base.data_handler import is_enter_pressed
from robot.utils.worker.time_scheduler import TimeScheduler
from robot.utils.worker.worker import Worker
from robot.data.collect_any import CollectAny

# ================= 1. 定义采集机器人 =================
# 在硬件主从模式下，由于主从臂物理并联在同一个CAN口，
# 工控机只需要读取总线上的状态即可，无需在代码中做 action_transform。
class PiperDualTeleop(Robot):
    def __init__(self, condition: Dict[str, Any]=None, move_check=True, start_episode=0):
        if condition is None:
            condition = {}  # 或者使用默认配置
        super().__init__(condition=condition, move_check=move_check, start_episode=start_episode)
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
 
    def set_up(self, teleop=False):  # 添加 teleop 参数，可以设置默认值
        super().set_up()  # 调用父类的 set_up
        # 左侧主从臂并联在can0，右侧主从臂并联在can1 
        # 如果只传入一个，可能会出问题，保存数据的时候只保留了一帧
        self.controllers["arm"]["left_arm"].set_up("can0")
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

# ================= 2. 定义工作节点 (Workers) =================
class TeleopWorker(Worker):
    def __init__(self, process_name: str, condition: Dict[str, Any], start_event, end_event):
        super().__init__(process_name, start_event, end_event)
        self.manager = Manager()
        self.condition = condition
        self.data_buffer = self.manager.dict()

    def component_init(self):
        # update 4.4
        self.component = PiperDualTeleop(condition=self.condition)
        # robot_cls = build_robot_node(PiperDualTeleop)
        # self.component = robot_cls(condition=self.condition)
        self.component.set_up()

    def handler(self):
        # 仅读取状态，不发送动作
        data = self.component.get()

        self.data_buffer["controller"] = self.manager.dict()
        self.data_buffer["sensor"] = self.manager.dict()

        for key, value in data[0].items():
            self.data_buffer["controller"]["slave_"+key] = value
        
        for key, value in data[1].items():
            self.data_buffer["sensor"]["slave_"+key] = value

class DataWorker(Worker):
    def __init__(self, process_name: str, start_event, end_event, condition: Dict[str, Any], collect_data_buffer: Manager, episode_id=0, resume=False):
        super().__init__(process_name, start_event, end_event)
        self.collect_data_buffer = collect_data_buffer
        self.episode_id = episode_id
        self.resume = resume
        self.condition = condition

    def component_init(self):
        self.collection = CollectAny(condition=self.condition, start_episode=self.episode_id, move_check=True, resume=self.resume)

    def handler(self):
        data = dict(self.collect_data_buffer)
        if "controller" in data and "sensor" in data:
            self.collection.collect(data["controller"], data["sensor"])

    def finish(self):
        self.collection.write()
    
# ================= 3. 主进程调度 =================
if __name__ == "__main__":
    os.environ["INFO_LEVEL"] = "INFO"

    parser = argparse.ArgumentParser(description='parameter for teleop data collection')
    parser.add_argument('--save_path', type=str, default="./save/dual_piper_data/", help='path to save collected data')
    parser.add_argument('--task_name', type=str, default="teleop_task", help='name of the task, will be used as part of the filename')
    parser.add_argument('--save_format', type=str, default="hdf5", choices=["hdf5"], help='format to save collected data')
    parser.add_argument('--save_freq', type=int, default=30)
    parser.add_argument('--collect_type', type=str, default='teleop')
    parser.add_argument('--enable_which_arm', type=str, default="both", choices=["left", "right", "both"], help="which arm to enable for teleop")
    parser.add_argument('--enable_which_camera', type=str, default="all", choices=["left", "right", "high", "all"], help="which camera to enable for teleop")
    parser.add_argument('--num_episode', type=int, default=10)
    parser.add_argument('--left_cam_serial', type=str, default="344322073012")
    parser.add_argument('--high_cam_serial', type=str, default="323422071854")
    parser.add_argument('--right_cam_serial', type=str, default="335522070790")
    args = parser.parse_args()

    condition = { 'save_path': args.save_path, 'task_name': args.task_name, 'save_format': args.save_format,
                 'save_freq': args.save_freq, 'collect_type': args.collect_type, 'left_cam_serial': args.left_cam_serial, 
                 'high_cam_serial': args.high_cam_serial, 'right_cam_serial': args.right_cam_serial, 'enable_which_arm': args.enable_which_arm,
                 'enable_which_camera': args.enable_which_camera }
    
    for i in range(args.num_episode):
        is_start = False
        start_event, end_event = Event(), Event()
        
        teleop = TeleopWorker(process_name="teleop_arms", start_event=start_event, end_event=end_event, condition=condition)
        data = DataWorker(process_name="collect_data", start_event=start_event, end_event=end_event, 
                          condition=condition, collect_data_buffer=teleop.data_buffer, episode_id=i, resume=True)

        time_scheduler = TimeScheduler(work_events=[teleop.forward_event], time_freq=30, end_events=[data.next_event])
        teleop.next_to(data)

        teleop.start()
        data.start()

        # ipdb.set_trace()  # 可以在这里检查组件状态，确认是否准备就绪
        print(f"--- 准备开始录制 Episode {i} ---")
        print("按 [Enter] 键开始录制...")
        
        while not is_start:
            time.sleep(0.01)
            if is_enter_pressed():
                is_start = True
                start_event.set()
            else:
                time.sleep(1)

        time_scheduler.start()
        print("录制中，按 [Enter] 键结束当前Episode...")
        
        while is_start:
            time.sleep(0.01)
            if is_enter_pressed():
                end_event.set()  
                time_scheduler.stop()  
                is_start = False

        time.sleep(1.5) # 给数据写入一定缓冲时间

        teleop.stop()
        data.stop()