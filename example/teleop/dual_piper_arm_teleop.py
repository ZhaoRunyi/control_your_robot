import os
import time
from typing import Dict, Any
from multiprocessing import Manager, Event

from robot.robot.base_robot import Robot
from robot.controller.Piper_controller import PiperController
from robot.sensor.Realsense_sensor import RealsenseSensor

from robot.utils.base.data_handler import is_enter_pressed
from robot.utils.worker.time_scheduler import TimeScheduler
from robot.utils.worker.worker import Worker
from robot.data.collect_any import CollectAny

condition = {
    "save_path": "./save/dual_piper_data/", 
    "task_name": "teleop_task", 
    "save_format": "hdf5", 
    "save_freq": 30,
    "collect_type": "teleop",
}

# ================= 1. 定义采集机器人 =================
# 在硬件主从模式下，由于主从臂物理并联在同一个CAN口，
# 工控机只需要读取总线上的状态即可，无需在代码中做 action_transform。
class PiperDualTeleop(Robot):
    def __init__(self, condition=condition, move_check=True, start_episode=0):
        super().__init__(condition=condition, move_check=move_check, start_episode=start_episode)
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
        # 左侧主从臂并联在can0，右侧主从臂并联在can1
        self.controllers["arm"]["left_arm"].set_up("can0")
        self.controllers["arm"]["right_arm"].set_up("can1")
        # 【请填入D435的真实序列号】可以通过终端输入 rs-enumerate-devices 查看
        self.sensors["image"]["cam_high"].set_up("填入D435的真实序列号")
        
        # 记录关节角度和夹爪（注意不要录制具有歧义的qpos(现在已修正为ee_pose)）
        self.set_collect_type({
            "arm": ["joint", "ee_pose", "gripper"],
            "image": ["color"]
        })

# ================= 2. 定义工作节点 (Workers) =================
class TeleopWorker(Worker):
    def __init__(self, process_name: str, start_event, end_event):
        super().__init__(process_name, start_event, end_event)
        self.manager = Manager()
        self.data_buffer = self.manager.dict()

    def component_init(self):
        self.component = PiperDualTeleop()
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
    def __init__(self, process_name: str, start_event, end_event, collect_data_buffer: Manager, episode_id=0, resume=False):
        super().__init__(process_name, start_event, end_event)
        self.collect_data_buffer = collect_data_buffer
        self.episode_id = episode_id
        self.resume = resume

    def component_init(self):
        self.collection = CollectAny(condition=condition, start_episode=self.episode_id, move_check=True, resume=self.resume)

    def handler(self):
        data = dict(self.collect_data_buffer)
        if "controller" in data and "sensor" in data:
            self.collection.collect(data["controller"], data["sensor"])

    def finish(self):
        self.collection.write()

# ================= 3. 主进程调度 =================
if __name__ == "__main__":
    import os
    os.environ["INFO_LEVEL"] = "INFO"
    num_episode = 10

    for i in range(num_episode):
        is_start = False
        start_event, end_event = Event(), Event()
        
        teleop = TeleopWorker("teleop_arms", start_event, end_event)
        data = DataWorker("collect_data", start_event, end_event, teleop.data_buffer, episode_id=i, resume=True)

        time_scheduler = TimeScheduler(work_events=[teleop.forward_event], time_freq=30, end_events=[data.next_event])
        teleop.next_to(data)

        teleop.start()
        data.start()

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

        time.sleep(1) # 给数据写入一定缓冲时间

        teleop.stop()
        data.stop()