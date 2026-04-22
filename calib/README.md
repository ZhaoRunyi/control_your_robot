# Dual Piper 外参标定

这个目录放双臂 Piper 的外部相机外参标定脚本、默认 tag 生成脚本、工具函数和每次标定输出。

## 先生成默认 tag

```bash
cd /home/edemlab/challenge_ws/control_your_robot
source .venv/bin/activate
python calib/generate_default_tag.py
```

这会固定生成一个默认 tag：

- `dictionary`: `DICT_APRILTAG_36h11`
- `id`: `0`
- `size`: `100.0 mm`

输出文件在：

- `calib/default_tag_a4.png`
- `calib/default_tag_metadata.json`

## 再运行标定

```bash
cd /home/edemlab/challenge_ws/control_your_robot
source .venv/bin/activate
python calib/dual_piper_arm_extrinsic_calib.py --camera high
```

启动后会先交互式要求输入 `l` 或 `r`：

- `l -> left_arm -> slave_left_arm -> can0`
- `r -> right_arm -> slave_right_arm -> can1`

主脚本不再生成 tag，也不再暴露 tag 相关 flag，默认只检测上面这一个 tag。

## 只做离线计算

如果已经有一个现成的 run 目录，只想重新读取 `samples.json` 和 `camera_intrinsics.json` 计算外参，不连接机械臂和相机，可以运行：

```bash
cd /home/edemlab/challenge_ws/control_your_robot
source .venv/bin/activate
python calib/dual_piper_arm_extrinsic_calib.py --offline-run-dir 20260415_120531_left_arm
```

`--offline-run-dir` 支持三种写法：

- `20260415_120531_left_arm`：会按 `calib/runs/<run_name>/` 解析
- `calib/runs/20260415_120531_left_arm`
- `/home/edemlab/challenge_ws/control_your_robot/calib/runs/20260415_120531_left_arm`

这个模式不会递归遍历整个 `runs/`，只会直接读取你给定目录下的：

- `samples.json`
- `camera_intrinsics.json`

然后把新的：

- `sample_all_raw.png`
- `calibration_result.json`
- `calibration_report.txt`
- `copyable_matrices.txt`

写回同一个 run 目录。

## 交互方式

- 只使用终端：
  输入 `c` 后回车保存当前样本，输入 `q` 后回车结束采样
- `cv2` 窗口只负责显示预览，不处理键盘输入

窗口会持续显示：

- 当前相机画面
- tag 检测框和 `RGB axis`
- 当前被标定机械臂末端 `ee_pose`
- 当前选中臂和另一侧臂相对启动时的位移/转角变化量
- 已保存样本数
- 最近一次保存状态

标定脚本会像 `example/teleop/dual_piper_arm_teleop.py` 一样同时初始化左右两个 slave arm，并在后台持续轮询最新的双臂状态和相机画面。前台窗口只消费最新快照做显示和保存，不直接阻塞机械臂状态采样。

## 输出目录

每次运行都会在 `calib/runs/<年月日_时分秒>_<left_arm|right_arm>/` 下生成：

- `camera_intrinsics.json`：本次使用的相机内参
- `sample_XXX_raw.png`：原始图
- `sample_all_raw.png`：所有 `sample_XXX_raw.png` 按等权透明度叠加后的总览图
- `sample_XXX_debug.png`：带检测框和坐标轴的调试图
- `sample_XXX_preview.png`：带状态栏、末端 pose 和检测结果的采样预览图
- `samples.json`：每个样本的 `ee_pose`、双臂 `arm_ee_pose_snapshot`、`T_base_gripper`、`T_camera_tag`
- `calibration_result.json`：最终外参与误差指标
- `calibration_report.txt`：可直接阅读/复制的文本报告
- `copyable_matrices.txt`：最终 4x4 矩阵和采样矩阵列表

## 标定模型

脚本假设：

- 外部相机固定不动
- 单独脚本生成的默认 tag 固定安装在当前被标定机械臂末端
- 机械臂返回的是末端相对本臂底座的 `ee_pose`

标定时使用 OpenCV `calibrateHandEye` 的 eye-to-hand 形式，最终输出：

- `T_base_camera`
- `T_camera_base`
- `T_gripper_tag`
- `T_tag_gripper`

同时报告：

- tag PnP 重投影误差
- `T_base_camera` 的逐样本一致性误差
- `T_gripper_tag` 的逐样本一致性误差
- 采样运动范围
