import h5py
import numpy as np
import os
from PIL import Image

# 文件路径
hdf5_path = "/home/edemlab/challenge_ws/control_your_robot/example/teleop/save/dual_piper_data_tmp/teleop_task/0.hdf5"
output_dir = "/home/edemlab/challenge_ws/control_your_robot/example/teleop/save/dual_piper_data_tmp"
output_txt = os.path.join(output_dir, "arm_data.txt")

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 打开HDF5文件
with h5py.File(hdf5_path, 'r') as f:
    # ----------------- 保存相机图片 -----------------
    cam_high_group = f['slave_cam_high']
    color_images = cam_high_group['color'][:]          # shape: (N, 480, 640, 3)
    timestamps_cam = cam_high_group['timestamp'][:]    # shape: (N,)

    print(f"共有 {len(color_images)} 张 high 相机图片")

    for i, (img_array, ts) in enumerate(zip(color_images, timestamps_cam)):
        # 将 numpy 数组转换为 PIL Image 并保存
        img = Image.fromarray(img_array.astype('uint8'), 'RGB')
        img_filename = f"cam_high_{i:04d}_ts_{ts}.png"
        img.save(os.path.join(output_dir, img_filename))

    print(f"所有 high 相机图片已保存至 {output_dir}")

    # ----------------- 提取左右臂数据 -----------------
    # 左臂数据
    left_arm = f['slave_left_arm']
    left_ee = left_arm['ee_pose'][:]          # shape: (N, 6)
    left_gripper = left_arm['gripper'][:]     # shape: (N,)
    left_joint = left_arm['joint'][:]         # shape: (N, 6)
    left_ts = left_arm['timestamp'][:]        # shape: (N,)

    # 右臂数据
    right_arm = f['slave_right_arm']
    right_ee = right_arm['ee_pose'][:]
    right_gripper = right_arm['gripper'][:]
    right_joint = right_arm['joint'][:]
    right_ts = right_arm['timestamp'][:]

    # 数据条数（假设左右臂长度一致）
    num_steps = len(left_ee)
    print(f"共有 {num_steps} 条手臂数据")

    # 写入文本文件
    with open(output_txt, 'w') as txt:
        txt.write("=== 左右臂数据记录 ===\n")
        txt.write("格式：索引 | 时间戳(左) | 时间戳(右) | 左臂末端位姿(6) | 左夹爪 | 左关节角度(6) | 右臂末端位姿(6) | 右夹爪 | 右关节角度(6)\n")
        txt.write("末端位姿顺序: [x, y, z, roll, pitch, yaw] (单位待定)\n")
        txt.write("关节角度顺序: [j1, j2, j3, j4, j5, j6] (单位弧度或度)\n")
        txt.write("-" * 120 + "\n")

        for i in range(num_steps):
            line = (f"{i:04d} | "
                    f"{left_ts[i]} | {right_ts[i]} | "
                    f"{np.array2string(left_ee[i], precision=6, separator=', ')} | "
                    f"{left_gripper[i]:.4f} | "
                    f"{np.array2string(left_joint[i], precision=6, separator=', ')} | "
                    f"{np.array2string(right_ee[i], precision=6, separator=', ')} | "
                    f"{right_gripper[i]:.4f} | "
                    f"{np.array2string(right_joint[i], precision=6, separator=', ')}\n")
            txt.write(line)

    print(f"手臂数据已保存至 {output_txt}")