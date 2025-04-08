import pinocchio as pin
import numpy as np
import random

# URDF 文件路径
urdf_filename = "diffusion_policy/real_world/leap_hand_utils/assets/leap_hand_dottip/leap_hand_right_for_pointcloud.urdf"

 
# 使用 Pinocchio 从 URDF 构建模型
model = pin.buildModelFromUrdf(urdf_filename)
visual_model = model.visual_model
data = model.createData()