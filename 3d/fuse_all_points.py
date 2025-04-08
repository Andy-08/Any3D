import open3d as o3d
import numpy as np
from yourdfpy import URDF
import copy
from scipy.spatial.transform import Rotation as R
import time
from pathlib import Path
from sys import argv
import numpy as np
from scipy.spatial.transform import Rotation as R
import pinocchio.visualize as visualize
import matplotlib.pyplot as plt
import pinocchio

from pointcloud_utils import pointcloud_processor, Leaphand_FK, transform_right_leap_pointcloud_to_base_frame, draw_coordinate_frame, cal_tactile_pointcloud

def main():
    urdf_filename = "leap_hand_utils/assets/leap_hand_dottip/leap_hand_right_for_pointcloud.urdf"

    t0 = time.time()
    p = pointcloud_processor(urdf_filename)
    Tac = Leaphand_FK(urdf_filename)
    pos = np.ones(16) * 0.3


    pc = p.get_mesh_pointcloud(joint_pos=pos) 
    Tac.update_model(pos)
    tip_frame = Tac.get_placement_all_tips_frame()
    print(f"Time elapsed 1: {time.time() - t0}")
    # pc = transform_right_leap_pointcloud_to_base_frame(pc, [0.1,0.1,0.1, 3.141, 0, 0])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    # Coloring the point cloud with light blue
    pcd.paint_uniform_color([0.7, 0.7, 1])  # Light blue color
    coordinate_frame = draw_coordinate_frame(size=0.05)
    frame = []
    fixed_joint_names_to_visualize = ["if_tip", "mf_tip", "rf_tip", "tf_tip"]
    for key in tip_frame:
        frame.append(draw_coordinate_frame(size=0.03, origin=tip_frame[key]["translation"], orientation=tip_frame[key]["rotation_matrix"]))
    print(f"Time elapsed 2: {time.time() - t0}")

    # Adding a tactile point
    theta = 0
    phi = 0
    contact_point_frame = tip_frame["if_tip"]["rotation_matrix"] @ R.from_euler('XYZ', [theta, phi, 0], 
                                degrees=True).as_matrix() 
    # contact_point = tip_frame["if_tip"]["translation"] +  contact_point_frame[:, 2] / np.linalg.norm(contact_point_frame[:, 2]) * 0.016
    point = cal_tactile_pointcloud(tip_frame["if_tip"]["translation"], tip_frame["if_tip"]["rotation_matrix"], theta, phi, 0.1)
    # point.append(contact_point)
    tac_point_pcd = o3d.geometry.PointCloud()
    tac_point_pcd.points = o3d.utility.Vector3dVector(point)
    tac_point_pcd.paint_uniform_color([1, 0, 0])  # Red color



    o3d.visualization.draw_geometries([pcd, tac_point_pcd, coordinate_frame, frame[0], frame[1], frame[2], frame[3]], window_name="Point Cloud with Coordinate Frame", width=800, height=600, left=50, top=50, mesh_show_back_face=True)

if __name__ == "__main__":
    main()