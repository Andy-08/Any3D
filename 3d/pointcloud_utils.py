import open3d as o3d
import numpy as np
from yourdfpy import URDF
import copy
from scipy.spatial.transform import Rotation as R
import time
from pathlib import Path
from sys import argv
import pinocchio.visualize as visualize
import matplotlib.pyplot as plt
import pinocchio

class Leaphand_FK():
    def __init__(self,
                 urdf_filename="",
                 ):
        
        # load the model
        self.model = pinocchio.buildModelFromUrdf(urdf_filename)
        self.data = self.model.createData()
        init_state=np.zeros(self.model.nq)
        pinocchio.forwardKinematics(self.model, self.data, init_state)
        pinocchio.updateFramePlacements(self.model, self.data)        

    def update_model(self, state):
        # swap the MCP1 and MCP2 joint
        state[0], state[1] = state[1], state[0]
        state[4], state[5] = state[5], state[4]
        state[8], state[9] = state[9], state[8]
        state[12], state[13] = state[13], state[12]
        # Perform the forward kinematics over the kinematic tree
        pinocchio.forwardKinematics(self.model, self.data, state)
        pinocchio.updateFramePlacements(self.model, self.data)


    def get_placement_all_joint(self):
        """cal the FK of the model and get the placement of all joints

        Returns:
            DICT: key: joint name, 
                value: {translation: [x,y,z], rotation_matrix: [[x,y,z],[x,y,z],[x,y,z]]}
        """
        joint_frame = {}
        for name, oMi in zip(self.model.names, self.data.oMi):
            translation = oMi.translation.T
            rotation_matrix = oMi.rotation

            joint_frame[name] = {
                "translation": translation,
                "rotation_matrix": rotation_matrix,
            }
        return joint_frame

    def get_placement_all_tips_frame(self):
        """cal the FK of the model and get the placement of all tips
            the frame is the same as the dottip frame in dottip stl file

        Returns:
            DICT: key: dottip name, 
                 value: {translation: [x,y,z], rotation_matrix: [[x,y,z],[x,y,z],[x,y,z]]}
       """
        # Visualize specific fixed joints
        fixed_joint_names_to_visualize = ["if_tip", "mf_tip", "rf_tip", "tf_tip"]
        dottip_frame = {}
        for fixed_joint_name in fixed_joint_names_to_visualize:
            frameId = self.model.getFrameId(fixed_joint_name)

            oMi_fixed = self.data.oMf[frameId]
            translation_fixed = oMi_fixed.translation.T
            rotation_matrix_fixed = (R.from_matrix(oMi_fixed.rotation) * R.from_euler('ZXY', [-np.pi/2, np.pi/2, 0])).as_matrix()

            dottip_frame[fixed_joint_name] = {
                "translation": translation_fixed,
                "rotation_matrix": rotation_matrix_fixed,
            }
        return dottip_frame



class pointcloud_processor:
    def __init__(self, urdf_filename="leap_hand_utils/assets/leap_hand_dottip/leap_hand_right_for_pointcloud.urdf"):

        self.hand_states = []
        self.hand_stl_list = []
        self.meshes = {}
        self.urdf_dict = {}
        self.Leap_urdf = URDF.load(urdf_filename)
        self.urdf_dict["right_leap"] = {
            "scene": self.Leap_urdf.scene,
            "mesh_list": self._load_meshes(self.Leap_urdf.scene),
        }

    def fetch_contact_pointcloud(self):
        pass

    def _load_meshes(self, scene):
        mesh_list = []
        for name, g in scene.geometry.items():
            mesh = g.as_open3d
            mesh_list.append(mesh)

        return mesh_list

    def _update_meshes(self, type):
        mesh_new = o3d.geometry.TriangleMesh()
        for idx, name in enumerate(self.urdf_dict[type]["scene"].geometry.keys()):
            mesh_new += copy.deepcopy(self.urdf_dict[type]["mesh_list"][idx]).transform(
                self.urdf_dict[type]["scene"].graph.get(name)[0]
            )
        return mesh_new

    def get_mesh_pointcloud(self, joint_pos, joint_pos_left=None):
        self.Leap_urdf.update_cfg(joint_pos)
        right_mesh = self._update_meshes("right_leap")  # Get the new updated mesh
        robot_pc = right_mesh.sample_points_uniformly(number_of_points=80000)

        # self.Leap_urdf_2.update_cfg(joint_pos_left)
        # left_mesh = self._update_meshes("left_leap")  # Get the new updated mesh
        # robot_pc_left = left_mesh.sample_points_uniformly(number_of_points=80000)

        # Convert the sampled mesh point cloud to the format expected by Open3D
        new_points = np.asarray(robot_pc.points)  # Convert to numpy array for points
        # new_points_left = np.asarray(robot_pc_left.points)  # Convert to numpy array for points
        # new_points_left[:, 1] = -1.0 * new_points_left[:, 1] # flip the right hand mesh to left hand mesh
        filtered_points = new_points[new_points[:, 2] <= -0.030]
        return filtered_points #, new_points_left

def transform_right_leap_pointcloud_to_base_frame(hand_pointcloud, pose_data):
    init_hand_rotation = R.from_euler('ZXY', [-np.pi/2, -np.pi, 0])
    pos = pose_data[:3]
    rotation = R.from_rotvec(pose_data[3:]).as_matrix()
    rotation = np.matmul(rotation, init_hand_rotation.as_matrix())

    points = hand_pointcloud[:, :3]
    # points = np.matmul(init_hand_rotation.as_matrix(), points.T).T
    points = (np.dot(points, rotation.T) + pos)

    return points

def cal_tactile_pointcloud(tip_frame_xyz, tip_frame_ori, theta, phi, force):
    rad_of_dottip = 0.0152
    point = []
    contact_point_frame = tip_frame_ori @ R.from_euler('XYZ', [theta, phi, 0], 
                            degrees=True).as_matrix()
    contact_point_0 =  contact_point_frame[:, 2] / np.linalg.norm(contact_point_frame[:, 2]) * rad_of_dottip
    
    contact_point = tip_frame_xyz + create_circle_points(0.0016, 8, center=contact_point_0, ori=contact_point_frame)
    point = list(contact_point)
    point.append(contact_point_0 + tip_frame_xyz)
    return point
        
def create_circle_points(radius, num_points, center, ori):
    """
    create a circle points in 3D space, the normal vector of the circle is the z axis of the frame
    
    parameters:
    - radius: 
    - num_points: number of sumple points
    - center: center of circle (3D point)
    - ori: frame of circle (3D vector)
    
    return:
    - points: sample points in 3D space
    """
    tangent = ori[:, 0] / np.linalg.norm(ori[:, 0])
    bitangent = ori[:, 1] / np.linalg.norm(ori[:, 1])
    
    angles = np.linspace(0, 2 * np.pi, num_points)
    points = np.array([
        center + radius * (np.cos(angle) * tangent + np.sin(angle) * bitangent)
        for angle in angles
    ])
    return points

# ==============for visualization=====================
# 创建一个函数来绘制坐标系
def draw_coordinate_frame(size=0.1, origin=[0, 0, 0], orientation=np.diag([1, 1, 1])):
    # 定义原点和三个轴的终点
    origin = np.array(origin)
    x_axis = origin + orientation[:, 0]*size
    y_axis = origin + orientation[:, 1]*size
    z_axis = origin + orientation[:, 2]*size

    # 创建线段集合 (LineSet)
    points = [origin, x_axis, origin, y_axis, origin, z_axis]
    lines = [[0, 1], [2, 3], [4, 5]]  # 连接点的索引
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # 红色 (x), 绿色 (y), 蓝色 (z)

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


# ==============unit test=====================
if __name__ == "__main__":
    t0 = time.time()
    p = pointcloud_processor()
    pos = np.ones(16)* 0
    # pos[3] = 0.65
    # pos[0] = -0.65

    pc = p.get_mesh_pointcloud(joint_pos=pos) 
    print(f"Time elapsed 1: {time.time() - t0}")
    pc = transform_right_leap_pointcloud_to_base_frame(pc, [0.1,0.1,0.1, 3.141, 0, 0])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    coordinate_frame = draw_coordinate_frame(size=0.5)
    print(f"Time elapsed 2: {time.time() - t0}")

    t1 = time.time()
    for i in range(100):

        pos = np.ones(16) * i / 100
        pc = p.get_mesh_pointcloud(joint_pos=pos)
        pc = transform_right_leap_pointcloud_to_base_frame(pc, [0.1,0.1,0.1, 3.141, 0, 0])

    print(f"Time elapsed 3: {time.time() - t1}")

    # visualize the point cloud
    # o3d.visualization.draw_geometries([pcd, coordinate_frame])