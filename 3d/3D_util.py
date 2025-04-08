import open3d as o3d
import numpy as np
from yourdfpy import URDF
import copy
from scipy.spatial.transform import Rotation as R
import time

class pointcloud_processor:
    def __init__(self):

        self.hand_states = []
        self.hand_stl_list = []
        self.meshes = {}
        self.urdf_dict = {}
        self.Leap_urdf = URDF.load("diffusion_policy/real_world/leap_hand_utils/assets/leap_hand_dottip/leap_hand_right_for_pointcloud.urdf")
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
        robot_pc = right_mesh.sample_points_uniformly(number_of_points=8000)

        # self.Leap_urdf_2.update_cfg(joint_pos_left)
        # left_mesh = self._update_meshes("left_leap")  # Get the new updated mesh
        # robot_pc_left = left_mesh.sample_points_uniformly(number_of_points=80000)

        # Convert the sampled mesh point cloud to the format expected by Open3D
        new_points = np.asarray(robot_pc.points)  # Convert to numpy array for points
        # new_points_left = np.asarray(robot_pc_left.points)  # Convert to numpy array for points
        # new_points_left[:, 1] = -1.0 * new_points_left[:, 1] # flip the right hand mesh to left hand mesh

        return new_points #, new_points_left

def transform_right_leap_pointcloud_to_camera_frame(hand_pointcloud, pose_data):
    init_hand_rotation = R.from_euler('ZXY', [-np.pi/2, -np.pi, 0])
    pos = pose_data[:3]
    rotation = R.from_rotvec(pose_data[3:]).as_matrix()
    rotation = np.matmul(rotation, init_hand_rotation.as_matrix())

    points = hand_pointcloud[:, :3]
    # points = np.matmul(init_hand_rotation.as_matrix(), points.T).T
    points = (np.dot(points, rotation.T) + pos)

    return points
        
        


# ==============辅助代码=====================
# 创建一个函数来绘制坐标系
def draw_coordinate_frame(size=1.0, origin=[0, 0, 0]):
    # 定义原点和三个轴的终点
    origin = np.array(origin)
    x_axis = origin + np.array([size, 0, 0])
    y_axis = origin + np.array([0, size, 0])
    z_axis = origin + np.array([0, 0, size])

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

if __name__ == "__main__":
    t0 = time.time()
    p = pointcloud_processor()
    pos = np.ones(16)* 0
    # pos[3] = 0.65
    # pos[0] = -0.65

    pc = p.get_mesh_pointcloud(joint_pos=pos) 
    print(f"Time elapsed 1: {time.time() - t0}")
    pc = transform_right_leap_pointcloud_to_camera_frame(pc, [0.1,0.1,0.1, 3.141, 0, 0])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    coordinate_frame = draw_coordinate_frame(size=0.5)
    print(f"Time elapsed 2: {time.time() - t0}")

    t1 = time.time()
    for i in range(100):
        
        pos = np.ones(16) * i / 100
        pc = p.get_mesh_pointcloud(joint_pos=pos)
        pc = transform_right_leap_pointcloud_to_camera_frame(pc, [0.1,0.1,0.1, 3.141, 0, 0])

    print(f"Time elapsed 3: {time.time() - t1}")

    # visualize the point cloud
    # o3d.visualization.draw_geometries([pcd, coordinate_frame])