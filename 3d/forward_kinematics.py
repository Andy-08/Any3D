from pathlib import Path
from sys import argv
import numpy as np
from scipy.spatial.transform import Rotation as R
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
    

if __name__ == "__main__":
    urdf_filename = "leap_hand_utils/assets/leap_hand_dottip/leap_hand_right_for_pointcloud.urdf"
    leapFK = Leaphand_FK(urdf_filename)
    q = np.zeros(16)
    q[1] = -0.5
    q[0] = 0.5
    q[2] = 0.2
    leapFK.update_model(q)
    joint_frame = leapFK.get_placement_all_joint()
    dottip_frame = leapFK.get_placement_all_tips_frame()

    # viz the placement of each joint and dottip frame
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for key in joint_frame:
        translation = joint_frame[key]["translation"]
        rotation_matrix = joint_frame[key]["rotation_matrix"]
        ax.scatter(translation[0], translation[1], translation[2], c='r', label=key)
        ax.quiver(
            translation[0], translation[1], translation[2],  # Starting point
            rotation_matrix[:, 0][0], rotation_matrix[:, 0][1], rotation_matrix[:, 0][2],  # X-axis direction
            length=0.02, color='r', label='X-axis' if key == "j01" else None
        )
        ax.quiver(
            translation[0], translation[1], translation[2],  # Starting point
            rotation_matrix[:, 1][0], rotation_matrix[:, 1][1], rotation_matrix[:, 1][2],  # Y-axis direction
            length=0.02, color='g', label='Y-axis' if key == "j01" else None
        )
        ax.quiver(
            translation[0], translation[1], translation[2],  # Starting point
            rotation_matrix[:, 2][0], rotation_matrix[:, 2][1], rotation_matrix[:, 2][2],  # Z-axis direction
            length=0.02, color='b', label='Z-axis' if key == "j01" else None
        )

    for key in dottip_frame:
        print(key)
        translation = dottip_frame[key]["translation"]
        rotation_matrix = dottip_frame[key]["rotation_matrix"]
        ax.scatter(translation[0], translation[1], translation[2], c='g', label=key)
        ax.quiver(
            translation[0], translation[1], translation[2],  # Starting point
            rotation_matrix[:, 0][0], rotation_matrix[:, 0][1], rotation_matrix[:, 0][2],  # X-axis direction
            length=0.02, color='m', label='X-axis' if key == "if_tip" else None
        )
        ax.quiver(
            translation[0], translation[1], translation[2],  # Starting point
            rotation_matrix[:, 1][0], rotation_matrix[:, 1][1], rotation_matrix[:, 1][2],  # Y-axis direction
            length=0.02, color='c', label='Y-axis' if key == "if_tip" else None
        )
        ax.quiver(
            translation[0], translation[1], translation[2],  # Starting point
            rotation_matrix[:, 2][0], rotation_matrix[:, 2][1], rotation_matrix[:, 2][2],  # Z-axis direction
            length=0.02, color='y', label='Z-axis' if key == "if_tip" else None
        )

    plt.show()

