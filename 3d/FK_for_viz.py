from pathlib import Path
from sys import argv
import numpy as np
from scipy.spatial.transform import Rotation as R
import pinocchio.visualize as visualize
import matplotlib.pyplot as plt
import pinocchio

#>>>>>>>>> load the model <<<<<<<<<<<<<<<<<<
 
# This path refers to Pinocchio source code but you can define your own directory here.
pinocchio_model_dir = Path(__file__).parent.parent / "models"
 
# You should change here to set up your own URDF file or just pass it as an argument of
# this example.
urdf_filename = "diffusion_policy/real_world/leap_hand_utils/assets/leap_hand_dottip/leap_hand_right_for_pointcloud.urdf"
mesh_dir = pinocchio_model_dir / "diffusion_policy/real_world/leap_hand_utils/assets/leap_hand_dottip/meshes"

 
# Load the urdf model
# model, collision_model, visual_model = pinocchio.buildModelsFromUrdf(
#     urdf_filename, mesh_dir
# )
model = pinocchio.buildModelFromUrdf(
    urdf_filename
)
print("model name: " + model.name)
 
# Create data required by the algorithms
data = model.createData()

#>>>>>>>>> update the model and forward kinematics<<<<<<<<<<<<<<<<<<
 
# Sample a random configuration
# q = pinocchio.randomConfiguration(model)
q = np.zeros(model.nq)
q[1] = -0.5
q[0] = 0.5
q[2] = 1
print(f"q: {q.T}")
 
# Perform the forward kinematics over the kinematic tree
pinocchio.forwardKinematics(model, data, q)
pinocchio.updateFramePlacements(model, data)

translations = []
rotvecs = []
joint_names = []

#>>>>>>>>>> print out the placement of each joint <<<<<<<<<<<<<<<<<<
 
# Print out the placement of each joint of the kinematic tree
for name, oMi in zip(model.names, data.oMi):
    translation = oMi.translation.T
    rotation_matrix = oMi.rotation
    rotvec = R.from_matrix(rotation_matrix).as_rotvec()  # Convert rotation matrix to rotation vector
    print(name, ":" , translation, rotvec)
    # print("{:<24} : {: .2f} {: .2f} {: .2f} | rotvec: {: .2f} {: .2f} {: .2f}".format(
    #     name, *translation, *rotvec))
    translations.append(translation)
    rotvecs.append(rotvec)
    joint_names.append(name)
    
# Convert to numpy arrays for easier manipulation
translations = np.array(translations)
rotvecs = np.array(rotvecs)


##### for viz
# Plot the translations and rotation vectors
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot translations as points
ax.scatter(translations[:, 0], translations[:, 1], translations[:, 2], c='b', label='Joint Positions')
ax.scatter(0, 0, 0, c='r', label='Origin')

# Plot frames as quivers (X, Y, Z axes)
for i, (translation, rotation_matrix) in enumerate(zip(translations, [oMi.rotation for oMi in data.oMi])):
    x_axis = rotation_matrix[:, 0]  # X-axis
    y_axis = rotation_matrix[:, 1]  # Y-axis
    z_axis = rotation_matrix[:, 2]  # Z-axis
    ax.quiver(
        translation[0], translation[1], translation[2],  # Starting point
        x_axis[0], x_axis[1], x_axis[2],                # X-axis direction
        length=0.02, color='r', label='X-axis' if i == 0 else None
    )
    ax.quiver(
        translation[0], translation[1], translation[2],  # Starting point
        y_axis[0], y_axis[1], y_axis[2],                # Y-axis direction
        length=0.02, color='g', label='Y-axis' if i == 0 else None
    )
    ax.quiver(
        translation[0], translation[1], translation[2],  # Starting point
        z_axis[0], z_axis[1], z_axis[2],                # Z-axis direction
        length=0.02, color='b', label='Z-axis' if i == 0 else None
    )

##### for viz


#>>>>>>>>> visualize the dottips' frame <<<<<<<<<<<<<<<<<<<<

# Visualize specific fixed joints
fixed_joint_names_to_visualize = ["if_tip", "mf_tip", "rf_tip", "tf_tip"]
dottip_frame = {}
for fixed_joint_name in fixed_joint_names_to_visualize:
    frameId = model.getFrameId(fixed_joint_name)
    print(f"Frame ID for {fixed_joint_name}: {frameId}")

    oMi_fixed = data.oMf[frameId]
    translation_fixed = oMi_fixed.translation.T
    rotation_matrix_fixed = (R.from_matrix(oMi_fixed.rotation) * R.from_euler('ZXY', [-np.pi/2, np.pi/2, 0])).as_matrix()
    rotvec_fixed = R.from_matrix(rotation_matrix_fixed).as_rotvec()

    dottip_frame[fixed_joint_name] = {
        "translation": translation_fixed,
        "rotation_matrix": rotation_matrix_fixed,
    }


    #####

    print(f"Fixed Joint '{fixed_joint_name}': Translation={translation_fixed}, RotVec={rotvec_fixed}")

    # Plot the fixed joint
    ax.scatter(
        translation_fixed[0], translation_fixed[1], translation_fixed[2],
        label=f'Fixed Joint: {fixed_joint_name}'
    )
    ax.quiver(
        translation_fixed[0], translation_fixed[1], translation_fixed[2],
        rotation_matrix_fixed[:, 0][0], rotation_matrix_fixed[:, 0][1], rotation_matrix_fixed[:, 0][2],
        length=0.03, color='m', label='X-axis' if fixed_joint_name == "if_tip" else None
    )
    ax.quiver(
        translation_fixed[0], translation_fixed[1], translation_fixed[2],
        rotation_matrix_fixed[:, 1][0], rotation_matrix_fixed[:, 1][1], rotation_matrix_fixed[:, 1][2],
        length=0.03, color='c', label='Y-axis' if fixed_joint_name == "if_tip" else None
    )
    ax.quiver(
        translation_fixed[0], translation_fixed[1], translation_fixed[2],
        rotation_matrix_fixed[:, 2][0], rotation_matrix_fixed[:, 2][1], rotation_matrix_fixed[:, 2][2],
        length=0.03, color='y', label='Z-axis' if fixed_joint_name == "if_tip" else None
    )

    ##### for viz

#>>>>>>> plot the others <<<<<<<<

dottip_contact_info = [0, 0, 0, 0, 0]

#
theta = 45
phi = 0
contact_point = dottip_frame["if_tip"]["rotation_matrix"] @ R.from_euler('XYZ', [theta, phi, 0], 
                             degrees=True).as_matrix() 
ax.quiver(
    dottip_frame["if_tip"]["translation"][0], dottip_frame["if_tip"]["translation"][1], dottip_frame["if_tip"]["translation"][2],
    contact_point[:, 2][0], contact_point[:, 2][1], contact_point[:, 2][2],
    length=0.04, color='r', label='contact point'
)
# Add labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
ax.set_title('Joint Translations and Frames')

# Annotate joint names
for i, name in enumerate(joint_names):
    ax.text(translations[i, 0], translations[i, 1], translations[i, 2], name, fontsize=8)

plt.show()