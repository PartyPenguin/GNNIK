import time

import h5py
import mplib
import numpy as np
import sapien.core as sapien
import torch
from numpy import ndarray
from sapien.utils.viewer import Viewer

from graphs.graph_from_obs import graph_from_state
from graphs.graph_structure import OnlyRobotGraphStructure

ASSET_PATH = "../assets"
DATASET_FILE = "dataset/your_dataset.h5"
DOF = 7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class RobotPlanner:
    def __init__(self, robot_urdf, robot_srdf, active_joints, link_names):
        self.planner = mplib.Planner(
            urdf=robot_urdf,
            srdf=robot_srdf,
            user_link_names=link_names,
            user_joint_names=active_joints,
            move_group="panda_hand",
            joint_vel_limits=np.ones(DOF),
            joint_acc_limits=np.ones(DOF),
        )

    def plan_path(self, pose, current_qpos):
        result = self.planner.plan_screw(pose, current_qpos, time_step=1 / 40)
        if result["status"] != "Success":
            result = self.planner.plan(pose, current_qpos, time_step=1 / 40)
        if result["status"] != "Success" or len(result["position"]) > 1000:
            return {}
        return result


class RobotController:
    def __init__(self, robot):
        self.robot = robot
        self.active_joints = robot.get_active_joints()

    def next_step_model(self, pose: np.ndarray):
        graph_structure = OnlyRobotGraphStructure()
        model = torch.load('model.pt')
        model.eval()
        model.to(DEVICE)

        state = []
        current_qpos = self.robot.get_qpos()[:7]
        current_qvel = self.robot.get_qvel()[:7]
        current_tcp_p = self.robot.get_links()[12].get_pose().p
        current_tcp_q = self.robot.get_links()[12].get_pose().q
        current_tcp_pose = np.concatenate([current_tcp_p, current_tcp_q])

        robot_joint = np.vstack(
            [
                current_qpos,
                current_qvel,
                np.tile(
                    current_tcp_pose, (current_qvel.shape[0], 1)
                ).T,
                np.tile(pose, (current_qvel.shape[0], 1)).T,
            ]
        ).T
        state.extend(robot_joint)

        edges = graph_structure.get_edges()
        graph = graph_from_state(state, edges).to(DEVICE)
        pred = model(graph.x, graph.edge_index).detach().cpu().numpy()

        next_qpos = current_qpos.reshape(current_qpos.shape[0], 1).squeeze() + pred[:, 0]
        next_qvel = pred[:, 1]
        return next_qpos, next_qvel

    def follow_path(self, result):
        for i in range(len(result["position"])):
            qf = self.robot.compute_passive_force(
                gravity=True, coriolis_and_centrifugal=True
            )
            self.robot.set_qf(qf)
            for j in range(len(self.active_joints)):
                self.active_joints[j].set_drive_target(result["position"][i][j])
                self.active_joints[j].set_drive_velocity_target(
                    result["velocity"][i][j]
                )
            yield


class RobotSimulator:
    """
    Create a simulator for the robot.
    Provides methods to setup the scene and follow a path.

    Args:
        robot_urdf: The path to the robot urdf file.
    """

    def __init__(self, robot_urdf):
        self.engine = sapien.Engine()
        self.renderer = sapien.SapienRenderer()
        self.engine.set_renderer(self.renderer)
        scene_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene(scene_config)
        self.scene.set_timestep(1 / 240.0)

        loader: sapien.URDFLoader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        self.robot: sapien.Articulation = loader.load(robot_urdf)
        self.pinocchio = self.robot.create_pinocchio_model()
        self.robot.set_root_pose(sapien.Pose())

        init_qpos = [
            0,
            0.19634954084936207,
            0.0,
            -2.617993877991494,
            0.0,
            2.941592653589793,
            0.7853981633974483,
            0,
            0,
        ]
        self.robot.set_qpos(np.array(init_qpos, dtype=np.float32))

        self.active_joints = self.robot.get_active_joints()
        for joint in self.active_joints:
            joint.set_drive_property(stiffness=1000, damping=200)

    def setup_scene(self):
        self.scene.add_ground(0)
        physical_material = self.scene.create_physical_material(1, 1, 0.0)
        self.scene.default_physical_material = physical_material

        self.scene.set_ambient_light(color=np.array([0.5, 0.5, 0.5]))
        self.scene.add_directional_light(
            direction=np.array([0, 1, -1], dtype=np.float32),
            color=np.array([0.5, 0.5, 0.5], dtype=np.float32),
            shadow=True,
        )
        self.scene.add_point_light(
            position=np.array([1, 2, 2], dtype=np.float32),
            color=np.array([1, 1, 1], dtype=np.float32),
            shadow=True,
        )
        self.scene.add_point_light(
            position=np.array([1, -2, 2], dtype=np.float32),
            color=np.array([1, 1, 1], dtype=np.float32),
            shadow=True,
        )
        self.scene.add_point_light(
            position=np.array([-1, 0, 1], dtype=np.float32),
            color=np.array([1, 1, 1], dtype=np.float32),
            shadow=True,
        )

    def add_viewer(self):
        if not hasattr(self, "viewer"):
            self.viewer = Viewer(self.renderer)
            self.viewer.set_scene(self.scene)
            self.viewer.set_camera_xyz(x=1.2, y=0.25, z=0.4)
            self.viewer.set_camera_rpy(r=0, p=-0.4, y=2.7)

    def follow_path(self, result, render=False):
        n_step = result["position"].shape[0]
        for i in range(n_step):
            qf = self.robot.compute_passive_force(
                gravity=True, coriolis_and_centrifugal=True
            )
            self.robot.set_qf(qf)
            for j in range(7):
                self.active_joints[j].set_drive_target(result["position"][i][j])
                self.active_joints[j].set_drive_velocity_target(
                    result["velocity"][i][j]
                )
            self.scene.step()
            if i % 4 == 0 and render:
                self.scene.update_render()
                self.viewer.render()

    def move_to(self, goal_pose: np.ndarray, controller: RobotController):
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(half_size=[0.02, 0.02, 0.02], color=[1, 0, 0])
        self.red_cube = builder.build_static(name='red_cube')
        self.red_cube.set_pose(sapien.Pose(goal_pose[:3], goal_pose[3:]))
        print("Goal pose:", goal_pose)

        is_there = False
        threshold = 0.1
        max_time = 10  # seconds
        start_time = time.time()
        steps = 0
        while not is_there and time.time() - start_time < max_time:
            q_pos, q_vel = controller.next_step_model(goal_pose)
            qf = self.robot.compute_passive_force(
                gravity=True, coriolis_and_centrifugal=True, external=False
            )
            self.robot.set_qf(qf)
            for j in range(7):
                self.active_joints[j].set_drive_target(q_pos[j])
                self.active_joints[j].set_drive_velocity_target(q_vel[j])
            self.scene.step()
            if steps % 4 == 0 and self.renderer:
                self.scene.update_render()
                self.viewer.render()
            steps += 1
            if np.linalg.norm(self.robot.get_links()[12].get_pose().p - goal_pose[:3]) < threshold:
                print("Reached the goal!")
                is_there = True
        if not is_there:
            print("Failed to reach the goal!")


class DatasetGenerator:
    """
    Generate a dataset of observations and actions for the robot.
    """

    def __init__(self, robot_simulator: RobotSimulator, robot_planner: RobotPlanner):
        self.robot_simulator = robot_simulator
        self.robot_planner = robot_planner

    def generate_dataset(self, num_experiments: int):
        with h5py.File(DATASET_FILE, "w") as hdf_file:
            starting_pose = self.robot_simulator.robot.get_qpos()
            saved_experiments = 0
            while saved_experiments < num_experiments:
                joint_pose = self.generate_random_joint_pose()
                result = self.robot_planner.plan_path(joint_pose, starting_pose)
                if not result:
                    continue
                obs, action = self.prepare_data(result, goal_joint_pose=joint_pose)
                self.save_experiment_data(hdf_file, saved_experiments, obs, action)
                starting_pose = self.update_starting_pose(result)
                saved_experiments += 1

    def generate_random_joint_pose(self):
        return np.random.uniform(-1, 1, size=DOF)

    def save_experiment_data(self, hdf_file, experiment_number, obs, action):
        group_name = f"experiment_{experiment_number}"
        group = hdf_file.create_group(group_name)
        group.create_dataset("obs", data=obs)
        group.create_dataset("action", data=action)
        print(f"Data for experiment {experiment_number} saved.")

    def update_starting_pose(self, result):
        return np.concatenate([result["position"][-1], np.zeros(2)])

    def compute_end_effector_positions(self, result, gripper_pos):
        """Compute the end effector positions for each step in the result."""
        n_steps = result["position"].shape[0]
        ee_pos = []
        for j in range(n_steps):
            q_pos = result["position"][j]
            q_pos_all = np.concatenate([q_pos, gripper_pos], axis=0).astype(np.float64)
            self.robot_simulator.pinocchio.compute_forward_kinematics(q_pos_all)
            ee_pos.append(self.robot_simulator.pinocchio.get_link_pose(12))
        return ee_pos

    def extract_position_and_rotation(self, ee_pos):
        """Extract the position and rotation components of the end effector poses."""
        ee_pos_pos = np.array([pose.p for pose in ee_pos])
        ee_pos_rot = np.array([pose.q for pose in ee_pos])
        return np.concatenate([ee_pos_pos, ee_pos_rot], axis=1)

    def compute_joint_positions_and_velocities(self, result):
        """Compute the joint positions and velocities from the result."""
        q_pos = np.array(result["position"])
        q_pos_delta = np.concatenate([(q_pos[:-1] - q_pos[1:]).reshape(-1, 7), np.zeros((1, 7))])
        q_vel = result["velocity"]
        return q_pos, q_pos_delta, q_vel

    def compute_goal_pose(self, goal_joint_pose: ndarray, gripper_pos: ndarray) -> ndarray:
        """Compute the goal pose based on the goal joint pose using forward kinematics."""
        self.robot_simulator.pinocchio.compute_forward_kinematics(
            np.concatenate([goal_joint_pose, gripper_pos], axis=0))
        goal_pose = self.robot_simulator.pinocchio.get_link_pose(12)
        goal_pose_p: ndarray = goal_pose.p
        goal_pose_q: ndarray = goal_pose.q
        goal_pose_raw = np.concatenate([goal_pose_p, goal_pose_q])
        return goal_pose_raw

    def prepare_data(self, result: dict, goal_joint_pose: np.ndarray):
        """Prepare the observation and action data from the result and the goal joint pose."""
        gripper_pos = np.array([1, 1], dtype=np.float64)
        ee_pos = self.compute_end_effector_positions(result, gripper_pos)
        ee_pos = self.extract_position_and_rotation(ee_pos)
        q_pos, q_pos_delta, q_vel = self.compute_joint_positions_and_velocities(result)
        goal_pose = self.compute_goal_pose(goal_joint_pose, gripper_pos)
        goal_pose = np.asarray(goal_pose).reshape(goal_joint_pose.shape[0], 1)
        goal_pose_tile = np.tile(goal_pose, (q_pos.shape[0])).T
        obs = np.concatenate([q_pos, q_vel, ee_pos, goal_pose_tile], axis=1)
        q_vel_shifted = np.concatenate([np.zeros((1, 7)), q_vel[:-1]], axis=0)
        action = np.concatenate([q_pos_delta, q_vel_shifted], axis=1)
        return obs, action


def main():
    robot_simulator = RobotSimulator(
        robot_urdf=ASSET_PATH + "/robot/franka_panda/panda.urdf"
    )
    robot = robot_simulator.robot
    robot_planner = RobotPlanner(
        robot_urdf=ASSET_PATH + "/robot/franka_panda/panda.urdf",
        robot_srdf=ASSET_PATH + "/robot/franka_panda/panda.srdf",
        active_joints=[joint.get_name() for joint in robot.get_active_joints()],
        link_names=[link.get_name() for link in robot.get_links()],
    )
    robot_controller = RobotController(robot)

    # dataset_generator = DatasetGenerator(robot_simulator, robot_planner)
    # dataset_generator.generate_dataset(num_experiments=1000)
    # result = robot_planner.plan_from_model(robot)

    # Load and follow a path from the generated dataset
    robot_simulator.add_viewer()
    robot_simulator.setup_scene()

    # Generate Random pose that is reachable by the robot
    valid_pose = False
    while not valid_pose:
        # cartesian_pose = np.random.uniform(-1, 1, size=7)
        cartesian_pose = np.array([-0.3, 0.3, 0.3, 0, 0, 0, 1])
        # calculate inverse kinematics
        status, result = robot_planner.planner.IK(cartesian_pose, robot.get_qpos())
        print(f'IK status: {status}')
        if status == "Success":
            valid_pose = True

    robot_simulator.move_to(cartesian_pose, robot_controller)

    # with h5py.File(DATASET_FILE, "r") as hdf_file:
    #     for group_name in hdf_file.keys():
    #         obs = hdf_file[group_name]["obs"][:]
    #         q_pos = obs[:, :7]
    #         q_vel = obs[:, 7:14]
    #         path = {"position": q_pos, "velocity": q_vel}
    #         robot_simulator.follow_path(path, render=True)


if __name__ == "__main__":
    main()
