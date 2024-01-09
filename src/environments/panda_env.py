import numpy as np
import sapien.core
from environments.sapien_env import SapienEnv
from gymnasium import spaces
from sapien.core import Pose
from sapien.utils.viewer import Viewer


class PandaEnv(SapienEnv):
    def __init__(self):
        self.init_qpos = [0, 0.19634954084936207, 0.0, -2.617993877991494,
                          0.0, 2.941592653589793, 0.7853981633974483, 0, 0]
        super().__init__(control_freq=20, timestep=0.01)

        self.robot = self.get_articulation('panda')
        self.end_effector = self.robot.get_links()[8]
        self.dof = self.robot.dof
        assert self.dof == 9, 'Panda should have 9 DoF'
        self.links = self.robot.get_links()
        self.active_joints = self.robot.get_active_joints()
        self.cube: sapien.core.Articulation = self.get_actor('cube')
        # Randomize initial position of the joints given the joint limits
        self.init_qpos = [np.random.default_rng().uniform(low=ql, high=qu) for ql, qu in self.robot.get_qlimits()]

        # The arm is controlled by the internal velocity drive
        for joint in self.active_joints[:5]:
            joint.set_drive_property(stiffness=1000, damping=200)
        for joint in self.active_joints[5:7]:
            joint.set_drive_property(stiffness=1000, damping=200)
        # The gripper will be controlled directly by the torque

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=[self.dof * 2 + 13], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=[self.dof], dtype=np.float32)

    # ---------------------------------------------------------------------------- #
    # Simulation world
    # ---------------------------------------------------------------------------- #
    def _build_world(self):
        physical_material = self._scene.create_physical_material(1.0, 1.0, 0.0)
        self._scene.default_physical_material = physical_material
        # self._scene.add_ground(0.0)

        # cube
        builder: sapien.core.ActorBuilder = self._scene.create_actor_builder()
        builder.add_box_visual(half_size=[0.02, 0.02, 0.02], color=[1, 0, 0])
        cube = builder.build_static(name='cube')

        # robot
        loader: sapien.core.ActorBuilder = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        robot = loader.load('../assets/robot/franka_panda/panda.urdf')
        robot.set_name('panda')
        robot.set_root_pose(Pose([0, 0, 0]))
        robot.set_qpos(self.init_qpos)

    # ---------------------------------------------------------------------------- #
    # RL
    # ---------------------------------------------------------------------------- #
    def step(self, action):
        # Use internal velocity drive
        qf = self.robot.compute_passive_force(True, True, False)
        self.robot.set_qf(qf)
        for idx in range(7):
            # self.active_joints[idx].set_drive_velocity_target(action[idx])
            self.active_joints[idx].set_drive_target(action[idx])
        # Control the gripper directly by torque

        for i in range(self.control_freq):
            self._scene.step()

        obs = self.get_obs()
        reward = self._get_reward()

        cube_pose = self.cube.get_pose()
        ee_pose = self.end_effector.get_pose()
        cube_to_ee = ee_pose.p - cube_pose.p
        done = np.sum(cube_to_ee) < 0.1
        if done:
            reward += 100.0

        return obs, reward, done, {}

    def reset(self):
        self.robot.set_qpos(self.init_qpos)
        self.cube.set_pose(Pose([-0.4, 0.2, 0.3]))
        self._scene.step()
        return self.get_obs()

    def get_obs(self):
        qpos = self.robot.get_qpos()
        qvel = self.robot.get_qvel()
        cube_pose = self.cube.get_pose()
        ee_pose = self.end_effector.get_pose()
        cube_to_ee = ee_pose.p - cube_pose.p
        return np.hstack([qpos, qvel, cube_pose.p, cube_pose.q, cube_to_ee])

    def _get_reward(self):
        # reaching reward
        cube_pose = self.cube.get_pose()
        ee_pose = self.end_effector.get_pose()
        distance = np.linalg.norm(ee_pose.p - cube_pose.p)
        reaching_reward = 1 - np.tanh(10.0 * distance)

        # lifting reward
        lifting_reward = max(
            0, self.cube.pose.p[2] - 0.02) / 0.02

        return reaching_reward + lifting_reward

    # ---------------------------------------------------------------------------- #
    # Visualization
    # ---------------------------------------------------------------------------- #
    def _setup_lighting(self):

        self._scene.set_ambient_light([.4, .4, .4])
        self._scene.add_directional_light([1, -1, -1], [0.3, 0.3, 0.3])
        self._scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _setup_viewer(self):
        self._setup_lighting()
        self.viewer = Viewer(self._renderer)
        self.viewer.set_scene(self._scene)
        self.viewer.set_camera_xyz(x=1.5, y=0.0, z=2.0)
        self.viewer.set_camera_rpy(y=3.14, p=-0.5, r=0)
