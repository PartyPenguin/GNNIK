import mplib
import numpy as np

from environments.panda_env import PandaEnv


class MotionPlanner:
    def __init__(self, env: PandaEnv):
        self.planner = None
        self.env = env

    def setup_planner(self):
        link_names = [link.get_name() for link in self.env.links]
        joint_names = [joint.get_name() for joint in self.env.active_joints]

        self.planner = mplib.Planner(
            urdf="../assets/robot/franka_panda/panda.urdf",
            srdf="../assets/robot/franka_panda/panda.srdf",
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group="panda_hand",
            joint_vel_limits=np.ones(7),
            joint_acc_limits=np.ones(7)
        )

    def open_gripper(self):
        for joint in self.active_joints[-2:]:
            joint.set_drive_target(0.4)
        for i in range(100):
            qf = self.robot.compute_passive_force(
                gravity=True,
                coriolis_and_centrifugal=True)
            self.robot.set_qf(qf)
            self.scene.step()
            if i % 4 == 0:
                self.scene.update_render()
                self.viewer.render()

    def close_gripper(self):
        for joint in self.active_joints[-2:]:
            joint.set_drive_target(0)
        for i in range(100):
            qf = self.robot.compute_passive_force(
                gravity=True,
                coriolis_and_centrifugal=True)
            self.robot.set_qf(qf)
            self.scene.step()
            if i % 4 == 0:
                self.scene.update_render()
                self.viewer.render()

    def move_to_pose_with_RRTConnect(self, pose):
        result = self.planner.plan(pose, self.env.robot.get_qpos(), time_step=self.env.timestep, planning_time=2)
        if result['status'] != "Success":
            return -1
        return result

    def move_to_pose_with_screw(self, pose):
        result = self.planner.plan_screw(pose, self.env.robot.get_qpos(), time_step=self.env.timestep)
        if result['status'] != "Success":
            result = self.planner.plan(pose, self.env.robot.get_qpos(), time_step=self.env.timestep, planning_time=2)
            if result['status'] != "Success":
                return -1
        return result

    def move_to_pose(self, pose, with_screw):
        if with_screw:
            return self.move_to_pose_with_screw(pose)
        else:
            return self.move_to_pose_with_RRTConnect(pose)

    def check_for_collision(self, qpos=None):
        self_col_result = self.planner.check_for_self_collision(qpos=qpos)

        if self_col_result:
            return True
        return False
