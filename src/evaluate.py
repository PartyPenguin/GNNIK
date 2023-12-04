"""
This script runs a simulation of the "PickCube" environment using a trained model to control the robot.
The model takes in a graph representation of the current state of the environment and outputs a joint action for the robot.
"""

import torch
import numpy as np
import gymnasium as gym
import mani_skill2.envs
from graphs.graph_from_obs import hetero_graph_from_state
from graphs.graph_structure import FirstGraphStructure


def get_poses(env, obs):
    """
    Helper function to extract the poses of the cube, goal, and TCP from the environment observation.

    Args:
        env (gym.Env): The environment object.
        obs (dict): The current observation from the environment.

    Returns:
        tuple: A tuple containing the pose of the cube, goal, and TCP.
    """
    cube_pose = np.concatenate(
        [
            env.unwrapped.get_actors()[1].get_pose().p,
            env.unwrapped.get_actors()[1].get_pose().q,
        ]
    )
    goal_pose = np.concatenate((obs["extra"]["goal_pos"], np.array([0, 0, 0, 1])))
    tcp_pose = obs["extra"]["tcp_pose"]
    return cube_pose, goal_pose, tcp_pose


def get_joint_features(env, obs, tcp_to_obj_pos, obj_to_goal_pos):
    """
    Helper function to extract the joint positions and velocities of the robot from the environment observation.

    Args:
        env (gym.Env): The environment object.
        obs (dict): The current observation from the environment.

    Returns:
        list: A list of joint positions and velocities.
    """
    robot_joint = [
        ["joint", s]
        for s in np.hstack(
            (
                np.stack((obs["agent"]["qpos"], obs["agent"]["qvel"])).T,
                np.tile(
                    tcp_to_obj_pos,
                    (obs["agent"]["qpos"].shape[0], 1),
                ),
                np.tile(
                    obj_to_goal_pos,
                    (obs["agent"]["qpos"].shape[0], 1),
                ),
            )
        )
    ]
    return robot_joint


def main():
    """
    The main function that runs the simulation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("src/models/model.pt")
    model.eval()
    model.to(device)

    graph_structure = FirstGraphStructure()

    env = gym.make(
        "PickCube-v0",
        obs_mode="rgbd",
        control_mode="pd_joint_delta_pos",
        render_mode="human",
    )
    print("Observation space", env.observation_space)
    print("Action space", env.action_space)

    obs, reset_info = env.reset()  # reset with a seed for randomness
    terminated, truncated = False, False

    while not terminated and not truncated:
        state = []  # The state of the current timestep
        cube_pose, goal_pose, tcp_pose = get_poses(env, obs)

        tcp_to_obj_pos = cube_pose[:3] - tcp_pose[:3]
        obj_to_goal_pos = goal_pose[:3] - cube_pose[:3]

        robot_joint = get_joint_features(env, obs, tcp_to_obj_pos, obj_to_goal_pos)

        state.extend(robot_joint)
        state.append(["pose", np.concatenate((tcp_pose, tcp_to_obj_pos))])
        state.append(["pose", np.concatenate((cube_pose, obj_to_goal_pos))])
        state.append(["pose", np.concatenate((goal_pose, obj_to_goal_pos))])

        edges = graph_structure.get_edges()
        graph = hetero_graph_from_state(state, edges).to(device)

        action = (
            model(graph.x_dict, graph.edge_index_dict)["joint"]
            .cpu()
            .detach()
            .numpy()[:-1]
            .flatten()
        )
        obs, reward, terminated, truncated, info = env.step(action)
        for _ in range(10):
            env.render()

    env.close()


if __name__ == "__main__":
    main()
