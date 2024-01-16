import h5py
import numpy as np
import sapien.core as sapien
import torch

from environments.panda_env import PandaEnv
from graphs.graph_from_obs import graph_from_state
from graphs.graph_structure import OnlyRobotGraphStructure
from motion_planning import MotionPlanner
from graph_learning import MLP

from matplotlib import pyplot as plt
graph_structure = OnlyRobotGraphStructure()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# SCALER_PARAMS = joblib.load("models/scaler_params.pkl")
# MEAN = SCALER_PARAMS['mean']
# STD = SCALER_PARAMS['std']


def save_to_file(obs, actions, id):
    with h5py.File('dataset/raw/experiment.h5', 'a') as f:
        g = f.create_group(f'experiment_{id}')
        g.create_dataset('obs', data=obs)
        g.create_dataset('actions', data=actions)
    print(f'Experiment No.{id}')


def init_model():
    model = torch.load('models/model_4.pt')
    model.eval()
    model.to(DEVICE)
    return model


def generate_target_pose():
    radius = 0.5
    offset = [0.2, 0, 0]
    # Sample spherical coordinates
    theta = np.random.uniform(0, 2 * np.pi)  # Azimuthal angle
    phi = np.arccos(2 * np.random.rand() - 1)  # Inclination angle

    # Convert spherical coordinates to Cartesian coordinates
    x = radius * np.sin(phi) * np.cos(theta) + offset[0]
    y = radius * np.sin(phi) * np.sin(theta) + offset[1]
    z = radius * np.cos(phi) + offset[2]

    # Generate a target pose that faces towards the robot. The orientation is represented as a quaternion
    position = np.asarray([x, y, z])
    # Calculate direction to robot
    direction_vector = - (position / np.linalg.norm(position))
    up_vector = np.asarray([0, 0, 1])
    # Calculate the rotation matrix
    right_vector = np.cross(direction_vector, up_vector)
    right_vector /= np.linalg.norm(right_vector)
    up_vector = np.cross(right_vector, direction_vector)

    # Create rotation matrix
    rotation_matrix = np.array([right_vector, up_vector, direction_vector])
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = position
    pose = sapien.pysapien.Pose.from_transformation_matrix(transformation_matrix)
    # Return target pose
    return pose


def run_policy(obs, model: torch.nn.Module, env: PandaEnv):
    robot_joint_pos = obs[:7]  # 7 dim joint positions
    robot_joint_vel = obs[9:16]  # 7 dim joint velocities
    goal_pos = obs[18:21]  # 7 dim goal pose
    goal_to_ee = obs[25:28]  # 3 dim goal to ee distance

    # Add robot joint positions, velocities, tcp pose and goal position to feature
    robot_joint = np.vstack(
        [
            robot_joint_pos,
            robot_joint_vel,
            np.tile(
                goal_to_ee, (robot_joint_pos.shape[0], 1)
            ).T,
            np.tile(goal_pos, (robot_joint_pos.shape[0], 1)).T,
        ]
    ).T
    features = []
    features.extend(robot_joint)
    graph = graph_from_state(features, graph_structure.get_edges()).to(DEVICE)

    # normalize node feature
    # graph.x = graph.x.cpu()
    # graph.x = (graph.x - MEAN) / STD
    # graph.x = graph.x.to(DEVICE)

    pred = model(graph).detach().cpu().numpy()

    return pred


def generate_dataset(env):
    traj = 0
    num_traj = 500
    max_steps = 500
    mp = MotionPlanner(env=env)
    mp.setup_planner()
    while traj < num_traj:
        target_pose = generate_target_pose()
        env.cube.set_pose(target_pose)
        env.render()
        trajectory = mp.move_to_pose(np.concatenate([target_pose.p, target_pose.q]), with_screw=True)
        if trajectory == -1 or trajectory['time'].shape[0] > max_steps:
            continue
        all_obs = []
        all_actions = []
        for pos in trajectory['position']:
            env.render()

            action = pos
            obs, _, _, _ = env.step(action)
            all_obs.append(obs)
            all_actions.append(action)
        save_to_file(all_obs, all_actions, traj)
        traj += 1


# def main():
#     env = PandaEnv()
#     env.reset()
#     generate_dataset(env)


def main():
    env = PandaEnv()
    env.reset()
    model = init_model()
    q_limits = env.robot.get_qlimits()
    # Initialize plot
    plt.ion()  # Turn on interactive mode
    fig, axs = plt.subplots(7, 1, sharex=True)
    fig.suptitle('Robot joint positions over time')

    all_actions = [[] for _ in range(7)]  # Initialize list to store joint positions
    all_obs = [[] for _ in range(7)]  # Initialize list to store joint positions
    all_mp_actions = [[] for _ in range(7)]  # Initialize list to store joint positions

    obs = env.get_obs()
    target_pose = [0.4, -0.3, 0.3, 0, 1 , 0, 0]
    env.cube.set_pose(sapien.Pose(target_pose[:3]))

    # Get motion planner result
    mp = MotionPlanner(env=env)
    mp.setup_planner()
    trajectory = mp.move_to_pose(target_pose, with_screw=True)
    if trajectory == -1:
        print('No solution found')
        return
    step = 0
    while step < 2000:
        print(step)
        env.render()
        model_actions = run_policy(obs, model, env)

        # Append current joint positions to list
        for i, joint_position in enumerate(model_actions):
            all_actions[i].append(joint_position)
            all_obs[i].append(obs[i])
            if step < len(trajectory['position']):
                all_mp_actions[i].append(trajectory['position'][step][i])
            else:
                all_mp_actions[i].append(trajectory['position'][-1][i])

                # Update plot
        for i, ax in enumerate(axs):
            ax.clear()
            # Action value
            ax.plot(all_actions[i], 'b--')
            # Joint position
            ax.plot(all_obs[i] * np.ones(len(all_actions[i])), 'c--')
            # Upper joint limits
            ax.plot(q_limits[i][1] * np.ones(len(all_actions[i])), 'r-')
            # Lower joint limits
            ax.plot(q_limits[i][0] * np.ones(len(all_actions[i])), 'r-')

            # Motion planner result
            ax.plot(all_mp_actions[i], 'g--')

            ax.set_ylabel(f'Joint {i + 1}')
        axs[-1].set_xlabel('Time step')
        plt.pause(0.001)  # Small delay to allow plot to update

        obs, reward, done, info = env.step(model_actions)
        print(obs)
        step += 1

    print('Done')
    env.close()
    plt.ioff()  # Turn off interactive mode
    plt.show()


if __name__ == '__main__':
    main()
