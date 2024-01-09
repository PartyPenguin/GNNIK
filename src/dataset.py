import os
import os.path as osp
import shutil
# add path to src
from typing import List

import h5py
import numpy as np
import torch
from numpy import ndarray
from torch_geometric.data import Dataset
from tqdm import tqdm

from graphs.graph_from_obs import graph_from_state
from graphs.graph_structure import OnlyRobotGraphStructure

DATASET_PATH = "dataset/your_dataset.h5"
GRAPH_STRUCTURE = OnlyRobotGraphStructure()


def extract_feature_from_obs(obs: ndarray, actions: ndarray = None, i: int = 0) -> (ndarray, ndarray):
    """
    Extract node features and actions from observation. Each step in the trajectory has one observation which contains
    the joint positions, velocities, tcp pose, and goal position. The observations and actions are extracted and added
    to the corresponding node features.

    For example the joint positions are a 7 dimensional vector. They get transposed such that each joint node gets its
    own feature vector. The same is done for the joint velocities, tcp pose and goal position.

    Args:
        obs: observation [1, 28]
        actions: actions
        i: index of the observation

    Returns:
        feature: node features [num_nodes, num_features]
        action: action

    """
    robot_joint_pos = obs[:, :7]  # 7 dim joint positions
    robot_joint_vel = obs[:, 9:16]  # 7 dim joint velocities
    goal_pos = obs[:, 18:21]  # 7 dim goal pose
    goal_to_ee = obs[:, 25:28]  # 3 dim goal to ee distance

    # Add robot joint positions, velocities, tcp pose and goal position to feature
    robot_joint = np.vstack(
        [
            robot_joint_pos[i - 1],
            robot_joint_vel[i - 1],
            np.tile(
                goal_to_ee[i], (robot_joint_pos[i].shape[0], 1)
            ).T,
            np.tile(goal_pos[i], (robot_joint_pos[i].shape[0], 1)).T,
        ]
    ).T
    # Repeat the last joint position and velocity for the gripper node
    # robot_joint = np.concatenate((robot_joint, robot_joint[-1][None, :]), axis=0)
    # robot_joint_next = np.vstack(
    #     [
    #         robot_joint_pos[i],
    #         robot_joint_vel[i],
    #         # np.tile(
    #         #     goal_to_ee[i], (robot_joint_pos[i].shape[0], 1)
    #         # ).T,
    #         # np.tile(goal_pos[i], (robot_joint_pos[i].shape[0], 1)).T,
    #     ]
    # ).T
    # # Repeat the last joint position and velocity for the gripper node
    # robot_joint_next = np.concatenate((robot_joint_next, robot_joint_next[-1][None, :]), axis=0)
    node_feature = []
    node_feature.extend(robot_joint)
    target_feature = []
    target_feature.extend(actions[i])

    # # Add actions to the graph
    # if actions is not None:
    #     edge_feature = actions[i]
    # else:
    #     edge_feature = []
    edge_feature = None

    return node_feature, edge_feature, target_feature


class RobotGraph(Dataset):
    def __init__(self, root, mask=None, transform=None, pre_transform=None):
        self.mask = mask
        self.num_links = 7
        super(RobotGraph, self).__init__(
            root=root, transform=transform, pre_transform=pre_transform
        )

        # Count number of files in processed directory with data prefix
        self.total_data = len(
            [
                name
                for name in os.listdir(self.processed_dir)
                if name.startswith("data")
            ]
        )

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, "raw")

    @property
    def processed_dir(self) -> str:
        if self.mask is None:
            return osp.join(self.root, "processed")
        return osp.join(self.root, "processed", self.mask)

    @property
    def raw_file_names(self) -> List[str]:
        return ["experiment.h5"]  # Replace with actual raw file names if needed

    @property
    def processed_file_names(self) -> List[str]:
        return ["data_0.pt"]

    @property
    def num_output_features(self) -> int:
        return 12

    def download(self):
        # write file to raw_dir
        shutil.copy(DATASET_PATH, self.raw_paths[0])

    def save_graph(self, node_feature, edge_feature, target_feature, total_graphs):
        data = graph_from_state(node_feature, GRAPH_STRUCTURE.get_edges(), edge_feature, target_feature)
        torch.save(
            data,
            osp.join(self.processed_dir, "data_%d.pt" % (total_graphs)),
        )

    def process(self):
        with h5py.File(self.raw_paths[0], "r") as f:
            total_graphs = 0
            data_group = f
            self.total_data = len(list(data_group))
            for group_name in tqdm(data_group.keys(), desc="Processing trajectories"):
                obs = np.array(
                    f["%s/obs" % group_name]
                )  # Convert Datatype to numpy array
                actions = np.array(f["%s/actions" % group_name])

                # For each sequence, create a graph
                for i in range(len(obs) - 1):
                    edge_lists = GRAPH_STRUCTURE.get_edges()
                    node_feature, edge_feature, target_feature = extract_feature_from_obs(obs, actions, i )
                    self.save_graph(node_feature, edge_feature, target_feature, total_graphs)
                    total_graphs += 1

    def len(self):
        return self.total_data

    @property
    def num_nodes(self):
        return len(GRAPH_STRUCTURE.nodes)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, "data_%d.pt" % (idx)))
        return data

# dataset = RobotGraph(root="dataset")
#
# dataset.len()
