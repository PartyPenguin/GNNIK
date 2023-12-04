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

dataset_path = "dataset/your_dataset.h5"
graph_structure = OnlyRobotGraphStructure()


def extract_feature_from_obs(obs: ndarray, actions: ndarray, i: int) -> (ndarray, ndarray):
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
    robot_joint_vel = obs[:, 7:14]  # 7 dim joint velocities
    robot_tcp_pose = obs[:, 14:21]  # 7 dim tcp pose
    goal_pos = obs[:, 21:28]  # 7 dim goal pose

    # Add robot joint positions, velocities, tcp pose and goal position to feature
    robot_joint = np.vstack(
        [
            robot_joint_pos[i],
            robot_joint_vel[i],
            np.tile(
                robot_tcp_pose[i], (robot_joint_pos[i].shape[0], 1)
            ).T,
            np.tile(goal_pos[i], (robot_joint_pos[i].shape[0], 1)).T,
        ]
    ).T
    feature = []
    feature.extend(robot_joint)

    # Add actions to the graph
    action = actions[i]

    return feature, action


class RobotGraph(Dataset):
    def __init__(self, root, mask=None, transform=None, pre_transform=None):
        self.mask = mask
        self.num_links = 7
        super(RobotGraph, self).__init__(
            root=root, transform=transform, pre_transform=pre_transform
        )
        with h5py.File(self.raw_paths[0], "r") as f:
            self.total_data = len(list(f))

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
        return ["recorded_dataset.h5"]  # Replace with actual raw file names if needed

    @property
    def processed_file_names(self) -> List[str]:
        return ["data_0.pt"]

    @property
    def num_output_features(self) -> int:
        return 12

    def download(self):
        # write file to raw_dir
        shutil.copy(dataset_path, self.raw_paths[0])

    def save_graph(self, state, edge_lists, action, total_graphs):
        data = graph_from_state(state, edge_lists, action)
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
                actions = np.array(f["%s/action" % group_name])

                # For each sequence, create a graph
                for i in range(len(obs)):
                    edge_lists = graph_structure.get_edges()
                    state, action = extract_feature_from_obs(obs, actions[:, :7], i)
                    self.save_graph(state, edge_lists, action, total_graphs)
                    total_graphs += 1

    def len(self):
        return self.total_data

    @property
    def num_nodes(self):
        return len(graph_structure.nodes)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, "data_%d.pt" % (idx)))
        return data


dataset = RobotGraph(root="src/dataset")

dataset.len()
