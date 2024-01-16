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

DATASET_PATH = "dataset/raw/experiment.h5"
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

    node_feature = []
    node_feature.extend(robot_joint)
    target_feature = []
    target_feature.extend(actions[i])

    edge_feature = None

    return node_feature, edge_feature, target_feature


class RobotGraph(Dataset):
    def __init__(self, root: str, transform=None, pre_transform=None):
        self.raw_name = ["experiment.h5"]  # Define raw_name here
        super().__init__(root=root, transform=transform, pre_transform=pre_transform)
        self.total_data = len(torch.load(osp.join(self.processed_dir, "data.pt"))[0])
        # Load the dataset into memory
        self.node_features, self.edge_features, self.target_features = torch.load(
            osp.join(self.processed_dir, "data.pt"))

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, "processed")

    @property
    def raw_file_names(self) -> List[str]:
        return self.raw_name

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt"]

    @property
    def num_output_features(self) -> int:
        return 8

    def download(self):
        # write file to raw_dir
        shutil.copy(DATASET_PATH, self.raw_paths[0])

    def process(self):
        node_features = []
        edge_features = []
        target_features = []
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
                    node_feature, edge_feature, target_feature = extract_feature_from_obs(obs, actions, i)
                    node_features.append(torch.from_numpy(np.asarray(node_feature)))
                    # edge_features.append(torch.from_numpy(edge_feature))
                    target_features.append(torch.from_numpy(np.asarray(target_feature)))
                    total_graphs += 1

        node_features = torch.stack(node_features)
        # edge_features = torch.stack(edge_features)
        target_features = torch.stack(target_features)
        torch.save((node_features, None, target_features), osp.join(self.processed_dir, "data.pt"))

    def get(self, idx):
        edge_list = GRAPH_STRUCTURE.get_edges()
        # Access the required data from the loaded dataset
        node_feature = self.node_features[idx]
        # edge_feature = self.edge_features[idx]
        target_feature = self.target_features[idx]
        data = graph_from_state(node_feature, edge_list, None, target_feature)
        return data

    def update_dataset(self, new_file_path):
        # Save the original raw file path
        original_raw_path = self.raw_paths[0]

        # Update the raw file path to the new file
        self.raw_name = [new_file_path]

        # Load the existing dataset
        existing_graph_list = torch.load(osp.join(self.processed_dir, "data.pt"))

        # Process the new data into graphs
        self.process()

        # Load the new graphs
        new_graph_list = torch.load(osp.join(self.processed_dir, "data.pt"))

        # Create a list to store the merged graphs
        merged_graph_list = [None, None, None]
        # Append the new graphs to the existing dataset
        for i in range(3):
            if existing_graph_list[i] is None:
                merged_graph_list[i] = None
            else:
                merged_graph_list[i] = torch.cat([existing_graph_list[i], new_graph_list[i]], dim=0)

        # Save the updated dataset
        torch.save(merged_graph_list, osp.join(self.processed_dir, "data.pt"))

        # Restore the original raw file path
        self.raw_paths[0] = original_raw_path

    def len(self):
        return self.total_data

    @property
    def num_nodes(self):
        return len(GRAPH_STRUCTURE.nodes)

# dataset = RobotGraph(root="dataset")
#
# dataset.len()
