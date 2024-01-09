from collections import namedtuple

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.data import HeteroData
from torch_geometric.typing import OptTensor

EdgeStruct = namedtuple("EdgeStruct", ["list", "types"])


def hetero_graph_from_state(state, edges):
    """
    Creates a PyTorch graph from the state of the environment and a list of edges.

    Args:
        state (list): A list of nodes in the environment, where each node is represented as a list
            containing the node type, node features, and node actions (optional).
        edges (EdgeStruct): A namedtuple containing a list of edges and a list of corresponding edge types.

    Returns:
        A PyTorch graph representing the environment, with nodes and edges defined by the state and edges.
    """

    graph = HeteroData()

    # Create nodes
    for node in state:
        node_type, node_features, node_actions = (
            node[0],
            torch.tensor(node[1], dtype=torch.float),
            torch.tensor(node[2], dtype=torch.float) if len(node) > 2 else None,
        )

        if not hasattr(graph[node_type], "x"):
            graph[node_type].x = node_features
        else:
            graph[node_type].x = torch.vstack((graph[node_type].x, node_features))

        # Add actions to the node if they exist
        if node_actions is not None:
            if not hasattr(graph[node_type], "y"):
                graph[node_type].y = node_actions
            else:
                graph[node_type].y = torch.vstack((graph[node_type].y, node_actions))

    # Create edges
    for idx, edge_list in edges.list.items():
        edge_type = edges.types[idx]
        edge_key = (f"{edge_type[0]}", f"{edge_type[1]}", f"{edge_type[2]}")
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        if hasattr(graph[edge_key], "edge_index"):
            graph[edge_key].edge_index = torch.cat(
                (graph[edge_key].edge_index, edge_index), dim=1
            )
        else:
            graph[edge_key].edge_index = edge_index

    graph = T.ToUndirected()(graph)

    return graph


def graph_from_state(
        node_feature: list,
        edges: EdgeStruct,
        edge_feature: list = None,
        target_feature=None,
):
    """
    Creates a homogenous graph from a state and a set of edges.

    Args:
        node_feature (list): A list of floats representing the state of the graph. Also known as features.
        edge_feature (list, optional): A list of floats representing the actions of the graph. Defaults to []. Also known as labels.
        edges (EdgeStruct): An EdgeStruct object containing the edges of the graph.
        target_feature (list, optional): A list of floats representing the target features of the graph. Defaults to None.

    Returns:
        Data: A PyTorch Geometric Data object representing the graph. This object includes node features (x), edge indices (edge_index), edge attributes (edge_attr), and optionally target features (y) if provided.
    """
    edge_attr: OptTensor = None

    # Initialize y as None. It will be used to store target features if they exist.
    y = None

    # Assert that only one type of edge is passed in
    assert len(edges.types) == 1, "Only one type of edge is expected."

    # Convert node features to a tensor
    x = torch.tensor(np.array(node_feature), dtype=torch.float)

    # Convert edge features to a tensor
    if edge_feature is not None:
        edge_attr = torch.tensor(np.array(edge_feature), dtype=torch.float)

    # If target features are provided, convert them to a tensor
    if target_feature is not None:
        y = torch.tensor(target_feature, dtype=torch.float)

    # Convert edge list to a tensor
    edge_index = torch.tensor(edges.list[0], dtype=torch.long).t().contiguous()

    # Create a PyTorch Geometric Data object representing the graph
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    # Convert the graph to an undirected graph
    graph = T.ToUndirected()(graph)

    return graph
