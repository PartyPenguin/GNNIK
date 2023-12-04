from collections import namedtuple

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.data import HeteroData

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
        state: list,
        edges: EdgeStruct,
        actions: list = [],
):
    """
    Creates a homogenous graph from a state and a set of edges.

    Args:
        state (list): A list of floats representing the state of the graph. Also known as features.
        edges (EdgeStruct): An EdgeStruct object containing the edges of the graph.
        actions (list, optional): A list of floats representing the actions of the graph. Defaults to []. Also known as labels.

    Returns:
        Data: A PyTorch Geometric Data object representing the graph.
    """
    # Assert that only one type of edge is passed in
    assert len(edges.types) == 1

    # Create nodes
    x = torch.tensor(np.array(state), dtype=torch.float)

    # Add actions to the graph if they exist
    if len(actions) > 0:
        y = torch.tensor(actions, dtype=torch.float)

    # Create edges
    edge_index = torch.tensor(edges.list[0], dtype=torch.long).t().contiguous()

    if len(actions) > 0:
        graph = Data(x=x, edge_index=edge_index, y=y)
    else:
        graph = Data(x=x, edge_index=edge_index)

    graph = T.ToUndirected()(graph)

    return graph
