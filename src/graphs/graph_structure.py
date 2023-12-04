from abc import ABC, abstractmethod
from collections import namedtuple


class GraphStructure(ABC):
    def __init__(self):
        self.name = "Default Graph Structure"
        self.edge_struct = namedtuple("edge_struct", ["list", "types"])

    @abstractmethod
    def get_edges(self):
        pass


class FirstGraphStructure(GraphStructure):
    def __init__(self):
        super(FirstGraphStructure, self).__init__()
        self.edge_types = None
        self.name = "First Graph Structure"
        self.node_types = ["robot_link", "pose"]
        self.edge_struct.types = {
            0: ["joint", "connects", "joint"],
            1: ["joint", "connects", "pose"],
            2: ["pose", "connects", "pose"],
        }
        self.nodes = {
            0: self.node_types[0],  # Link0
            1: self.node_types[0],  # Link1
            2: self.node_types[0],  # Link2
            3: self.node_types[0],  # Link3
            4: self.node_types[0],  # Link4
            5: self.node_types[0],  # Link5
            6: self.node_types[0],  # Link6
            7: self.node_types[0],  # Link7 (Gripper)
            8: self.node_types[0],  # Link8 (Gripper)
            9: self.node_types[1],  # TCP Pose
            10: self.node_types[1],  # Cube Pose
            11: self.node_types[1],  # Goal Pose
        }
        self.edge_lists = {0: [[i, i + 1] for i in range(8)], 1: [[i, 0] for i in range(8)], 2: [[1, 0], [2, 0]]}
        # Connect each link to the next link
        # Conect TCP to all links (TCP Pose is the first node of type pose so it´s id is 0)
        # Connect Cube and goal to TCP (Cube is second node of type pose so it's id is 1. Goal is third > id is 2)

        self.edge_struct.list = self.edge_lists

    def get_edges(self):
        return self.edge_struct

    def get_edge_types(self):
        return self.edge_types

    def get_node_types(self):
        return self.node_types

    def get_node_type(self, node_id):
        return self.node_types[node_id]


class OnlyRobotGraphStructure(GraphStructure):
    def __init__(self):
        super(OnlyRobotGraphStructure, self).__init__()
        self.edge_types = None
        self.name = "Only Robot Graph Structure"
        self.node_types = ["robot_link"]
        self.edge_struct.types = {
            0: ["joint", "connects", "joint"],
        }
        self.nodes = {
            0: self.node_types[0],  # Link0
            1: self.node_types[0],  # Link1
            2: self.node_types[0],  # Link2
            3: self.node_types[0],  # Link3
            4: self.node_types[0],  # Link4
            5: self.node_types[0],  # Link5
            6: self.node_types[0],  # Link6
        }
        #self.edge_lists = {0: [[i, i + 1] for i in range(6)]}
        # Connect each link to the next link

        # Create fully connected graph
        self.edge_lists = {0: [[i, j] for i in range(7) for j in range(7) if i != j]}

        self.edge_struct.list = self.edge_lists

    def get_edges(self):
        return self.edge_struct

    def get_edge_types(self):
        return self.edge_types

    def get_node_types(self):
        return self.node_types

    def get_node_type(self, node_id):
        return self.node_types[node_id]
