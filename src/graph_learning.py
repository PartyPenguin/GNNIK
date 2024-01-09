# %%
import joblib
import matplotlib.pyplot as plt
import numpy as np
import sapien.core as sapien
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.transforms as T
from numpy import ndarray
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv

from src.dataset import RobotGraph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

urdf_path = "../assets/robot/franka_panda/panda.urdf"
scene_config = sapien.SceneConfig()
engine = sapien.Engine()
scene = engine.create_scene(scene_config)
DOF = 7

loader: sapien.URDFLoader = scene.create_urdf_loader()
loader.fix_root_link = True
robot: sapien.Articulation = loader.load(urdf_path)
pinocchio = robot.create_pinocchio_model()


def forward_kinematics(joint_positions: ndarray) -> ndarray:
    """
    Calculate the forward kinematics of the robot.

    Args:
        joint_positions: joint positions

    Returns:
        ndarray: forward kinematics

    """
    joint_positions_np = joint_positions.detach().cpu().numpy()
    dummy_joints = torch.zeros(2).to(device)
    ee_pos = torch.zeros((joint_positions.shape[0], 3)).to(device)

    with torch.no_grad():
        for i, jp in enumerate(joint_positions_np):
            # Add two dummy joints to match the pinocchio model
            jp = torch.cat((torch.tensor(jp).to(device), dummy_joints))
            pinocchio.compute_forward_kinematics(jp.detach().cpu().numpy())
            ee_pose = pinocchio.get_link_pose(12)
            ee_pos[i] = torch.from_numpy(ee_pose.p).to(device)

    return ee_pos


# %%
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, data):
        x = data.x
        x = self.mlp(x)
        return x


class GCN(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GATv2Conv(8, hidden_dim, heads=8, dropout=0.6)
        self.conv2 = GATv2Conv(hidden_dim * 8, output_dim, heads=1, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        return x


# %%
def normalize_node_features(dataset):
    print("Normalizing node features...")

    # Try to load the scaler params
    try:
        scaler_params = joblib.load("models/scaler_params.pkl")
        mean = scaler_params['mean']
        std = scaler_params['std']
    except FileNotFoundError:
        print("Scaler params not found. Calculating mean and std...")
        # Get all node features
        node_features = np.vstack([data.x.numpy() for data in dataset])

        # Per-feature normalization
        mean = np.mean(node_features, axis=0)
        std = np.std(node_features, axis=0)

        # Save the scaler with mean and std
        scaler_params = {'mean': mean, 'std': std}
        joblib.dump(scaler_params, "models/scaler_params.pkl")

    # Transform node features
    print("Transforming node features...")
    for i, data in enumerate(dataset):
        normalized_x = (data.x.numpy() - mean) / std
        dataset[i].x = torch.tensor(normalized_x, dtype=torch.float32)

    print("Done.")
    return dataset


def train(train_loader, test_loader, writer, model, optimizer, num_epochs=50, l1_lambda=0.01):
    print("Device:", device)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.975)
    model.to(device)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = F.huber_loss(pred.squeeze(), batch.y)
            # L1 regularization
            l1_reg = torch.tensor(0., device=device, requires_grad=True)
            for name, param in model.named_parameters():
                if 'weight' in name:
                    l1_reg = l1_reg + torch.norm(param, 1)
            loss_no_reg = loss.detach().cpu().numpy()
            loss = loss + l1_lambda * l1_reg
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(train_loader.dataset)
        writer.add_scalar("loss", total_loss, epoch)

        if epoch % 1 == 0:
            test_loss = test(test_loader, model)
            print(
                f"Epoch {epoch}. L1 Loss: {total_loss:.4f}. Huber Loss:{loss_no_reg: .4f}. Test loss: {test_loss:.4f}")
            writer.add_scalar("test_loss", test_loss, epoch)
        if epoch % 20 == 0:
            scheduler.step()

    return model


# %%
def test(loader, model):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data)
        target = data.y.reshape(pred.shape[0], pred.shape[1])

        # loss = F.mse_loss(pred.squeeze(), target).to(device)
        loss = F.huber_loss(pred, target).to(device)
        total_loss += loss.item() * data.num_graphs
        total_samples += data.num_graphs

        avg_loss = total_loss / total_samples
    return avg_loss


def plot_goal_pos(dataset):
    data = np.asarray([data.x for data in dataset])

    data_goal_pos = data[:, :, 5:8]

    goal_pos = np.unique(np.unique(data_goal_pos, axis=0), axis=1).squeeze()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(goal_pos[:, 0], goal_pos[:, 1], goal_pos[:, 2])
    plt.show()


# %%


def main():
    writer = SummaryWriter("../runs/test")
    transform = T.NormalizeFeatures()
    dataset = RobotGraph("dataset")

    train_ratio = 0.8
    num_samples = len(dataset)
    num_train = int(train_ratio * num_samples)
    indices = list(range(num_samples))
    train_indices, test_indices = train_test_split(
        indices, train_size=num_train, shuffle=True, random_state=42
    )

    train_dataset = Subset(dataset, train_indices)
    # plot_goal_pos(train_dataset)
    test_dataset = Subset(dataset, test_indices)


    # Normalize node features in the training dataset
    # train_dataset = normalize_node_features(train_dataset)
    # test_dataset = normalize_node_features(test_dataset)

    train_loader = DataLoader(
        train_dataset, batch_size=512, shuffle=True, drop_last=True, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=512, shuffle=True, drop_last=True, num_workers=4
    )

    # model = GCN(128, 1)
    model = MLP(8, 128, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)

    trained_model = train(train_loader, test_loader, writer, model, optimizer)
    print("Training complete.")
    # Save the model
    torch.save(trained_model, "models/model.pt")


if __name__ == "__main__":
    main()
