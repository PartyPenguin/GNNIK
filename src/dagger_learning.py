import logging

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from environments.panda_env import PandaEnv
from motion_planning import MotionPlanner
from simulator import generate_target_pose
from simulator import run_policy
from src.dataset import RobotGraph

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_STEPS = 250
NUM_TRAJECTORIES = 250
TARGET_POSE_QUATERNION = [0, 1, 0, 0]
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
BATCH_SIZE = 512
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 5e-4
ITERATIONS = 5
WRITER = SummaryWriter("../runs/test")

logging.basicConfig(level=logging.INFO)


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


def calculate_regularization_loss(model, device, l1_lambda):
    """Calculate L1 regularization loss for the model."""
    l1_reg = torch.tensor(0., device=device, requires_grad=True)
    for name, param in model.named_parameters():
        if 'weight' in name:
            l1_reg = l1_reg + torch.norm(param, 1)
    return l1_lambda * l1_reg


def train_model_for_one_epoch(model, device, train_loader, optimizer, l1_lambda):
    """Train the model for one epoch and return the total loss."""
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch).float()
        loss = F.huber_loss(pred.squeeze(), batch.y.float())
        loss += calculate_regularization_loss(model, device, l1_lambda)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(train_loader.dataset)


def train_model(model, device, train_loader, test_loader, optimizer, num_epochs, l1_lambda):
    """Train the model and update the weights."""
    logging.info(f"Device: {device}")
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.975)
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        total_loss = train_model_for_one_epoch(model, device, train_loader, optimizer, l1_lambda)
        logging.info(f"Epoch {epoch}. L1 Loss: {total_loss:.4f}.")

        if epoch % 1 == 0:
            test_loss = test(test_loader, model)
            logging.info(f"Test loss: {test_loss:.4f}")

        if epoch % 20 == 0:
            scheduler.step()

    return model


def test_model(loader, model, device):
    """Test the model by calculating the average loss over all batches in the loader."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            predictions = model(data)
        target = data.y.reshape(predictions.shape[0], predictions.shape[1])
        loss = F.huber_loss(predictions, target).to(device)
        total_loss += loss.item() * data.num_graphs
        total_samples += data.num_graphs

    avg_loss = total_loss / total_samples
    return avg_loss


def reset_environment(env):
    env.reset()
    return env.get_obs()


def execute_trajectory_steps(model, env, mp, target_pose, traj_num, iteration):
    steps = 0
    all_obs = []
    all_actions = []
    obs = env.get_obs()
    while steps < MAX_STEPS:
        next_action = run_policy(obs, model, env)
        is_collision = mp.check_for_collision(next_action.squeeze())
        if is_collision:
            print("Collision detected. Using Expert instead of policy.")
            model_trajectory = mp.move_to_pose(np.concatenate([target_pose.p, target_pose.q]), with_screw=True)
            if model_trajectory == -1:
                break
            next_action = model_trajectory['position'][0]

        obs, _, _, _ = env.step(next_action)

        # Generate expert action
        model_trajectory = mp.move_to_pose(np.concatenate([target_pose.p, target_pose.q]), with_screw=True)
        if model_trajectory == -1:
            break
        expert_action = model_trajectory['position'][0]
        all_obs.append(obs)
        all_actions.append(expert_action)
        steps += 1

    if len(all_obs) == 0:
        return None

    print("Saving trajectory number:", traj_num)
    with h5py.File('dataset/raw/experiment_{}.h5'.format(iteration), 'a') as f:
        g = f.create_group(f'experiment_{traj_num}')
        g.create_dataset('obs', data=all_obs)
        g.create_dataset('actions', data=all_actions)
    return reset_environment(env)


def split_dataset(dataset, train_ratio, seed):
    """Split the dataset into training and test sets."""
    num_samples = dataset.len()
    num_train = int(train_ratio * num_samples)
    indices = list(range(num_samples))
    train_indices, test_indices = train_test_split(
        indices, train_size=num_train, shuffle=True, random_state=seed
    )
    return train_indices, test_indices


def create_data_loaders(dataset, train_indices, test_indices, batch_size):
    """Create data loaders for the training and test set."""
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4
    )
    return train_loader, test_loader


def initialize_model(input_dim, hidden_dim, latent_dim):
    """Initialize the model if not provided."""
    return MLP(input_dim, hidden_dim, latent_dim)


def run_training(run_id, dataset, model=None):
    """
    Train the model and save it.

    Parameters:
    run_id (int): The run identifier.
    dataset (RobotGraph): The dataset to use for training.
    model (nn.Module, optional): The model to train. If None, a new model is initialized. Defaults to None.
    """

    # Split the dataset into training and test sets
    train_indices, test_indices = split_dataset(dataset, TRAIN_RATIO, RANDOM_SEED)

    # Create data loaders for the training and test set
    train_loader, test_loader = create_data_loaders(dataset, train_indices, test_indices, BATCH_SIZE)

    # Initialize the model if not provided
    if model is None:
        model = initialize_model(8, 128, 1)

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Train the model
    trained_model = train(train_loader, test_loader, WRITER, model, optimizer)
    logging.info("Training complete.")

    # Save the trained model
    torch.save(trained_model, f"models/model_{run_id}.pt")


def initialize_environment_and_planner():
    """Initialize the environment and motion planner."""
    env = PandaEnv()
    mp = MotionPlanner(env=env)
    mp.setup_planner()
    return env, mp


def execute_trajectory(model, env, mp, current_trajectory, iteration):
    """Execute a trajectory and return the new observation."""
    target_pose = generate_target_pose()
    env.cube.set_pose(target_pose)
    trajectory = mp.move_to_pose(np.concatenate([target_pose.p, target_pose.q]), with_screw=True)

    if trajectory == -1:
        return reset_environment(env)

    print(f"Generating trajectory number: {current_trajectory}")
    obs = execute_trajectory_steps(model, env, mp, target_pose, current_trajectory, iteration)
    return obs


def evaluate_policy(model: nn.Module, iteration: int):
    """
    Evaluate the policy by executing a number of trajectories.

    Parameters:
    model (nn.Module): The model to evaluate.
    iteration (int): The current iteration number.
    """
    # Constants
    NUM_TRAJECTORIES = 250

    # Initialize environment and motion planner
    env, mp = initialize_environment_and_planner()

    # Reset the environment and get the initial observation
    obs = reset_environment(env)

    # Iterate over trajectories
    current_trajectory = 0
    while current_trajectory < NUM_TRAJECTORIES:
        obs = execute_trajectory(model, env, mp, current_trajectory, iteration)
        if obs is None:
            obs = reset_environment(env)
            continue
        current_trajectory += 1

    env.close()


def main():
    """Main function."""
    for i in range(ITERATIONS):
        dataset = RobotGraph("dataset")
        print("Running iteration:", i)
        run_training(i, dataset)
        model_path = f"models/model_{i}.pt"
        if os.path.exists(model_path):
            model = torch.load(model_path)
            evaluate_policy(model, i)
            dataset.update_dataset(f"experiment_{i}.h5")
        else:
            print(f"Model not found at {model_path}. Skipping...")


if __name__ == "__main__":
    main()
