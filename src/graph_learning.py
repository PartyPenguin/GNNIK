# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.nn as pyg_nn
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GAE

from src.dataset import RobotGraph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
class GCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(GCNEncoder, self).__init__()
        self.conv1 = pyg_nn.GCNConv(input_dim, hidden_dim)
        self.conv2 = pyg_nn.GCNConv(hidden_dim, latent_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# %%
def train(train_loader, test_loader, writer, model, optimizer, num_epochs=1000, l1_lambda=0.01):
    print("Device:", device)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.975)
    model.to(device)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index)
            loss = F.mse_loss(pred.squeeze(), batch.y.reshape([pred.shape[0], pred.shape[1]]))

            # L1 regularization
            l1_reg = torch.tensor(0., requires_grad=True)
            for name, param in model.named_parameters():
                if 'weight' in name:
                    l1_reg = l1_reg + torch.norm(param, 1)
            #loss = loss + l1_lambda * l1_reg

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(train_loader.dataset)
        writer.add_scalar("loss", total_loss, epoch)

        if epoch % 10 == 0:
            test_loss = test(test_loader, model)
            print(f"Epoch {epoch}. Loss: {total_loss:.4f}. Test loss: {test_loss:.4f}")
            writer.add_scalar("test_loss", test_loss, epoch)
        if epoch % 100 == 0:
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
            pred = model(data.x, data.edge_index)
        target = data.y.reshape([pred.shape[0], pred.shape[1]])

        loss = torch.sqrt(F.mse_loss(pred.squeeze(), target)).to(device)
        total_loss += loss.item() * data.num_graphs
        total_samples += data.num_graphs

    avg_loss = total_loss / total_samples
    return avg_loss


# %%
def main():
    writer = SummaryWriter("../runs/test")
    dataset = RobotGraph("dataset")

    train_ratio = 0.8
    num_samples = len(dataset)
    num_train = int(train_ratio * num_samples)
    indices = list(range(num_samples))
    train_indices, test_indices = train_test_split(
        indices, train_size=num_train, shuffle=True, random_state=42
    )

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=True, drop_last=True, num_workers=4
    )

    model = GAE(GCNEncoder(16, 64, 1))
    # model = to_hetero(model, train_dataset[0].metadata(), aggr="sum")
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    trained_model = train(train_loader, test_loader, writer, model, optimizer)
    print("Training complete.")
    # Save the model
    torch.save(trained_model, "model.pt")


if __name__ == "__main__":
    main()
