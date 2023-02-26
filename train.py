from __future__ import print_function
import argparse
from torch_geometric.loader import DataLoader
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from data import train_dataset, test_dataset


class SAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(SAGEConv, self).__init__(aggr="max")
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.act = torch.nn.ReLU()
        self.update_lin = torch.nn.Linear(
            in_channels + out_channels, in_channels, bias=False
        )
        self.update_act = torch.nn.ReLU()

    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        x_j = self.lin(x_j)
        x_j = self.act(x_j)

        return x_j

    def update(self, aggr_out, x):
        new_embedding = torch.cat([aggr_out, x], dim=1)
        new_embedding = self.update_lin(new_embedding)
        new_embedding = self.update_act(new_embedding)

        return new_embedding


embed_dim = 128


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = SAGEConv(embed_dim, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = SAGEConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = SAGEConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=300, embedding_dim=embed_dim
        )
        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, 1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.item_embedding(x)

        x = F.relu(self.conv1(x, edge_index))

        x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))

        x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))

        x, edge_index, _, batch, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = self.lin1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.lin2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.lin3(x).squeeze(1)
        x = torch.sigmoid(x)

        return x


def train(model, crit, device, train_loader, optimizer, epoch):
    model.train()
    print(f"epoch{epoch}")

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        label = data.y.to(device)
        loss = crit(output, label)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


def evaluate(loader):
    model.eval()

    loss_all = 0
    correct = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            label = data.y.to(device)
            loss = crit(output, label)
            loss_all += data.num_graphs * loss.item()

            pred = output.round().squeeze()
            correct += pred.eq(label).sum().item()

    acc = correct / len(loader.dataset)
    loss = loss_all / len(loader.dataset)

    print("Test set: Average loss: {:.4f}, Accuracy: {:.2f}%\n".format(
        loss, acc * 100))
    return loss, acc


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    args = parser.parse_args()

    # Setup device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load data
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False)

    # Setup model
    model = Net().to(device)
    crit = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, crit, device, train_loader, optimizer, epoch)
        test_loss, test_acc = evaluate(test_loader)
        scheduler.step()

    print("Training finished\n")


if __name__ == "__main__":
    main()
    # run with --device cpu or --device mps
