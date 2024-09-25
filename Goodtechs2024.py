import warnings
import PIL
from PIL import Image, PngImagePlugin
warnings.filterwarnings("ignore")
warnings.simplefilter('ignore', Image.DecompressionBombWarning)
warnings.simplefilter('ignore', PIL.Image.DecompressionBombError)
warnings.simplefilter('ignore', Image.DecompressionBombError)
warnings.simplefilter("ignore", UserWarning)


PIL.Image.MAX_IMAGE_PIXELS = 250000000
Image.MAX_IMAGE_PIXELS = 250000000

from pathlib import Path
# Setup path to data folder
data_path = Path("C:/")
image_path = data_path / "FL-dataset"

# Setup train and testing paths
train_dir = image_path

# Use ImageFolder to create dataset(s)
from torchvision import datasets
train_data = datasets.ImageFolder(root=train_dir, target_transform=None)

from collections import OrderedDict
from typing import List, Tuple

import flwr as fl
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from flwr.common import Metrics
from torch.utils.data import DataLoader, random_split

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(2)

verbose_train_log_file = "clients_2_seed_2.txt"


CLASSES = ('bar_charts', 'maps', 'piecharts', 'slides', 'tables', 'technical_drawings', 'x_y_plots')

from flwr.common import (
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

# For learning algorithm
AdamW_lr = 0.01
AdamW_weight_decay = 0.95
AdamW_betas = (0.9, 0.99999)

# For dederateld learning client-server configuration
NUM_CLIENTS = 2
FRACTION_FIT = 1.0
FRACTION_EVALUATE = 1.0
MIN_FIT_CLIENTS = NUM_CLIENTS
MIN_EVALUATE_CLIENTS = NUM_CLIENTS
MIN_AVAILABLE_CLIENTS = NUM_CLIENTS

# For data preparation
BATCH_SIZE = 32

# dederated learning rounds and epoch per round
NUM_ROUNDS = 100
EPOCHS_PER_ROUND = 1

# For the TinyVGG model
HIDDEN_UNITS = 100

# Define model
class TinyVGG(nn.Module):

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,  
                      stride=1,  
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  
            nn.Dropout(p=0.05, inplace=True)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.05, inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 16 * 16, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_shape)          
        )

    def weights_init(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.1)
                nn.init.constant_(module.bias, 0.05)

    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x
        

# train and test procedure
def train(net, trainloader, epochs: int, verbose=True):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    #optimizer = torch.optim.AdamW(net.parameters(), lr=AdamW_lr, amsgrad=True, weight_decay=AdamW_weight_decay, betas = AdamW_betas)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.95)
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        if verbose:
            f = open(verbose_train_log_file, 'a')
            f.write(f"{epoch_acc}\n")
            f.close()
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy




# Dataloader
def load_datasets():
    from torchvision import datasets
    data_transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.ToTensor()
    ])

    trainset = datasets.ImageFolder(root=train_dir,  # target folder of images
                                    transform=data_transform,  # transforms to perform on data (images)
                                    target_transform=None)  # transforms to perform on labels (if necessary)

    partition_size = len(trainset) // NUM_CLIENTS
    lengths = [partition_size] * NUM_CLIENTS

    try:
        datasets = random_split(trainset, lengths, generator=torch.Generator())

    except ValueError:
        trainset = list(trainset)
        trainset = trainset[:sum(lengths)]
        trainset = tuple(trainset)
        datasets = random_split(trainset, lengths, generator=torch.Generator())

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = len(ds) // 25  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, generator=torch.Generator())
        trainloaders.append(DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=BATCH_SIZE))

    return trainloaders, valloaders
    



trainloaders, valloaders = load_datasets()


params = []
names = []

params = np.array(params, dtype=object)


# Serialize ndarrays to `Parameters`
parameters = fl.common.ndarrays_to_parameters(params)


## Step 2: Federated Learning with Flower
def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=EPOCHS_PER_ROUND)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        #set_parameters(self.net, parameters)
        #loss, accuracy = test(self.net, self.valloader)
        return None #float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""

    # Load model
    net = TinyVGG(input_shape=3,  # number of color channels (3 for RGB)
                  hidden_units=HIDDEN_UNITS,
                  output_shape=len(train_data.classes)).to(DEVICE)

    # net = model_ft

    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]

    # Create a  single Flower client representing a single organization
    return FlowerClient(net, trainloader, valloader)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)

    return {"accuracy": sum(accuracies) / sum(examples)}

RAY_DISABLE_MEMORY_MONITOR=1

# Create strategies
strategy_FedAvg = fl.server.strategy.FedAvg(
        fraction_fit=FRACTION_FIT, # Sample x% of available clients for training
        fraction_evaluate=FRACTION_EVALUATE, # Sample x% of available clients for evaluation
        min_fit_clients=MIN_FIT_CLIENTS,
        min_evaluate_clients=MIN_EVALUATE_CLIENTS,
        min_available_clients=MIN_AVAILABLE_CLIENTS,
        evaluate_metrics_aggregation_fn=weighted_average,
)


# Start simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy_FedAvg
)


