import torch
import torchvision.datasets as datasets

# Define the path to the ECG5000 dataset
data_path = "./dataset"

# Define the transform to apply to the data
transform = transforms.ToTensor()

# Load the ECG5000 dataset
ecg_dataset = datasets.DatasetFolder(
    root=data_path, loader=torch.load, transform=transform
)

# Define the train/test split
train_ratio = 0.8
n_train_examples = int(len(ecg_dataset) * train_ratio)
n_test_examples = len(ecg_dataset) - n_train_examples

# Split the dataset into training and testing sets
train_dataset, test_dataset = torch.utils.data.random_split(
    ecg_dataset, [n_train_examples, n_test_examples]
)
