import torch

# main parameter
input_size = 12
num_classes = 2
num_workers = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
# Neural network learning parameter
lr = 0.001
n_epochs = 100
batch_size = 128
valid_size = 0.2

