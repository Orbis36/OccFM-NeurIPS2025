import torch.nn as nn

class EmptyPlanner(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, batch_dict):
        return batch_dict
