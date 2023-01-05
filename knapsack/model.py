import torch
import torch.nn as nn
from torch.distributions import Categorical


# create model
class Model(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(Model, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
            # nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        
        # flatten the observation space Box to linear tensor
        # out = torch.flatten(x, start_dim=-3, end_dim=-1).to(torch.float32)
        out = self.net(x)
        return out
