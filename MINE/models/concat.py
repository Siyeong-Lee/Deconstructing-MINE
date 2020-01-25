import torch
import torch.nn as nn

class ConcatNet(nn.Module):
    def __init__(self, input_state, hidden_state):
        super(ConcatNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(2*input_state, hidden_state), nn.ReLU(),
            nn.Linear(hidden_state, 1, bias=False)
            )

    def forward(self, x, y):
        xy = torch.cat((x, y), dim=1)
        return self.network(xy)

def ConcatNet20(input_state):
    return ConcatNet(input_state=input_state, hidden_state=20)
