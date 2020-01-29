import torch
import torch.nn as nn

class EmbeddedNet(nn.Module):
    def __init__(self, input_state, hidden_state):
        super(EmbeddedNet, self).__init__()
        self.subnet1 = nn.Sequential(
            nn.Linear(input_state, hidden_state//2), nn.ReLU()
            )
        self.subnet2 = nn.Sequential(
            nn.Linear(input_state, hidden_state//2), nn.ReLU()
            )
        self.network = nn.Linear(hidden_state, 1, bias=False)

    def forward(self, x, y):
        x = self.subnet1(x)
        y = self.subnet2(y)
        xy = torch.cat((x, y), dim=1)
        return self.network(xy)

def EmbeddedNet20(input_state):
    return EmbeddedNet(input_state=input_state, hidden_state=20)
