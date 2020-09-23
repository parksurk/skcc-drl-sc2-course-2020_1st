import torch
import torch.nn as nn

class DuelingQNet(nn.Module):

    def __init__(self,
                 input_dim: int,
                 output_dim: int):
        super(DuelingQNet, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc_value = nn.Linear(64, 256)
        self.fc_adv = nn.Linear(64, 256)

        self.value = nn.Linear(256, 1)
        self.adv = nn.Linear(256, output_dim)

    def forward(self, xs):
        y = self.relu(self.fc1(xs))
        value = self.relu(self.fc_value(y))
        adv = self.relu(self.fc_adv(y))

        value = self.value(value)
        adv = self.adv(adv)

        advAverage = torch.mean(adv, dim=1, keepdim=True)
        Q = value + adv - advAverage

        return Q

if __name__ == '__main__':
    net = DuelingQNet(10, 1, [20, 12], 'ReLU', 'Identity')
    print(net)

    xs = torch.randn(size=(12, 10))
    ys = net(xs)
    print(ys)
