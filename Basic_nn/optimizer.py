import torch
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt

torch.manual_seed(1)

LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1*torch.normal(torch.zeros(*x.size()))

torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, _x):
        _x = F.relu(self.hidden(_x))
        _x = self.out(_x)
        return _x


def train():
    net_SGD = Net(1, 20, 1)
    net_Momentum = Net(1, 20, 1)
    net_RMSProp = Net(1, 20, 1)
    net_Adam = Net(1, 20, 1)
    nets = [net_SGD, net_Momentum, net_RMSProp, net_Adam]

    opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
    opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
    opt_RMSProp = torch.optim.RMSprop(net_RMSProp.parameters(), lr=LR, alpha=0.9)
    opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
    optimizers = [opt_SGD, opt_Momentum, opt_RMSProp, opt_Adam]

    loss_func = torch.nn.MSELoss()
    losses = [[], [], [], []]

    for epoch in range(EPOCH):
        print('Epoch: ', epoch)
        for step, (b_x, b_y) in enumerate(loader):

            # 对每个优化器, 优化属于他的神经网络
            for net, opt, l_his in zip(nets, optimizers, losses):
                output = net(b_x)
                loss = loss_func(output, b_y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                l_his.append(loss.data.numpy())

    labels = ['SGD', 'Momentum', 'RMSProp', 'Adam']
    for i, l_his in enumerate(losses):
        plt.plot(l_his, label=labels[i])
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.ylim((0, 0.2))
    plt.show()

if __name__ == "__main__":
    train()