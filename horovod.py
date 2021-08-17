import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import os
import torch.optim as optim
import horovod.torch as hvd
hvd.init()
torch.cuda.set_device(hvd.local_rank())

input_size = 5
output_size = 2
batch_size = 30
data_size = 90

class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)
        self.labels = torch.cat([torch.zeros((length//2,)), torch.ones((length - length//2,))], 0)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.len

dataset = RandomDataset(input_size, data_size)

train_sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank())

# 3）使用DistributedSampler
rand_loader = DataLoader(dataset=dataset,
                         batch_size=batch_size,
                         sampler=train_sampler)

class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("  In Model: input size", input.size(),
              "output size", output.size())
        return output
    
model = Model(input_size, output_size)
model.to(torch.device('cuda:0'))

loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
for i in range(100):
    for data, label in rand_loader:
        if torch.cuda.is_available():
            data = data.to('cuda')
            label = label.to('cuda')

        output = model(data)
        loss = loss_function(output, label.long())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Loss: ", loss.item(), "Outside: input size", data.size(), "output_size", output.size())