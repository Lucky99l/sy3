import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ResBlock
# class ResBlock(nn.Module):
#     def __init__(self, num_hidden):
#         super(ResBlock, self).__init__()
#         self.conv = nn. Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=1, bias=False)
#         # self.batch_norm = nn.BatchNorm2d(num_features=num_hidden)
#         nn.init.kaiming_normal_(self.conv.weight,mode='fan_out', nonlinearity='relu')
#         # nn.init.constant_(self.batch_norm.weight, 0.5)
#         # nn.init.zeros_(self.batch_norm.bias)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         out = self.conv(x)
#         out = self.relu(out)
#         return x + out

# class Model(nn.Module):
#     def __init__(self, input_shape, n_actions, num_hidden1=4, num_hidden2=8, num_block=1):
#         super(Model, self).__init__()
#         self.num_hidden1 = num_hidden1

#         self.conv1 = nn.Conv2d(2, num_hidden1, kernel_size=3, stride=2, padding=1)
#         self.relu1 = nn.ReLU()
#         self.pool1 = nn.MaxPool2d(1, 1)
#         self.block1 = nn.Sequential(*(num_block * [ResBlock(num_hidden=num_hidden1)]))
#         self.pool2 = nn.MaxPool2d(1, 1)

#         # self.conv2 = nn.Conv2d(num_hidden1, num_hidden2, kernel_size=3, stride=2, padding=1)
#         # self.relu2 = nn.ReLU()
#         # self.pool3 = nn.MaxPool2d(1, 1)
#         # self.block2 = nn.Sequential(*(num_block * [ResBlock(num_hidden=num_hidden2)]))
#         # self.pool4 = nn.MaxPool2d(1, 1)

#         self.fc1 = nn.Linear(num_hidden1 * 16 * 50, 256) # num_hidden12 * 8 * 25
#         self.fc2 = nn.Linear(256, n_actions)
#         # self.fc3 = nn.Linear(64, n_actions)
#         # self.dropout = nn.Dropout(0.1)
#         # self.sigmoid = nn.Sigmoid()

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 # nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         out = self.pool1(self.relu1(self.conv1(x)))
#         out = self.block1(out)
#         out = self.pool2(out)

#         # out = self.pool3(self.relu2(self.conv2(out)))
#         # out = self.block2(out)
#         # out = self.pool4(out)

#         out = out.view(out.size(0), -1)
#         out = self.relu1(self.fc1(out))
#         out = self.relu1(self.fc2(out))
#         # out = self.fc3(out)
#         return out



# create model
class Model(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(Model, self).__init__()
        
        # self.net = nn.Sequential(
        #     nn.Linear(input_shape[0] * input_shape[1] * input_shape[2], 1600),
        #     nn.ReLU(),
        #     nn.Linear(1600, 400),
        #     nn.ReLU(),
        #     nn.Linear(400, n_actions),
        #     nn.Sigmoid()
        # )
        
        self.conv1 = nn.Conv2d(2, 4, kernel_size=(1, 3), stride=(1, 2), padding=0)
        self.bn1 = nn.BatchNorm2d(4)
        self.pool1 = nn.MaxPool2d(1, 1)
        self.conv2 = nn.Conv2d(4, 6, kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.bn2 = nn.BatchNorm2d(6)
        self.pool2 = nn.MaxPool2d(1, 1)
        # self.conv3 = nn.Conv2d(6, 8, kernel_size=(2, 2), stride=1, padding=0)
        # self.bn3 = nn.BatchNorm2d(8)
        # self.pool3 = nn.MaxPool2d(1, 1)
        self.fc1 = nn.Linear(6 * 16 * 24, 128)
        self.fc2 = nn.Linear(128, n_actions)
        # self.fc3 = nn.Linear(64, n_actions)
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        
        # flatten the observation space Box to linear tensor
        # out = torch.flatten(x, start_dim=-3, end_dim=-1).to(torch.float32)
        # out = self.net(out)
        # print(self.fc1.weight)

        out = self.pool1(self.relu(self.bn1(self.conv1(x))))
        # print(out.shape)
        out = self.pool2(self.relu(self.bn2(self.conv2(out))))
        # print(out.shape)
        # out = self.pool3(self.relu(self.bn3(self.conv3(out))))
        # print(out.shape)
        out = out.view(out.size(0), -1)
        # out = self.sigmoid(self.fc3(self.leakyrelu(self.fc2(self.leakyrelu(self.fc1(out))))))
        # out = self.sigmoid(self.fc2(self.leakyrelu(self.fc1(out))))
        out = self.fc2(self.relu(self.fc1(out)))
        return F.softmax(out, dim=1)


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.2):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        #else:
        # weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
        # bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)
            
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x


class NoisyDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(NoisyDQN, self).__init__()
        self.conv1 = nn.Conv2d(2, 4, kernel_size=(1, 3), stride=(1, 2), padding=0)
        self.bn1 = nn.BatchNorm2d(4)
        self.pool1 = nn.MaxPool2d(1, 1)
        self.conv2 = nn.Conv2d(4, 6, kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.bn2 = nn.BatchNorm2d(6)
        self.pool2 = nn.MaxPool2d(1, 1)

        self.noisy1 = NoisyLinear(6 * 16 * 24, 128)
        self.noisy2 = NoisyLinear(128, n_actions)
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.relu(self.noisy1(x.view(x.size(0), -1)))
        return self.noisy2(x)

    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()

