import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class ConvNN(nn.Module):
    def __init__(self, num_classes=36):
        super(ConvNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))
        '''
            After conv1:        W - F + 1 = 128 - 5 + 1 = 124   (=> 16 Channels of 124x124)
            After max pooling:  W/2 = 124/2 = 62                (=> 16 Channels of 62x62)
            After conv2:        W - F + 1 = 62 - 5 + 1 = 58     (=> 32 Channels of 58x58)
            After adaptive pooling: 32*5*5                      (=> 32 Channels of 5x5)
        '''
        self.fc1 = nn.Linear(32*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def init_weights(self):
        for layer in [self.conv1, self.conv2]:
            init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if layer.bias is not None:
                init.constant_(layer.bias, 0)
        for layer in [self.fc1, self.fc2, self.fc3]:
            init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, 0)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = x.view(-1, 32*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x