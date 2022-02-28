import torch.nn as nn
import torch.nn.functional as F

printFeatureSize = True

class Net(nn.Module):

    # Defining the LeNet layers
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 9, 5, padding=2) # input channel 3 (RGB), out channel 9, 5x5 filter
        self.bn2d1 = nn.BatchNorm2d(9)
        self.pool = nn.MaxPool2d(2, 2) # 2x2 filter, stride = 2
        self.conv2 = nn.Conv2d(9, 16, 5) # input channel 9, out channel 16, 5x5 filter
        self.bn2d2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 6 * 6, 180) # 16 channels * feature sizes, out 180 channel
        self.bn1d1 = nn.BatchNorm1d(180)
        self.fc2 = nn.Linear(180, 60) # in 180 channels, out 60 channels
        self.bn1d2 = nn.BatchNorm1d(60)
        self.fc3 = nn.Linear(60, numberClasses) # in 60 channels, out 10 channels
        self.dropout = nn.Dropout(0.5)

    # Defining how layers are connected, with extra print functions to check out the feature size
    def forward(self, x):
        if (printFeatureSize):
            print(x.size())
        x = self.pool(F.relu(self.bn2d1(self.conv1(x)))) # Pooling layer + relu activation + batch normalisation + convolution
        if (printFeatureSize):
            print(x.size())
        x = self.pool(F.relu(self.bn2d2(self.conv2(x)))) # Pooling layer + relu activation + batch normalisation + convolution
        if (printFeatureSize):
            print(x.size())
        x = x.view(-1, 16 * 6 * 6) # Flatten the feature, we know the feature size from above
        if (printFeatureSize):
            print(x.size())
        x = F.relu(self.bn1d1(self.fc1(x))) # Relu activation + batch normalisation + fully-connected layer
        if (printFeatureSize):
            print(x.size())
        x = F.relu(self.bn1d2(self.fc2(x))) # Relu activation + batch normalisation + fully-connected layer
        if (printFeatureSize):
            print(x.size())
        x = self.fc3(x)
        if (printFeatureSize):
            print(x.size())
        return x

# Creating one network
net = Net()
