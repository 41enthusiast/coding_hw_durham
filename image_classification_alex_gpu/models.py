class AlexNet(nn.Module):

    bDisplayFeatureSize = False;
    
    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        
        # The convolution layers part of the network
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), # This means to train 64 filters of size 11 x 11 (i.e. the next layer has 64 channels), and the channel number of this layer is 3 (i.e. RGB from the image)
            nn.ReLU(inplace=True), # inplace=True is to save memory
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2), # This means to train 192 filters of size 5x5 (i.e. the next layer has 192 channels), and the channel number of this layer is 64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # A single pooling layer to ensure that the output feature is 6x6
        # N.B. Adaptive average pooling - the input parameter is the output feature size we want, and Pytorch will calculate the pooling filter size
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6)) 
        
        # The fully connected layer part of the network
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096), # depth * feature W * feature H
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    # The forward operations
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (self.bDisplayFeatureSize): print(str(x.size()) + " - input")
            
        x = self.features(x)
        if (self.bDisplayFeatureSize): print(str(x.size()) + " - after convolution layers")
            
        x = self.avgpool(x)
        if (self.bDisplayFeatureSize): print(str(x.size()) + " - after average pooling")
            
        x = torch.flatten(x, 1)
        if (self.bDisplayFeatureSize): print(str(x.size()) + " - after the flatten operation")
            
        x = self.classifier(x)
        if (self.bDisplayFeatureSize): print(str(x.size()) + " - after fully connected layers")
            
        return x

# Initialize one AlexNet, 10 is the size of the output layer (i.e. the number of classes)
#net = AlexNet(10)
class Fire(nn.Module):
    def __init__(self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int) -> None:
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat(
            [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        )

class Residual(nn.Module):  #@save
    """The Residual block of ResNet."""
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

class SqueezeExcitation(nn.Module):
    """
    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
    """

    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
    ) -> None:
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = torch.nn.Conv2d(squeeze_channels, input_channels, 1)

    def _scale(self, input: Tensor) -> Tensor:
        scale = self.avgpool(input)
        scale = nn.ReLU(self.fc1(scale))
        scale = nn.Sigmoid(self.fc2(scale))
        return scale

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input)
        return scale * input

def swish(x):
    return x * F.sigmoid(x)

from keras.layers import Conv2D, DepthwiseConv2D, Add
def inverted_residual_block(x, expand=64, squeeze=16):
    block = Conv2D(expand, (1,1), activation=’relu’)(x)
    block = DepthwiseConv2D((3,3), activation=’relu’)(block)
    block = Conv2D(squeeze, (1,1), activation=’relu’)(block)
    return Add()([block, x])

