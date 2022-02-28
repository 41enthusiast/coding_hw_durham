import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import torch.optim as optim

def train():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # The normalisation transform

    batchSize = 32
    numberClasses = 10

    # The training set
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=2)

    # The testing set
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=2)

    # The class labels
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    printFeatureSize = False

    # Looping over the dataset multiple times
    for epoch in range(2):  

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            
            # Getting the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # Resetting the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Printing statistics
            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    # Save the trained network parameters into a file
    PATH = './basic_cifar10.pth'
    torch.save(net.state_dict(), PATH)
    pass

if __name__=='__main__':
    pass