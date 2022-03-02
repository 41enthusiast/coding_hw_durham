from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import visdom
import argparse

def train(args, model, device, train_loader, optimizer, epoch):
    # Loop over the whole dataset iTotalEpochToTrain times
    model.train()

    running_loss = 0.0
    
    # Loop over each batch of training data
    for i, data in enumerate(trainloader, 0):
        
        # Get the batch of training data and send it to the GPU
        inputs, labels = data[0].to(device), data[1].to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Do one forward operation, one loss calculation, and one backward operation
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print the loss every iPrintEveryIteration loops
        running_loss += loss.item()
        if i % args.log_interval == args.log_interval-1:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            
            vis.line(
                    X=torch.tensor([(epoch-1) * len(trainloader) + i]).cpu(),
                    Y=running_loss / 2000,
                    win=loss_function_plot,
                    update='append')
            
            running_loss = 0.0

print('Training completed!')

# Save the trained network parameters into a file



def test(args, model, device, train_loader, optimizer, epoch):
    model.eval()
    pass

def main():
    global vis, loss_function_plot

    PATH = './checkpoints/alexnet_cifar10.pth'

    # Training settings
    parser = argparse.ArgumentParser(description='Image Classification with AlexNet')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    parser.add_argument('--visdom_server', default='http://localhost',
                        help='Visdom server')
    parser.add_argument('--visdom_port', default=8097,
                        help='Visdom port')
    parser.add_argument('--visdom_env_name', default='main',
                        help='Visdom main experiment type')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # Connect to the Visdom server
    vis = visdom.Visdom(server=args.visdom_server, port=args.visdom_port
            , env = args.visdom_env_name)

    # Create a new plot for the loss function
    loss_function_plot = vis.line(X=torch.zeros((1,)).cpu(), Y=torch.zeros((1,)).cpu(),
            opts={
                "xlabel":"Iteration",
                "ylabel":"Loss",
                "title":"Loss over time",
                "legend":["Classification loss"],
                "layoutopts":{
                    "plotly": {
                        "yaxis": { "type": "log" }
                        }
                    }
                })
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=iBatchSize, shuffle=True, num_workers=4) #num_worker = 4 * num_GPU

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=iBatchSize, shuffle=False, num_workers=4)

    classes = ('Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')

    model = AlexNet(10).to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    #scheduler = StepLR(optimizer, step_size = 1, gamma =args.gamma)

    for epoch in range(iTotalEpochToTrain):  
        train(args, model, device, trainloader, optimizer, epoch)
        #test(args, model, device, testloader)
        #scheduler.step()
        optimizer.step()
    
    if args.save_model:
        torch.save(model.state_dict(), PATH)
