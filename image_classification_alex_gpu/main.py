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

def main():
    global vis, loss_function_plot

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
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
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