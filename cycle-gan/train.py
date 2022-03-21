import torch
import sys
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.utils import save_image

import argparse

from dataset import HorseZebraDataset
from model import Generator, Discriminator

def train_fn(args, disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    H_reals = 0
    H_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (zebra, horse) in enumerate(loop):
        zebra = zebra.to(device)
        horse = horse.to(device)

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_horse = gen_H(zebra)
            D_H_real = disc_H(horse)
            D_H_fake = disc_H(fake_horse.detach())
            H_reals += D_H_real.mean().item()
            H_fakes += D_H_fake.mean().item()
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            fake_zebra = gen_Z(horse)
            D_Z_real = disc_Z(zebra)
            D_Z_fake = disc_Z(fake_zebra.detach())
            D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            # put it togethor
            D_loss = (D_H_loss + D_Z_loss)/2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_H_fake = disc_H(fake_horse)
            D_Z_fake = disc_Z(fake_zebra)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

            # cycle loss
            cycle_zebra = gen_Z(fake_horse)
            cycle_horse = gen_H(fake_zebra)
            cycle_zebra_loss = l1(zebra, cycle_zebra)
            cycle_horse_loss = l1(horse, cycle_horse)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_zebra = gen_Z(zebra)
            identity_horse = gen_H(horse)
            identity_zebra_loss = l1(zebra, identity_zebra)
            identity_horse_loss = l1(horse, identity_horse)

            # add all togethor
            G_loss = (
                loss_G_Z
                + loss_G_H
                + cycle_zebra_loss * args.lambda_cycle 
                + cycle_horse_loss * args.lambda_cycle
                + identity_horse_loss * args.lambda_identity
                + identity_zebra_loss * args.lambda_identity
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % args.log_interval == 0:
            save_image(fake_horse*0.5+0.5, f"outputs/horse_{idx}.png")
            save_image(fake_zebra*0.5+0.5, f"outputs/zebra_{idx}.png")

        loop.set_postfix(H_real=H_reals/(idx+1), H_fake=H_fakes/(idx+1))



def main():
    #global vis, loss_function_plot

    # Training settings
    parser = argparse.ArgumentParser(description='Cycle GAN')

    parser.add_argument('--train-dir', type=str, default="datasets/horse2zebra", metavar='D',
                        help='training dataset directory (default:datasets/horse2zebra)')
    parser.add_argument('--val-dir', type=str, default="datasets/horse2zebra", metavar='D',
                        help='validation dataset directory (default:datasets/horse2zebra)')

    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--val-batch-size', type=int, default=16, metavar='N',
                        help='input batch size for testing (default: 16)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate (default: 1e-5)')
    
    parser.add_argument('--lambda-identity', type=float, default=0.0, metavar='LI',
                        help='Lambda identity (default: 0.0)')
    parser.add_argument('--lambda-cycle', type=int, default=10, metavar='LC',
                        help='Lambda cycle (default: 10)')
    
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                        help='Number of workers (default: 4)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--ckpt-path', type=str, default="checkpoints/cycle-gan-horses2zebras-v1", metavar='CKPT',
                        help='training dataset directory (default: checkpoints/cycle-gan-horses2zebras-v1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--load-model', action='store_true', default=False,
                        help='For Loading the Model')

    #parser.add_argument('--visdom_server', default='http://localhost',
    #                    help='Visdom server')
    #parser.add_argument('--visdom_port', default=8097,
    #                    help='Visdom port')
    #parser.add_argument('--visdom_env_name', default='main',
    #                    help='Visdom main experiment type')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # Connect to the Visdom server
    #vis = visdom.Visdom(server=args.visdom_server, port=args.visdom_port
    #        , env = args.visdom_env_name)

    # Create a new plot for the loss function
    #loss_function_plot = vis.line(X=torch.zeros((1,)).cpu(), Y=torch.zeros((1,)).cpu(),
    #        opts={
    #            "xlabel":"Iteration",
    #            "ylabel":"Loss",
    #            "title":"Loss over time",
    #            "legend":["Classification loss"],
    #            "layoutopts":{
    #                "plotly": {
    #                    "yaxis": { "type": "log" }
    #                    }
    #                }
    #            })
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if not args.no_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if not args.no_cuda else {}


    transforms = A.Compose(
        [
            A.Resize(width=256, height=256),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
            ToTensorV2(),
        ],
        additional_targets={"image0": "image"},
    )

    disc_H = Discriminator(in_channels=3).to(device)
    disc_Z = Discriminator(in_channels=3).to(device)
    gen_Z = Generator(img_channels=3, num_residuals=9).to(device)
    gen_H = Generator(img_channels=3, num_residuals=9).to(device)
    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if args.load_model:
        load_checkpoint(
            args.CKPT+'genh.pth.tar', gen_H, opt_gen, args.lr,
        )
        load_checkpoint(
            args.CKPT+'genz.pth.tar', gen_Z, opt_gen, args.lr,
        )
        load_checkpoint(
            args.CKPT+'critich.pth.tar', disc_H, opt_disc, args.lr,
        )
        load_checkpoint(
            args.CKPT+'criticz.pth.tar', disc_Z, opt_disc, args.lr,
        )

    dataset = HorseZebraDataset(
        
        root_horse=args.train_dir+"/trainA", root_zebra=args.train_dir+"/trainB", transform=transforms
    )
    val_dataset = HorseZebraDataset(
       root_horse=args.train_dir+"/testA", root_zebra=args.train_dir+"/testB", transform=transforms
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.epochs):
        train_fn(args, disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)

        if args.save_model:
            save_checkpoint(gen_H, opt_gen, filename=args.CKPT+'genh.pth.tar')
            save_checkpoint(gen_Z, opt_gen, filename=args.CKPT+'genz.pth.tar')
            save_checkpoint(disc_H, opt_disc, filename=args.CKPT+'critich.pth.tar')
            save_checkpoint(disc_Z, opt_disc, filename=args.CKPT+'criticz.pth.tar')

if __name__ == "__main__":
    main()