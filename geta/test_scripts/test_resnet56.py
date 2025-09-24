# HESSO on resnet56 Cifar10 dataset

import sys
sys.path.append('..')
from sanity_check.backends.resnet20_cifar10 import resnet56_cifar10
from only_train_once import OTO
import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import argparse

def get_config():
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.1,
        help="Sparsity",
    )

    # Parse arguments
    config = parser.parse_args()

    return config

def main(config):
    model = resnet56_cifar10()
    dummy_input = torch.rand(1, 3, 32, 32)
    oto = OTO(model=model.cuda(), dummy_input=dummy_input.cuda())

    # A ResNet_zig.gv.pdf will be generated to display the depandancy graph.
    oto.visualize(view=False, out_dir='../cache')

    trainset = CIFAR10(root='cifar10', train=True, download=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
    testset = CIFAR10(root='cifar10', train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

    trainloader =  torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)

    optimizer = oto.hesso(
        variant='sgd', 
        lr=0.1, 
        weight_decay=1e-4,
        target_group_sparsity=config.sparsity,
        start_pruning_step=10 * len(trainloader), 
        pruning_periods=10,
        pruning_steps=10 * len(trainloader)
    )

    from utils.utils import check_accuracy

    max_epoch = 200
    model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    # Every 50 epochs, decay lr by 10.0
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.1) 

    for epoch in range(max_epoch):
        f_avg_val = 0.0
        model.train()
        lr_scheduler.step()
        for X, y in trainloader:
            X = X.cuda()
            y = y.cuda()
            y_pred = model.forward(X)
            f = criterion(y_pred, y)
            optimizer.zero_grad()
            f.backward()
            f_avg_val += f
            optimizer.step()
        opt_metrics = optimizer.compute_metrics()
        # group_sparsity, param_norm, _ = optimizer.compute_group_sparsity_param_norm()
        # norm_important, norm_redundant, num_grps_important, num_grps_redundant = optimizer.compute_norm_groups()
        accuracy1, accuracy5 = check_accuracy(model, testloader)
        f_avg_val = f_avg_val.cpu().item() / len(trainloader)
        
        print("Ep: {ep}, loss: {f:.2f}, norm_all:{param_norm:.2f}, grp_sparsity: {gs:.2f}, acc1: {acc1:.4f}, norm_import: {norm_import:.2f}, norm_redund: {norm_redund:.2f}, num_grp_import: {num_grps_import}, num_grp_redund: {num_grps_redund}"\
            .format(ep=epoch, f=f_avg_val, param_norm=opt_metrics.norm_params, gs=opt_metrics.group_sparsity, acc1=accuracy1,\
            norm_import=opt_metrics.norm_important_groups, norm_redund=opt_metrics.norm_redundant_groups, \
            num_grps_import=opt_metrics.num_important_groups, num_grps_redund=opt_metrics.num_redundant_groups
            ))
        
    # save the .pt file
    torch.save(model, f"resnet56_best_{str(config.sparsity)}.pt")

if __name__ == "__main__":
    main(get_config())