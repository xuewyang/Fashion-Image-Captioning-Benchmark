import argparse
import os
import pdb
import sys
import numpy as np
from sklearn.metrics import average_precision_score
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from resnet import ResNet
from dataset import FICDataset
from torch import nn
import pytorch_warmup as warmup

# Data parameters

# Model parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_folder', type=str,
                        help='folder with data files saved.', default='')
    parser.add_argument('--model_folder', type=str,
                        help='base folder to save models.', default='')
    parser.add_argument('--checkpoint', type=str,
                        help='Resume training using checkpoint.', default='')
    parser.add_argument('--epochs', type=int,
                        help='number of epochs to train for (if early stopping is not triggered).', default=200)
    parser.add_argument('--batch_size', type=int,
                        help='batch size for training.', default=100)
    parser.add_argument('--iters_epoch', type=int,
                        help='iterations per epoch.', default=10000)
    parser.add_argument('--patience', type=int,
                        help='early stopping.', default=30)
    parser.add_argument('--workers', type=int,
                        help='for data-loading; right now, only 1 worker with h5py.', default=1)
    parser.add_argument('--print_freq', type=int,
                        help='print training/validation stats every __ batches.', default=200)
    parser.add_argument('--lr', type=float,
                        help='learning rate for cnn if fine-tuning.', default=1e-5)
    parser.add_argument('--dropout', type=float,
                        help='dropout rate.', default=0.5)
    parser.add_argument('--grad_clip', type=float,
                        help='clip gradients at an absolute value of.', default=5)
    parser.add_argument('--num_att', type=int,
                        help='number of attributes.', default=990)
    parser.add_argument('--num_cate', type=int,
                        help='number of categories.', default=78)
    return parser.parse_args(argv)


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(model_folder, epoch, epochs_since_improvement, model, optimizer, acc, is_best):
    """
    Saves model checkpoint.
    :param model_folder: where to save checkpoint
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in metric
    :param model: resnet model
    :param optimizer: optimizer to update resnet's weights, if fine-tuning
    :param acc: validation acc score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    if torch.cuda.device_count() > 1:
        state = {'epoch': epoch,
                 'epochs_since_improvement': epochs_since_improvement,
                 'acc': acc,
                 'model': model.module.state_dict(),
                 'optimizer': optimizer.state_dict()}
    else:
        state = {'epoch': epoch,
                 'epochs_since_improvement': epochs_since_improvement,
                 'acc': acc,
                 'model': model.state_dict(),
                 'optimizer': optimizer.state_dict()}

    filename = 'resnet_101_fashion_' + str(epoch) + '.pth'
    torch.save(state, os.path.join(model_folder, filename))
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, os.path.join(model_folder, 'resnet_101_fashion_best.pth'))


def main(args):
    """
    Training and validation.
    """
    start_epoch = 0
    epochs_since_improvement = 0
    best_acc = 0

    # Initialize / load checkpoint
    if len(args.checkpoint) == 0:
        model = ResNet(num_cate=args.num_cate)
        model = model.to(device)
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)
    else:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = ResNet(num_cate=args.num_cate)
        model.load_state_dict(checkpoint['model'])
        model = model.to(device)
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_loader = torch.utils.data.DataLoader(
        FICDataset(args.data_folder, 'TRAIN', transform=preprocess),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        FICDataset(args.data_folder, 'VAL', transform=preprocess),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        FICDataset(args.data_folder, 'TEST', transform=preprocess),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)
    # Custom dataloaders

    # Epochs
    for epoch in range(start_epoch, args.epochs):
        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == args.patience:
            break

        # One epoch's training
        train(args=args, train_loader=train_loader, model=model, criterion=criterion,
              optimizer=optimizer, epoch=epoch)

        # One epoch's validation
        recent_acc = validate(args=args, val_loader=val_loader, model=model, criterion=criterion)
        # Check if there was an improvement
        is_best = recent_acc > best_acc
        best_acc = max(recent_acc, best_acc)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0
        if not os.path.exists(args.model_folder):
            os.mkdir(args.model_folder)
        save_checkpoint(args.model_folder, epoch, epochs_since_improvement, model, optimizer, recent_acc, is_best)
        test_acc = test(args=args, test_loader=test_loader, model=model, criterion=criterion)


def train(args, train_loader, model, criterion, optimizer, epoch):
    """
    Performs one epoch's training.
    :param train_loader: DataLoader for training data
    :param model: cnn model
    :param criterion: loss layer
    :param optimizer: optimizer to update cnn's weights
    :param epoch: epoch number
    """
    model.train()
    # Batches
    loss_total = []
    correct = 0
    for i, (imgs, cates) in enumerate(train_loader):
        # Move to GPU, if available
        imgs = imgs.to(device)
        cates = cates.to(device)
        # Forward prop.
        logits = model(imgs)
        loss = criterion(logits, cates)
        optimizer.zero_grad()
        loss.backward()
        # Clip gradients
        if args.grad_clip is not None:
            clip_gradient(optimizer, args.grad_clip)
        # Update weights
        optimizer.step()
        # Keep track of metrics
        predictions = torch.argmax(logits, dim=-1)
        batch_correct = (predictions == cates).float().sum()
        correct += batch_correct
        batch_acc = batch_correct / imgs.shape[0]
        loss_total.append(loss)

        # Print status
        if i % args.print_freq == 0 and i != 0:
            print('Epoch: [{}][{}/{}]\t Loss {:.4f}\t ACC {:.4f}'.format(epoch, i, len(train_loader), loss, batch_acc))
    print('*******************************************************************')
    accuracy = correct / len(train_loader)
    print('Training Epoch: [{}] Loss {:.4f}\t ACC {:.4f}\t'.
          format(epoch, sum(loss_total) / len(loss_total), accuracy))
    print('*******************************************************************')


def validate(args, val_loader, model, criterion):
    """
    Performs one epoch's validation.
    :param val_loader: DataLoader for validation data.
    :param model: cnn model
    :param criterion: loss layer
    """
    model.eval()
    loss_total = []
    with torch.no_grad():
        # Batches
        correct = 0
        for i, (imgs, cates) in enumerate(val_loader):
            # Move to device, if available
            imgs = imgs.to(device)
            cates = cates.to(device)
            logits = model(imgs)
            # Calculate loss
            loss = criterion(logits, cates)
            # Keep track of metrics
            predictions = torch.argmax(logits, dim=-1)
            batch_correct = (predictions == cates).float().sum()
            correct += batch_correct
            batch_acc = batch_correct / imgs.shape[0]
            loss_total.append(loss)
            if i % args.print_freq == 0 and i != 0:
                print('Epoch: [{}/{}]\t Loss {:.4f}\t ACC {:.4f}'.format(i, len(val_loader), loss, batch_acc))

            loss_total.append(loss)
        accuracy = correct / len(val_loader)

        print('Validation: Loss {:.4f}\t ACC {:.4f}\t'.
              format(sum(loss_total) / len(loss_total), accuracy))
    return accuracy


def test(args, test_loader, model, criterion):
    """
    Performs one epoch's validation.
    :param test_loader: DataLoader for test data.
    :param model: resnet model
    :param criterion: loss layer
    """
    model.eval()
    loss_total = []
    correct = 0
    with torch.no_grad():
        # Batches
        for i, (imgs, cates) in enumerate(test_loader):
            # Move to device, if available
            imgs = imgs.to(device)
            cates = cates.to(device)
            logits = model(imgs)

            # Calculate loss
            loss = criterion(logits, cates)
            # Keep track of metrics
            predictions = torch.argmax(logits, dim=-1)
            batch_correct = (predictions == cates).float().sum()
            correct += batch_correct
            batch_acc = batch_correct / imgs.shape[0]
            if i % args.print_freq == 0 and i != 0:
                print('Epoch: [{}/{}]\t Loss {:.4f}\t ACC {:.4f}'.format(i, len(test_loader), loss, batch_acc))
            loss_total.append(loss)
        accuracy = correct / len(test_loader)
        print('Test: Loss {:.4f}\t ACC {:.4f}\t'.
              format(sum(loss_total) / len(loss_total), accuracy))
    return accuracy


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
