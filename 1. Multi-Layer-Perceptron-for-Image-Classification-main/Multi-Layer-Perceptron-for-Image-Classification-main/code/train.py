import enum
import os
import argparse
from sklearn.metrics import label_ranking_loss
import torch
import torch.nn as nn
from dataset import SignDigitDataset
from torch.utils.data import DataLoader
from utils import *
from model import MLP
from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/sign_digits_experiment_1')

parser = argparse.ArgumentParser()
# Hyper-parameters
parser.add_argument('--n_epochs', type=int, default=100,
                    required=True, help='number of epochs for training')
parser.add_argument('--print_every', type=int, default=10,
                    help='print the loss every n epochs')
parser.add_argument('--img_size', type=int, default=64,
                    help='image input size')
parser.add_argument('--n_classes', type=int, default=6,
                    help='number of classes')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--learning_rate', type=float,
                    default=0.001, help='learning rate')
parser.add_argument('--hidden_layers', type=int, required=True, nargs='+',
                    help='number of units per layer (except input and output layer)')
parser.add_argument('--activation', type=str, default='relu',
                    choices=['relu', 'tanh'], help='activation layers')
args = parser.parse_args()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("WARNING: You are not using gpu!")

#####################################################################################
# TODO: Complete the script to do the following steps                               #
# 0. Create train/test datasets
# 1. Create train and test data loaders with respect to some hyper-parameters       #
# 2. Get an instance of your MLP model.                                             #
# 3. Define an appropriate loss function (e.g. cross entropy loss)                  #
# 4. Define an optimizers with proper hyper-parameters such as (learning_rate, ...).#
# 5. Implement the main loop function with n_epochs iterations which the learning   #
#    and evaluation process occurred there.                                         #
# 6. Save the model weights                                                         #
#####################################################################################


# 0. creating train_dataset and test_dataset
train_dataset = SignDigitDataset(root_dir='data/',
                                 h5_name='train_signs.h5',
                                 train=True,
                                 transform=get_transformations(64))

test_dataset = SignDigitDataset(root_dir='data/',
                                h5_name='test_signs.h5',
                                train=False,
                                transform=get_transformations(64))
# 1. Data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=False)
# 2. get an instance of the model
input_size = args.img_size*args.img_size*3
hidden_size = args.hidden_layers
num_classes = args.n_classes

units = [input_size] + hidden_size + [num_classes]

print(units)

model = MLP(units, args.activation, 'uniform')
model.apply(init_weights)
# 3, 4. loss function and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
# 5. Train the model
n_train_batches = 0
n_test_batches = 0
for epoch in range(args.n_epochs):
    print(epoch)
    train_running_loss, test_running_loss = 0.0, 0.0

    for i, image_label_dict_train in enumerate(train_loader):
        # extract the images and labels from train_loader
        images = image_label_dict_train['image'].reshape(-1, input_size)
        labels = image_label_dict_train['label']
        # cast the labels tensor from long to float (because of the CrossEntropyLoss)
        labels = labels.float()
        # Forward pass
        outputs = model(images)
        # print(outputs)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n_train_batches = i

    print(n_train_batches)

    train_running_loss = loss.item()

    # print('Epoch [{}/{}]: Loss: {:.4f}'.format(epoch +
    #                                            1, args.n_epochs, loss.item()))

    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        correct = 0
        total = 0
        for i, image_label_dict_train in enumerate(test_loader):
            images = image_label_dict_train['image'].reshape(-1, input_size)
            labels = image_label_dict_train['label']
            # cast the labels tensor from long to float (because of the CrossEntropyLoss)
            labels = labels.float()

            outputs = model(images)
            loss = criterion(outputs, labels)

            n_test_batches = i

        # test_running_loss = loss.item()

            # _, predicted = torch.max(outputs.data, 1)
            # predicted = torch.empty(8, 6)

            # for i, out in enumerate(outputs):
            #     max_out = float(out.max())
            #     out = (out == max_out)
            #     predicted[i] = out
            #     pred_equal_label = torch.equal(labels[i], predicted[i])

            #     if(pred_equal_label):
            #         correct += 1
            # correct += (predicted == labels).sum().item()

    print('correct: {}'.format(correct))
    total = len(test_dataset)

    # print(labels)
    # print(outputs)
    # print(predicted)
    # correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(
        100 * correct / total))

    # print('total: {}'.format(total))
    # print('correct: {}'.format(correct))
    # print(is_equal)

    # # ...log the running loss
    # writer.add_scalar('Train Loss', train_running_loss / n_train_batches, epoch)
    # writer.add_scalar('Test Loss', test_running_loss / n_test_batches, epoch)

    # if epoch % args.print_every == 0:
    #     # You have to log the accuracy as well
    #     print('Epoch [{}/{}]:\t Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch + 1,
    #                                                                            args.n_epochs,
    #                                                                            train_running_loss / n_train_batches,
    #                                                                            test_running_loss / n_test_batches))

#####################################################################################
#                                 END OF YOUR CODE                                  #
#####################################################################################


# save the model weights
checkpoint_dir = 'checkpoints/my_model.pth'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

torch.save(model.state_dict(), checkpoint_dir)
