# Reference: https://github.com/pytorch/examples/tree/master/mnist

from __future__ import print_function, division


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
from torchvision import transforms, utils
import math
from skimage import io

######################################################################
# Setting
# ====================
import argparse
parser = argparse.ArgumentParser(description='GFGnet')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='Number of images training in one batch (default 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

######################################################################
# Data
# ====================
from data_load import ChicagoFaceDatabase, InstagramDatabase, ToTensor, Rescale, CropSquare

CFD_train_dataset = ChicagoFaceDatabase(root_dir='../data/ChicagoFaceDatabase/',
                                        transform=transforms.Compose([
                                            CropSquare(),
                                            Rescale(256),
                                            ToTensor()]),
                                        train=True)
CFD_train_loader = torch.utils.data.DataLoader(dataset=CFD_train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True)

CFD_test_dataset = ChicagoFaceDatabase(root_dir='../data/ChicagoFaceDatabase/',
                                       transform=transforms.Compose([
                                           CropSquare(),
                                           Rescale(256),
                                           ToTensor()]),
                                       train=False)
CFD_test_loader = torch.utils.data.DataLoader(dataset=CFD_test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True)

INS_train_dataset = InstagramDatabase(root_dir='../data/InstagramSelfieDatabase/',
                                      transform=transforms.Compose([
                                          CropSquare(),
                                          Rescale(256),
                                          ToTensor()]),
                                      train=True)
INS_train_loader = torch.utils.data.DataLoader(dataset=INS_train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True)

INS_test_dataset = InstagramDatabase(root_dir='../data/InstagramSelfieDatabase',
                                     transform=transforms.Compose([
                                         CropSquare(),
                                         Rescale(256),
                                         ToTensor()]),
                                     train=False)
INS_test_loader = torch.utils.data.DataLoader(dataset=INS_test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True)


######################################################################
# Creating the Network
# ====================
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1,  bias=True)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=True)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.max_pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.max_pool5 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(512 * 8 * 8, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.max_pool1(F.relu(x))

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool2(F.relu(x))

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.max_pool3(F.relu(x))

        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.max_pool4(F.relu(x))

        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.max_pool5(F.relu(x))

        x = x.view(-1, 512 * 8 * 8)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


######################################################################
# Initialize network
# =====================
model = Net().double()
if args.cuda:
    model = model.cuda()


######################################################################
# Train the Network
# ====================
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


def train(epoch, train_loader=CFD_train_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        # print(output)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), loss.data[0]))



######################################################################
# Test the Network
# ====================
def test(test_loader=CFD_test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    correct5 = 0
    correct3 = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

        # currently only support 1 batch for reporting top3 and top5 accuracy
        (preddata, predIdx) = output.data.sort(descending=True)
        # print(predIdx)
        # print(target)

        for i in range(0, 5):
            pred = predIdx[0][i]
            # print(pred)
            correct5 += pred.eq(target.data.view_as(pred)).long().cpu().sum()

            if i < 4:
                correct3 += pred.eq(target.data.view_as(pred)
                                    ).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.0f}%), Top 3 Accuracy: {}/{} ({:.0f}%), Top 5 Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset),
        correct3, len(test_loader.dataset),
        100. * correct3 / len(test_loader.dataset),
        correct5, len(test_loader.dataset),
        100. * correct5 / len(test_loader.dataset)))


######################################################################
# Run on new image input
# ====================
def predict(imagePath, modelPath=None):
    # load trained model
    if (modelPath == None):
        predict_model = model
    else:
        checkpoint = torch.load(modelPath,map_location=lambda storage, loc:storage)
        predict_model = Net().double()
        if args.cuda:
            predict_model = predict_model.cuda()

        predict_model.load_state_dict(checkpoint['state_dict'])

    # load image
    image = io.imread(imagePath)
    transform = transforms.Compose([CropSquare(),
                                    Rescale(256),
                                    ToTensor()])
    image = transform(image)
    data = image.unsqueeze(0)
    data = Variable(data)

    output = predict_model(data)
    # get the index of the max log-probability
    pred = output.data.max(1, keepdim=True)[1]
    print(pred.item())
    return pred.item()


# set the dataset here
dataset = 'CFD'

if dataset == 'CFD':
    train_loader = CFD_train_loader
    test_loader = CFD_test_loader
else:
    train_loader = INS_train_loader
    test_loader = INS_test_loader

# start training
for epoch in range(1, args.epochs + 1):

    train(epoch, train_loader)
    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, 'trained-' + str(epoch) + '.model')
    test(test_loader)

   
