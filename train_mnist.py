import torch
import torch.nn as nn
import numpy as np
import time
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from torchvision.transforms import RandomRotation
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
from model.architecture import E4_net
from model.architecture_large import E4_netL
from PIL import Image
from datetime import datetime
import argparse
now=datetime.now()
now=datetime.strftime(now,'%Y-%m-%d %H:%M:%S') 




parser = argparse.ArgumentParser(description='net')
parser.add_argument('--model', '-a', default='normal')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 200) ')
parser.add_argument('--learning_rate', type=float, default=2e-2, help='learning rate (default: 0.1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--kernel', default=5, type=int, help='kernel_size')
parser.add_argument('--bias', default=False, type=bool, help='bias')
parser.add_argument('--reduction', default=1, type=float, help='reduction_ratio')
parser.add_argument('--groups', default=8, type=int, help='groups')
parser.add_argument('--dropout', default=0.2, type=float, help='dropout_rate')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
cudnn.benchmark = False  # Should make training should go faster for large models
cudnn.deterministic = True
cudnn.enabled = True

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

print(args)




def compute_param(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    # print('Parameters of the net: {}M'.format(params/(10**6)))
    return params


class MnistRotDataset(Dataset):
    
    def __init__(self, mode, transform=None):
        assert mode in ['train', 'test']
            
        if mode == "train":
            file = "data/mnist_rotation_new/mnist_all_rotation_normalized_float_train_valid.amat"
        else:
            file = "data/mnist_rotation_new/mnist_all_rotation_normalized_float_test.amat"
        
        self.transform = transform

        data = np.loadtxt(file, delimiter=' ')
            
        self.images = data[:, :-1].reshape(-1, 28, 28).astype(np.float32)
        self.labels = data[:, -1].astype(np.int64)
        self.num_samples = len(self.labels)
    
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.labels)


totensor = ToTensor()

train_transform = Compose([
    totensor,
])

mnist_train = MnistRotDataset(mode='train', transform=train_transform)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=args.batch_size, shuffle=True)


test_transform = Compose([
    totensor,
])
mnist_test = MnistRotDataset(mode='test', transform=test_transform)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=128)


if args.model=='normal':
    net=E4_net(args.kernel, args.groups, args.reduction, args.dropout)
elif args.model=='large':
    net=E4_netL(args.kernel, args.groups, args.reduction, args.dropout)



# net=cnn()
param=compute_param(net)
model = nn.DataParallel(net, device_ids=range(torch.cuda.device_count())).cuda()
loss_function = torch.nn.CrossEntropyLoss()



schedule=[60,120,160]
learning_rate = args.learning_rate
start=schedule[0]
best=0.
best_epoch=0
for epoch in range(1,args.epochs):
    if(epoch==schedule[0]):
        learning_rate=learning_rate*0.1
    elif(epoch==schedule[1]):
        learning_rate=learning_rate*0.1
    elif(epoch==schedule[2]):
        learning_rate=learning_rate*0.1


    t1=time.time()
   
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    model.train()
    total = 0
    correct = 0
    print('Parameters of the net: {}M'.format(param/(10**6)))
    for i, (x, t) in enumerate(train_loader):
        optimizer.zero_grad()

        x = x.cuda()
        t = t.cuda()
        y = model(x)
        _, prediction = torch.max(y.data, 1)
        total += t.shape[0]
        correct += (prediction == t).sum().item()
        loss = loss_function(y, t)
        # writer.add_scalar('training loss', loss/x.size(0), epoch * len(train_loader) + i)
        loss.backward()

        optimizer.step()
    print(f"epoch {epoch} | train accuracy: {correct/total*100.}")
    if(epoch>=start):
        total = 0
        correct = 0
        with torch.no_grad():
            model.eval()
            for i, (x, t) in enumerate(test_loader):

                x = x.cuda()
                t = t.cuda()
                
                y = model(x)

                _, prediction = torch.max(y.data, 1)
                total += t.shape[0]
                correct += (prediction == t).sum().item()
        print(f"epoch {epoch} | test accuracy: {correct/total*100.}")
        if(correct/total*100.>best):
            best=correct/total*100
            best_epoch=epoch
            print('Best test acc: {},Best epoch: {}'.format(best,best_epoch))
        
    t2=time.time()
    print('Comsuming {}s'.format(t2-t1))
    
    print('\n')
    
print('Best test acc: {}, Best epoch: {}'.format(best,best_epoch))

#Right