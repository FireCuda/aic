import os
import torch.utils.data
from torch import nn
from torch.nn import DataParallel
from config import BATCH_SIZE, SAVE_DIR, TOTAL_EPOCH, IMG_SIZE, NUM_CLASSES, LEARNING_RATE, SAVED_EPOCH
from config import CASIA_DATA_DIR
from model import MobileFaceNet, ArcMarginProduct
from CASIA_Face_loader import CASIA_Face
from torch.optim import lr_scheduler
import torch.optim as optim
import time
import numpy as np
import scipy.io
os.environ["CUDA_VISIBLE_DEVICES"]="1"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# define trainloader and testloader
trainset = CASIA_Face(root_path=CASIA_DATA_DIR, image_size = IMG_SIZE)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2, drop_last=False)
# define model
net = MobileFaceNet()
ArcMargin = ArcMarginProduct(512, NUM_CLASSES)

net = net.to(device)
ArcMargin = ArcMargin.to(device)

# define optimizers
ignored_params = list(map(id, net.linear.parameters()))
ignored_params += list(map(id, ArcMargin.weight))
prelu_params_id = []
prelu_params = []
for m in net.modules():
    if isinstance(m, nn.PReLU):
        ignored_params += list(map(id, m.parameters()))
        prelu_params += m.parameters()
base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

optimizer_ft = optim.SGD([
    {'params': base_params, 'weight_decay': 4e-5},
    {'params': net.linear.parameters(), 'weight_decay': 4e-4},
    {'params': ArcMargin.weight, 'weight_decay': 4e-4},
    {'params': prelu_params, 'weight_decay': 0.0}
], lr=LEARNING_RATE, momentum=0.9, nesterov=True)

exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[200000], gamma=0.1)

# Using cross-entropy loss for classification human ID
criterion = torch.nn.CrossEntropyLoss()

best_acc = 0.0
best_epoch = 0
iters = 0
for epoch in range(0, TOTAL_EPOCH+1):
    # train model
    print('Train Epoch: {}/{} ...'.format(epoch, TOTAL_EPOCH))
    net.train()

    train_total_loss = 0.0
    total = 0
    since = time.time()
    print('iteration for 1 epoch: ', iters)
    for data in trainloader:
        iters = iters + 1 
        img, label = data[0].cuda(), data[1].cuda()
        batch_size = img.size(0)
        optimizer_ft.zero_grad()

        raw_logits = net(img)

        output = ArcMargin(raw_logits, label)
        total_loss = criterion(output, label)
        total_loss.backward()
        optimizer_ft.step()
        exp_lr_scheduler.step()

        train_total_loss += total_loss.item() * batch_size
        total += batch_size

    train_total_loss = train_total_loss / total
    time_elapsed = time.time() - since
    loss_msg = 'total_loss: {:.4f} time: {:.0f}m {:.0f}s'\
        .format(train_total_loss, time_elapsed // 60, time_elapsed % 60)
    print(loss_msg)

    # save model
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    if epoch % SAVED_EPOCH == 0:
        torch.save(net.state_dict(),
            os.path.join(SAVE_DIR, '%03d.pth' % epoch))
print('finishing training')
