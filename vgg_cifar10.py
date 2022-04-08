import time
import os

import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn


from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from PIL import Image
import matplotlib.pyplot as plt

import argparse

import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

# train process
print("\nLOAD DATA\n")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64

#Create train and test dataset
train_dataset = datasets.CIFAR10(root='data',
								 train=True,
								 transform=transforms.ToTensor(),
								 download=True)

test_dataset = datasets.CIFAR10(root='data',
								train=False,
								transform=transforms.ToTensor())

#Create train and test dataloader
train_loader = DataLoader(dataset=train_dataset,
						  batch_size=BATCH_SIZE,
						  shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
						 batch_size=BATCH_SIZE,
						 shuffle=False)

#Create VGG19 model
class VGG19(torch.nn.Module):
	def __init__(self, num_classes):
		super(VGG19, self).__init__()
		self.block1 = nn.Sequential(
			nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(3,3),stride=(1,1),padding=1),
			nn.BatchNorm2d(num_features=64),
			nn.ReLU(),
			nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),stride=(1,1),padding=1),
			nn.BatchNorm2d(num_features=64),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
		)
		self.block2 = nn.Sequential(
			nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
			nn.BatchNorm2d(num_features=128),
			nn.ReLU(),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
			nn.BatchNorm2d(num_features=128),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		)
		self.block3 = nn.Sequential(
			nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
			nn.BatchNorm2d(num_features=256),
			nn.ReLU(),
			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
			nn.BatchNorm2d(num_features=256),
			nn.ReLU(),
			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
			nn.BatchNorm2d(num_features=256),
			nn.ReLU(),
			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
			nn.BatchNorm2d(num_features=256),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		)
		self.block4 = nn.Sequential(
			nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
			nn.BatchNorm2d(num_features=512),
			nn.ReLU(),
			nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
			nn.BatchNorm2d(num_features=512),
			nn.ReLU(),
			nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
			nn.BatchNorm2d(num_features=512),
			nn.ReLU(),
			nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
			nn.BatchNorm2d(num_features=512),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		)
		self.block5 = nn.Sequential(
			nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
			nn.BatchNorm2d(num_features=512),
			nn.ReLU(),
			nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
			nn.BatchNorm2d(num_features=512),
			nn.ReLU(),
			nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
			nn.BatchNorm2d(num_features=512),
			nn.ReLU(),
			nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
			nn.BatchNorm2d(num_features=512),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		)
		self.classifier = nn.Sequential(
			nn.Linear(512*1*1,4096),
			nn.ReLU(),
			nn.Dropout(0.5),

			nn.Linear(4096,4096),
			nn.ReLU(),
			nn.Dropout(0.5),

			nn.Linear(4096,num_classes)
		)
	def forward(self,x):
		x = self.block1(x)
		x = self.block2(x)
		x = self.block3(x)
		x = self.block4(x)
		x = self.block5(x)
		logits = self.classifier(x.view(-1, 512*1*1))
		probas = F.softmax(logits, dim=1)
		return logits, probas


net = VGG19(10)
#print(net)
#print(net(torch.randn([1,3,32,32])))

NUM_EPOCHS = 15
model = VGG19(num_classes=10)
model = model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#training process
valid_loader = test_loader

def computer_accuracy_and_loss(model, data_loader, device):
	correct_pred, num_examples = 0, 0
	cross_entropy = 0
	for i , (features, targets) in enumerate(data_loader):

		features = features.to(DEVICE)
		targets = targets.to(DEVICE)

		logits, probas = model(features)
		cross_entropy += F.cross_entropy(logits, targets).item()
		_, predicted_labels = torch.max(probas, 1)
		num_examples += targets.size(0)
		correct_pred += (predicted_labels == targets).sum()
	return correct_pred.float()/num_examples *100, cross_entropy/num_examples

start_time = time.perf_counter()
train_acc_list, valid_acc_list = [], []
train_loss_list, valid_loss_list = [], []

for epoch in range(NUM_EPOCHS):

	model.train()

	for batch_idx , (features, targets) in enumerate(train_loader):
		#prepare minibatch
		features = features.to(DEVICE)
		targets = targets.to(DEVICE)

		#forward and back prop
		logits, probas = model(features)
		cost = F.cross_entropy(logits, targets)
		optimizer.zero_grad()

		cost.backward()
		#update model parameters
		optimizer.step()

		#logging
		if not  batch_idx % 300:
			print(f'Epoch:{epoch+1:03d}/{NUM_EPOCHS:03d} |'
				  f'Batch:{batch_idx:03d}/{len(train_loader):03d}|'
				  f'Loss:{cost:.4f}')

	#no need to build the computation graph for backprop when computing accuracy
	model.eval()
	with torch.set_grad_enabled(False):
		train_acc, train_loss = computer_accuracy_and_loss(model, train_loader, device=DEVICE)
		valid_acc, valid_loss = computer_accuracy_and_loss(model, test_loader, device=DEVICE)
		train_acc_list.append(train_acc)
		valid_acc_list.append(valid_acc)
		train_loss_list.append(train_loss)
		valid_loss_list.append(valid_loss)
		print(f'Epch:{epoch+1:03d}/{NUM_EPOCHS:03d} Train Acc:{train_acc:.2f}'
			  f'| Test Acc:{valid_acc:.2f}%')

	elapsed = (time.time() - start_time)/60
	print(f'Time elapsed:{elapsed:.2f} minutes')

elapsed = (time.time() - start_time)/60
print(f'Total Training Time:{elapsed:.2f} minutes')


#test process
model.eval()
with torch.set_grad_enabled(False):
	test_acc, test_loss = computer_accuracy_and_loss(model, test_loader, DEVICE)
	print(f'Test accuracy:{test_acc:.2f}%')

import matplotlib.pyplot as plt

#plotting process

#训练和测试的损失值
plt.plot(range(1, NUM_EPOCHS + 1), train_loss_list, label='Training loss')
plt.plot(range(1, NUM_EPOCHS + 1), valid_loss_list, label='Validation loss')
plt.legend(loc='upper right')
plt.ylabel('Cross entropy')
plt.xlabel('Epoch')
plt.show()



#训练和测试的准确值
plt.plot(range(1, NUM_EPOCHS + 1), train_acc_list, label='Training accuracy')
plt.plot(range(1, NUM_EPOCHS + 1), valid_acc_list, label='Validation accuracy')
plt.legend(loc='upper left')
plt.ylabel('Cross entropy')
plt.xlabel('Epoch')
plt.show()


#预测环节
_, predictions = model.forward(features[:8].to(DEVICE))
predictions = torch.argmax(predictions, dim=1)
print(predictions)

features = features[:7]
fig = plt.figure()
tname = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

for i in range(6):
	plt.subplot(2,3,i+1)
	plt.tight_layout()
	tmp = features[i]
	plt.imshow(np.transpose(tmp, (1,2,0)))
	plt.title("Actual value:{}".format(tname[targets[i]])+'\n'+"Prediction value:{}".format(tname(predictions[i])),size=10)
plt.show()
























































































