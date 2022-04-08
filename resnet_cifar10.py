import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt

# train process
print("\nLOAD DATA\n")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64
train_dataset = datasets.CIFAR10(root='data',
								 train=True,
								 transform=transforms.ToTensor(),
								 download=True)

test_dataset = datasets.CIFAR10(root='data',
								train=False,
								transform=transforms.ToTensor())

##################################################### Using a GPU #####################################################
cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

print("\nUSING", device)
if cuda:
	num_dev = torch.cuda.current_device()
	print(torch.cuda.get_device_name(num_dev), "\n")


train_loader = DataLoader(dataset=train_dataset,
						  batch_size=batch_size,
						  shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
						 batch_size=batch_size,
						 shuffle=False)


def conv3x3(in_planes, out_planes, stride=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=1, bias=False)


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
							   padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes * 4)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out


class ResNet(nn.Module):

	def __init__(self, block, layers, num_classes, grayscale):
		self.inplanes = 64
		if grayscale:
			in_dim = 1
		else:
			in_dim = 3
		super(ResNet, self).__init__()
		self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
							   bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
		self.avgpool = nn.AvgPool2d(7, stride=1)
		self.fc = nn.Linear(512 * block.expansion, num_classes)

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		# because MNIST is already 1x1 here:
		# disable avg pooling
		# x = self.avgpool(x)

		x = x.view(x.size(0), -1)
		logits = self.fc(x)
		probas = F.softmax(logits, dim=1)
		return logits, probas


def resnet101(num_classes):
	"""Constructs a ResNet-34 model."""
	model = ResNet(block=Bottleneck,
				   layers=[3, 4, 23, 3],
				   num_classes=num_classes,
				   grayscale=False)
	return model


NUM_EPOCHS = 16
model = resnet101(num_classes=10)
model = model.to(DEVICE)

# 选用Adam优化比SGD的效果要好
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

valid_loader = test_loader


def compute_accuracy_and_loss(model, data_loader, device):
	correct_pred, num_examples = 0, 0
	cross_entropy = 0
	for i, (features, targets) in enumerate(data_loader):
		features = features.to(device)
		targets = targets.to(device)

		logits, probas = model(features)
		cross_entropy += F.cross_entropy(logits, targets).item()
		_, predicted_labels = torch.max(probas, 1)
		num_examples += targets.size(0)
		correct_pred += (predicted_labels == targets).sum()
	return correct_pred.float() / num_examples * 100, cross_entropy / num_examples


start_time = time.perf_counter()
train_acc_list, valid_acc_list = [], []
train_loss_list, valid_loss_list = [], []

for epoch in range(NUM_EPOCHS):

	model.train()

	for batch_idx, (features, targets) in enumerate(train_loader):

		# Prepare Minibatch
		features = features.to(DEVICE)
		targets = targets.to(DEVICE)

		# Forward and back prop
		logits, probas = model(features)
		cost = F.cross_entropy(logits, targets)
		optimizer.zero_grad()

		cost.backward()

		# Update model parameters
		optimizer.step()

		# Logging
		if not batch_idx % 200:
			print(f'Epoch:{epoch + 1:03d}/{NUM_EPOCHS:03d}|'
				  f'Batch:{batch_idx:04d}/{len(train_loader):04d} |'
				  f'Cost:{cost:.4f}')

	# No need to build the computation graph for backprop when computing accuracy
	model.eval()
	with torch.set_grad_enabled(False):
		train_acc, train_loss = compute_accuracy_and_loss(model, train_loader, device=DEVICE)
		valid_acc, valid_loss = compute_accuracy_and_loss(model, valid_loader, device=DEVICE)
		train_acc_list.append(train_acc)
		valid_acc_list.append(valid_acc)
		train_loss_list.append(train_loss)
		valid_loss_list.append(valid_loss)
		print(f'Epoch:{epoch + 1:03d}/{NUM_EPOCHS:03d} Train Accuracy:{train_acc:.2f}'
			  f'|Test Accuracy:{valid_acc:.2f}')

	elapsed = (time.time() - start_time) / 60
	print(f'Time elapsed:{elapsed:.2f} minutes')

elapsed = (time.time() - start_time) / 60
print(f'Total Training Time:{elapsed:.2f} minutes')


#plotting process
#训练损失和测试损失图
plt.plot(range(1, NUM_EPOCHS+1), train_loss_list, label='Training loss')
plt.plot(range(1, NUM_EPOCHS+1), valid_loss_list, label='Validation loss')
plt.legend(loc='upper right')
plt.ylabel('Cross entropy')
plt.xlabel('Epoch')
plt.show()

#训练精度和测试精度
plt.plot(range(1, NUM_EPOCHS+1), train_acc_list, label='Training accuracy')
plt.plot(range(1, NUM_EPOCHS+1), valid_acc_list, label='Validation accuracy')
plt.legend(loc='upper left')
plt.ylabel('Cross entropy')
plt.xlabel('Epoch')
plt.show()





# Test process
model.eval()
with torch.set_grad_enabled(False):  # save memory during inference
	test_acc, test_loss = compute_accuracy_and_loss(model, test_loader, DEVICE)
	print(f'Test accuracy: {test_acc:.2f}%')





# plotting process
#训练损失和测试损失图
plt.plot(range(1, NUM_EPOCHS + 1), train_loss_list, label='Training loss')
plt.plot(range(1, NUM_EPOCHS + 1), valid_loss_list, label='Validation loss')
plt.legend(loc='upper right')
plt.ylabel('Cross entropy')
plt.xlabel('Epoch')
plt.show()

#训练精度和测试精度
plt.plot(range(1, NUM_EPOCHS + 1), train_acc_list, label='Training accuracy')
plt.plot(range(1, NUM_EPOCHS + 1), valid_acc_list, label='Validation accuracy')
plt.legend(loc='upper left')
plt.ylabel('Cross entropy')
plt.xlabel('Epoch')
plt.show()


import matplotlib.pyplot as plt

for features, targets in train_loader:
	break
# 预测环节
_, predictions = model.forward(features[:8].to(DEVICE))
predictions = torch.argmax(predictions, dim=1)
print(predictions)

features = features[:7]
fig = plt.figure()
tname = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# print(features[i].size())


for i in range(6):
	plt.subplot(2, 3, i + 1)
	plt.tight_layout()
	tmp = features[i]
	plt.imshow(np.transpose(tmp, (1, 2, 0)))
	plt.title(
		"Actual value: {}".format(tname[targets[i]]) + '\n' + "Prediction value: {}".format(tname[predictions[i]]),
		size=10)

#plt.title("Prediction value: {}".format(tname[targets[i]]))
plt.show()
























