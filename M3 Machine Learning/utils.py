import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt  

class Model(nn.Module):
	def __init__(self, layers, lr=1e-3):
		super().__init__()
		self.layers = layers
		self.lr = lr

	def __call__(self, x):
		out = x
		for layer in self.layers:
			out = layer(out)
		return out
	
	def update(self):
		for layer in self.layers[::-1]:
			if layer.__class__.__name__ == "MatrixLayer":
				layer.update(self.lr)

	def __str__(self):
		res = "Model(\n"
		for layer in self.layers:
			res += "\t" + layer.__str__() + "\n"
		res += ")"
		return res

def imshow(img):
		img = img / 2 + 0.5     # unnormalize
		npimg = img.numpy()
		plt.imshow(np.transpose(npimg, (1, 2, 0)))
		plt.show()

def check_matrix(matrix_layer):
	torch.manual_seed(0)
	X = torch.randn(2, 3)
	layer = matrix_layer(3, 3)
	out = layer(X)
	if out == None:
		print("You didn't return anything. Make sure you are returning the right variable.")
		return
	error = torch.sum(torch.abs(out - torch.tensor([[ 2.7291231155, -1.4978368282,  1.5452182293],
										[ 1.9875525236, -0.9821926951,  1.2650012970]], requires_grad=False)))
	print("Error: ", error.item())
	if error < 1e-6:
		print("You got it! Please wait for further instructions.")
	else:
		print("Error was too high. Please try again")

def check_activation(activation):
	torch.manual_seed(0)
	X = torch.randn(2, 3, requires_grad=False)
	layer = activation()
	out = layer(X)
	if out == None:
		print("You didn't return anything. Make sure you are returning the right variable.")
		return
	if layer.__class__.__name__ == "ReLU":
		ref = X
		ref[X < 0] = 0
	elif layer.__class__.__name__ == "Sigmoid":
		ref = 1/(1 + torch.exp(-X))
	elif layer.__class__.__name__ == "Tanh":
		ref = torch.tanh(X)
	else:
		raise NotImplementedError(f"Activation {activation} not implemented")
	error = torch.sum(torch.abs(out - ref))
	print("Error: ", error.item())
	if error < 1e-6:
		print("You got it! Please wait for further instructions.")
	else:
		print("Error was too high. Please try again")

def check_loss(loss):
	torch.manual_seed(0)
	y_true, y_pred = torch.randn(10), torch.randn(10)
	layer = loss()
	out = layer(y_true, y_pred)
	if out == None:
		print("You didn't return anything. Make sure you are returning the right variable.")
		return
	if layer.__class__.__name__ == "MSE":
		ref = torch.mean((y_true  - y_pred) ** 2)
	else:
		raise NotImplementedError(f"Loss function {name} not implemented")
	error = torch.sum(torch.abs(out - ref))
	print("Error: ", error.item())
	if error < 1e-6:
		print("You got it! Please wait for further instructions.")
	else:
		print("Error was too high. Please try again")

def check_grad(matrix_layer):
	torch.manual_seed(0)
	layer = matrix_layer(3, 3)
	matrix_old = layer.matrix.detach().clone()
	bias_old = layer.bias.detach().clone()
	layer.matrix.grad = torch.randn(3, 3)
	layer.bias.grad = torch.randn(3)
	layer.update(1e-3)
	matrix_error = torch.sum(torch.abs(layer.matrix - torch.tensor([[[ 1.5418527126, -0.2945294976, -2.1777181625],
																																	 [ 0.5683085322, -1.0839560032, -1.3989685774],
																																	 [ 0.4042388201,  0.8395354748, -0.7196279764]]], requires_grad=False)))
	bias_error = torch.sum(torch.abs(layer.bias - torch.tensor([-0.4048000276, -0.5975751281,  0.1812616438], requires_grad=False)))
	if torch.sum(torch.abs(layer.matrix - matrix_old)) < 1e-6:
		print("You didn't update your matrix. Make sure you are updating the right variable.")
	elif torch.sum(torch.abs(layer.bias - bias_old)) < 1e-6:
		print("You didn't update your bias. Make sure you are updating the right variable.")
	elif matrix_error > 1e-6:
		print("Matrix error was too high. Please try again")
	elif bias_error > 1e-6:
		print("Bias error was too high. Please try again")
	else:
		print("You got it! Please wait for further instructions.")

def evaluate(model, testloader):
	correct = 0
	total = 0
	with torch.no_grad():
			for i, data in enumerate(testloader):
					images, labels = data
					images = images.reshape(-1, 784)
					outputs = model(images)
					_, predicted = torch.max(outputs.data, 1)
					total += labels.size(0)
					correct += (predicted == labels).sum().item()

	print('Accuracy of the network on the 10000 test images: %d %%' % (
			100 * correct / total))

	imshow(torchvision.utils.make_grid(images.reshape(-1, 1, 28, 28)))
	_, predicted = torch.max(outputs.data, 1)
	batch_size = images.shape[0]
	print(' '.join('%5s' % predicted[j].item() for j in range(batch_size)))