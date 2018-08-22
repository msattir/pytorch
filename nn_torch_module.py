import torch

dtype = torch.float
device = torch.device("cuda:0")

N, D_in, H, D_out = 64, 1000, 100, 10

#Create random input and ouput
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

#Random weights
w1 = torch.randn(D_in, H)
w2 = torch.randn(H, D_out)

lr = 1e-6
for t in range(500):
	#Compute predicted y
	h = x.mm(w1)
	h_relu = h.clamp(min=0)
	y_pred = h_relu.mm(w2)

	#Loss
	loss = (y - y_pred).pow(2).sum().item()
	print(t, loss)

	#Backprop
	grad_y_pred = 2.0*(y_pred - y)
	grad_w2 = h_relu.t().mm(grad_y_pred)
	grad_h_relu = grad_y_pred.mm(w2.t())
	grad_h = grad_h_relu.clone()
	grad_h[h < 0] = 0
	grad_w1 = x.t().mm(grad_h)

	#Update
	w1 -= lr*grad_w1
	w2 -= lr*grad_w2

