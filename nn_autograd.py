import torch

class myRelu(torch.autograd.Function):
	@staticmethod
	def forward(ctx, input):
		#ctx statsh input for backprop
		ctx.save_for_backward(input)
		return input.clamp(min=0)

	@staticmethod
	def backward(ctx, grad_output):
		"""
        	In the backward pass we receive a Tensor containing the gradient of the loss
        	with respect to the output, and we need to compute the gradient of the loss
        	with respect to the input.
		"""
		
		input, = ctx.saved_tensors
		grad_input = grad_output.clone()
		grad_input[input < 0] = 0
		return grad_input

dtype = torch.float
device = torch.device("cuda:0")

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

lr = 1e-6
relu = myRelu.apply

for t in range(500):
	y_pred = relu(x.mm(w1)).mm(w2)
	loss = (y - y_pred).pow(2).sum()
	print (t, loss.item())

	loss.backward()

#	torch.optim.SGD([w1, w2], lr)	

	with torch.no_grad():
		w1 -= lr*w1.grad
		w2 -= lr*w2.grad

		w1.grad.zero_()
		w2.grad.zero_()
