import torch, torchvision


model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(1, 3, 64, 64) # 1 image; 3 channels; height 64 and weight 64
labels = torch.rand(1, 1000) #labels are torch type

# ## Vanilla NN training--------------------------------
# ## ---------------------------------------------------

# # forward pass
prediction = model(data) # forward pass
# # calculate loss 
loss = (prediction-labels).sum()
print(loss)
# # back propagation
loss.backward()
# # load optimizer
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
# # train neural network one step
optim.step()



#-------------------------------------------
# x = torch.tensor([2.,3.,4.], requires_grad=True)
# y = x**2

# #y.backward() # error

# y.backward(torch.tensor([1.,1.,1.])) # work
# x.grad # tensor([4.,6.,8.])