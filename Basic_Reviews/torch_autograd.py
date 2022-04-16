import torch, torchvision


model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(1, 3, 64, 64) # 1 image; 3 channels; height 64 and weight 64
labels = torch.rand(1, 1000) #labels are torch type

## Vanilla NN training--------------------------------
## ---------------------------------------------------

# forward pass
prediction = model(data) # forward pass
# calculate loss 
loss = (prediction-labels).sum()
# back propagation
loss.backward()
# load optimizer
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
# train neural network one step
optim.step()