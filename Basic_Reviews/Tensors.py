import  torch
import numpy as np 

# Tensor Initialization

# method-1: from data 
data=[[1,2],[3,4]]
x_data = torch.tensor(data)
print(f"x_data: {x_data}")
print(f"x_data dimensions: {x_data.dim()}\n")

# method-2: from array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
np_array = x_np.numpy() # from tensor to numpy

# method-3: from another tensor
x_ones = torch.ones_like(x_data)  # ones_like: fill the 1s in the element in x_data, retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")
x_random =  torch.rand_like(x_data,dtype=torch.float64) 
print(f"random like tensor: \n{x_random}\n")

# method-4: with random or onstant values
shape = (2,3,1) # shape is a tuple that determines the tensor dimensions   (row num, col num, page num)
rand_tensor = torch.rand(shape)
print(f"rand_tensor: \n {rand_tensor} \n")
print(f"rand_tensor: \n {rand_tensor[1][1][0]} \n")
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

#-------------------------------------------------------------------------
#-------------------------------------------------------------------------

# Tensor Arritibutes
tensor = torch.rand((2,3,1),dtype=torch.float64)
print(f"shape of tensor: {tensor.shape}")
print(f"data type of tensor: {tensor.dtype}")
print(f"device on which tensor is stored: {tensor.device}" )
print(f"tensor data: {tensor.data}")


#-------------------------------------------------------------------------
#-------------------------------------------------------------------------


# Tensor Operations
if torch.cuda.is_available():
    tensor = tensor.to('cuda')
print(f"Device tensor is stored on: {tensor.device}")

## try element-wise multiplication -- "*"
ten_1 = torch.rand((1,2,))
ten_2 = torch.rand((2,2,))
print(f"Tensor multiplication 1: {ten_1.mul(ten_2)}")
print(f"Tensor multiplication 1: {ten_1*ten_2}")

## try matrix multiplication -- "@"
print(f"Tensor multiplication 1: {ten_1.matmul(ten_1.T)}")
print(f"Tensor multiplication 1: {ten_1@ten_1.T}")