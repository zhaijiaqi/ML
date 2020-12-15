import numpy as np
import torch

np_data = np.arange(6).reshape((2,3))
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()
print(
    '\nnumpy',np_data,
    '\ntorch',torch_data,
    '\ntensor2array',tensor2array
)


data = [-1,-2,1,2]
tensor = torch.FloatTensor(data)
print(
    '\nabs',
    '\nnumpy',np.mean(data),
    '\ntorch',torch.mean(tensor)
)


data = [[1,2],[3,4]]
tensor = torch.FloatTensor(data)
data = np.array(data)
print(
    '\nnumpy:',np.matmul(data,data),
    '\ntorch',torch.mm(tensor,tensor)
)
print(
    '\nnumpy:',data.dot(data),
    '\ntorch',torch.dot(tensor)
)