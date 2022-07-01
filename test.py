print('in python')
import os
# print(os.environ['CONDA_DEFAULT_ENV'])
import torch
print(torch.cuda.is_available())
print(torch.cuda.mem_get_info())
