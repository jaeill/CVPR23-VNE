import torch
import numpy as np

# N   : batch size
# d   : embedding dimension
# H   : embeddings, Tensor, shape=[N, d]

# def get_vne(H):
#     Z = torch.nn.functional.normalize(H, dim=1)
#     rho = torch.matmul(Z.T, Z) / Z.shape[0]
#     eig_val = torch.linalg.eigh(rho)[0][-Z.shape[0]:]
#     return - (eig_val * torch.log(eig_val)).nansum()

# the following is equivalent and faster when N < d (for the most cases)
def get_vne(H):
    Z = torch.nn.functional.normalize(H, dim=1)
    sing_val = torch.svd(Z / np.sqrt(Z.shape[0]))[1]
    eig_val = sing_val ** 2
    return - (eig_val * torch.log(eig_val)).nansum()
