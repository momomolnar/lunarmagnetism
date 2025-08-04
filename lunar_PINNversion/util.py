import torch
import torch.nn as nn


# Laplacian computation
def laplacian(phi, coords):

    grads = torch.autograd.grad(phi, coords, grad_outputs=torch.ones_like(phi),
                                create_graph=True)[0]
    d2 = []
    for i in range(coords.shape[1]):
        grad2 = torch.autograd.grad(grads[:, i], coords,
                                    grad_outputs=torch.ones_like(grads[:, i]),
                                    create_graph=True)[0][:, i]
        d2.append(grad2)
    return sum(d2)
