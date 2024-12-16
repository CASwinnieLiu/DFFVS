import math
import numpy as np
from torch import torch
from manifolds.manifold import Manifold 
from manifolds.complex_product import ComplexProduct
from torch.optim.optimizer import Optimizer, required


class DFFVS_Cp(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, eight_decay = 0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=0.0001)
        super(DFFVS_Cp, self).__init__(params, defaults)
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()     
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                        continue
                grads = p.grad.data
                state = self.state[p] 
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                #  gradient values
                    state['g_0'] = grads.new().resize_as_(grads).zero_()
                state['step'] += 1 
                
                # Projector  #0728
                #  def proj(self, X, U):
                #        
                gradst = (grads*grads).grad
                #
                gradst = p.grad.data
                gradst_1 = gradst.grad
                gradst_1 = p.grad.data
                # Dual #0708
                df = p * gradst_1 - 0.1
                df = p.grad.data
                gradst_1 = df.grad
                gradst_1 = p.grad.data
                gradst_2 = gradst_1.grad
                gradst_2 = p.grad.data
                # Accelerated Gradient Descent 
                gradsa = gradst_1 + gradst_1 - gradst
                gradsa_d = gradst_2 + gradst_2 - gradst
                gradsa = p.grad.data
                gradsa_d = p.grad.data
                # Retr
                gradsr = gradsa
                gradsr = p.grad.data
                # Dual_Retr 0708
                gradsr = gradsa_d
                gradsr = p.grad.data
                grads = gradsr
                
        return loss
 
