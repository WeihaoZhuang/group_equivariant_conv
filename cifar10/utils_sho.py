import groupy
import torch
import numpy as np
mask = torch.tensor([[0,1,0],
                     [1,1,1],
                     [0,1,0]]).cuda()

P4Conv_type = (groupy.gconv.pytorch_gconv.splitgconv2d.P4MConvZ2, 
               groupy.gconv.pytorch_gconv.splitgconv2d.P4ConvP4, 
               groupy.gconv.pytorch_gconv.splitgconv2d.P4MConvZ2,
               groupy.gconv.pytorch_gconv.splitgconv2d.P4MConvP4M)

def grad_mul_mask(self, grad_input, grad_output):
    """
    """
    if self.weight.grad != None:
        self.weight.grad *= mask                     
               
def mask_bwhook(net):
    print("register backward hook")
    for mod in select_layers(net):
        if mod.kernel_size == (3,3):
            mod.register_backward_hook(grad_mul_mask)         
               
def select_layers(model, ltype = P4Conv_type):
    """
        Filter the submodules of model according to ltype.
    """
    check_ltype = lambda x: type(x) in ltype 
    return list(filter(check_ltype, model.modules()))    


def check_para(tensor):
    h = tensor.shape[-1]
    mask_check = torch.zeros(h,h).cuda()
    pos = int(mask_check.shape[0]/2)
    mask_check[pos] += 999.
    mask_check[:,pos] += 999.
    mask_check[pos,pos] = 999.
    temp = (tensor != mask_check)
    mask_expand = mask.expand_as(tensor)
    np.testing.assert_allclose(temp.cpu().numpy(), mask_expand.cpu().numpy())
    print("pass para checking")     
               
               
               
               