import groupy
import torch
import numpy as np

# ker_hw = 5

# Conv_type = (torch.nn.modules.conv.Conv2d,)
# Conv_type = (groupy.gconv.pytorch_gconv.splitgconv2d.P4MConvZ2, 
#                groupy.gconv.pytorch_gconv.splitgconv2d.P4ConvP4, 
#                groupy.gconv.pytorch_gconv.splitgconv2d.P4MConvZ2,
#                groupy.gconv.pytorch_gconv.splitgconv2d.P4MConvP4M)

def create_mask(h):
    mask = torch.zeros(size=(h,h), device='cuda')
    pos = int(mask.shape[0]/2)
    mask[pos] += 1.
    mask[:,pos] += 1.
    mask[pos,pos] = 1.
    return mask

# mask = create_mask(ker_hw)

def grad_mul_mask_grad(self,*para):
    if self.weight.grad != None:
        a = self.weight.grad * mask 
        self.weight.grad = a 
                
def grad_mul_mask_data(self,*para):
#     if self.weight.data != None:
    mask = create_mask(self.weight.data.shape[-1])
    a = self.weight.data * mask 
    self.weight.data = a 

def mask_hook(net, ltype, hook_type = 'forward_pre'):
    for mod in select_layers(net, ltype):
        if (mod.kernel_size == (3,3)) or (mod.kernel_size == (5,5)):
            if hook_type == 'forward_pre':
                print("register forward pre hook:", mod)
                mod.register_forward_pre_hook(grad_mul_mask_data) 
            elif hook_type == 'backward':
                print("register backward hook:", mod)
                mod.register_backward_hook(grad_mul_mask_grad)         
                
def init_skeleton_weight(net, ltype):
    for mod in select_layers(net, ltype):
        if (mod.kernel_size == (3,3)) or (mod.kernel_size == (5,5)):
            mask = create_mask(mod.kernel_size[-1])
            mod.weight.data *= mask
    
def select_layers(model, ltype):
    """
        Filter the submodules of model according to ltype.
    """
    check_ltype = lambda x: type(x) in ltype 
    return list(filter(check_ltype, model.modules()))    


def check_para(tensor):
    mask = create_mask(tensor.shape[-1])
    h = tensor.shape[-1]
    mask_check = torch.zeros(h,h).cuda()
    pos = int(mask_check.shape[0]/2)
    mask_check[pos] += 999.
    mask_check[:,pos] += 999.
    mask_check[pos,pos] = 999.
    temp = (tensor != mask_check)
    mask_expand = mask.expand_as(tensor)
    np.testing.assert_allclose(temp.cpu().numpy(), mask_expand.cpu().numpy())
    print("Pass para checking")     

def check_net_paras(net, ltype):
    for mod in select_layers(net, ltype):
        if (mod.kernel_size == (3,3)) or (mod.kernel_size == (5,5)):
            print("layer name:",mod," layer weight shape:",mod.weight.data.shape)
            check_para(mod.weight.data)

# mask = torch.tensor([[0,1,0],
#                      [1,1,1],
#                      [0,1,0]]).cuda()

# P4Conv_type = (groupy.gconv.pytorch_gconv.splitgconv2d.P4MConvZ2, 
#                groupy.gconv.pytorch_gconv.splitgconv2d.P4ConvP4, 
#                groupy.gconv.pytorch_gconv.splitgconv2d.P4MConvZ2,
#                groupy.gconv.pytorch_gconv.splitgconv2d.P4MConvP4M)

# def grad_mul_mask(self, grad_input, grad_output):
#     """
#     """
#     if self.weight.grad != None:
#         self.weight.grad *= mask                     
               
# def mask_bwhook(net):
#     print("register backward hook")
#     for mod in select_layers(net):
#         if mod.kernel_size == (3,3):
#             mod.register_backward_hook(grad_mul_mask)         
               
# def select_layers(model, ltype = P4Conv_type):
#     """
#         Filter the submodules of model according to ltype.
#     """
#     check_ltype = lambda x: type(x) in ltype 
#     return list(filter(check_ltype, model.modules()))    


# def check_para(tensor):
#     h = tensor.shape[-1]
#     mask_check = torch.zeros(h,h).cuda()
#     pos = int(mask_check.shape[0]/2)
#     mask_check[pos] += 999.
#     mask_check[:,pos] += 999.
#     mask_check[pos,pos] = 999.
#     temp = (tensor != mask_check)
#     mask_expand = mask.expand_as(tensor)
#     np.testing.assert_allclose(temp.cpu().numpy(), mask_expand.cpu().numpy())
#     print("pass para checking")     
               
               
               
               