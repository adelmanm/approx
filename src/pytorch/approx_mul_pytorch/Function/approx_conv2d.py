import torch
from ..functional.approx_conv2d import approx_conv2d_func_bwd, approx_conv2d_func_wu, approx_conv2d_func_forward
#import approx_conv2d

class approx_conv2d_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias = None, stride = 1, padding = 0, dilation = 1, groups = 1,
                sample_ratio=1.0, minimal_k=1, sample_ratio_bwd=None, minimal_k_bwd=None, sample_ratio_wu=None, minimal_k_wu=None):
        ctx.save_for_backward(input,weight, bias)
        #store non-tensor objects in ctx
        ctx.stride=stride
        ctx.padding=padding
        ctx.dilation=dilation
        ctx.groups = groups
        ctx.sample_ratio_bwd = sample_ratio_bwd
        ctx.minimal_k_bwd = minimal_k_bwd
        ctx.sample_ratio_wu = sample_ratio_wu
        ctx.minimal_k_wu = minimal_k_wu
        
        return torch.nn.functional.conv2d(input, weight, bias, stride, padding, dilation, groups)
        #return approx_conv2d_func_forward(input, weight, bias, stride, padding, dilation, groups, sample_ratio, minimal_k)
        #return approx_conv2d.forward(input, weight, bias, stride, padding, dilation, groups, sample_ratio, minimal_k, False, False)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_variables
        #grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output, ctx.stride, ctx.padding, ctx.dilation, ctx.groups)
        #grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, ctx.stride, ctx.padding, ctx.dilation, ctx.groups)
        
        if ctx.minimal_k_bwd is None:
            grad_input = approx_conv2d.backward_input(input.shape, weight, grad_output, ctx.stride, ctx.padding, ctx.dilation, ctx.groups, False, False)
        else:
            grad_input = approx_conv2d_func_bwd(input.shape, weight, grad_output, ctx.stride, ctx.padding, ctx.dilation, ctx.groups, ctx.sample_ratio_bwd, ctx.minimal_k_wu)
        if ctx.minimal_k_wu is None:
            grad_weight = approx_conv2d.backward_weight(input, weight.shape, grad_output, ctx.stride, ctx.padding, ctx.dilation, ctx.groups, False, False)
        else:
            grad_weight = approx_conv2d_func_wu(input, weight.shape, grad_output, ctx.stride, ctx.padding, ctx.dilation, ctx.groups, ctx.sample_ratio_wu, ctx.minimal_k_wu)
        grad_bias = None
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0,2,3))       
 
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None, None, None
