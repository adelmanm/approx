import torch
#import approx_conv2d
from ..modules.utils import topk_indices

''' shape - shape of grad_input. expected: (batch, in_channels,h,w)
    weight - weight tensor, shape (out_channels, in_channels, h,w)
    grad_output - grad output tensor, shape (batch, out_channels, h,w)
    stride, padding, dilation, groups as in regular conv2d
    sample_ratio - Ratio of out_channels to sample
    minimal_k - Minimal number of out_channels to keep in the sampling
'''
def conv2d_bwd_topk(shape, weight, grad_output, stride, padding, dilation, groups, sample_ratio,minimal_k):
    #print("Sanity check - conv2d_bwd_topk is used with sample_ratio = " + str(sample_ratio) + " and minimal_k = " + str(minimal_k))
    #print("shape: {}".format(shape))
    #print("weight size: {}".format(weight.size()))
    #print("grad_output size: {}".format(grad_output.size()))
    #print("sample_ratio: {}".format(sample_ratio))
    #print("minimal_k: {}".format(minimal_k))
    
    out_channels = weight.size()[0]

    # calculate the number of input channels to sample
    k_candidate = int(float(out_channels)*sample_ratio)

    # make k at least minimal_k
    k = min(max(k_candidate,minimal_k),out_channels)
    
    # if because of minimal_k or sample_ratio k equals the number of features, perform full conv2d_bwd instead of approximating
    if k == out_channels:
        return approx_conv2d.backward_input(shape, weight, grad_output, stride, padding, dilation, groups, False, False, True)

    # calculate norms of output channels
    weight_out_channels_norms = torch.norm(weight.view(out_channels,-1),dim=1, p=2)
    grad_output_out_channels_norms = torch.norm(grad_output.view(grad_output.size()[0],out_channels,-1), dim=2, p=2)
    grad_output_out_channels_norms = torch.norm(grad_output_out_channels_norms, dim=0, p=2)
    grad_output_out_channels_norms = torch.squeeze(grad_output_out_channels_norms)            

    # multiply both norms element-wise to and pick the indices of the top K channels
    norm_mult = torch.mul(weight_out_channels_norms, grad_output_out_channels_norms)

    # top_k_indices = torch.topk(norm_mult,k)[1]
    top_k_indices = topk_indices(norm_mult,k)

    # pick top-k channels to form new smaller tensors
    weight_top_k_channels = torch.index_select(weight,dim = 0, index = top_k_indices)
    grad_output_top_k_channels = torch.index_select(grad_output,dim = 1, index = top_k_indices)

    # compute sampled tensors
    grad_input_approx = approx_conv2d.backward_input(shape, weight_top_k_channels, grad_output_top_k_channels, stride, padding, dilation, groups, False, False, True)
    return grad_input_approx

''' shape - shape of grad_input. expected: (batch, in_channels,h,w)
    weight - weight tensor, shape (out_channels, in_channels, h,w)
    grad_output - grad output tensor, shape (batch, out_channels, h,w)
    stride, padding, dilation, groups as in regular conv2d
    sample_ratio - Ratio of out_channels to sample
    minimal_k - Minimal number of out_channels to keep in the sampling
'''
def conv2d_bwd_random_sampling(shape, weight, grad_output, stride, padding, dilation, groups, sample_ratio,minimal_k, with_replacement, optimal_prob, scale):
    #print("Sanity check - conv2d_bwd_random_sampling is used with sample_ratio = " + str(sample_ratio) + "and minimal_k = " + str(minimal_k))
    #print("shape: {}".format(shape))
    #print("weight size: {}".format(weight.size()))
    #print("grad_output size: {}".format(grad_output.size()))
    #print("sample_ratio: {}".format(sample_ratio))
    #print("minimal_k: {}".format(minimal_k))
    #print("with_replacement: {}".format(with_replacement))
    #print("optimal_prob: {}".format(optimal_prob))
    #print("scale: {}".format(scale))
    
    out_channels = weight.size()[0]
    device = weight.device    

    # calculate the number of input channels to sample
    k_candidate = int(float(out_channels)*sample_ratio)

    # make k at least minimal_k
    k = min(max(k_candidate,minimal_k),out_channels)
    
    # if because of minimal_k or sample_ratio k equals the number of features, perform full conv2d_bwd instead of approximating
    if k == out_channels:
        return approx_conv2d.backward_input(shape, weight, grad_output, stride, padding, dilation, groups, False, False, True)
    
    if optimal_prob == True:
        
        # calculate norms of output channels
        weight_out_channels_norms = torch.norm(weight.view(out_channels,-1),dim=1, p=2)
        grad_output_out_channels_norms = torch.norm(grad_output.view(grad_output.size()[0],out_channels,-1),p=2, dim=2)
        grad_output_out_channels_norms = torch.norm(grad_output_out_channels_norms, dim=0, p=2)
        grad_output_out_channels_norms = torch.squeeze(grad_output_out_channels_norms)            
        
        # multiply both norms element-wise
        norm_mult = torch.mul(weight_out_channels_norms, grad_output_out_channels_norms)
            
        # use epsilon-optimal sampling to allow learning random weights and to bound the scaling factor
        epsilon = 0.1
        if epsilon > 0:
            sum_norm_mult = torch.sum(norm_mult)
            norm_mult = torch.div(norm_mult, sum_norm_mult)
            uniform = torch.ones_like(norm_mult)/out_channels 
            norm_mult = (1-epsilon)*norm_mult + epsilon*uniform
        
        # no need to normalize, it is already done by torch.multinomial

        # calculate number of nonzero elements in norm_mult. this serves 
        # two purposes:
        # 1. Possibly reduce number of sampled pairs, as zero elements in norm_mult will not contribute to the result
        # 2. Prevents scaling of zero values
        nnz = (norm_mult!=0).sum()
        if nnz == 0:
            #print("zero multiply detected! scenario not optimzied (todo)")
            return approx_conv2d.backward_input(shape, weight, grad_output, stride, padding, dilation, groups, False, False, True)
            
        k = min(k,nnz)
        indices = torch.multinomial(norm_mult,k,replacement=with_replacement)
    
        # pick top-k channels to form new smaller tensors
        weight_top_k_channels = torch.index_select(weight,dim = 0, index = indices)
        grad_output_top_k_channels = torch.index_select(grad_output,dim = 1, index = indices)
        
        if scale == True:
            # when sampling without replacement a more complicated scaling factor is required (see Horvitz and Thompson, 1952)
            assert(with_replacement == True)
            # scale out_channels by 1/(k*p_i) to get unbiased estimation
            sum_norm_mult = torch.sum(norm_mult)
            scale_factors = torch.div(sum_norm_mult,torch.mul(norm_mult,k))
            weight_top_k_channels = torch.mul(weight_top_k_channels, scale_factors[indices].view(-1,1,1,1))            
 
    else:
        # uniform sampling    
        if with_replacement == True:
            indices = torch.randint(low=0,high=out_channels,size=(k,),device=device)
        else:
            uniform_dist = torch.ones(out_channels,device=device)
            indices = torch.multinomial(uniform_dist,k,replacement=False)
        
        # pick k channels to form new smaller tensors
        weight_top_k_channels = torch.index_select(weight,dim = 0, index = indices)
        grad_output_top_k_channels = torch.index_select(grad_output,dim = 1, index = indices)

        if scale == True:
            # scale column-row pairs by 1/(k*p_i) to get unbiased estimation
            # in case of uniform distribution, p_i = 1/in_features when sampling with replacement 
            # when sampling without replacement a different scaling factor is required (see Horvitz and Thompson, 1952), but
            # for uniform sampling it turns to be in_features/k as well
            scale_factor = out_channels/k
            weight_top_k_channels = torch.mul(weight_top_k_channels, scale_factor) 
        
    # compute sampled tensors
    grad_input_approx = approx_conv2d.backward_input(shape, weight_top_k_channels, grad_output_top_k_channels, stride, padding, dilation, groups, False, False, True)
    return grad_input_approx

''' shape - shape of grad_input. expected: (batch, in_channels,h,w)
    weight - weight tensor, shape (out_channels, in_channels, h,w)
    grad_output - grad output tensor, shape (batch, out_channels, h,w)
    stride, padding, dilation, groups as in regular conv2d
    sample_ratio - Ratio of out_channels to sample
    minimal_k - Minimal number of out_channels to keep in the sampling
'''
def conv2d_bwd_bernoulli_sampling(shape, weight, grad_output, stride, padding, dilation, groups, sample_ratio,minimal_k, scale):
    #print("Sanity check - conv2d_bwd_bernoulli_sampling is used with sample_ratio = " + str(sample_ratio) + "and minimal_k = " + str(minimal_k))
    #print("shape: {}".format(shape))
    #print("weight size: {}".format(weight.size()))
    #print("grad_output size: {}".format(grad_output.size()))
    #print("sample_ratio: {}".format(sample_ratio))
    #print("minimal_k: {}".format(minimal_k))
    #print("scale: {}".format(scale))
    
    out_channels = weight.size()[0]
    device = weight.device    

    # calculate the number of input channels to sample
    k_candidate = int(float(out_channels)*sample_ratio)

    # make k at least minimal_k
    k = min(max(k_candidate,minimal_k),out_channels)
    
    # if because of minimal_k or sample_ratio k equals the number of features, perform full conv2d_bwd instead of approximating
    if k == out_channels:
        return approx_conv2d.backward_input(shape, weight, grad_output, stride, padding, dilation, groups, False, False, True)
    
    # calculate norms of output channels
    weight_out_channels_norms = torch.norm(weight.view(out_channels,-1),dim=1, p=2)
    grad_output_out_channels_norms = torch.norm(grad_output.view(grad_output.size()[0],out_channels,-1),p=2, dim=2)
    grad_output_out_channels_norms = torch.norm(grad_output_out_channels_norms, dim=0, p=2)
    grad_output_out_channels_norms = torch.squeeze(grad_output_out_channels_norms)            
    
    # multiply both norms element-wise
    norm_mult = torch.mul(weight_out_channels_norms, grad_output_out_channels_norms)
    sum_norm_mult = norm_mult.sum()       
    
    # calculate number of nonzero elements in norm_mult. this serves 
    # two purposes:
    # 1. Possibly reduce number of sampled pairs, as zero elements in norm_mult will not contribute to the result
    # 2. Prevents scaling of zero values
    nnz = (norm_mult!=0).sum()
    if nnz == 0:
        #print("zero multiply detected! scenario not optimzied (todo)")
        return approx_conv2d.backward_input(shape, weight, grad_output, stride, padding, dilation, groups, False, False, True)
        
    k = min(k,nnz)
    prob_dist = k * torch.div(norm_mult,sum_norm_mult)
    prob_dist = prob_dist.clamp(min=0, max=1)
    
    # use epsilon-optimal sampling to allow learning random weights and to bound the scaling factor
    epsilon = 0.1
    if epsilon > 0:
        uniform = torch.ones_like(prob_dist)/out_channels 
        prob_dist = (1-epsilon)*prob_dist + epsilon*uniform
    
    indices = torch.bernoulli(prob_dist).nonzero(as_tuple=True)[0]
    if len(indices) == 0:
        print("no elements selected - hmm")
        indices = torch.arange(k, device=device)

    # pick top-k channels to form new smaller tensors
    weight_top_k_channels = torch.index_select(weight,dim = 0, index = indices)
    grad_output_top_k_channels = torch.index_select(grad_output,dim = 1, index = indices)
    
    if scale == True:
        # scale out_channels by 1/(p_i) to get unbiased estimation
        scale_factors = torch.div(1,prob_dist)
        weight_top_k_channels = torch.mul(weight_top_k_channels, scale_factors[indices].view(-1,1,1,1))            
 
        
    # compute sampled tensors
    grad_input_approx = approx_conv2d.backward_input(shape, weight_top_k_channels, grad_output_top_k_channels, stride, padding, dilation, groups, False, False,True)
    return grad_input_approx



''' shape - shape of grad_input. expected: (batch, in_channels,h,w)
    weight - weight tensor, shape (out_channels, in_channels, h,w)
    grad_output - grad output tensor, shape (batch, out_channels, h,w)
    stride, padding, dilation, groups as in regular conv2d
    sample_ratio - Ratio of out_channels to sample
    minimal_k - Minimal number of out_channels to keep in the sampling
'''
def approx_conv2d_func_bwd(shape, weight, grad_output, stride, padding, dilation, groups, sample_ratio,minimal_k):
    #return approx_conv2d.backward_input(shape, weight, grad_output, stride, padding, dilation, groups, False, False, True)
    #return conv2d_bwd_topk(shape, weight, grad_output, stride, padding, dilation, groups, sample_ratio,minimal_k)
    #return conv2d_bwd_random_sampling(shape, weight, grad_output, stride, padding, dilation, groups, sample_ratio,minimal_k, with_replacement=True, optimal_prob=True, scale=True)
    return conv2d_bwd_bernoulli_sampling(shape, weight, grad_output, stride, padding, dilation, groups, sample_ratio,minimal_k, scale=True)

''' input - input tensor, shape (batch, in_channels, h, w)
    weight_shape - shape of grad_weight. expected: (out_channels, in_channels,h,w)
    grad_output - grad output tensor, shape (batch, out_channels, h,w)
    stride, padding, dilation, groups as in regular conv2d
    sample_ratio - Ratio of out_channels to sample
    minimal_k - Minimal number of out_channels to keep in the sampling
'''
def conv2d_wu_topk(input, weight_shape, grad_output, stride, padding, dilation, groups, sample_ratio,minimal_k):
    #print("Sanity check - conv2d_bwd_wu is used with sample_ratio = " + str(sample_ratio) + "and minimal_k = " + str(minimal_k))
    #print("input: {}".format(input))
    #print("weight_shape: {}".format(weight_shape))
    #print("input size: {}".format(input.size()))
    #print("grad_output size: {}".format(grad_output.size()))
    #print("sample_ratio: {}".format(sample_ratio))
    #print("minimal_k: {}".format(minimal_k))
    
    batch = input.size()[0]

    # calculate the number of minibatch examples to sample
    k_candidate = int(float(batch)*sample_ratio)

    # make k at least minimal_k
    k = min(max(k_candidate,minimal_k),batch)
    
    # if because of minimal_k or sample_ratio k equals the minibatch size, perform full conv2d_wu instead of approximating
    if k == batch:
        return approx_conv2d.backward_weight(input, weight_shape, grad_output, stride, padding, dilation, groups, False, False, True)

    # calculate norms of minibatch examples
    input_batch_norms = torch.norm(input.view(batch,-1),dim=1, p=2)
    grad_output_batch_norms = torch.norm(grad_output.view(batch,-1),dim=1, p=2)

    # multiply both norms element-wise to and pick the indices of the top K minibatch examples
    norm_mult = torch.mul(input_batch_norms, grad_output_batch_norms)

    # top_k_indices = torch.topk(norm_mult,k)[1]
    top_k_indices = topk_indices(norm_mult,k)

    # pick top-k batch examples to form new smaller tensors
    input_top_k_batch = torch.index_select(input,dim = 0, index = top_k_indices)
    grad_output_top_k_batch = torch.index_select(grad_output,dim = 0, index = top_k_indices)

    # compute sampled tensors
    grad_weight_approx = approx_conv2d.backward_weight(input_top_k_batch, weight_shape, grad_output_top_k_batch, stride, padding, dilation, groups, False, False, True)
    return grad_weight_approx

''' input - input tensor, shape (batch, in_channels, h, w)
    weight_shape - shape of grad_weight. expected: (out_channels, in_channels,h,w)
    grad_output - grad output tensor, shape (batch, out_channels, h,w)
    stride, padding, dilation, groups as in regular conv2d
    sample_ratio - Ratio of out_channels to sample
    minimal_k - Minimal number of out_channels to keep in the sampling
'''
def conv2d_wu_random_sampling(input, weight_shape, grad_output, stride, padding, dilation, groups, sample_ratio,minimal_k, with_replacement, optimal_prob, scale):
    #print("Sanity check - conv2d_bwd_wu is used with sample_ratio = " + str(sample_ratio) + "and minimal_k = " + str(minimal_k))
    #print("input: {}".format(input))
    #print("weight_shape: {}".format(weight_shape))
    #print("input size: {}".format(input.size()))
    #print("grad_output size: {}".format(grad_output.size()))
    #print("sample_ratio: {}".format(sample_ratio))
    #print("minimal_k: {}".format(minimal_k))
    #print("with_replacement: {}".format(with_replacement))
    #print("optimal_prob: {}".format(optimal_prob))
    #print("scale: {}".format(scale))
    
    batch = input.size()[0]
    device = input.device    

    # calculate the number of minibatch examples to sample
    k_candidate = int(float(batch)*sample_ratio)

    # make k at least minimal_k
    k = min(max(k_candidate,minimal_k),batch)
    
    # if because of minimal_k or sample_ratio k equals the minibatch size, perform full conv2d_wu instead of approximating
    if k == batch:
        return approx_conv2d.backward_weight(input, weight_shape, grad_output, stride, padding, dilation, groups, False, False, True)

    if optimal_prob == True:
        
        # calculate norms of output channels
        input_batch_norms = torch.norm(input.view(batch, -1),dim=1, p=2)
        grad_output_batch_norms = torch.norm(grad_output.view(batch, -1) ,dim=1, p=2)
        
        # multiply both norms element-wise
        norm_mult = torch.mul(input_batch_norms, grad_output_batch_norms)

        # use epsilon-optimal sampling to allow learning random weights and to bound the scaling factor
        epsilon = 0.1
        if epsilon > 0:
            sum_norm_mult = torch.sum(norm_mult)
            norm_mult = torch.div(norm_mult, sum_norm_mult)
            uniform = torch.ones_like(norm_mult)/batch 
            norm_mult = (1-epsilon)*norm_mult + epsilon*uniform

        # no need to normalize, it is already done by torch.multinomial

        # calculate number of nonzero elements in norm_mult. this serves 
        # two purposes:
        # 1. Possibly reduce number of sampled pairs, as zero elements in norm_mult will not contribute to the result
        # 2. Prevents scaling of zero values
        nnz = (norm_mult!=0).sum()
        if nnz == 0:
            #print("zero multiply detected! scenario not optimzied (todo)")
            return approx_conv2d.backward_weight(input, weight_shape, grad_output, stride, padding, dilation, groups, False, False, True)
            
        k = min(k,nnz)
        indices = torch.multinomial(norm_mult,k,replacement=with_replacement)
    
        # pick top-k minibatch examples to form new smaller tensors
        input_top_k_batch = torch.index_select(input,dim = 0, index = indices)
        grad_output_top_k_batch = torch.index_select(grad_output,dim = 0, index = indices)
        
        if scale == True:
            # when sampling without replacement a more complicated scaling factor is required (see Horvitz and Thompson, 1952)
            assert(with_replacement == True)
            # scale out_channels by 1/(k*p_i) to get unbiased estimation
            sum_norm_mult = torch.sum(norm_mult)
            scale_factors = torch.div(sum_norm_mult,torch.mul(norm_mult,k))
            input_top_k_batch = torch.mul(input_top_k_batch, scale_factors[indices].view(-1,1,1,1))            
 
    else:
        # uniform sampling    
        if with_replacement == True:
            indices = torch.randint(low=0,high=batch,size=(k,),device=device)
        else:
            uniform_dist = torch.ones(batch,device=device)
            indices = torch.multinomial(uniform_dist,k,replacement=False)
        
        # pick top-k minibatch examples to form new smaller tensors
        input_top_k_batch = torch.index_select(input,dim = 0, index = indices)
        grad_output_top_k_batch = torch.index_select(grad_output,dim = 0, index = indices)

        if scale == True:
            # scale sampled batch examples by 1/(k*p_i) to get unbiased estimation
            # in case of uniform distribution, p_i = 1/in_features when sampling with replacement 
            # when sampling without replacement a different scaling factor is required (see Horvitz and Thompson, 1952), but
            # for uniform sampling it turns to be in_features/k as well
            scale_factor = batch/k
            input_top_k_batch = torch.mul(input_top_k_batch, scale_factor) 
        
    # compute sampled tensors
    grad_weight_approx = approx_conv2d.backward_weight(input_top_k_batch, weight_shape, grad_output_top_k_batch, stride, padding, dilation, groups, False, False, True)
    return grad_weight_approx


''' input - input tensor, shape (batch, in_channels, h, w)
    weight_shape - shape of grad_weight. expected: (out_channels, in_channels,h,w)
    grad_output - grad output tensor, shape (batch, out_channels, h,w)
    stride, padding, dilation, groups as in regular conv2d
    sample_ratio - Ratio of out_channels to sample
    minimal_k - Minimal number of out_channels to keep in the sampling
'''
def conv2d_wu_bernoulli_sampling(input, weight_shape, grad_output, stride, padding, dilation, groups, sample_ratio,minimal_k, scale):
    #print("Sanity check - conv2d_wu_bernoulli is used with sample_ratio = " + str(sample_ratio) + "and minimal_k = " + str(minimal_k))
    #print("input: {}".format(input))
    #print("weight_shape: {}".format(weight_shape))
    #print("input size: {}".format(input.size()))
    #print("grad_output size: {}".format(grad_output.size()))
    #print("sample_ratio: {}".format(sample_ratio))
    #print("minimal_k: {}".format(minimal_k))
    #print("scale: {}".format(scale))
    
    batch = input.size()[0]
    device = input.device    

    # calculate the number of minibatch examples to sample
    k_candidate = int(float(batch)*sample_ratio)

    # make k at least minimal_k
    k = min(max(k_candidate,minimal_k),batch)
    
    # if because of minimal_k or sample_ratio k equals the minibatch size, perform full conv2d_wu instead of approximating
    if k == batch:
        return approx_conv2d.backward_weight(input, weight_shape, grad_output, stride, padding, dilation, groups, False, False, True)

    # calculate norms of output channels
    input_batch_norms = torch.norm(input.view(batch, -1),dim=1, p=2)
    grad_output_batch_norms = torch.norm(grad_output.view(batch, -1) ,dim=1, p=2)
    
    # multiply both norms element-wise
    norm_mult = torch.mul(input_batch_norms, grad_output_batch_norms)
    sum_norm_mult = norm_mult.sum()       
       
    # calculate number of nonzero elements in norm_mult. this serves 
    # two purposes:
    # 1. Possibly reduce number of sampled pairs, as zero elements in norm_mult will not contribute to the result
    # 2. Prevents scaling of zero values
    nnz = (norm_mult!=0).sum()
    if nnz == 0:
        #print("zero multiply detected! scenario not optimzied (todo)")
        return approx_conv2d.backward_weight(input, weight_shape, grad_output, stride, padding, dilation, groups, False, False, True)
        
    k = min(k,nnz)
    prob_dist = k * torch.div(norm_mult,sum_norm_mult)
    prob_dist = prob_dist.clamp(min=0, max=1)
        
    # use epsilon-optimal sampling to allow learning random weights and to bound the scaling factor
    epsilon = 0.1
    if epsilon > 0:
        uniform = torch.ones_like(prob_dist)/batch 
        prob_dist = (1-epsilon)*prob_dist + epsilon*uniform
    
    indices = torch.bernoulli(prob_dist).nonzero(as_tuple=True)[0]
    if len(indices) == 0:
        print("no elements selected - hmm")
        indices = torch.arange(k, device=device)

    # pick top-k minibatch examples to form new smaller tensors
    input_top_k_batch = torch.index_select(input,dim = 0, index = indices)
    grad_output_top_k_batch = torch.index_select(grad_output,dim = 0, index = indices)
    
    if scale == True:
        # scale out_channels by 1/(p_i) to get unbiased estimation
        scale_factors = torch.div(1,prob_dist)
        input_top_k_batch = torch.mul(input_top_k_batch, scale_factors[indices].view(-1,1,1,1))            
 
        
    # compute sampled tensors
    grad_weight_approx = approx_conv2d.backward_weight(input_top_k_batch, weight_shape, grad_output_top_k_batch, stride, padding, dilation, groups, False, False, True)
    return grad_weight_approx





''' input - input tensor, shape (batch, in_channels, h, w)
    weight_shape - shape of grad_weight. expected: (out_channels, in_channels,h,w)
    grad_output - grad output tensor, shape (batch, out_channels, h,w)
    stride, padding, dilation, groups as in regular conv2d
    sample_ratio - Ratio of out_channels to sample
    minimal_k - Minimal number of out_channels to keep in the sampling
'''
def approx_conv2d_func_wu(input, weight_shape, grad_output, stride, padding, dilation, groups, sample_ratio,minimal_k):
    #return approx_conv2d.backward_weight(input, weight_shape, grad_output, stride, padding, dilation, groups, False, False, True)
    #return conv2d_wu_topk(input, weight_shape, grad_output, stride, padding, dilation, groups, sample_ratio,minimal_k)
    #return conv2d_wu_random_sampling(input, weight_shape, grad_output, stride, padding, dilation, groups, sample_ratio,minimal_k, with_replacement=True, optimal_prob=True, scale=True)
    return conv2d_wu_bernoulli_sampling(input, weight_shape, grad_output, stride, padding, dilation, groups, sample_ratio,minimal_k, scale=True)

def approx_conv2d_func_forward(A,B,bias, stride, padding, dilation, groups, sample_ratio,minimal_k):
    #return torch.nn.functional.conv2d(input=A, weight=B, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    return conv2d_top_k(A=A.float(),B=B.float(),bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups, sample_ratio=sample_ratio, minimal_k=minimal_k)
   # return conv2d_top_k_weights(A=A.float(),B=B.float(),bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups, sample_ratio=sample_ratio, minimal_k=minimal_k)
    #return conv2d_top_k_approx(A=A.float(),B=B.float(),bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups, sample_ratio=sample_ratio, minimal_k=minimal_k)
    #return conv2d_top_k_adaptive(A=A.float(),B=B.float(),bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups, sample_ratio=sample_ratio, minimal_k=minimal_k)
    #return approx_conv2d.forward(A, B, bias, stride, padding, dilation, groups, sample_ratio, minimal_k, False, False, True)
    #return conv2d_narrow(A=A.float(),B=B.float(),bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups, sample_ratio=sample_ratio, minimal_k=minimal_k)
    #return conv2d_uniform_sampling(A=A.float(),B=B.float(),bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups, sample_ratio=sample_ratio, minimal_k=minimal_k)
    #return conv2d_random_sampling(A=A.float(),B=B.float(),bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups, sample_ratio=sample_ratio, minimal_k=minimal_k, with_replacement=True, optimal_prob=True, scale=True)
    #return conv2d_random_sampling_adaptive(A=A.float(),B=B.float(),bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups, sample_ratio=sample_ratio, minimal_k=minimal_k, with_replacement=False, optimal_prob=True, scale=False)
    #return conv2d_bernoulli_sampling(A=A.float(),B=B.float(),bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups, sample_ratio=sample_ratio, minimal_k=minimal_k, scale=True)

''' Approximates 2d convolution with channel sampling
    A - input tensor, shape (batch, in_channels, h, w)
    B - input matrices, shape (out_channels, in_channels, kw, kw)
    bias - bias vector, shape (out_channels)
    stride, padding, dilation, groups as in regular conv2d
    sample_ratio - Ratio of in_channels to sample
    minimal_k - Minimal number of in_channels to keep in the sampling
    with_replacement - True means sampling is done with replacement, False means sampling without replacement
    optimal_prob - True means sampling probability is proportional to |Ai|*|Bi|. False means uniform distribution.
    scale - True means each input channel is scaled by 1/sqrt(K*pi) to ensure bias 0
'''
def conv2d_random_sampling(A,B,bias, stride, padding, dilation, groups, sample_ratio, minimal_k, with_replacement, optimal_prob, scale):
    #print("Sanity check - conv2d_random_sampling is used with sample_ratio = " + str(sample_ratio) + "and minimal_k = " + str(minimal_k))
    #print("A size: {}".format(A.size()))
    #print("B size: {}".format(B.size()))
    #print("bias size: {}".format(bias.size()))
    #print("sample_ratio: {}".format(sample_ratio))
    #print("minimal_k: {}".format(minimal_k))
    #print("with_replacement: {}".format(with_replacement))
    #print("optimal_prob: {}".format(optimal_prob))
    #print("scale: {}".format(scale))
    #print("A mean: {}".format(A.mean()))
    #print("A std: {}".format(A.std()))
    #print("B mean: {}".format(B.mean()))
    #print("B std: {}".format(B.std()))
    
    in_channels = A.size()[1]
    device = A.device    

    # calculate the number of input channels to sample
    k_candidate = int(float(in_channels)*sample_ratio)

    # make k at least minimal_k
    k = min(max(k_candidate,minimal_k),in_channels)
    
    # if because of minimal_k or sample_ratio k equals the number of features, perform full conv2d instead of approximating
    if k == in_channels:
        return torch.nn.functional.conv2d(input=A, weight=B, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

    if optimal_prob == True:
        with torch.no_grad():
            # calculate norms of the input channels of A and B
            a_channel_norms = torch.norm(A.view(A.size()[0],A.size()[1],-1),p=2, dim=2)
            a_channel_norms = torch.norm(a_channel_norms, dim=0, p=2)
            a_channel_norms = torch.squeeze(a_channel_norms)

            b_channel_norms = torch.norm(B.view(B.size()[0],B.size()[1],-1),p=2, dim=2)
            b_channel_norms = torch.norm(b_channel_norms, dim=0, p=2)
            b_channel_norms = torch.squeeze(b_channel_norms)

            # multiply both norms element-wise
            norm_mult = torch.mul(a_channel_norms,b_channel_norms)
            
            # use epsilon-optimal sampling to allow learning random weights and to bound the scaling factor
            epsilon = 0.1
            if epsilon > 0:
                sum_norm_mult = torch.sum(norm_mult)
                norm_mult = torch.div(norm_mult, sum_norm_mult)
                uniform = torch.ones_like(norm_mult)/in_channels 
                norm_mult = (1-epsilon)*norm_mult + epsilon*uniform
            
            # no need to normalize, it is already done by torch.multinomial

            # calculate number of nonzero elements in norm_mult. this serves 
            # two purposes:
            # 1. Possibly reduce number of sampled pairs, as zero elements in norm_mult will not contribute to the result
            # 2. Prevents scaling of zero values
            nnz = (norm_mult!=0).sum()
            if nnz == 0:
                #print("zero multiply detected! scenario not optimzied (todo)")
                return torch.nn.functional.conv2d(input=A, weight=B, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
                
            k = min(k,nnz)
            indices = torch.multinomial(norm_mult,k,replacement=with_replacement)
        
        # pick k channels to form new smaller tensors
        A_top_k_channels = torch.index_select(A,dim = 1, index = indices)
        B_top_k_channels = torch.index_select(B,dim = 1, index = indices)
        
        if scale == True:
            # when sampling without replacement a more complicated scaling factor is required (see Horvitz and Thompson, 1952)
            assert(with_replacement == True)
            # scale column-row pairs by 1/(k*p_i) to get unbiased estimation
            with torch.no_grad():
                sum_norm_mult = torch.sum(norm_mult)
                scale_factors = torch.div(sum_norm_mult,torch.mul(norm_mult,k))
            A_top_k_channels = torch.mul(A_top_k_channels, scale_factors[indices].view(1,-1,1,1))            
 
    else:
        # uniform sampling    
        if with_replacement == True:
            indices = torch.randint(low=0,high=in_channels,size=(k,),device=device)
        else:
            uniform_dist = torch.ones(in_channels,device=device)
            indices = torch.multinomial(uniform_dist,k,replacement=False)
        
        # pick k column-row pairs to form new smaller matrices
        A_top_k_channels = torch.index_select(A, dim=1, index=indices)
        B_top_k_channels = torch.index_select(B, dim=1, index=indices)

        if scale == True:
            # scale column-row pairs by 1/(k*p_i) to get unbiased estimation
            # in case of uniform distribution, p_i = 1/in_features when sampling with replacement 
            # when sampling without replacement a different scaling factor is required (see Horvitz and Thompson, 1952), but
            # for uniform sampling it turns to be in_features/k as well
            scale_factor = in_channels/k
            A_top_k_channels = torch.mul(A_top_k_channels, scale_factor) 
        
    # convolve smaller tensors 
    C_approx = torch.nn.functional.conv2d(input=A_top_k_channels, weight=B_top_k_channels, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    return C_approx

''' Approximates 2d convolution with channel sampling
    the number of channels sampled will vary according to norm concentration
    A - input tensor, shape (batch, in_channels, h, w)
    B - input matrices, shape (out_channels, in_channels, kw, kw)
    bias - bias vector, shape (out_channels)
    stride, padding, dilation, groups as in regular conv2d
    sample_ratio - Ratio of in_channels to sample
    minimal_k - Minimal number of in_channels to keep in the sampling
    with_replacement - True means sampling is done with replacement, False means sampling without replacement
    optimal_prob - True means sampling probability is proportional to |Ai|*|Bi|. False means uniform distribution.
    scale - True means each input channel is scaled by 1/sqrt(K*pi) to ensure bias 0
'''
def conv2d_random_sampling_adaptive(A,B,bias, stride, padding, dilation, groups, sample_ratio, minimal_k, with_replacement, optimal_prob, scale):
    #print("Sanity check - conv2d_random_sampling adaptive is used with sample_ratio = " + str(sample_ratio) + "and minimal_k = " + str(minimal_k))
    #print("A size: {}".format(A.size()))
    #print("B size: {}".format(B.size()))
    #print("bias size: {}".format(bias.size()))
    #print("sample_ratio: {}".format(sample_ratio))
    #print("minimal_k: {}".format(minimal_k))
    #print("with_replacement: {}".format(with_replacement))
    #print("optimal_prob: {}".format(optimal_prob))
    #print("scale: {}".format(scale))
    #print("A mean: {}".format(A.mean()))
    #print("A std: {}".format(A.std()))
    #print("B mean: {}".format(B.mean()))
    #print("B std: {}".format(B.std()))
    
    in_channels = A.size()[1]
    device = A.device    

    # if because of minimal_k or sample_ratio k equals the number of features, perform full conv2d instead of approximating
    if minimal_k >= in_channels or sample_ratio == 1.0:
        return torch.nn.functional.conv2d(input=A, weight=B, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

    if optimal_prob == True:
        with torch.no_grad():
            # calculate norms of the input channels of A and B
            a_channel_norms = torch.norm(A.view(A.size()[0],A.size()[1],-1),p=2, dim=2)
            a_channel_norms = torch.norm(a_channel_norms, dim=0, p=2)
            a_channel_norms = torch.squeeze(a_channel_norms)

            b_channel_norms = torch.norm(B.view(B.size()[0],B.size()[1],-1),p=2, dim=2)
            b_channel_norms = torch.norm(b_channel_norms, dim=0, p=2)
            b_channel_norms = torch.squeeze(b_channel_norms)

            # multiply both norms element-wise
            norm_mult = torch.mul(a_channel_norms,b_channel_norms)
            
            # use epsilon-optimal sampling to allow learning random weights and to bound the scaling factor
            epsilon = 0.1
            if epsilon > 0:
                sum_norm_mult = torch.sum(norm_mult)
                norm_mult = torch.div(norm_mult, sum_norm_mult)
                uniform = torch.ones_like(norm_mult)/in_channels 
                norm_mult = (1-epsilon)*norm_mult + epsilon*uniform
            
            # no need to normalize, it is already done by torch.multinomial

            # calculate number of nonzero elements in norm_mult. this serves 
            # two purposes:
            # 1. Possibly reduce number of sampled pairs, as zero elements in norm_mult will not contribute to the result
            # 2. Prevents scaling of zero values
            nnz = (norm_mult!=0).sum()
            if nnz == 0:
                #print("zero multiply detected! scenario not optimzied (todo)")
                return torch.nn.functional.conv2d(input=A, weight=B, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        
            sum_norm_mult = torch.sum(norm_mult) 
            sorted_indices = topk_indices(norm_mult,in_channels)
            for k in range(minimal_k, in_channels):
                if norm_mult[sorted_indices[:k]].sum() >= sum_norm_mult*sample_ratio:
                    break
                
            indices = torch.multinomial(norm_mult,k,replacement=with_replacement)
        
        # pick k channels to form new smaller tensors
        A_top_k_channels = torch.index_select(A,dim = 1, index = indices)
        B_top_k_channels = torch.index_select(B,dim = 1, index = indices)
        
        if scale == True:
            # when sampling without replacement a more complicated scaling factor is required (see Horvitz and Thompson, 1952)
            assert(with_replacement == True)
            # scale column-row pairs by 1/(k*p_i) to get unbiased estimation
            with torch.no_grad():
                sum_norm_mult = torch.sum(norm_mult)
                scale_factors = torch.div(sum_norm_mult,torch.mul(norm_mult,k))
            A_top_k_channels = torch.mul(A_top_k_channels, scale_factors[indices].view(1,-1,1,1))            
 
    else:
        # uniform sampling    
        print('adaptive sampling not implemented yet for uniform sampling')
        exit()
        
    # convolve smaller tensors 
    C_approx = torch.nn.functional.conv2d(input=A_top_k_channels, weight=B_top_k_channels, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    return C_approx

''' Approximates 2d convolution with channel sampling according to largest norm
    A - input tensor, shape (batch, in_channels, h, w)
    B - input matrices, shape (out_channels, in_channels, kw, kw)
    bias - bias vector, shape (out_channels)
    stride, padding, dilation, groups as in regular conv2d
    sample_ratio - Ratio of in_channels to sample
    minimal_k - Minimal number of in_channels to keep in the sampling
'''
def conv2d_top_k(A,B,bias, stride, padding, dilation, groups, sample_ratio, minimal_k):
    #print("Sanity check - conv2d_top_k is used with sample_ratio = " + str(sample_ratio) + " and minimal_k = " + str(minimal_k))

    in_channels = A.size()[1]
    
    # calculate the number of channels to sample for the forward propagation phase
    k_candidate = int(float(in_channels)*sample_ratio)

    # make k at least min_clrows (similar to meProp)
    k = min(max(k_candidate,minimal_k),in_channels)
    
    # if because of minimal_k or sample_ratio k equals the number of features, perform full conv2d instead of approximating
    if k == in_channels:
        return torch.nn.functional.conv2d(input=A, weight=B, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

    # calculate norms of the columns of A and rows of B
    with torch.no_grad():
        a_channel_norms = torch.norm(A.view(A.size()[0],A.size()[1],-1),p=2, dim=2)
        a_channel_norms = torch.norm(a_channel_norms, dim=0, p=2)
        a_channel_norms = torch.squeeze(a_channel_norms)

        b_channel_norms = torch.norm(B.view(B.size()[0],B.size()[1],-1),p=2, dim=2)
        b_channel_norms = torch.norm(b_channel_norms, dim=0, p=2)
        b_channel_norms = torch.squeeze(b_channel_norms)

        # multiply both norms element-wise to and pick the indices of the top K column-row pairs
        norm_mult = torch.mul(a_channel_norms,b_channel_norms)

        #top_k_indices = torch.topk(norm_mult,k)[1]
        top_k_indices = topk_indices(norm_mult,k)

    # pick top-k column-row pairs to form new smaller matrices
    A_top_k_cols = torch.index_select(A,dim = 1, index = top_k_indices)
    B_top_k_rows = torch.index_select(B,dim = 1, index = top_k_indices)

    # multiply smaller matrices
    C_approx = torch.nn.functional.conv2d(input=A_top_k_cols, weight=B_top_k_rows, bias=bias, stride=stride, padding=padding,
                                          dilation=dilation, groups=groups)
    return C_approx


''' Approximates 2d convolution with channel sampling according to largest norm
    A - input tensor, shape (batch, in_channels, h, w)
    B - input matrices, shape (out_channels, in_channels, kw, kw)
    bias - bias vector, shape (out_channels)
    stride, padding, dilation, groups as in regular conv2d
    sample_ratio - Ratio of in_channels to sample
    minimal_k - Minimal number of in_channels to keep in the sampling
'''
def conv2d_top_k_weights(A,B,bias, stride, padding, dilation, groups, sample_ratio, minimal_k):
    #print("Sanity check - conv2d_top_k_weights is used with sample_ratio = " + str(sample_ratio) + " and minimal_k = " + str(minimal_k))

    in_channels = A.size()[1]
    
    # calculate the number of channels to sample for the forward propagation phase
    k_candidate = int(float(in_channels)*sample_ratio)

    # make k at least min_clrows (similar to meProp)
    k = min(max(k_candidate,minimal_k),in_channels)
    
    # if because of minimal_k or sample_ratio k equals the number of features, perform full conv2d instead of approximating
    if k == in_channels:
        return torch.nn.functional.conv2d(input=A, weight=B, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

    # calculate norms of rows of B
    with torch.no_grad():
        b_channel_norms = torch.norm(B.view(B.size()[0],B.size()[1],-1),p=2, dim=2)
        b_channel_norms = torch.norm(b_channel_norms, dim=0, p=2)
        b_channel_norms = torch.squeeze(b_channel_norms)

        #top_k_indices = torch.topk(norm_mult,k)[1]
        top_k_indices = topk_indices(b_channel_norms,k)

    # pick top-k column-row pairs to form new smaller matrices
    A_top_k_cols = torch.index_select(A,dim = 1, index = top_k_indices)
    B_top_k_rows = torch.index_select(B,dim = 1, index = top_k_indices)

    # multiply smaller matrices
    C_approx = torch.nn.functional.conv2d(input=A_top_k_cols, weight=B_top_k_rows, bias=bias, stride=stride, padding=padding,
                                          dilation=dilation, groups=groups)
    return C_approx

''' Approximates 2d convolution with channel sampling according to largest norm
    A - input tensor, shape (batch, in_channels, h, w)
    B - input matrices, shape (out_channels, in_channels, kw, kw)
    bias - bias vector, shape (out_channels)
    stride, padding, dilation, groups as in regular conv2d
    sample_ratio - Ratio of in_channels to sample
    minimal_k - Minimal number of in_channels to keep in the sampling
    returns conv result and selected indices
'''
def conv2d_top_k_weights_dist(A,B,bias, stride, padding, dilation, groups, sample_ratio, minimal_k):
    #print("Sanity check - conv2d_top_k_weights is used with sample_ratio = " + str(sample_ratio) + " and minimal_k = " + str(minimal_k))

    in_channels = A.size()[1]
    
    # calculate the number of channels to sample for the forward propagation phase
    k_candidate = int(float(in_channels)*sample_ratio)

    # make k at least min_clrows (similar to meProp)
    k = min(max(k_candidate,minimal_k),in_channels)
    
    # if because of minimal_k or sample_ratio k equals the number of features, perform full conv2d instead of approximating
    if k == in_channels:
        return torch.nn.functional.conv2d(input=A, weight=B, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups), torch.arange(in_channels, device=A.device)

    # calculate norms of rows of B
    with torch.no_grad():
        b_channel_norms = torch.norm(B.view(B.size()[0],B.size()[1],-1),p=2, dim=2)
        b_channel_norms = torch.norm(b_channel_norms, dim=0, p=2)
        b_channel_norms = torch.squeeze(b_channel_norms)

        # add explicit sorting because of strange indeterministic behavior across multiple GPUs
        top_k_indices = torch.topk(b_channel_norms,k)[1].sort()[0]
        #top_k_indices = topk_indices(b_channel_norms,k).sort()[0]

    # pick top-k column-row pairs to form new smaller matrices
    A_top_k_cols = torch.index_select(A,dim = 1, index = top_k_indices)
    B_top_k_rows = torch.index_select(B,dim = 1, index = top_k_indices)

    # multiply smaller matrices
    C_approx = torch.nn.functional.conv2d(input=A_top_k_cols, weight=B_top_k_rows, bias=bias, stride=stride, padding=padding,
                                          dilation=dilation, groups=groups)
    return C_approx, top_k_indices

''' Approximates 2d convolution with channel sampling according to largest norm
    the norm is sampled from a subset of the reduction dimension
    A - input tensor, shape (batch, in_channels, h, w)
    B - input matrices, shape (out_channels, in_channels, kw, kw)
    bias - bias vector, shape (out_channels)
    stride, padding, dilation, groups as in regular conv2d
    sample_ratio - Ratio of in_channels to sample
    minimal_k - Minimal number of in_channels to keep in the sampling
'''
def conv2d_top_k_approx(A,B,bias, stride, padding, dilation, groups, sample_ratio, minimal_k):
    #print("Sanity check - conv2d_top_k_approx is used with sample_ratio = " + str(sample_ratio) + "and minimal_k = " + str(minimal_k))

    in_channels = A.size()[1]
    
    # calculate the number of channels to sample for the forward propagation phase
    k_candidate = int(float(in_channels)*sample_ratio)

    # make k at least min_clrows (similar to meProp)
    k = min(max(k_candidate,minimal_k),in_channels)
    
    # if because of minimal_k or sample_ratio k equals the number of features, perform full conv2d instead of approximating
    if k == in_channels:
        return torch.nn.functional.conv2d(input=A, weight=B, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

    # calculate norms of the columns of A and rows of B
    with torch.no_grad():
        a_channel_norms = torch.norm(A[:,:,torch.randint(A.size()[2],size=[1],dtype=torch.long), torch.randint(A.size()[3],size=[1],dtype=torch.long)], dim=0, p=2)
        a_channel_norms = torch.squeeze(a_channel_norms)

        b_channel_norms = torch.norm(B[:,:,torch.randint(B.size()[2],size=[1],dtype=torch.long), torch.randint(B.size()[3],size=[1],dtype=torch.long)], dim=0, p=2)
        b_channel_norms = torch.squeeze(b_channel_norms)

        # multiply both norms element-wise to and pick the indices of the top K column-row pairs
        norm_mult = torch.mul(a_channel_norms,b_channel_norms)

        #top_k_indices = torch.topk(norm_mult,k)[1]
        top_k_indices = topk_indices(norm_mult,k)

    # pick top-k column-row pairs to form new smaller matrices
    A_top_k_cols = torch.index_select(A,dim = 1, index = top_k_indices)
    B_top_k_rows = torch.index_select(B,dim = 1, index = top_k_indices)

    # multiply smaller matrices
    C_approx = torch.nn.functional.conv2d(input=A_top_k_cols, weight=B_top_k_rows, bias=bias, stride=stride, padding=padding,
                                          dilation=dilation, groups=groups)
    return C_approx

''' Approximates 2d convolution with channel sampling according to largest norm
    the number of channels sampled will vary according to norm concentration
    A - input tensor, shape (batch, in_channels, h, w)
    B - input matrices, shape (out_channels, in_channels, kw, kw)
    bias - bias vector, shape (out_channels)
    stride, padding, dilation, groups as in regular conv2d
    sample_ratio - Ratio of in_channels to sample
    minimal_k - Minimal number of in_channels to keep in the sampling
'''
def conv2d_top_k_adaptive(A,B,bias, stride, padding, dilation, groups, sample_ratio, minimal_k):
    #print("Sanity check - conv2d_top_k_adaptive is used with sample_ratio = " + str(sample_ratio) + "and minimal_k = " + str(minimal_k))

    in_channels = A.size()[1]
    
    # if because of minimal_k k equals the number of features, perform full conv2d instead of approximating
    if sample_ratio == 1.0 or minimal_k >= in_channels:
        return torch.nn.functional.conv2d(input=A, weight=B, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

    # calculate norms of the columns of A and rows of B
    with torch.no_grad():
        a_channel_norms = torch.norm(A.view(A.size()[0],A.size()[1],-1),p=2, dim=2)
        a_channel_norms = torch.norm(a_channel_norms, dim=0, p=2)
        a_channel_norms = torch.squeeze(a_channel_norms)

        b_channel_norms = torch.norm(B.view(B.size()[0],B.size()[1],-1),p=2, dim=2)
        b_channel_norms = torch.norm(b_channel_norms, dim=0, p=2)
        b_channel_norms = torch.squeeze(b_channel_norms)

        # multiply both norms element-wise to and pick the indices of the top K column-row pairs
        norm_mult = torch.mul(a_channel_norms,b_channel_norms)

        sum_norm_mult = torch.sum(norm_mult) 
        #top_k_indices = torch.topk(norm_mult,k)[1]
        
        sorted_indices = topk_indices(norm_mult,in_channels)
        k = minimal_k
        for k in range(minimal_k, in_channels):
            if norm_mult[sorted_indices[:k]].sum() >= sum_norm_mult*sample_ratio:
                top_k_indices = sorted_indices[:k]
                break

    # pick top-k column-row pairs to form new smaller matrices
    A_top_k_cols = torch.index_select(A,dim = 1, index = top_k_indices)
    B_top_k_rows = torch.index_select(B,dim = 1, index = top_k_indices)

    # multiply smaller matrices
    C_approx = torch.nn.functional.conv2d(input=A_top_k_cols, weight=B_top_k_rows, bias=bias, stride=stride, padding=padding,
                                          dilation=dilation, groups=groups)
    return C_approx

def conv2d_narrow(A,B,bias, stride, padding, dilation, groups, sample_ratio, minimal_k):

    #print("Sanity check - conv2d_narrow is used with sample_ratio = " + str(sample_ratio) + "and minimal_k = " + str(minimal_k))

    # calculate the number of channels to sample for the forward propagation phase
    k_candidate = int(float(B.size()[1])*sample_ratio)

    # make k at least min_clrows (similar to meProp)
    k = min(max(k_candidate,minimal_k),B.size()[1])

    A_top_k_cols = torch.narrow(A,dim = 1, start=0, length=k)
    B_top_k_rows = torch.narrow(B,dim = 1, start=0, length=k)

    # multiply smaller matrices
    C_approx = torch.nn.functional.conv2d(input=A_top_k_cols, weight=B_top_k_rows, bias=bias, stride=stride, padding=padding,
                                          dilation=dilation, groups=groups)
    return C_approx



def conv2d_uniform_sampling(A,B,bias, stride, padding, dilation, groups, sample_ratio, minimal_k):

    # print("Sanity check - conv2d_uniform_sampling is used, sample_ratio = " + str(sample_ratio) + " minimal_k = " + str(minimal_k))

    # calculate the number of input channels to sample for the forward propagation phase
    k_candidate = int(float(B.size()[1])*sample_ratio)

    # make k at least min_clrows (similar to meProp)
    k = min(max(k_candidate,minimal_k),B.size()[1])

    indices = torch.randperm(B.size()[1])[:k].cuda()

    # pick top-k column-row pairs to form new smaller matrices
    A_top_k_cols = torch.index_select(A, dim=1, index=indices)
    B_top_k_rows = torch.index_select(B,dim = 1, index = indices)


    # convolve smaller tensors
    C_approx = torch.nn.functional.conv2d(input=A_top_k_cols, weight=B_top_k_rows, bias=bias, stride=stride, padding=padding,
                                          dilation=dilation, groups=groups)
    return C_approx

''' Approximates 2d convolution with bernoulli channel sampling
    A - input tensor, shape (batch, in_channels, h, w)
    B - input matrices, shape (out_channels, in_channels, kw, kw)
    bias - bias vector, shape (out_channels)
    stride, padding, dilation, groups as in regular conv2d
    sample_ratio - Ratio of in_channels to sample
    minimal_k - Minimal number of in_channels to keep in the sampling
    scale - True means each input channel is scaled by 1/sqrt(K*pi) to ensure bias 0
'''
def conv2d_bernoulli_sampling(A,B,bias, stride, padding, dilation, groups, sample_ratio, minimal_k, scale):
    #print("Sanity check - conv2d_bernoulli_sampling is used with sample_ratio = " + str(sample_ratio) + "and minimal_k = " + str(minimal_k))
    #print("A size: {}".format(A.size()))
    #print("B size: {}".format(B.size()))
    #print("bias size: {}".format(bias.size()))
    #print("sample_ratio: {}".format(sample_ratio))
    #print("minimal_k: {}".format(minimal_k))
    #print("scale: {}".format(scale))
    #print("A mean: {}".format(A.mean()))
    #print("A std: {}".format(A.std()))
    #print("B mean: {}".format(B.mean()))
    #print("B std: {}".format(B.std()))
    
    in_channels = A.size()[1]
    device = A.device    

    # calculate the number of input channels to sample
    k_candidate = int(float(in_channels)*sample_ratio)

    # make k at least minimal_k
    k = min(max(k_candidate,minimal_k),in_channels)
    
    # if because of minimal_k or sample_ratio k equals the number of features, perform full conv2d instead of approximating
    if k == in_channels:
        return torch.nn.functional.conv2d(input=A, weight=B, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

    with torch.no_grad():
        # calculate norms of the input channels of A and B
        a_channel_norms = torch.norm(A.view(A.size()[0],A.size()[1],-1),p=2, dim=2)
        a_channel_norms = torch.norm(a_channel_norms, dim=0, p=2)
        a_channel_norms = torch.squeeze(a_channel_norms)

        b_channel_norms = torch.norm(B.view(B.size()[0],B.size()[1],-1),p=2, dim=2)
        b_channel_norms = torch.norm(b_channel_norms, dim=0, p=2)
        b_channel_norms = torch.squeeze(b_channel_norms)

        # multiply both norms element-wise
        norm_mult = torch.mul(a_channel_norms,b_channel_norms)
        sum_norm_mult = norm_mult.sum()       
        

        # calculate number of nonzero elements in norm_mult. this serves 
        # two purposes:
        # 1. Possibly reduce number of sampled pairs, as zero elements in norm_mult will not contribute to the result
        # 2. Prevents scaling of zero values
        nnz = (norm_mult!=0).sum()
        if nnz == 0:
            #print("zero multiply detected! scenario not optimzied (todo)")
            return torch.nn.functional.conv2d(input=A, weight=B, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            
        k = min(k,nnz)
        prob_dist = k * torch.div(norm_mult,sum_norm_mult)
        prob_dist = prob_dist.clamp(min=0, max=1)
        
        # use epsilon-optimal sampling to allow learning random weights and to bound the scaling factor
        epsilon = 0.1
        if epsilon > 0:
            uniform = torch.ones_like(prob_dist)/in_channels 
            prob_dist = (1-epsilon)*prob_dist + epsilon*uniform

        indices = torch.bernoulli(prob_dist).nonzero(as_tuple=True)[0]
        if len(indices) == 0:
            print("no elements selected - hmm")
            indices = torch.arange(k, device=device)
    
    # pick k channels to form new smaller tensors
    A_top_k_channels = torch.index_select(A,dim = 1, index = indices)
    B_top_k_channels = torch.index_select(B,dim = 1, index = indices)
    
    if scale == True:
        # scale column-row pairs by 1/(p_i) to get unbiased estimation
        with torch.no_grad():
            scale_factors = torch.div(1,prob_dist)
        A_top_k_channels = torch.mul(A_top_k_channels, scale_factors[indices].view(1,-1,1,1))            
 
    # convolve smaller tensors 
    C_approx = torch.nn.functional.conv2d(input=A_top_k_channels, weight=B_top_k_channels, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    return C_approx


