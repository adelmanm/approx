import torch
from ..modules.utils import *

def approx_linear_forward(input,weight,bias,sample_ratio,minimal_k,sample_ratio_bwd,minimal_k_bwd,sample_ratio_wu,minimal_k_wu):
    r"""
    Applies approximate linear transformation to the incoming data: :math:`y = xA^T + b`.
    the matrix multiply xA^T is approximated
    note: weight transposition is done in this function
    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """


    #return torch.nn.functional.linear(input,weight,bias)
    return approx_linear_forward_xA_b(input,weight.t(),bias,sample_ratio, minimal_k,sample_ratio_bwd,minimal_k_bwd,sample_ratio_wu,minimal_k_wu)

def approx_linear_forward_xA_b(input,weight,bias,sample_ratio,minimal_k,sample_ratio_bwd,minimal_k_bwd,sample_ratio_wu,minimal_k_wu):
    r"""
    Applies approximate linear transformation to the incoming data: :math:`y = xA + b`.
    Note: A is assumed not transposed
    the matrix multiply xA is approximated
    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Weight: :math:`(in\_features, out\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """
    #return linear_top_k(input,weight,bias,sample_ratio, minimal_k,sample_ratio_bwd,minimal_k_bwd,sample_ratio_wu,minimal_k_wu)
    #return linear_top_k_approx(input,weight,bias,sample_ratio, minimal_k,sample_ratio_bwd,minimal_k_bwd,sample_ratio_wu,minimal_k_wu)
    #return linear_uniform_sampling(input,weight,bias,sample_ratio, minimal_k)
    #return linear_random_sampling(input,weight,bias,sample_ratio, minimal_k, with_replacement=True, optimal_prob=True, scale=True,sample_ratio_bwd=sample_ratio_bwd, minimal_k_bwd=minimal_k_bwd, sample_ratio_wu=sample_ratio_wu, minimal_k_wu=minimal_k_wu)
    #return approx_linear_xA_b.topk(input,weight,bias,sample_ratio, minimal_k)
    return linear_bernoulli_sampling(input,weight,bias,sample_ratio, minimal_k, scale=True,sample_ratio_bwd=sample_ratio_bwd, minimal_k_bwd=minimal_k_bwd, sample_ratio_wu=sample_ratio_wu, minimal_k_wu=minimal_k_wu)

''' Approximates the matrix multiply A*B+b by sampling the column-row pairs with the largest norm
    A - input matrix, shape (N,*,in_features) where '*' means any number of additional dimensions
    B - input matrices, shape (in_features, out_features)
    bias - bias vector, shape (out_features)
    sample_ratio - Ratio of column-row pairs to sample
    minimal_k - Minimal number of column-row pairs to keep in the sampling
    note: B is not transposed
    output: A*B+b, shape (N,*,out_features)
'''
def linear_top_k(A,B,bias,sample_ratio, minimal_k,sample_ratio_bwd=None,minimal_k_bwd=None,sample_ratio_wu=None,minimal_k_wu=None):

    #print("Sanity check - top_k is used")
    #print("A size: {}".format(A.size()))
    #print("B size: {}".format(B.size()))
    #print("bias size: {}".format(bias.size()))
    #print("sample_ratio: {}".format(sample_ratio))
    #print("minimal_k: {}".format(minimal_k))
    #print("sample_ratio_bwd: {}".format(sample_ratio_bwd))
    #print("minimal_k_bwd: {}".format(minimal_k_bwd))
    #print("sample_ratio_wu: {}".format(sample_ratio_wu))
    #print("minimal_k_wu: {}".format(minimal_k_wu))

    in_features = A.size()[-1]

    # calculate the number of column-row pairs to sample for the forward propagation phase
    k_candidate = int(float(in_features)*sample_ratio)

    # make k at least minimal_k
    k = min(max(k_candidate,minimal_k),in_features)

    # if because of minimal_k or sample_ratio k equals the number of features, perform full matmul instead of approximating
    if k == in_features:
        #no need to sample. perform normal matmul
        if A.dim() == 2 and bias is not None:
            # fused op is marginally faster
            C = torch.addmm(bias, A, B)
        else:
            C = torch.matmul(A, B)
            if bias is not None:
                C += bias
        return C
    
    with torch.no_grad():
        # calculate norms of the columns of A and rows of B
        if A.dim() == 2:   
            a_col_norms = torch.norm(A,dim=0)
        else:
            # since we sample across in_featuers, consider other dimensions as a single dimension for sampling purpuses
            a_col_norms = torch.norm(A.view(-1,in_features),dim=0)    

        b_row_norms = torch.norm(B,dim=1)

        # multiply both norms element-wise to and pick the indices of the top K column-row pairs
        norm_mult = torch.mul(a_col_norms,b_row_norms)

        #top_k_indices = torch.topk(norm_mult,k)[1]
        top_k_indices = topk_indices(norm_mult,k)

    # pick top-k column-row pairs to form new smaller matrices
    A_top_k_cols = torch.index_select(A,dim = -1, index = top_k_indices)
    B_top_k_rows = torch.index_select(B,dim = 0, index = top_k_indices)

    # multiply smaller matrices
    if sample_ratio_bwd is None and sample_ratio_wu is None:
        if A.dim() == 2 and bias is not None:
            # fused op is marginally faster
            C_approx = torch.addmm(bias, A_top_k_cols, B_top_k_rows)
        else:
            C_approx = torch.matmul(A_top_k_cols, B_top_k_rows)
            if bias is not None:
                C_approx += bias
    else:
        # The following code will be used to apply additional sampling in the backward pass but update only the
        # sub-tensors sampled in the forward pass.
        # For simplicity, we don't optimize for torch.addmm usage in this case
        C_approx = matmul_approx_bwd_func.apply(A_top_k_cols, B_top_k_rows,sample_ratio_bwd,minimal_k_bwd,sample_ratio_wu,minimal_k_wu)
        if bias is not None:
            C_approx += bias
 
    return C_approx

''' Approximates the matrix multiply A*B+b by sampling the column-row pairs with the largest norm
    the norm is sampled from a subset of the reduction dimension
    A - input matrix, shape (N,*,in_features) where '*' means any number of additional dimensions
    B - input matrices, shape (in_features, out_features)
    bias - bias vector, shape (out_features)
    sample_ratio - Ratio of column-row pairs to sample
    minimal_k - Minimal number of column-row pairs to keep in the sampling
    note: B is not transposed
    output: A*B+b, shape (N,*,out_features)
'''
def linear_top_k_approx(A,B,bias,sample_ratio, minimal_k,sample_ratio_bwd=None,minimal_k_bwd=None,sample_ratio_wu=None,minimal_k_wu=None):

    #print("Sanity check - top_k_approx is used")
    #print("A size: {}".format(A.size()))
    #print("B size: {}".format(B.size()))
    #print("bias size: {}".format(bias.size()))
    #print("sample_ratio: {}".format(sample_ratio))
    #print("minimal_k: {}".format(minimal_k))

    in_features = A.size()[-1]

    # calculate the number of column-row pairs to sample for the forward propagation phase
    k_candidate = int(float(in_features)*sample_ratio)

    # make k at least minimal_k
    k = min(max(k_candidate,minimal_k),in_features)
    
    # if because of minimal_k or sample_ratio k equals the number of features, perform full matmul instead of approximating
    if k == in_features:
        # no need to sample. perform normal matmul
        if A.dim() == 2 and bias is not None:
            # fused op is marginally faster
            C = torch.addmm(bias, A, B)
        else:
            C = torch.matmul(A, B)
            if bias is not None:
                C += bias
        return C

    # calculate norms of the columns of A and rows of B
    # instead of calculating the exact norms, we sample a subset of A rows and B columns
    # and calculate the norm over them. This serves two purposes:
    # 1. faster estimation of the norm
    # 2. introduces some randomness to avoid always sampling the same high-norm features

    with torch.no_grad():
        if A.dim() == 2:   
            a_num_rows = A.size()[0]
            a_sample_start = torch.randint(a_num_rows-9,size=(1,),dtype=torch.long)
            a_col_norms = torch.norm(A[a_sample_start:a_sample_start+10:],dim=0)
        else:
            # since we sample across in_featuers, consider other dimensions as a single dimension for sampling purpuses
            a_num_rows = A.view(-1,in_features).size()[0]
            a_sample_start = torch.randint(a_num_rows-9,size=(1,),dtype=torch.long)
            a_col_norms = torch.norm(A.view(-1,in_features)[a_sample_start:a_sample_start+10,:],dim=0)    

        b_num_cols = B.size()[1]
        b_sample_start = torch.randint(b_num_cols-9,size=(1,),dtype=torch.long)
        b_row_norms = torch.norm(B[:,b_sample_start:b_sample_start+10],dim=1)

        # multiply both norms element-wise to and pick the indices of the top K column-row pairs
        norm_mult = torch.mul(a_col_norms,b_row_norms)

        #top_k_indices = torch.topk(norm_mult,k)[1]
        top_k_indices = topk_indices(norm_mult,k)

    # pick top-k column-row pairs to form new smaller matrices
    A_top_k_cols = torch.index_select(A,dim = -1, index = top_k_indices)
    B_top_k_rows = torch.index_select(B,dim = 0, index = top_k_indices)

    # multiply smaller matrices
    if sample_ratio_bwd is None and sample_ratio_wu is None:
        if A.dim() == 2 and bias is not None:
            # fused op is marginally faster
            C_approx = torch.addmm(bias, A_top_k_cols, B_top_k_rows)
        else:
            C_approx = torch.matmul(A_top_k_cols, B_top_k_rows)
            if bias is not None:
                C_approx += bias
    else:
        # The following code will be used to apply additional sampling in the backward pass but update only the
        # sub-tensors sampled in the forward pass.
        # For simplicity, we don't optimize for torch.addmm usage in this case
        C_approx = matmul_approx_bwd_func.apply(A_top_k_cols, B_top_k_rows,sample_ratio_bwd,minimal_k_bwd,sample_ratio_wu,minimal_k_wu)
        if bias is not None:
            C_approx += bias
    
    return C_approx
''' Approximates the matrix multiply A*B+b
    A - input matrix, shape (N,*,in_features) where '*' means any number of additional dimensions
    B - input matrices, shape (in_features, out_features)
    bias - bias vector, shape (out_features)
    sample_ratio - Ratio of column-row pairs to sample
    minimal_k - Minimal number of column-row pairs to keep in the sampling
    with_replacement - True means sampling is done with replacement, False means sampling without replacement
    optimal_prob - True means sampling probability is proportional to |Ai|*|Bj|. False means uniform distribution.
    scale - True means each column-row is scaled by 1/sqrt(K*pi) to ensure bias 0
    
'''
def linear_random_sampling(A,B,bias,sample_ratio, minimal_k, with_replacement, optimal_prob, scale,sample_ratio_bwd=None,minimal_k_bwd=None,sample_ratio_wu=None,minimal_k_wu=None):
    #print("Sanity check - linear_sampling is used")
    #print("A size: {}".format(A.size()))
    #print("B size: {}".format(B.size()))
    #if bias is not None:
    #    print("bias size: {}".format(bias.size()))
    #else:
    #    print("no bias")
    #print("sample_ratio: {}".format(sample_ratio))
    #print("minimal_k: {}".format(minimal_k))
    #print("with_replacement: {}".format(with_replacement))
    #print("optimal_prob: {}".format(optimal_prob))
    #print("scale: {}".format(scale))
    #print("A mean: {}".format(A.mean()))
    #print("A std: {}".format(A.std()))
    #print("B mean: {}".format(B.mean()))
    #print("B std: {}".format(B.std()))
    
    in_features = A.size()[-1]
    device = A.device    

    # calculate the number of column-row pairs to sample
    k_candidate = int(float(in_features)*sample_ratio)

    # make k at least minimal_k
    k = min(max(k_candidate,minimal_k),in_features)
    
    # if because of minimal_k or sample_ratio k equals the number of features, perform full matmul instead of approximating
    if k == in_features:
        # no need to sample. perform normal matmul
        if A.dim() == 2 and bias is not None:
            # fused op is marginally faster
            C = torch.addmm(bias, A, B)
        else:
            C = torch.matmul(A, B)
            if bias is not None:
                C += bias
        return C

    if optimal_prob == True:
        with torch.no_grad():
            # calculate norms of the columns of A and rows of B
            if A.dim() == 2:   
                a_col_norms = torch.norm(A,dim=0)
            else:
                # since we sample across in_featuers, consider other dimensions as a single dimension for sampling purpuses
                a_col_norms = torch.norm(A.view(-1,in_features),dim=0)    
            
            b_row_norms = torch.norm(B,dim=1)

            # multiply both norms element-wise
            norm_mult = torch.mul(a_col_norms,b_row_norms)
            
            # use epsilon-optimal sampling to allow learning random weights and to bound the scaling factor
            epsilon = 0.1
            if epsilon > 0:
                sum_norm_mult = torch.sum(norm_mult)
                norm_mult = torch.div(norm_mult, sum_norm_mult)
                uniform = torch.ones_like(norm_mult)/in_features 
                norm_mult = (1-epsilon)*norm_mult + epsilon*uniform
            
            # no need to normalize, it is already done by torch.multinomial

            # calculate number of nonzero elements in norm_mult. this serves 
            # two purposes:
            # 1. Possibly reduce number of sampled pairs, as zero elements in norm_mult will not contribute to the result
            # 2. Prevents scaling of zero values
            nnz = (norm_mult!=0).sum()
            if nnz == 0:
                #print("zero multiply detected! scenario not optimzied (todo)")
                return torch.nn.functional.linear(A, B.t(),bias)
                
            k = min(k,nnz)
            indices = torch.multinomial(norm_mult,k,replacement=with_replacement)
        
        # pick k column-row pairs to form new smaller matrices
        A_top_k_cols = torch.index_select(A, dim=-1, index=indices)
        B_top_k_rows = torch.index_select(B, dim=0, index=indices)
        
        if scale == True:
            # when sampling without replacement a more complicated scaling factor is required (see Horvitz and Thompson, 1952)
            assert(with_replacement == True)
            # scale column-row pairs by 1/(k*p_i) to get unbiased estimation
            with torch.no_grad():
                sum_norm_mult = torch.sum(norm_mult)
                scale_factors = torch.div(sum_norm_mult,torch.mul(norm_mult,k))
                scale_matrix = torch.diag(scale_factors[indices])
            A_top_k_cols = torch.matmul(A_top_k_cols, scale_matrix)
            
 
    else:
        # uniform sampling    
        if with_replacement == True:
            indices = torch.randint(low=0,high=in_features,size=(k,),device=device)
        else:
            uniform_dist = torch.ones(in_features,device=device)
            indices = torch.multinomial(uniform_dist,k,replacement=False)
        
        # pick k column-row pairs to form new smaller matrices
        A_top_k_cols = torch.index_select(A, dim=-1, index=indices)
        B_top_k_rows = torch.index_select(B, dim=0, index=indices)

        if scale == True:
            # scale column-row pairs by 1/(k*p_i) to get unbiased estimation
            # in case of uniform distribution, p_i = 1/in_features when sampling with replacement 
            # when sampling without replacement a different scaling factor is required (see Horvitz and Thompson, 1952), but
            # for uniform sampling it turns to be in_features/k as well
            scale_factor = in_features/k
            
            scale_matrix = torch.diag(torch.empty((k,), device=device).fill_(scale_factor))
            
            A_top_k_cols = torch.matmul(A_top_k_cols, scale_matrix)
        
    # multiply smaller matrices
    if sample_ratio_bwd is None and sample_ratio_wu is None:
        if A.dim() == 2 and bias is not None:
            # fused op is marginally faster
            C_approx = torch.addmm(bias, A_top_k_cols, B_top_k_rows)
        else:
            C_approx = torch.matmul(A_top_k_cols, B_top_k_rows)
            if bias is not None:
                C_approx += bias
    else:
        # The following code will be used to apply additional sampling in the backward pass but update only the
        # sub-tensors sampled in the forward pass.
        # For simplicity, we don't optimize for torch.addmm usage in this case
        C_approx = matmul_approx_bwd_func.apply(A_top_k_cols, B_top_k_rows,sample_ratio_bwd,minimal_k_bwd,sample_ratio_wu,minimal_k_wu)
        if bias is not None:
            C_approx += bias
    
    return C_approx

def linear_uniform_sampling(A,B,bias,sample_ratio, minimal_k):

    #print("Sanity check - uniform_sampling is used")

    # calculate the number of column-row pairs to sample for the forward propagation phase
    k_candidate = int(float(B.size()[1])*sample_ratio)

    # make k at least min_clrows (similar to meProp)
    k = min(max(k_candidate,minimal_k),B.size()[1])

    indices = torch.randperm(B.size()[1])[:k].cuda()

    # pick top-k column-row pairs to form new smaller matrices
    A_top_k_cols = torch.index_select(A, dim=1, index=indices)
    B_top_k_rows = torch.index_select(B, dim=1, index=indices)

    # multiply smaller matrices
    C_approx = torch.nn.functional.linear(A_top_k_cols, B_top_k_rows,bias)
    return C_approx


''' Approximates the matrix multiply A*B+b using Bernoulli sampling
    A - input matrix, shape (N,*,in_features) where '*' means any number of additional dimensions
    B - input matrices, shape (in_features, out_features)
    bias - bias vector, shape (out_features)
    sample_ratio - Ratio of column-row pairs to sample
    minimal_k - Minimal number of column-row pairs to keep in the sampling
    scale - True means each column-row is scaled by 1/sqrt(K*pi) to ensure bias 0
    
'''
def linear_bernoulli_sampling(A,B,bias,sample_ratio, minimal_k, scale,sample_ratio_bwd=None,minimal_k_bwd=None,sample_ratio_wu=None,minimal_k_wu=None):
    #print("Sanity check - bernoulli_sampling is used")
    #print("A size: {}".format(A.size()))
    #print("B size: {}".format(B.size()))
    #if bias is not None:
    #    print("bias size: {}".format(bias.size()))
    #else:
    #    print("no bias")
    #print("sample_ratio: {}".format(sample_ratio))
    #print("minimal_k: {}".format(minimal_k))
    #print("scale: {}".format(scale))
    #print("A mean: {}".format(A.mean()))
    #print("A std: {}".format(A.std()))
    #print("B mean: {}".format(B.mean()))
    #print("B std: {}".format(B.std()))

    in_features = A.size()[-1]
    device = A.device    

    # calculate the number of column-row pairs to sample
    k_candidate = int(float(in_features)*sample_ratio)

    # make k at least minimal_k
    k = min(max(k_candidate,minimal_k),in_features)
    
    # if because of minimal_k or sample_ratio k equals the number of features, perform full matmul instead of approximating
    if k == in_features:
        # no need to sample. perform normal matmul
        if A.dim() == 2 and bias is not None:
            # fused op is marginally faster
            C = torch.addmm(bias, A, B)
        else:
            C = torch.matmul(A, B)
            if bias is not None:
                C += bias
        return C

    with torch.no_grad():
        # calculate norms of the columns of A and rows of B
        if A.dim() == 2:   
            a_col_norms = torch.norm(A,dim=0)
        else:
            # since we sample across in_featuers, consider other dimensions as a single dimension for sampling purpuses
            a_col_norms = torch.norm(A.view(-1,in_features),dim=0)    
        
        b_row_norms = torch.norm(B,dim=1)

        # multiply both norms element-wise
        norm_mult = torch.mul(a_col_norms,b_row_norms)
        sum_norm_mult = norm_mult.sum()       
 
        # calculate number of nonzero elements in norm_mult. this serves 
        # two purposes:
        # 1. Possibly reduce number of sampled pairs, as zero elements in norm_mult will not contribute to the result
        # 2. Prevents scaling of zero values
        nnz = (norm_mult!=0).sum()
        if nnz == 0:
            #print("zero multiply detected! scenario not optimzied (todo)")
            return torch.nn.functional.linear(A, B.t(),bias)
            
        k = min(k,nnz)
        
        prob_dist = k * torch.div(norm_mult,sum_norm_mult)
        prob_dist = prob_dist.clamp(min=0, max=1)

        # use epsilon-optimal sampling to allow learning random weights and to bound the scaling factor
        epsilon = 0.1
        if epsilon > 0:
            uniform = torch.ones_like(prob_dist)/in_features 
            prob_dist = (1-epsilon)*prob_dist + epsilon*uniform

        indices = torch.bernoulli(prob_dist).nonzero(as_tuple=True)[0]
        if len(indices) == 0:
            print("no elements selected - hmm")
            indices = torch.arange(k, device=device)
    
    # sample column-row pairs to form new smaller matrices
    A_top_k_cols = torch.index_select(A, dim=-1, index=indices)
    B_top_k_rows = torch.index_select(B, dim=0, index=indices)
    
    if scale == True:
        # scale column-row pairs by 1/(p_i) to get unbiased estimation
        with torch.no_grad():
            scale_factors = torch.div(1,prob_dist)
            scale_matrix = torch.diag(scale_factors[indices])
        A_top_k_cols = torch.matmul(A_top_k_cols, scale_matrix)
            
    # multiply smaller matrices
    if sample_ratio_bwd is None and sample_ratio_wu is None:
        if A.dim() == 2 and bias is not None:
            # fused op is marginally faster
            C_approx = torch.addmm(bias, A_top_k_cols, B_top_k_rows)
        else:
            C_approx = torch.matmul(A_top_k_cols, B_top_k_rows)
            if bias is not None:
                C_approx += bias
    else:
        # The following code will be used to apply additional sampling in the backward pass but update only the
        # sub-tensors sampled in the forward pass.
        # For simplicity, we don't optimize for torch.addmm usage in this case
        C_approx = matmul_approx_bwd_func.apply(A_top_k_cols, B_top_k_rows,sample_ratio_bwd,minimal_k_bwd,sample_ratio_wu,minimal_k_wu)
        if bias is not None:
            C_approx += bias
    
    return C_approx

# This function calculates exact matmul in the forward pass and approximate one in the backward pass
# it is indended to allow approximation of the sampled matrix multiply in approx_linear_func backward pass
# while updating all the matrix elements and not only the elements that were sampled in the forward pass 
class matmul_approx_bwd_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights, sample_ratio_bwd,minimal_k_bwd,sample_ratio_wu,minimal_k_wu):
        ctx.save_for_backward(inputs,weights)
        #store non-tensor objects in ctx
        ctx.sample_ratio_bwd = sample_ratio_bwd
        ctx.minimal_k_bwd = minimal_k_bwd
        ctx.sample_ratio_wu = sample_ratio_wu
        ctx.minimal_k_wu = minimal_k_wu
        return torch.matmul(inputs, weights)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weights = ctx.saved_tensors

        #print('calculating matmul_approx_bwd_func bwd pass! sample_ratio_bwd={},minimal_k_bwd={},sample_ratio_wu={},minimal_k_wu={}'.format(ctx.sample_ratio_bwd,ctx.minimal_k_bwd,ctx.sample_ratio_wu,ctx.minimal_k_wu))
        
        #grad_input = torch.matmul(grad_output, weights.t())
        grad_input = approx_linear_forward_xA_b(grad_output, weights.t(), None, ctx.sample_ratio_bwd, ctx.minimal_k_bwd,None,None,None,None)
        
        #grad_weight = torch.matmul(inputs.t(),grad_output)
        grad_weight = approx_linear_forward_xA_b(inputs.t(), grad_output, None, ctx.sample_ratio_wu, ctx.minimal_k_wu, None, None, None, None)
        
        return grad_input, grad_weight, None, None, None, None
