#include <torch/extension.h>

/***********************************************************************
    Applies approximate linear transformation to the incoming data: y = xA + b.
    Note: A is assumed not transposed
    the matrix multiply xA is approximated
    Shape:
        - Input: (N, *, in\_features) where `*` means any number of
          additional dimensions
        - Weight: (in\_features, out\_features)
        - Bias: (out\_features)
        - Output: (N, *, out\_features)
**********************************************************************/

torch::Tensor approx_linear_xA_b_topk(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const float sample_ratio,
    const int64_t minimal_k) 
{

  
    //std::cout << "sample_ratio= " << sample_ratio << "minimal_k = " << minimal_k << std::endl;  
    
    //calculate the number of column-row pairs to sample
    const int64_t in_features = weight.sizes()[0];
    int64_t k_candidate = (int)((float)in_features*sample_ratio);
    
    //make k at least minimal_k
    int64_t k = std::min(std::max(k_candidate,minimal_k),in_features);  
   
    //std::cout <<"k = " << k << std::endl;
   
    // if no sampling is required, perform exact computation
    if (k == in_features) {
        std::cout << "no sampling needed, executing regular matmul" << std::endl;
        if (input.dim() == 2 && bias.numel() != 0) {
            std::cout << "matmul with bias" << std::endl;
            return torch::addmm(bias,input,weight);
        }
        else {
            auto C_approx = torch::matmul(input, weight);
            if (bias.numel() == 0) {
                std::cout << "matmul without bias" << std::endl;
                return C_approx;
            }
            else {
                return torch::add(C_approx, bias); 
            }
        }    
    } 

    // calculate norms of the columns of A and rows of B
    auto a_col_norms = (input.dim()==2) ? torch::frobenius_norm(input,{0}) : torch::frobenius_norm(input.view({-1,in_features}),{0});
    auto b_row_norms = torch::frobenius_norm(weight,{1});
    
    //  multiply both norms element-wise to and pick the indices of the top K column-row pairs
    auto norm_mult = torch::mul(a_col_norms, b_row_norms);
    
    // pick topk indices. as of pytorch v1.1.0, it turns out that torch.sort[:k] is faster than torch.topk
    //auto top_k_indices = std::get<1>(torch::topk(norm_mult, k, -1,true,false));
    auto top_k_indices = torch::argsort(norm_mult,0,true).narrow(0,0,k);

    // pick top-k column-row pairs to form new smaller matrices
    auto A_top_k_cols = torch::index_select(input, -1, top_k_indices);
    auto B_top_k_rows = torch::index_select(weight, 0, top_k_indices);

    //multiply smaller matrices
    if (input.dim() == 2 && bias.numel() != 0) {
        //std::cout << "matmul with bias" << std::endl;
        return torch::addmm(bias,A_top_k_cols,B_top_k_rows);
    }
    else {
        auto C_approx = torch::matmul(A_top_k_cols, B_top_k_rows);
        if (bias.numel() == 0) {
            std::cout << "matmul without bias" << std::endl;
            return C_approx;
        }
        else {
            return torch::add(C_approx, bias); 
        }
    }    
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("topk", &approx_linear_xA_b_topk, "Approx linear topk cpp");
}


