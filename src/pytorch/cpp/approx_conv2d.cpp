#include <torch/extension.h>
#include <ATen/native/ConvUtils.h>

torch::Tensor approx_conv2d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::IntArrayRef stride,
    torch::IntArrayRef padding,
    torch::IntArrayRef dilation,
    int64_t groups,
    const float sample_ratio,
    const int64_t minimal_k,
    bool benchmark,
    bool deterministic,
    bool allow_tf32) {

    //std::cout << "sample_ratio= " << sample_ratio << "minimal_k = " << minimal_k << std::endl;  
    const int64_t ic = weight.sizes()[1];
    int64_t k_candidate = (int)((float)ic*sample_ratio);
    int64_t k = std::min(std::max(k_candidate,minimal_k),ic);  
   
    //std::cout <<"k = " << k << std::endl;
    
    //if k is equal to the input channels, avoid sampling overhead and call convolution directly
    if (k == ic) {
        //std::cout << "no sampling needed, executing regulst convolution" << std::endl;
        return torch::cudnn_convolution(input, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
    }
    else {
        //ver1: sampled norms

        //torch::autograd::GradMode::set_enabled(false);
	
	//auto input_select = torch::index_select(input,3,torch::randint(input.size(3),{1},torch::dtype(torch::kLong).device(input.device())));
        //input_select = torch::index_select(input_select,2,torch::randint(input_select.size(2),{1},torch::dtype(torch::kLong).device(input.device())));
        //auto weight_select = torch::index_select(weight,3,torch::randint(weight.size(3),{1},torch::dtype(torch::kLong).device(weight.device())));
        //weight_select = torch::index_select(weight_select,2,torch::randint(weight_select.size(2),{1},torch::dtype(torch::kLong).device(weight.device())));
        //auto input_select_ic_norms = torch::frobenius_norm(input_select, {0});
        //auto weight_select_ic_norms = torch::frobenius_norm(weight_select, {0});
        //auto norm_mult = torch::mul(input_select_ic_norms, weight_select_ic_norms).squeeze();
        //auto top_k_indices = std::get<1>(torch::topk(norm_mult, k, -1,true,false));
        
	//torch::autograd::GradMode::set_enabled(true);
        
        //std::cout << "input.sizes() = " << input.sizes() << std::endl;
        //std::cout << "weight.sizes() = " << weight.sizes() << std::endl;
        //std::cout << "bias.sizes() = " << weight.sizes() << std::endl;
        //std::cout << "input_select.sizes() = " << input_select.sizes() << std::endl;
        //std::cout << "weight_sapmled.sizes() = " << weight_select.sizes() << std::endl;
        //std::cout << "input_select_ic_norms.sizes() = " << input_select_ic_norms.sizes() << std::endl;
        //std::cout << "weight_select_ic_norms.sizes() = " << weight_select_ic_norms.sizes() << std::endl;
        //std::cout << "norm_mult.sizes() = " << norm_mult.sizes() << std::endl;
        
        //----------
        //ver2: full norms
        torch::autograd::GradMode::set_enabled(false);
        auto input_ic_norms = torch::frobenius_norm(input.view({input.sizes()[0],ic,-1}), {0,2});
        auto weight_ic_norms = torch::frobenius_norm(weight.view({weight.sizes()[0],ic,-1}), {0,2});
        auto norm_mult = torch::mul(input_ic_norms, weight_ic_norms);
        auto top_k_indices = std::get<1>(torch::topk(norm_mult, k, -1,true,false));
        torch::autograd::GradMode::set_enabled(true);
        //----------
        auto input_sampled = torch::index_select(input, 1, top_k_indices);//.set_requires_grad(false);
        auto weight_sampled = torch::index_select(weight, 1, top_k_indices);//.set_requires_grad(false);              
       
        //direct call to torch:cudnn_convolution is faster for 1x1 kernel sizes
        if (weight.sizes()[2] == 1 && weight.sizes()[3] == 1) {
            //std::cout << "1x1 kernel" << std::endl;
            if (bias.numel() == 0) {
    	        return torch::cudnn_convolution(input_sampled, weight_sampled, padding, stride, dilation, groups, benchmark, deterministic,allow_tf32);
	    }
	    else {
	    	return torch::cudnn_convolution(input_sampled, weight_sampled, padding, stride, dilation, groups, benchmark, deterministic,allow_tf32)+at::native::reshape_bias(input.dim(),bias);
	    }
	}
        else {
            //std::cout << "3x3 kernel" << std::endl;
            return torch::conv2d(input_sampled, weight_sampled, bias, stride, padding, dilation, groups);
        }
    }
}

// Note - the function signature uses the same pattern as torch.nn.grad.conv2d_input, which is different than torch::cudnn_convolution_backward_input
torch::Tensor approx_conv2d_backward_input(
    torch::IntArrayRef input_size,
    const torch::Tensor& weight,
    const torch::Tensor& grad_output,
    torch::IntArrayRef stride,
    torch::IntArrayRef padding,
    torch::IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32) {

  return torch::cudnn_convolution_backward_input(input_size, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);

}

// Note - the function signature uses the same pattern as torch.nn.grad.conv2d_weight, which is different than torch::cudnn_convolution_backward_weight
torch::Tensor approx_conv2d_backward_weight(
    const torch::Tensor& input,
    torch::IntArrayRef weight_size,
    const torch::Tensor& grad_output,
    torch::IntArrayRef stride,
    torch::IntArrayRef padding,
    torch::IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32) {

  //std::cout << "input.sizes() = " << input.sizes() << std::endl;
  //std::cout << "weight_size = " << weight_size << std::endl;
  //std::cout << "grad_output.sizes() = " << grad_output.sizes() << std::endl;
  //std::cout << "padding = " << padding << std::endl;
  //std::cout << "stride = " << stride << std::endl;
  //std::cout << "dilation = " << dilation << std::endl;
  //std::cout << "groups = " << groups << std::endl;
  return torch::cudnn_convolution_backward_weight(weight_size, grad_output, input, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &approx_conv2d_forward, "Approx Conv2d fowrard cudnn");
  m.def("backward_input", &approx_conv2d_backward_input, "Approx Conv2d backward inputs cudnn");
  m.def("backward_weight", &approx_conv2d_backward_weight, "Approx Conv2d backward weights cudnn");
}


