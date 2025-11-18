#include "filtering.h"

unsigned int compute_dim(std::uint64_t global_size, int block_size)
{
	return static_cast<unsigned int>((global_size / block_size) + (global_size % block_size > 0 ? 1 : 0));
}


__global__ void gray_scale_kernel(
	std::uint32_t* dst_data,
	std::uint32_t* src_data, 
	std::uint64_t w, std::uint64_t h
)
{
	//TODO: 1.2) Implement conversion
}
void to_grayscale(gpu_image& dst, gpu_image const& src)
{
	dim3 block_size = { 32, 32 };
	dim3 grid_size = { compute_dim(src.width, block_size.x), compute_dim(src.height, block_size.y) };
	gray_scale_kernel<<<grid_size, block_size>>>(dst.data.get(), src.data.get(), src.width, src.height);
}


__global__ void convolution_kernel(
	std::uint32_t* dst_data,
	std::uint32_t* src_data, 
	std::uint64_t w, std::uint64_t h, 
	float* filter_data,
	std::uint64_t fw, std::uint64_t fh,
	bool use_abs_value
)
{
	//TODO: 1.3) Implement convolution
}
void apply_convolution(gpu_image& dst, gpu_image const& src, gpu_filter const& filter, bool use_abs_value)
{
	dim3 block_size = { 32, 32 };
	dim3 grid_size = { compute_dim(src.width, block_size.x), compute_dim(src.height, block_size.y) };
	convolution_kernel<<<grid_size, block_size>>>(dst.data.get(), src.data.get(), src.width, src.height, filter.data.get(), filter.width, filter.height, use_abs_value);
}


constexpr int num_threads = 64;
__global__ void histogram_kernel(
	std::uint32_t* hist_data,
	std::uint32_t* img_data,
	std::uint64_t w, std::uint64_t h,
	std::uint8_t channel_flags = 1
)
{
	//TODO: 1.4) Implement histogram computation
}
void compute_histogram(gpu_matrix<std::uint32_t>& hist, gpu_image const& img)
{
	std::uint8_t channel_flags = 1;
	cudaMemset(hist.data.get(), 0, hist.width * hist.height * sizeof(std::uint32_t));
	dim3 block_size = { num_threads };
	dim3 grid_size = { compute_dim(img.width, block_size.x) };
	histogram_kernel<<<grid_size, block_size>>>(hist.data.get(), img.data.get(), img.width, img.height, channel_flags);
}