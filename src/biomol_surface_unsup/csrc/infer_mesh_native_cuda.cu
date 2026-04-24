#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <limits>
#include <vector>

namespace {

template <typename scalar_t>
__global__ void make_grid_block_kernel(
    const scalar_t* lo,
    const scalar_t* spacing,
    int64_t start_x,
    int64_t start_y,
    int64_t start_z,
    int64_t size_x,
    int64_t size_y,
    int64_t size_z,
    scalar_t* out
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = size_x * size_y * size_z;
    if (idx >= total) {
        return;
    }

    const int64_t yz = size_y * size_z;
    const int64_t ix = idx / yz;
    const int64_t rem = idx % yz;
    const int64_t iy = rem / size_z;
    const int64_t iz = rem % size_z;

    out[idx * 3 + 0] = lo[0] + static_cast<scalar_t>(start_x + ix) * spacing[0];
    out[idx * 3 + 1] = lo[1] + static_cast<scalar_t>(start_y + iy) * spacing[1];
    out[idx * 3 + 2] = lo[2] + static_cast<scalar_t>(start_z + iz) * spacing[2];
}

template <typename scalar_t>
__global__ void narrow_band_bbox_kernel(
    const scalar_t* sdf,
    int64_t size_x,
    int64_t size_y,
    int64_t size_z,
    double threshold,
    int64_t* bbox,
    int32_t* found
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = size_x * size_y * size_z;
    if (idx >= total) {
        return;
    }

    const double value = static_cast<double>(sdf[idx]);
    if (fabs(value) > threshold) {
        return;
    }
    atomicExch(found, 1);

    const int64_t yz = size_y * size_z;
    const int64_t x = idx / yz;
    const int64_t rem = idx % yz;
    const int64_t y = rem / size_z;
    const int64_t z = rem % size_z;

    atomicMin(reinterpret_cast<unsigned long long*>(&bbox[0]), static_cast<unsigned long long>(x));
    atomicMax(reinterpret_cast<unsigned long long*>(&bbox[1]), static_cast<unsigned long long>(x));
    atomicMin(reinterpret_cast<unsigned long long*>(&bbox[2]), static_cast<unsigned long long>(y));
    atomicMax(reinterpret_cast<unsigned long long*>(&bbox[3]), static_cast<unsigned long long>(y));
    atomicMin(reinterpret_cast<unsigned long long*>(&bbox[4]), static_cast<unsigned long long>(z));
    atomicMax(reinterpret_cast<unsigned long long*>(&bbox[5]), static_cast<unsigned long long>(z));
}

}  // namespace

at::Tensor make_grid_block_cuda(
    torch::Tensor lo,
    torch::Tensor spacing,
    std::vector<int64_t> start_indices,
    std::vector<int64_t> block_shape
) {
    auto lo_contig = lo.contiguous();
    auto spacing_contig = spacing.contiguous();
    const auto total = block_shape[0] * block_shape[1] * block_shape[2];
    auto out = torch::empty({total, 3}, lo.options());

    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(lo_contig.scalar_type(), "make_grid_block_cuda", [&] {
        make_grid_block_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
            lo_contig.data_ptr<scalar_t>(),
            spacing_contig.data_ptr<scalar_t>(),
            start_indices[0],
            start_indices[1],
            start_indices[2],
            block_shape[0],
            block_shape[1],
            block_shape[2],
            out.data_ptr<scalar_t>()
        );
    });

    return out;
}

at::Tensor narrow_band_bbox_cuda(torch::Tensor sdf_block, double threshold) {
    auto sdf = sdf_block.contiguous();
    TORCH_CHECK(sdf.dim() == 3, "sdf_block must be 3D");

    auto bbox_device = torch::tensor(
        {
            static_cast<int64_t>(sdf.size(0)),
            static_cast<int64_t>(-1),
            static_cast<int64_t>(sdf.size(1)),
            static_cast<int64_t>(-1),
            static_cast<int64_t>(sdf.size(2)),
            static_cast<int64_t>(0),
        },
        torch::TensorOptions().dtype(torch::kInt64).device(sdf.device())
    );
    auto found_device = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32).device(sdf.device()));

    const int64_t total = sdf.numel();
    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(sdf.scalar_type(), "narrow_band_bbox_cuda", [&] {
        narrow_band_bbox_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
            sdf.data_ptr<scalar_t>(),
            sdf.size(0),
            sdf.size(1),
            sdf.size(2),
            threshold,
            bbox_device.data_ptr<int64_t>(),
            found_device.data_ptr<int32_t>()
        );
    });

    auto bbox_cpu = bbox_device.to(torch::kCPU);
    auto found_cpu = found_device.to(torch::kCPU);
    if (found_cpu.data_ptr<int32_t>()[0] == 0) {
        auto* ptr = bbox_cpu.data_ptr<int64_t>();
        for (int i = 0; i < 6; ++i) {
            ptr[i] = -1;
        }
        return bbox_cpu;
    }
    auto* ptr = bbox_cpu.data_ptr<int64_t>();
    ptr[1] += 1;
    ptr[3] += 1;
    ptr[5] += 1;
    return bbox_cpu;
}
