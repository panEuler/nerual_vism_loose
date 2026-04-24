#include <torch/extension.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

at::Tensor make_grid_block_cuda(
    torch::Tensor lo,
    torch::Tensor spacing,
    std::vector<int64_t> start_indices,
    std::vector<int64_t> block_shape
);

at::Tensor narrow_band_bbox_cuda(torch::Tensor sdf_block, double threshold);

namespace {

void check_grid_args(
    const torch::Tensor& lo,
    const torch::Tensor& spacing,
    const std::vector<int64_t>& start_indices,
    const std::vector<int64_t>& block_shape
) {
    TORCH_CHECK(lo.numel() == 3, "lo must have exactly 3 elements");
    TORCH_CHECK(spacing.numel() == 3, "spacing must have exactly 3 elements");
    TORCH_CHECK(start_indices.size() == 3, "start_indices must have length 3");
    TORCH_CHECK(block_shape.size() == 3, "block_shape must have length 3");
}

at::Tensor make_grid_block_cpu(
    torch::Tensor lo,
    torch::Tensor spacing,
    const std::vector<int64_t>& start_indices,
    const std::vector<int64_t>& block_shape
) {
    auto lo_contig = lo.contiguous();
    auto spacing_contig = spacing.contiguous();
    const auto total = block_shape[0] * block_shape[1] * block_shape[2];
    auto out = torch::empty({total, 3}, lo.options().device(torch::kCPU));

    AT_DISPATCH_FLOATING_TYPES(lo_contig.scalar_type(), "make_grid_block_cpu", [&] {
        const auto* lo_ptr = lo_contig.data_ptr<scalar_t>();
        const auto* spacing_ptr = spacing_contig.data_ptr<scalar_t>();
        auto* out_ptr = out.data_ptr<scalar_t>();

        int64_t linear = 0;
        for (int64_t ix = 0; ix < block_shape[0]; ++ix) {
            const scalar_t x_val = lo_ptr[0] + static_cast<scalar_t>(start_indices[0] + ix) * spacing_ptr[0];
            for (int64_t iy = 0; iy < block_shape[1]; ++iy) {
                const scalar_t y_val = lo_ptr[1] + static_cast<scalar_t>(start_indices[1] + iy) * spacing_ptr[1];
                for (int64_t iz = 0; iz < block_shape[2]; ++iz, ++linear) {
                    out_ptr[linear * 3 + 0] = x_val;
                    out_ptr[linear * 3 + 1] = y_val;
                    out_ptr[linear * 3 + 2] =
                        lo_ptr[2] + static_cast<scalar_t>(start_indices[2] + iz) * spacing_ptr[2];
                }
            }
        }
    });

    return out.to(lo.device());
}

at::Tensor narrow_band_bbox_cpu(torch::Tensor sdf_block, double threshold) {
    auto sdf = sdf_block.contiguous().to(torch::kCPU);
    TORCH_CHECK(sdf.dim() == 3, "sdf_block must be 3D");

    int64_t min_x = sdf.size(0);
    int64_t max_x = -1;
    int64_t min_y = sdf.size(1);
    int64_t max_y = -1;
    int64_t min_z = sdf.size(2);
    int64_t max_z = -1;

    AT_DISPATCH_FLOATING_TYPES(sdf.scalar_type(), "narrow_band_bbox_cpu", [&] {
        auto accessor = sdf.accessor<scalar_t, 3>();
        for (int64_t x = 0; x < sdf.size(0); ++x) {
            for (int64_t y = 0; y < sdf.size(1); ++y) {
                for (int64_t z = 0; z < sdf.size(2); ++z) {
                    if (std::abs(static_cast<double>(accessor[x][y][z])) <= threshold) {
                        min_x = std::min(min_x, x);
                        max_x = std::max(max_x, x);
                        min_y = std::min(min_y, y);
                        max_y = std::max(max_y, y);
                        min_z = std::min(min_z, z);
                        max_z = std::max(max_z, z);
                    }
                }
            }
        }
    });

    auto out = torch::empty({6}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    auto* out_ptr = out.data_ptr<int64_t>();
    if (max_x < 0) {
        for (int i = 0; i < 6; ++i) {
            out_ptr[i] = -1;
        }
        return out;
    }
    out_ptr[0] = min_x;
    out_ptr[1] = max_x + 1;
    out_ptr[2] = min_y;
    out_ptr[3] = max_y + 1;
    out_ptr[4] = min_z;
    out_ptr[5] = max_z + 1;
    return out;
}

}  // namespace

at::Tensor make_grid_block(
    torch::Tensor lo,
    torch::Tensor spacing,
    std::vector<int64_t> start_indices,
    std::vector<int64_t> block_shape
) {
    check_grid_args(lo, spacing, start_indices, block_shape);
    if (lo.is_cuda()) {
#ifdef WITH_CUDA
        return make_grid_block_cuda(lo, spacing, std::move(start_indices), std::move(block_shape));
#else
        throw std::runtime_error("make_grid_block received CUDA tensors but extension was built without CUDA");
#endif
    }
    return make_grid_block_cpu(lo, spacing, start_indices, block_shape);
}

at::Tensor narrow_band_bbox(torch::Tensor sdf_block, double threshold) {
    if (sdf_block.is_cuda()) {
#ifdef WITH_CUDA
        return narrow_band_bbox_cuda(sdf_block, threshold);
#else
        throw std::runtime_error("narrow_band_bbox received CUDA tensors but extension was built without CUDA");
#endif
    }
    return narrow_band_bbox_cpu(sdf_block, threshold);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("make_grid_block", &make_grid_block, "Generate a flattened xyz query block");
    m.def("narrow_band_bbox", &narrow_band_bbox, "Find the bbox of |sdf| <= threshold");
}
