#include <algorithm>   // std::min, std::max, std::clamp
#include <cmath>       // std::floor, std::sqrt, ...
#include <cstdio>      // printf / fprintf
#include <stdexcept>   // std::runtime_error
#include <string>
#include <vector>
#include <sstream>
#include <sleekslam/mapping/tsdf_gpu.hpp>

namespace sfs {
    namespace {
        void cudaCheck(cudaError_t e, const char* what) {
            if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
        }
        struct DeviceVoxel { float tsdf; unsigned short weight; unsigned short pad; };
        static_assert(sizeof(DeviceVoxel) == 8, "Unexpected device voxel layout");

        __global__ void resetKernel(DeviceVoxel* voxels, std::size_t n) {
            const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < n) { voxels[i].tsdf = 1.0f; voxels[i].weight = 0; voxels[i].pad = 0; }
        }

        __global__ void integrateKernel(DeviceVoxel* voxels, const float* depth, int w, int h,
            float fx, float fy, float cx, float cy, const float* T,
            int nx, int ny, int nz, float vx, float ox, float oy, float oz,
            float trunc, float minD, float maxD, int maxW) {
            const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            const std::size_t total = static_cast<std::size_t>(nx) * ny * nz;
            if (i >= total) return;
            const int x = static_cast<int>(i % nx);
            const int y = static_cast<int>((i / nx) % ny);
            const int z = static_cast<int>(i / (static_cast<std::size_t>(nx) * ny));
            const float wx = ox + (x + 0.5f) * vx, wy = oy + (y + 0.5f) * vx, wz = oz + (z + 0.5f) * vx;
            const float px = T[0] * wx + T[1] * wy + T[2] * wz + T[3];
            const float py = T[4] * wx + T[5] * wy + T[6] * wz + T[7];
            const float pz = T[8] * wx + T[9] * wy + T[10] * wz + T[11];
            if (!(pz > 0.0f)) return;
            const int u = __float2int_rn(fx * px / pz + cx), v = __float2int_rn(fy * py / pz + cy);
            if (u < 0 || u >= w || v < 0 || v >= h) return;
            const float d = depth[static_cast<std::size_t>(v) * w + u];
            if (!isfinite(d) || d < minD || d > maxD) return;
            const float sdf = d - pz;
            if (sdf < -trunc) return;
            const float obs = fminf(1.0f, fmaxf(-1.0f, sdf / trunc));
            DeviceVoxel& q = voxels[i];
            const int oldW = q.weight;
            if (oldW < maxW) {
                const int nw = oldW + 1;
                q.tsdf = (q.tsdf * oldW + obs) / nw;
                q.weight = static_cast<unsigned short>(nw);
            }
            else {
                q.tsdf += (obs - q.tsdf) / maxW;
            }
        }
    }

    struct TsdfGpuVolume::Impl {
        TsdfGpuConfig cfg;
        DeviceVoxel* voxels = nullptr;
        float* depth = nullptr;
        float* pose = nullptr;
        std::size_t voxelCount = 0, depthCapacity = 0;
        explicit Impl(const TsdfGpuConfig& c) : cfg(c) {
            if (c.nx <= 1 || c.ny <= 1 || c.nz <= 1 || c.voxelSize <= 0 || c.truncation <= 0 || c.maxWeight < 1)
                throw std::invalid_argument("Invalid TSDF GPU config");
            voxelCount = static_cast<std::size_t>(c.nx) * c.ny * c.nz;
            int devices = 0; cudaCheck(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
            if (devices == 0) throw std::runtime_error("No CUDA device found");
            cudaCheck(cudaMalloc(&voxels, voxelCount * sizeof(DeviceVoxel)), "cudaMalloc voxels");
            cudaCheck(cudaMalloc(&pose, 12 * sizeof(float)), "cudaMalloc pose");
            const int threads = 256;
            const int blocks = static_cast<int>((voxelCount + threads - 1) / threads);
            resetKernel << <blocks, threads >> > (voxels, voxelCount);
            cudaCheck(cudaGetLastError(), "reset kernel launch");
            cudaCheck(cudaDeviceSynchronize(), "reset kernel");
        }
        ~Impl() { if (depth) cudaFree(depth); if (pose) cudaFree(pose); if (voxels) cudaFree(voxels); }
    };

    TsdfGpuVolume::TsdfGpuVolume(const TsdfGpuConfig& c) : impl_(new Impl(c)) {}
    TsdfGpuVolume::~TsdfGpuVolume() { delete impl_; }
    TsdfGpuVolume::TsdfGpuVolume(TsdfGpuVolume&& o) noexcept : impl_(std::exchange(o.impl_, nullptr)) {}
    TsdfGpuVolume& TsdfGpuVolume::operator=(TsdfGpuVolume&& o) noexcept {
        if (this != &o) { delete impl_; impl_ = std::exchange(o.impl_, nullptr); } return *this;
    }
    void TsdfGpuVolume::reset() {
        if (!impl_) throw std::runtime_error("Moved-from TSDF volume");
        const int threads = 256, blocks = static_cast<int>((impl_->voxelCount + threads - 1) / threads);
        resetKernel << <blocks, threads >> > (impl_->voxels, impl_->voxelCount);
        cudaCheck(cudaGetLastError(), "reset kernel launch"); cudaCheck(cudaDeviceSynchronize(), "reset kernel");
    }
    void TsdfGpuVolume::integrate(const float* depth, int w, int h, float fx, float fy, float cx, float cy, const float Tcw[12]) {
        if (!impl_ || !depth || !Tcw || w <= 0 || h <= 0 || fx <= 0 || fy <= 0) throw std::invalid_argument("Invalid GPU integration input");
        const std::size_t n = static_cast<std::size_t>(w) * h;
        if (n > impl_->depthCapacity) {
            if (impl_->depth) cudaCheck(cudaFree(impl_->depth), "cudaFree old depth");
            cudaCheck(cudaMalloc(&impl_->depth, n * sizeof(float)), "cudaMalloc depth");
            impl_->depthCapacity = n;
        }
        cudaCheck(cudaMemcpy(impl_->depth, depth, n * sizeof(float), cudaMemcpyHostToDevice), "copy depth");
        cudaCheck(cudaMemcpy(impl_->pose, Tcw, 12 * sizeof(float), cudaMemcpyHostToDevice), "copy pose");
        const auto& c = impl_->cfg;
        const int threads = 256, blocks = static_cast<int>((impl_->voxelCount + threads - 1) / threads);
        integrateKernel << <blocks, threads >> > (impl_->voxels, impl_->depth, w, h, fx, fy, cx, cy, impl_->pose,
            c.nx, c.ny, c.nz, c.voxelSize, c.originX, c.originY, c.originZ, c.truncation, c.minDepth, c.maxDepth, c.maxWeight);
        cudaCheck(cudaGetLastError(), "integration kernel launch");
        cudaCheck(cudaDeviceSynchronize(), "integration kernel");
    }
    std::vector<TsdfCpuVoxel> TsdfGpuVolume::downloadVolume() const {
        if (!impl_) throw std::runtime_error("Moved-from TSDF volume");
        std::vector<DeviceVoxel> raw(impl_->voxelCount);
        cudaCheck(cudaMemcpy(raw.data(), impl_->voxels, raw.size() * sizeof(DeviceVoxel), cudaMemcpyDeviceToHost), "download TSDF");
        std::vector<TsdfCpuVoxel> out(raw.size());
        for (std::size_t i = 0; i < raw.size(); ++i) { out[i].tsdf = raw[i].tsdf; out[i].weight = raw[i].weight; }
        return out;
    }
    const TsdfGpuConfig& TsdfGpuVolume::config() const { return impl_->cfg; }
    std::size_t TsdfGpuVolume::deviceBytes() const {
        return impl_->voxelCount * sizeof(DeviceVoxel) + impl_->depthCapacity * sizeof(float) + 12 * sizeof(float);
    }
    std::string TsdfGpuVolume::deviceSummary() const {
        int dev = 0; cudaCheck(cudaGetDevice(&dev), "cudaGetDevice"); cudaDeviceProp p{};
        cudaCheck(cudaGetDeviceProperties(&p, dev), "cudaGetDeviceProperties");
        std::ostringstream s; s << p.name << ", compute " << p.major << '.' << p.minor << ", volume buffers " << (deviceBytes() / (1024 * 1024)) << " MiB";
        return s.str();
    }
} // namespace sfs
