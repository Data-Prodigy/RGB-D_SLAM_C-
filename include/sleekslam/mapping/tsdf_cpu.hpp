#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>   // std::size_t
#include <cstdint>   // std::uint8_t
#include <stdexcept>
#include <string>    // std::string
#include <vector>    // std::vector
namespace sfs {

    struct TsdfCpuConfig {
        int nx = 256;
        int ny = 256;
        int nz = 256;

        float voxelSize = 0.02f;

        // World-space position of the volume's minimum corner.
        float originX = -2.56f;
        float originY = -2.56f;
        float originZ = -1.06f;

        float truncation = 0.08f;
        float minDepth = 0.30f;
        float maxDepth = 5.0f;
        int maxWeight = 64;
    };

    struct TsdfCpuVoxel {
        float tsdf = 1.0f;
        std::uint16_t weight = 0;
    };

    class TsdfCpuVolume {
    public:
        explicit TsdfCpuVolume(const TsdfCpuConfig& cfg) : cfg_(cfg){
            if (cfg_.nx <= 1 || cfg_.ny <= 1 || cfg_.nz <= 1 || !(cfg_.voxelSize > 0.0f) ||!(cfg_.truncation > 0.0f) || cfg_.maxWeight <= 0) {
                throw std::invalid_argument("Invalid TSDF configuration");}

            const auto count = static_cast<std::size_t>(cfg_.nx) * static_cast<std::size_t>(cfg_.ny) * static_cast<std::size_t>(cfg_.nz);

            voxels_.resize(count);
        }

        void reset(){
            std::fill(voxels_.begin(), voxels_.end(), TsdfCpuVoxel{});
        }

        [[nodiscard]] const TsdfCpuConfig& config() const noexcept{
            return cfg_;
        }

        [[nodiscard]] const std::vector<TsdfCpuVoxel>& voxels() const noexcept{
            return voxels_;
        }

        [[nodiscard]] const TsdfCpuVoxel& at(int x, int y, int z) const{
            return voxels_.at(index(x, y, z));
        }

        // Depth is CV_32FC1 in metres.
        // T_cw is a row-major 3x4 world-to-camera rigid transform.
        // K is the pinhole intrinsic matrix for this depth image.
        void integrate(const float* depth, int width, int height, float fx, float fy, float cx, float cy, const float T_cw[12]){
            if (!depth || !T_cw || width <= 0 || height <= 0 ||
                !(fx > 0.0f) || !(fy > 0.0f)) {
                throw std::invalid_argument("Invalid TSDF integration input");
            }

            for (int z = 0; z < cfg_.nz; ++z) {
                for (int y = 0; y < cfg_.ny; ++y) {
                    for (int x = 0; x < cfg_.nx; ++x) {
                        const float wx = cfg_.originX + (x + 0.5f) * cfg_.voxelSize;
                        const float wy = cfg_.originY + (y + 0.5f) * cfg_.voxelSize;
                        const float wz = cfg_.originZ + (z + 0.5f) * cfg_.voxelSize;

                        const float px = T_cw[0] * wx + T_cw[1] * wy + T_cw[2] * wz + T_cw[3];
                        const float py = T_cw[4] * wx + T_cw[5] * wy + T_cw[6] * wz + T_cw[7];
                        const float pz = T_cw[8] * wx + T_cw[9] * wy + T_cw[10] * wz + T_cw[11];

                        if (!(pz > 0.0f)) {
                            continue;
                        }

                        const int u = static_cast<int>(std::lround(fx * px / pz + cx));
                        const int v = static_cast<int>(std::lround(fy * py / pz + cy));

                        if (u < 0 || u >= width || v < 0 || v >= height) {
                            continue;
                        }

                        const float measured = depth[
                            static_cast<std::size_t>(v) * width + u];

                        if (!std::isfinite(measured) || measured < cfg_.minDepth || measured > cfg_.maxDepth) {
                            continue;
                        }

                        const float sdf = measured - pz;

                        // Reject voxels well behind the observed surface.
                        if (sdf < -cfg_.truncation) {
                            continue;
                        }

                        const float observation = std::clamp(
                            sdf / cfg_.truncation, -1.0f, 1.0f);

                        auto& voxel = voxels_[index(x, y, z)];
                        const int oldWeight = voxel.weight;
                        const int newWeight =  std::min(oldWeight + 1, cfg_.maxWeight);

                        voxel.tsdf = (voxel.tsdf * static_cast<float>(oldWeight) + observation) / static_cast<float>(oldWeight + 1);

                        voxel.weight = static_cast<std::uint16_t>(newWeight);

                        // Once capped, use a bounded running average.
                        if (oldWeight >= cfg_.maxWeight) {
                            voxel.tsdf += (observation - voxel.tsdf) / static_cast<float>(cfg_.maxWeight);
                        }
                    }
                }
            }
        }

    private:
        [[nodiscard]] std::size_t index(int x, int y, int z) const{
            if (x < 0 || x >= cfg_.nx ||  y < 0 || y >= cfg_.ny || z < 0 || z >= cfg_.nz) {
                throw std::out_of_range("TSDF voxel index out of range");
            }

            return (static_cast<std::size_t>(z) * cfg_.ny + y) * cfg_.nx + x;
        }

        TsdfCpuConfig cfg_;
        std::vector<TsdfCpuVoxel> voxels_;
    };

} // namespace sfs
