#pragma once
#include <cstddef>   // std::size_t
#include <cstdint>   // std::uint8_t
#include <string>    // std::string
#include <vector>    // std::vector
#include <sleekslam/mapping/tsdf_cpu.hpp>


namespace sfs {
    struct TsdfGpuConfig : TsdfCpuConfig {};
    struct MeshVertex {
        float x, y, z;
        float nx, ny, nz;
        std::uint8_t r, g, b, a;
    };
    struct MeshData {
        std::vector<MeshVertex> vertices; // triangle soup: every 3 vertices form a triangle
        bool truncated = false;
    };
    class TsdfGpuVolume {
    public:
        explicit TsdfGpuVolume(const TsdfGpuConfig& config);
        ~TsdfGpuVolume();
        TsdfGpuVolume(TsdfGpuVolume&&) noexcept;
        TsdfGpuVolume& operator=(TsdfGpuVolume&&) noexcept;
        TsdfGpuVolume(const TsdfGpuVolume&) = delete;
        TsdfGpuVolume& operator=(const TsdfGpuVolume&) = delete;

        void reset();
        void integrate(const float* depth, int width, int height, float fx, float fy, float cx, float cy, const float T_cw[12]);
        std::vector<TsdfCpuVoxel> downloadVolume() const;
        const TsdfGpuConfig& config() const;
        std::size_t deviceBytes() const;
        std::string deviceSummary() const;
    private:
        struct Impl;
        Impl* impl_ = nullptr;
    };
} // namespace sfs
