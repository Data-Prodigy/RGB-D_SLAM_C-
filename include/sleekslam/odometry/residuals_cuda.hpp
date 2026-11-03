#pragma once

#include <cstddef>

// Plain POD interface shared by MSVC and NVCC. Do not include Eigen/Sophus here.
namespace vo::cuda_backend {

    struct Camera { double fx, fy, cx, cy; };
    struct Pose { double R[9]; double t[3]; }; // row-major R
    struct PhotoSample { double p[3]; double intensity; };
    struct GeoSample { double p[3]; double n[3]; };
    struct GeoGate { double maxDist; double minNormalCos; };
    struct Result { double r; double J[6]; int valid; };

    struct LevelHandle;

    LevelHandle* createLevel(int width, int height, Camera K, const float* gray, const float* gradX, const float* gradY,
        const float* pointsXYZ, const float* normalsXYZ, const PhotoSample* photo, std::size_t photoCount, const GeoSample* geo, std::size_t geoCount);

    void destroyLevel(LevelHandle* handle) noexcept;

    // Evaluates both batches and accumulates robustified normal equations on the GPU.
    // H is row-major 6x6, b has 6 entries. When withJacobian=0 only cost is returned.
    // robustKind: 0=None, 1=Huber, 2=Cauchy. Information weights are 1/sigma^2.
    void evaluateNormalEquations(LevelHandle* handle, const Pose& pose, GeoGate gate, int withJacobian, double infoPhoto, double infoGeo,
        int robustKind, double robustDelta, double* H, double* b, double* cost);

} // namespace vo::cuda_backend
