#pragma once
#include <sleekslam/backend/keyframe.hpp>
#include <sleekslam/mapping/mesh_io.hpp>
#include <sleekslam/mapping/tsdf_gpu.hpp>

namespace sfs {
    struct FusionConfig {
        TsdfGpuConfig volume{};
        KeyframeConfig keyframes{};
        std::size_t maxMeshTriangles = 500000;
    };
    inline std::array<float, 12> worldToCamera3x4(const Pose4f& p) {
        // Invert rigid T_wc: R_cw=R_wc^T, t_cw=-R_wc^T*t_wc.
        std::array<float, 12> t{};
        for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) t[r * 4 + c] = p.m[c * 4 + r];
        for (int r = 0; r < 3; ++r) t[r * 4 + 3] = -(t[r * 4] * p.m[3] + t[r * 4 + 1] * p.m[7] + t[r * 4 + 2] * p.m[11]);
        return t;
    }
    class FusionMapper {
    public:
        explicit FusionMapper(const FusionConfig& c) :cfg_(c), volume_(c.volume), selector_(c.keyframes) {}
        // Depth must be contiguous float32 metres; intrinsics must match that depth image.
        // T_wc is camera-to-world. trackingOk must reflect actual VO health.
        bool consider(const float* depth, int w, int h, float fx, float fy, float cx, float cy, const Pose4f& T_wc, bool trackingOk) {
            selector_.advanceFrame();
            if (!selector_.shouldFuse(T_wc, trackingOk)) return false;
            if (!depth || w <= 0 || h <= 0 || fx <= 0 || fy <= 0) return false;
            const auto Tcw = worldToCamera3x4(T_wc);
            volume_.integrate(depth, w, h, fx, fy, cx, cy, Tcw.data());
            selector_.commitFusedPose(T_wc); ++fusedFrames_;
            return true;
        }
        
            // Reintegrate a previously selected keyframe without applying
            // the keyframe-selection thresholds a second time.
            bool integrateKeyframe(const float* depth,int w, int h,float fx, float fy, float cx, float cy, const Pose4f& T_wc){
            if (!depth || w <= 0 || h <= 0 || fx <= 0 || fy <= 0)
                return false;

            const auto T_cw = worldToCamera3x4(T_wc);
            volume_.integrate(
                depth, w, h, fx, fy, cx, cy, T_cw.data());

            ++fusedFrames_;
            return true;
        }
        
        MeshData extractMesh() const {
            const auto v = volume_.downloadVolume();
            return extractMeshCpu(volume_.config(), v, cfg_.maxMeshTriangles);
        }
        bool savePly(const std::filesystem::path& p) const { return saveMeshPly(p, extractMesh()); }
        void reset() { volume_.reset(); selector_.reset(); fusedFrames_ = 0; }
        std::size_t fusedFrames() const { return fusedFrames_; }
        std::size_t deviceBytes() const { return volume_.deviceBytes(); }
        std::string deviceSummary() const { return volume_.deviceSummary(); }
    private:
        FusionConfig cfg_;
        TsdfGpuVolume volume_;
        KeyframeSelector selector_;
        std::size_t fusedFrames_ = 0;
    };
} // namespace sfs
