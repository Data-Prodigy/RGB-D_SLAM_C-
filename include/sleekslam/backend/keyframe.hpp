#pragma once
#include "../src/pch.h"

namespace sfs {
    // Row-major camera-to-world 4x4 rigid transform.
    struct Pose4f { float m[16]{}; };
    inline Pose4f identityPose() {
        Pose4f p{}; p.m[0] = p.m[5] = p.m[10] = p.m[15] = 1.0f; return p;
    }
    inline float poseTranslationDistance(const Pose4f& a, const Pose4f& b) {
        const float dx = a.m[3] - b.m[3], dy = a.m[7] - b.m[7], dz = a.m[11] - b.m[11];
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }
    inline float poseRotationDistance(const Pose4f& a, const Pose4f& b) {
        // Relative rotation R = R_a^T R_b; clamp trace-derived cosine.
        float tr = 0;
        for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) tr += a.m[r * 4 + c] * b.m[r * 4 + c];
        const float cosine = std::clamp((tr - 1.0f) * 0.5f, -1.0f, 1.0f);
        return std::acos(cosine);
    }
    struct KeyframeConfig { float minTranslation = 0.03f; float minRotationRadians = 0.05235988f; int maxFrameGap = 20; };
    class KeyframeSelector {
    public:
        explicit KeyframeSelector(KeyframeConfig c = {}) :cfg_(c) {}
        bool shouldFuse(const Pose4f& pose, bool trackingOk) const {
            if (!trackingOk) return false;
            if (!hasLast_) return true;
            return poseTranslationDistance(last_, pose) >= cfg_.minTranslation ||
                poseRotationDistance(last_, pose) >= cfg_.minRotationRadians ||
                framesSince_ >= cfg_.maxFrameGap;
        }
        void commitFusedPose(const Pose4f& pose) { last_ = pose; hasLast_ = true; framesSince_ = 0; }
        void advanceFrame() { ++framesSince_; }
        void reset() { hasLast_ = false; framesSince_ = 0; }
    private:
        KeyframeConfig cfg_; Pose4f last_{}; bool hasLast_ = false; int framesSince_ = 0;
    };
} // namespace sfs
