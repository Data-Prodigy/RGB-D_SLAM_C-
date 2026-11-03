// g2o_types.hpp  --  the g2o graph for one frame-to-frame alignment.
//
//   VertexPose        : T_cr in SE(3), updated by the LEFT perturbation  T <- exp(delta) * T
//   EdgePhotometric   : one scalar brightness residual per selected pixel   (Step 3)
//   EdgeGeometric     : one scalar point-to-plane residual per selected point (Step 4)
//
// The graph is "one vertex + many unary edges". The residual math lives in photometric.hpp /
// geometric.hpp; these classes only adapt it to g2o.
//
// An edge whose point leaves the image or loses its correspondence reports error 0 and a zero
// Jacobian, i.e. it simply stops voting for that iteration.
#pragma once
#include "../src/pch.h"
#include <sleekslam/odometry/geometric.hpp>
#include <sleekslam/odometry/photometric.hpp>

namespace vo {

    class VertexPose final : public g2o::BaseVertex<6, SE3> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            void setToOriginImpl() override { _estimate = SE3(); }

        void oplusImpl(const double* update) override {
            const Eigen::Map<const Vec6> delta(update);
            _estimate = perturbLeft(_estimate, delta);
        }

        bool read(std::istream&) override { return false; }
        bool write(std::ostream&) const override { return false; }
    };

    class EdgePhotometric final : public g2o::BaseUnaryEdge<1, double, VertexPose> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            // _measurement = brightness of the pixel in the reference image
            EdgePhotometric(const PyramidLevel* cur, Vec3 pRef) : cur_(cur), pRef_(std::move(pRef)) {}

        void computeError() override {
            const auto* v = static_cast<const VertexPose*>(_vertices[0]);
            const auto e = photometricEval(v->estimate(), { pRef_, _measurement }, *cur_, false);
            _error[0] = e ? e->r : 0.0;
        }

        void linearizeOplus() override {
            const auto* v = static_cast<const VertexPose*>(_vertices[0]);
            const auto e = photometricEval(v->estimate(), { pRef_, _measurement }, *cur_, true);
            _jacobianOplusXi = e ? e->J : JacRow::Zero();
        }

        bool read(std::istream&) override { return false; }
        bool write(std::ostream&) const override { return false; }

    private:
        const PyramidLevel* cur_;
        Vec3 pRef_;
    };

    class EdgeGeometric final : public g2o::BaseUnaryEdge<1, double, VertexPose> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            // _measurement is unused (the target distance is 0)
            EdgeGeometric(const PyramidLevel* cur, GeoSample sample, GeoGate gate) : cur_(cur), sample_(std::move(sample)), gate_(gate) {
        }

        void computeError() override {
            const auto* v = static_cast<const VertexPose*>(_vertices[0]);
            const auto e = geometricEval(v->estimate(), sample_, *cur_, gate_, false);
            _error[0] = e ? e->r : 0.0;
        }

        void linearizeOplus() override {
            const auto* v = static_cast<const VertexPose*>(_vertices[0]);
            const auto e = geometricEval(v->estimate(), sample_, *cur_, gate_, true);
            _jacobianOplusXi = e ? e->J : JacRow::Zero();
        }

        bool read(std::istream&) override { return false; }
        bool write(std::ostream&) const override { return false; }

    private:
        const PyramidLevel* cur_;
        GeoSample sample_;
        GeoGate gate_;
    };

}  // namespace vo