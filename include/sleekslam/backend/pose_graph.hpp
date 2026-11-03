#pragma once

#include "../src/pch.h"

namespace sfs {

    using GraphPose = Sophus::SE3d;
    using GraphVec6 = Eigen::Matrix<double, 6, 1>;

    class PoseVertex final : public g2o::BaseVertex<6, GraphPose> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            void setToOriginImpl() override {
            _estimate = GraphPose();
        }

        void oplusImpl(const double* update) override {
            Eigen::Map<const GraphVec6> delta(update);
            // Matches the left-perturbation convention used by this project.
            _estimate = GraphPose::exp(delta) * _estimate;
        }

        bool read(std::istream&) override { return false; }
        bool write(std::ostream&) const override { return false; }
    };

    class RelativePoseEdge final
        : public g2o::BaseBinaryEdge<6, GraphPose, PoseVertex, PoseVertex> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            void computeError() override {
            const auto* vi =
                static_cast<const PoseVertex*>(_vertices[0]);
            const auto* vj =
                static_cast<const PoseVertex*>(_vertices[1]);

            const GraphPose predicted =
                vi->estimate().inverse() * vj->estimate();

            _error = (_measurement.inverse() * predicted).log();
        }

        // Numerical Jacobians avoid depending on a particular Sophus
        // analytic-Jacobian implementation. Replace with analytic ones later.
        void linearizeOplus() override {
            auto* vi = static_cast<PoseVertex*>(_vertices[0]);
            auto* vj = static_cast<PoseVertex*>(_vertices[1]);

            const GraphPose originalI = vi->estimate();
            const GraphPose originalJ = vj->estimate();
            constexpr double eps = 1e-6;

            for (int k = 0; k < 6; ++k) {
                GraphVec6 delta = GraphVec6::Zero();
                delta[k] = eps;

                vi->setEstimate(GraphPose::exp(delta) * originalI);
                computeError();
                const GraphVec6 plusI = _error;

                vi->setEstimate(GraphPose::exp(-delta) * originalI);
                computeError();
                const GraphVec6 minusI = _error;

                vi->setEstimate(originalI);
                _jacobianOplusXi.col(k) = (plusI - minusI) / (2.0 * eps);
                vj->setEstimate(GraphPose::exp(delta) * originalJ);
                computeError();
                const GraphVec6 plusJ = _error;

                vj->setEstimate(GraphPose::exp(-delta) * originalJ);
                computeError();
                const GraphVec6 minusJ = _error;

                vj->setEstimate(originalJ);
                _jacobianOplusXj.col(k) = (plusJ - minusJ) / (2.0 * eps);
            }

            vi->setEstimate(originalI);
            vj->setEstimate(originalJ);
            computeError();
        }

        bool read(std::istream&) override { return false; }
        bool write(std::ostream&) const override { return false; }
    };

    class PoseGraph {
    public:
        PoseGraph() {
            using BlockSolver = g2o::BlockSolver<
                g2o::BlockSolverTraits<6, 6>>;
            using LinearSolver =
                g2o::LinearSolverEigen<BlockSolver::PoseMatrixType>;

            auto linearSolver = std::make_unique<LinearSolver>();
            auto blockSolver = std::make_unique<BlockSolver>(std::move(linearSolver));

            optimizer_.setAlgorithm(new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver)));

            optimizer_.setVerbose(false);
        }

        bool addPose(int id, const GraphPose& pose, bool fixed = false) {
            if (optimizer_.vertex(id))
                return false;

            auto* vertex = new PoseVertex();
            vertex->setId(id);
            vertex->setEstimate(pose);
            vertex->setFixed(fixed);

            if (!optimizer_.addVertex(vertex)) {
                delete vertex;
                return false;
            }
            return true;
        }

        // Measurement convention: Z_ij = T_wc_i^-1 * T_wc_j.
        bool addSequentialEdge(int idI, int idJ, const GraphPose& measurement, double informationWeight = 1.0) {
            return addEdge(idI, idJ, measurement, informationWeight, false);}

        bool addLoopEdge(int idI, int idJ, const GraphPose& measurement, double informationWeight = 1.0, double huberDelta = 1.0) {
            return addEdge(idI, idJ, measurement, informationWeight, true, huberDelta);}

        int optimize(int iterations = 30) {
            if (optimizer_.vertices().empty())
                return 0;

            optimizer_.initializeOptimization();
            return optimizer_.optimize(iterations);
        }

        bool getPose(int id, GraphPose& pose) const {
            const auto* vertex = dynamic_cast<const PoseVertex*>(optimizer_.vertex(id));
            if (!vertex)
                return false;

            pose = vertex->estimate();
            return true;
        }

        std::size_t size() const {
            return optimizer_.vertices().size();
        }

    private:
        bool addEdge(int idI, int idJ, const GraphPose& measurement, double weight, bool robust, double huberDelta = 1.0) {
            auto* vi = dynamic_cast<PoseVertex*>(optimizer_.vertex(idI));
            auto* vj = dynamic_cast<PoseVertex*>(optimizer_.vertex(idJ));

            if (!vi || !vj || idI == idJ || weight <= 0.0)
                return false;

            auto* edge = new RelativePoseEdge();
            edge->setVertex(0, vi);
            edge->setVertex(1, vj);
            edge->setMeasurement(measurement);
            edge->setInformation(
                Eigen::Matrix<double, 6, 6>::Identity() * weight);

            if (robust) {
                auto* kernel = new g2o::RobustKernelHuber();
                kernel->setDelta(huberDelta);
                edge->setRobustKernel(kernel);
            }

            if (!optimizer_.addEdge(edge)) {
                delete edge;
                return false;
            }
            return true;
        }

        g2o::SparseOptimizer optimizer_;
    };

} // namespace sfs