// pose.hpp  --  Step 2: how camera motion is represented.
//
// A pose is an element of SE(3) (Sophus::SE3d). Optimisation never touches the matrix
// directly: it moves a 6-vector xi in the tangent space and maps it back with exp().
//
//   Sophus tangent ordering:  xi = (rho, phi)   rho = translation part (3), phi = rotation part (3)
//   (same ordering as tau = (rho, theta) in the "micro Lie theory" paper)
//
// The update rule used by the g2o vertex is a LEFT perturbation:
//        T  <-  exp(delta) * T, and every Jacobian in this project is taken with respect to that delta.
#pragma once
#include "../src/pch.h"

namespace vo {

    using SE3 = Sophus::SE3d;
    using Vec3 = Eigen::Vector3d;
    using Vec6 = Eigen::Matrix<double, 6, 1>;
    using JacRow = Eigen::Matrix<double, 1, 6>;   // Jacobian of one scalar residual w.r.t. the 6 pose parameters

    [[nodiscard]] inline SE3 expMap(const Vec6& xi) { return SE3::exp(xi); }       // xi -> SE(3)
    [[nodiscard]] inline Vec6 logMap(const SE3& T) { return T.log(); }             // SE(3) -> xi
    [[nodiscard]] inline SE3 perturbLeft(const SE3& T, const Vec6& delta) { return SE3::exp(delta) * T; }

    [[nodiscard]] inline double rotationAngleDeg(const SE3& T) {
        return T.so3().log().norm() * 180.0 / std::numbers::pi;
    }

    // TUM trajectory format: "timestamp tx ty tz qx qy qz qw"   (pose of the camera in the world frame)
    [[nodiscard]] inline std::string tumLine(double stamp, const SE3& T_wc) {
        const Vec3& t = T_wc.translation();
        const Eigen::Quaterniond q = T_wc.unit_quaternion();
        return std::format("{:.6f} {:.6f} {:.6f} {:.6f} {:.8f} {:.8f} {:.8f} {:.8f}", stamp, t.x(), t.y(), t.z(), q.x(), q.y(), q.z(), q.w());
    }

}  // namespace vo