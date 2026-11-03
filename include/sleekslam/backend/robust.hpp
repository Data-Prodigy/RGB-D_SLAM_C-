// robust.hpp  -- this prevents one bad pixel from wrecking my pose estimate.
//
// g2o applies the kernel to the chi-square of each edge,  chi2 = e^2 / sigma^2  (information = 1/sigma^2).
// So `delta` is measured in WHITENED units, i.e. in multiples of the edge's sigma:
//
//     |e| / sigma <= delta   ->  behaves like ordinary least squares
//     |e| / sigma  > delta   ->  cost grows only linearly (outlier is down-weighted by delta/|e|)
//
// delta = 1.345 is the textbook Huber constant (95% efficiency on Gaussian noise).
#pragma once
#include "../src/pch.h"

namespace vo {

    enum class RobustKind { None, Huber, Cauchy };

    struct RobustConfig {
        RobustKind kind = RobustKind::Huber;
        double delta = 1.345;
    };

    // Returns a NEW kernel (ownership passes to the edge via setRobustKernel), or nullptr for plain least squares.
    [[nodiscard]] inline g2o::RobustKernel* makeRobustKernel(const RobustConfig& cfg) {
        switch (cfg.kind) {
        case RobustKind::Huber: {
            auto* k = new g2o::RobustKernelHuber();
            k->setDelta(cfg.delta);
            return k;
        }
        case RobustKind::Cauchy: {
            auto* k = new g2o::RobustKernelCauchy();
            k->setDelta(cfg.delta);
            return k;
        }
        case RobustKind::None:
            break;
        }
        return nullptr;
    }

}  // namespace vo