#include <sleekslam/mapping/fusion.hpp>


int main() {
    try {
        sfs::FusionConfig cfg;
        cfg.volume.nx = 96; cfg.volume.ny = 96; cfg.volume.nz = 96;
        cfg.volume.voxelSize = 0.03f;
        cfg.volume.originX = -1.44f; cfg.volume.originY = -1.44f; cfg.volume.originZ = 0.0f;
        cfg.volume.truncation = 0.09f; cfg.volume.minDepth = 0.2f; cfg.volume.maxDepth = 4.0f;
        cfg.maxMeshTriangles = 100000;
        sfs::FusionMapper mapper(cfg);
        std::cout << "GPU: " << mapper.deviceSummary() << "\n";
        constexpr int w = 160, h = 120;
        constexpr float fx = 140, fy = 140, cx = (w - 1) * 0.5f, cy = (h - 1) * 0.5f;
        std::vector<float> depth(static_cast<std::size_t>(w) * h, 1.5f);
        auto pose = sfs::identityPose();
        const bool fused = mapper.consider(depth.data(), w, h, fx, fy, cx, cy, pose, true);
        if (!fused) { std::cerr << "Initial synthetic frame was not fused\n"; return 2; }
        auto mesh = mapper.extractMesh();
        std::cout << "Fused frames: " << mapper.fusedFrames() << "\n"
            << "Mesh vertices: " << mesh.vertices.size() << "\n"
            << "Triangles: " << mesh.vertices.size() / 3 << "\n"
            << "Triangle buffer truncated: " << (mesh.truncated ? "yes" : "no") << "\n";
        if (mesh.vertices.empty()) { std::cerr << "No mesh extracted; inspect TSDF conventions/config\n"; return 3; }
        if (!sfs::saveMeshPly("phase2_plane_test.ply", mesh)) {
            std::cerr << "Failed to write PLY\n"; return 4;
        }
        std::cout << "Wrote phase2_plane_test.ply\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Phase 2 test failed: " << e.what() << "\n";
        return 1;
    }
}
