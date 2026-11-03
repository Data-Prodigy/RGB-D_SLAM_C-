#pragma once
#include <sleekslam/mapping/tsdf_cpu.hpp>
#include <sleekslam/mapping/tsdf_gpu.hpp>

namespace sfs {
    inline bool saveMeshPly(const std::filesystem::path& path, const MeshData& mesh) {
        if (mesh.vertices.size() % 3 != 0) return false;
        std::ofstream f(path, std::ios::binary);
        if (!f) return false;
        const std::size_t nv = mesh.vertices.size(), nf = nv / 3;
        f << "ply\nformat binary_little_endian 1.0\n"
            << "element vertex " << nv << "\nproperty float x\nproperty float y\nproperty float z\n"
            << "property float nx\nproperty float ny\nproperty float nz\n"
            << "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            << "element face " << nf << "\nproperty list uchar int vertex_indices\nend_header\n";
        for (const auto& v : mesh.vertices) {
            const float p[6] = { v.x,v.y,v.z,v.nx,v.ny,v.nz };
            f.write(reinterpret_cast<const char*>(p), sizeof(p));
            const unsigned char rgb[3] = { v.r,v.g,v.b };
            f.write(reinterpret_cast<const char*>(rgb), 3);
        }
        for (std::uint32_t i = 0; i < nv; i += 3) {
            const std::uint8_t count = 3;
            const std::uint32_t ids[3] = { i,i + 1,i + 2 };
            f.write(reinterpret_cast<const char*>(&count), 1);
            f.write(reinterpret_cast<const char*>(ids), sizeof(ids));
        }
        return static_cast<bool>(f);
    }

    // CPU marching tetrahedra. This is deliberately the first reliable extractor;
    // GPU extraction can replace it after the TSDF is validated.
    inline MeshData extractMeshCpu(const TsdfCpuConfig& c, const std::vector<TsdfCpuVoxel>& vox,
        std::size_t maxTriangles = 1500000) {
        const auto idx = [&](int x, int y, int z) {return (static_cast<std::size_t>(z) * c.ny + y) * c.nx + x; };
        if (vox.size() != static_cast<std::size_t>(c.nx) * c.ny * c.nz) throw std::invalid_argument("Wrong volume size");
        MeshData out;
        const int tet[6][4] = { {0,5,1,6},{0,1,2,6},{0,2,3,6},{0,3,7,6},{0,7,4,6},{0,4,5,6} };
        const int corner[8][3] = { {0,0,0},{1,0,0},{1,1,0},{0,1,0},{0,0,1},{1,0,1},{1,1,1},{0,1,1} };
        struct P { float x, y, z, s; };
        auto interp = [](const P& a, const P& b) {
            float t = a.s / (a.s - b.s); t = std::max(0.0f, std::min(1.0f, t));
            return P{ a.x + t * (b.x - a.x),a.y + t * (b.y - a.y),a.z + t * (b.z - a.z),0 };
            };
        auto emit = [&](P a, P b, P d) {
            if (out.vertices.size() / 3 >= maxTriangles) { out.truncated = true; return; }
            // Flat normal, orientation corrected toward increasing TSDF (free space).
            float ux = b.x - a.x, uy = b.y - a.y, uz = b.z - a.z, vx = d.x - a.x, vy = d.y - a.y, vz = d.z - a.z;
            float nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
            float len = std::sqrt(nx * nx + ny * ny + nz * nz); if (len < 1e-12f)return;
            nx /= len; ny /= len; nz /= len;
            MeshVertex v[3] = { {a.x,a.y,a.z,nx,ny,nz,200,200,200,255},
                             {b.x,b.y,b.z,nx,ny,nz,200,200,200,255},
                             {d.x,d.y,d.z,nx,ny,nz,200,200,200,255} };
            out.vertices.insert(out.vertices.end(), v, v + 3);
            };
        for (int z = 0; z < c.nz - 1; ++z) for (int y = 0; y < c.ny - 1; ++y) for (int x = 0; x < c.nx - 1; ++x) {
            if (out.truncated) return out;
            P p[8]; bool valid = true;
            for (int k = 0; k < 8; ++k) {
                int xx = x + corner[k][0], yy = y + corner[k][1], zz = z + corner[k][2];
                const auto& q = vox[idx(xx, yy, zz)];
                if (q.weight == 0) { valid = false; break; }
                p[k] = { c.originX + (xx + 0.5f) * c.voxelSize,c.originY + (yy + 0.5f) * c.voxelSize,
                      c.originZ + (zz + 0.5f) * c.voxelSize,q.tsdf };
            }
            if (!valid) continue;
            for (const auto& t : tet) {
                P inside[4], outside[4]; int ni = 0, no = 0;
                for (int j = 0; j < 4; ++j) { if (p[t[j]].s < 0)inside[ni++] = p[t[j]]; else outside[no++] = p[t[j]]; }
                if (ni == 0 || ni == 4)continue;
                if (ni == 1) {
                    P a = interp(inside[0], outside[0]), b = interp(inside[0], outside[1]), d = interp(inside[0], outside[2]);
                    emit(a, b, d);
                }
                else if (ni == 3) {
                    P a = interp(outside[0], inside[0]), b = interp(outside[0], inside[1]), d = interp(outside[0], inside[2]);
                    emit(a, d, b);
                }
                else {
                    P a = interp(inside[0], outside[0]), b = interp(inside[0], outside[1]);
                    P d = interp(inside[1], outside[0]), e = interp(inside[1], outside[1]);
                    emit(a, b, d); emit(b, e, d);
                }
            }
        }
        return out;
    }
} // namespace sfs
