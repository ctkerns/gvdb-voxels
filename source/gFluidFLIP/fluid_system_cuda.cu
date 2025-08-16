// Christopher Kerns 2025
#include "fluid_system_cuda.cuh"

#include "fluid_params.h"
#include "fluid_utils.h"

__constant__ FluidParams fp;

extern "C"  __global__ void integrateParticles(float3 *pos, float3 *vel) {
  uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i > fp.numpnts)
    return;

  pos[i] += vel[i] * fp.dt;
  vel[i] += fp.dt * fp.gravity;
}

__global__ void handleParticleCollision(float3 *pos, float3 *vel) {
  uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i > fp.numpnts)
    return;

  if (pos[i].x < fp.h) {
    pos[i].x = fp.h;
    vel[i].x = 0.0f;
  } else if (pos[i].x >= (fp.gridres.x - 1) * fp.h) {
    pos[i].x = (fp.gridres.x - 1) * fp.h - 0.001f;
    vel[i].x = 0.0f;
  }
  if (pos[i].y < fp.h) {
    pos[i].y = fp.h;
    vel[i].y = 0.0f;
  } else if (pos[i].y >= (fp.gridres.y - 1) * fp.h) {
    pos[i].y = (fp.gridres.y - 1) * fp.h - 0.001f;
    vel[i].y = 0.0f;
  }
  if (pos[i].z < fp.h) {
    pos[i].z = fp.h;
    vel[i].z = 0.0f;
  } else if (pos[i].z >= (fp.gridres.z - 1) * fp.h) {
    pos[i].z = (fp.gridres.z - 1) * fp.h - 0.001f;
    vel[i].z = 0.0f;
  }
}

__global__ void transferToGrid(VDBInfo *gvdb, int num_sc, Component component,
                               int *sc_nid, int *sc_cnt, int *sc_off,
                               int3 *sc_pos, float3 *sc_pnt_pos,
                               float3 *sc_pnt_vel) {
  int sc_id = blockIdx.x;
  if (sc_id >= num_sc) return;

  int3 wpos = sc_pos[sc_id] + make_int3(threadIdx); // World voxel position.
  VDBNode *node = getNode(gvdb, 0, sc_nid[sc_id]);
  int3 vox = node->mValue + (wpos - node->mPos); // Atlas index of voxel.

  float3 val = make_float3(0.0f, 0.0f, 0.0f);
  float r = 0.0f;

  for (int j = 0; j < sc_cnt[sc_id]; j++) {
    float3 ppos = sc_pnt_pos[sc_off[sc_id] + j];

    // Only take into account particles within this cell.
    if (ppos.x >= wpos.x - 1 && ppos.x < wpos.x + 1 &&
        ppos.y >= wpos.y - 1 && ppos.y < wpos.y + 1 &&
        ppos.z >= wpos.z - 1 && ppos.z < wpos.z + 1) {
      float weight = (1.0f - abs(ppos.x - wpos.x)) *
                     (1.0f - abs(ppos.y - wpos.y)) *
                     (1.0f - abs(ppos.z - wpos.z));
      val += sc_pnt_vel[sc_off[sc_id] + j] * weight;
      r += weight;
    }
  }

  if (r > 0.0f) val /= r;

  switch (component) {
  case Component::X:
    val = make_float3(val.x, 0.0f, 0.0f);
    break;
  case Component::Y:
    val = make_float3(0.0f, val.y, 0.0f);
    val += fxyz(surf3Dread<float4>(gvdb->volOut[1], vox.x * sizeof(float4),
                                  vox.y, vox.z));
    break;
  case Component::Z:
    val = make_float3(0.0f, 0.0f, val.z);
    val += fxyz(surf3Dread<float4>(gvdb->volOut[1], vox.x * sizeof(float4),
                                  vox.y, vox.z));
    break;
  }

  surf3Dwrite(make_float4(val), gvdb->volOut[1], vox.x * sizeof(float4), vox.y,
              vox.z);
}

__global__ void transferFromGrid(VDBInfo *gvdb, int num_sc, Component component,
                                 int *sc_nid, int *sc_cnt, int *sc_off,
                                 int3 *sc_pos, float3 *sc_pnt_pos,
                                 uint *sc_pnt_clr, float3 *vel) {
  int sc_id = blockIdx.x;
  if (sc_id >= num_sc) return;

  int3 wpos = sc_pos[sc_id] + make_int3(threadIdx); // World voxel position.
  VDBNode *node = getNode(gvdb, 0, sc_nid[sc_id]);
  int3 vox = node->mValue + (wpos - node->mPos); // Atlas index of voxel.
  
  int3 cell[8];
  getNeighborCellIndices(vox, cell);

  // Velocities from each corner.
  float3 gridvel[8];
  for (int i = 0; i < 8; i++) {
    gridvel[i] = fxyz(surf3Dread<float4>(gvdb->volOut[1], cell[i].x * sizeof(float4), cell[i].y, cell[i].z));
  }

  int3 offset =
      make_int3(component == Component::X, component == Component::Y,
                component == Component::Z);

  bool valid[8];
  for (int i = 0; i < 8; i++) {
    CellType c1 = (CellType)surf3Dread<uchar>(gvdb->volOut[2], cell[i].x * sizeof(uchar), cell[i].y, cell[i].z);
    CellType c2 = (CellType)surf3Dread<uchar>(
        gvdb->volOut[2], (cell[i].x - offset.x) * sizeof(uchar),
        cell[i].y - offset.y, cell[i].z - offset.z);
    // valid[i] = !(c1 == CellType::Air && c2 == CellType::Air);
    // TODO: Temp until we start using cell types.
    valid[i] = (c1 == CellType::Air && c2 == CellType::Air);
  }

  for (int j = 0; j < sc_cnt[sc_id]; j++) {
    float3 ppos = sc_pnt_pos[sc_off[sc_id] + j];
    uint idx = sc_pnt_clr[sc_off[sc_id] + j];

    // Only take into account particles within this cell.
    if (ppos.x >= wpos.x && ppos.x < wpos.x + 1 &&
        ppos.y >= wpos.y && ppos.y < wpos.y + 1 &&
        ppos.z >= wpos.z && ppos.z < wpos.z + 1) {
      float3 v = getVelocityFromGridCell(ppos - make_float3(wpos), gridvel,
                                         valid, component);
      switch (component) {
      case Component::X:
        vel[idx] = make_float3(v.x, 0.0f, 0.0f);
        break;
      case Component::Y:
        vel[idx] += make_float3(0.0f, v.y, 0.0f);
        break;
      case Component::Z:
        vel[idx] += make_float3(0.0f, 0.0f, v.z);
        break;
      }
    }
  }
}
