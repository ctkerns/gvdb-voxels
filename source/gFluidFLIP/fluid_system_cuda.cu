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
}

__global__ void handleParticleCollision(float3 *pos, float3 *vel) {
  uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i > fp.numpnts)
    return;

  float min = fp.h + fp.radius;
  float maxX = (fp.gridres.x - 1) * fp.h - fp.radius;
  float maxY = (fp.gridres.y - 1) * fp.h - fp.radius;
  float maxZ = (fp.gridres.z - 1) * fp.h - fp.radius;

  if (pos[i].x < min) {
    pos[i].x = min;
    vel[i].x = 0.0f;
  } else if (pos[i].x >= maxX) {
    pos[i].x = maxX;
    vel[i].x = 0.0f;
  }
  if (pos[i].y < min) {
    pos[i].y = min;
    vel[i].y = 0.0f;
  } else if (pos[i].y >= maxY) {
    pos[i].y = maxY;
    vel[i].y = 0.0f;
  }
  if (pos[i].z < min) {
    pos[i].z = min;
    vel[i].z = 0.0f;
  } else if (pos[i].z >= maxZ) {
    pos[i].z = maxZ;
    vel[i].z = 0.0f;
  }
}

// TODO: Optimization: We are dispatching kernels over all the same subcells,
// so why not cache the GVDB nodes for reuse?

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
    ppos = offsetGrid(fp, ppos, component);

    // TODO this could be done easier.
    // Only take into account particles within 2x2x2 grid.
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
    val += fxyz(surf3Dread<float4>(gvdb->volOut[CHAN_VELOCITY],
                                   vox.x * sizeof(float4), vox.y, vox.z));
    break;
  case Component::Z:
    val = make_float3(0.0f, 0.0f, val.z);
    val += fxyz(surf3Dread<float4>(gvdb->volOut[CHAN_VELOCITY],
                                   vox.x * sizeof(float4), vox.y, vox.z));
    break;
  }

  surf3Dwrite(make_float4(val), gvdb->volOut[CHAN_VELOCITY],
              vox.x * sizeof(float4), vox.y, vox.z);
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
    gridvel[i] = fxyz(surf3Dread<float4>(gvdb->volOut[CHAN_VELOCITY],
                                         cell[i].x * sizeof(float4), cell[i].y,
                                         cell[i].z));
  }

  int3 offset =
      make_int3(component == Component::X, component == Component::Y,
                component == Component::Z);

  bool valid[8];
  for (int i = 0; i < 8; i++) {
    CellType c1 = (CellType)surf3Dread<uchar>(gvdb->volOut[CHAN_CELL_TYPE],
                                              cell[i].x * sizeof(uchar),
                                              cell[i].y, cell[i].z);
    CellType c2 = (CellType)surf3Dread<uchar>(
        gvdb->volOut[CHAN_CELL_TYPE], (cell[i].x - offset.x) * sizeof(uchar),
        cell[i].y - offset.y, cell[i].z - offset.z);
    valid[i] = !(c1 == CellType::Air && c2 == CellType::Air);
  }

  for (int j = 0; j < sc_cnt[sc_id]; j++) {
    float3 ppos = sc_pnt_pos[sc_off[sc_id] + j];
    ppos = offsetGrid(fp, ppos, component);

    // TODO: This could be done way easier.
    // Only take into account particles within this cell.
    if (ppos.x >= wpos.x && ppos.x < wpos.x + 1 &&
        ppos.y >= wpos.y && ppos.y < wpos.y + 1 &&
        ppos.z >= wpos.z && ppos.z < wpos.z + 1) {
      float3 v = getVelocityFromGridCell(ppos - make_float3(wpos), gridvel,
                                         valid, component);
      uint idx = sc_pnt_clr[sc_off[sc_id] + j];
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

__global__ void applyGravity(VDBInfo *gvdb, int num_sc, int *sc_nid,
                             int3 *sc_pos) {
  int sc_id = blockIdx.x;
  if (sc_id >= num_sc) return;

  int3 wpos = sc_pos[sc_id] + make_int3(threadIdx); // World voxel position.
  VDBNode *node = getNode(gvdb, 0, sc_nid[sc_id]);
  int3 vox = node->mValue + (wpos - node->mPos); // Atlas index of voxel.

  float3 val = fxyz(surf3Dread<float4>(gvdb->volOut[CHAN_VELOCITY],
                                       vox.x * sizeof(float4), vox.y, vox.z));

  if (wpos.x < 1 || wpos.x >= fp.gridres.x - 1) val.x = 0.0f;
  if (wpos.y < 1 || wpos.y >= fp.gridres.y - 1) val.y = 0.0f;
  if (wpos.z < 1 || wpos.z >= fp.gridres.z - 1) val.z = 0.0f;

  val += fp.gravity * fp.dt;
  surf3Dwrite(make_float4(val), gvdb->volOut[CHAN_VELOCITY],
              vox.x * sizeof(float4), vox.y, vox.z);
}

__global__ void markCells(VDBInfo *gvdb, int num_sc, int *sc_nid, int *sc_cnt,
                          int *sc_off, int3 *sc_pos, float3 *sc_pnt_pos) {
  int sc_id = blockIdx.x;
  if (sc_id >= num_sc) return;

  int3 wpos = sc_pos[sc_id] + make_int3(threadIdx); // World voxel position.
  VDBNode *node = getNode(gvdb, 0, sc_nid[sc_id]);
  int3 vox = node->mValue + (wpos - node->mPos); // Atlas index of voxel.

  for (int j = 0; j < sc_cnt[sc_id]; j++) {
    int3 ppos = make_int3(sc_pnt_pos[sc_off[sc_id] + j]);

    if (ppos.x == wpos.x && ppos.y == wpos.y && ppos.z == wpos.z) {
      surf3Dwrite((uchar) CellType::Fluid, gvdb->volOut[CHAN_CELL_TYPE],
                  vox.x * sizeof(uchar), vox.y, vox.z);
      return;
    }
  }
}

__global__ void computeDivergence(VDBInfo *gvdb, int num_sc, int *sc_nid,
                                  int3 *sc_pos) {
  int sc_id = blockIdx.x;
  if (sc_id >= num_sc) return;

  int3 wpos = sc_pos[sc_id] + make_int3(threadIdx); // World voxel position.
  VDBNode *node = getNode(gvdb, 0, sc_nid[sc_id]);
  int3 vox = node->mValue + (wpos - node->mPos); // Atlas index of voxel.

  float div = 0.0f;
  div += surf3Dread<float4>(gvdb->volOut[CHAN_VELOCITY],
                            (vox.x + 1) * sizeof(float4), vox.y, vox.z).x;
  div += surf3Dread<float4>(gvdb->volOut[CHAN_VELOCITY], vox.x * sizeof(float4),
                            vox.y + 1, vox.z).y;
  div += surf3Dread<float4>(gvdb->volOut[CHAN_VELOCITY],
                             vox.x * sizeof(float4), vox.y, vox.z + 1).z;
  float3 val = fxyz(surf3Dread<float4>(gvdb->volOut[CHAN_VELOCITY],
                            vox.x * sizeof(float4), vox.y, vox.z));
  div -= (val.x + val.y + val.z);
  div /= fp.h;

  surf3Dwrite(div, gvdb->volOut[CHAN_DIVERGENCE], vox.x * sizeof(float), vox.y,
              vox.z);
}

__global__ void solveJacobi(VDBInfo *gvdb, int num_sc, int p_chan,
                            int p_tmp_chan, int *sc_nid, int3 *sc_pos) {
  int sc_id = blockIdx.x;
  if (sc_id >= num_sc) return;

  int3 wpos = sc_pos[sc_id] + make_int3(threadIdx); // World voxel position.
  VDBNode *node = getNode(gvdb, 0, sc_nid[sc_id]);
  int3 vox = node->mValue + (wpos - node->mPos); // Atlas index of voxel.

  // Air cells have zero pressure.
  if (surf3Dread<uchar>(gvdb->volOut[CHAN_CELL_TYPE], vox.x * sizeof(uchar),
                        vox.y, vox.z) != CellType::Fluid) {
    surf3Dwrite(0.0f, gvdb->volOut[p_tmp_chan], vox.x * sizeof(float), vox.y,
                vox.z);
    return;
  }

  float s_sum = float(wpos.x > 1) + float(wpos.x < fp.gridres.x - 2) +
                float(wpos.y > 1) + float(wpos.y < fp.gridres.y - 2) +
                float(wpos.z > 1) + float(wpos.z < fp.gridres.z - 2);

  if (s_sum == 0.0f) return;

  float div = surf3Dread<float>(gvdb->volOut[CHAN_DIVERGENCE],
                                vox.x * sizeof(float), vox.y, vox.z);

  // Neighbor cells' pressures.
  float p_sum = 0.0f;
  p_sum += surf3Dread<float>(gvdb->volOut[p_chan], (vox.x - 1) * sizeof(float),
                             vox.y, vox.z);
  p_sum += surf3Dread<float>(gvdb->volOut[p_chan], (vox.x + 1) * sizeof(float),
                             vox.y, vox.z);
  p_sum += surf3Dread<float>(gvdb->volOut[p_chan], vox.x * sizeof(float),
                             vox.y - 1, vox.z);
  p_sum += surf3Dread<float>(gvdb->volOut[p_chan], vox.x * sizeof(float),
                             vox.y + 1, vox.z);
  p_sum += surf3Dread<float>(gvdb->volOut[p_chan], vox.x * sizeof(float), vox.y,
                             vox.z - 1);
  p_sum += surf3Dread<float>(gvdb->volOut[p_chan], vox.x * sizeof(float), vox.y,
                             vox.z + 1);

  float pressure = (p_sum - (fp.h * fp.h * (fp.density / fp.dt)) * div) / s_sum;
  surf3Dwrite(pressure, gvdb->volOut[p_tmp_chan], vox.x * sizeof(float), vox.y,
              vox.z);
}

__global__ void applyPressure(VDBInfo *gvdb, int num_sc, int *sc_nid, int3 *sc_pos) {
  int sc_id = blockIdx.x;
  if (sc_id >= num_sc) return;

  int3 wpos = sc_pos[sc_id] + make_int3(threadIdx); // World voxel position.
  VDBNode *node = getNode(gvdb, 0, sc_nid[sc_id]);
  int3 vox = node->mValue + (wpos - node->mPos); // Atlas index of voxel.

  float dt_div_rho_0_h = fp.dt / (fp.density * fp.h);

  float3 vel = fxyz(surf3Dread<float4>(gvdb->volOut[CHAN_VELOCITY],
                                      vox.x * sizeof(float4), vox.y, vox.z));

  float p  = surf3Dread<float>(gvdb->volOut[CHAN_PRESSURE],
                              vox.x * sizeof(float), vox.y, vox.z);

  // TODO: Why > gridres instead of >= gridres?
  if (wpos.x <= 1 || wpos.x > fp.gridres.x - 1) vel.x = 0.0f;
  else
    vel.x -=
        (p - surf3Dread<float>(gvdb->volOut[CHAN_PRESSURE],
                               (vox.x - 1) * sizeof(float), vox.y, vox.z)) *
        dt_div_rho_0_h;
  if (wpos.y <= 1 || wpos.y > fp.gridres.y - 1) vel.y = 0.0f;
  else
    vel.y -= (p - surf3Dread<float>(gvdb->volOut[CHAN_PRESSURE],
                                    vox.x * sizeof(float), vox.y - 1, vox.z)) *
             dt_div_rho_0_h;
  if (wpos.z <= 1 || wpos.z > fp.gridres.z - 1) vel.z = 0.0f;
  else
    vel.z -= (p - surf3Dread<float>(gvdb->volOut[CHAN_PRESSURE],
                                    vox.x * sizeof(float), vox.y, vox.z - 1)) *
             dt_div_rho_0_h;

  surf3Dwrite(make_float4(vel), gvdb->volOut[CHAN_VELOCITY],
              vox.x * sizeof(float4), vox.y, vox.z);
}
