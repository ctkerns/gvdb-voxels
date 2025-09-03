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

  if (pos[i].x < fp.tankMin.x) {
    pos[i].x = fp.tankMin.x;
    vel[i].x = 0.0f;
  } else if (pos[i].x > fp.tankMax.x) {
    pos[i].x = fp.tankMax.x;
    vel[i].x = 0.0f;
  }
  if (pos[i].y < fp.tankMin.y) {
    pos[i].y = fp.tankMin.y;
    vel[i].y = 0.0f;
  } else if (pos[i].y > fp.tankMax.y) {
    pos[i].y = fp.tankMax.y;
    vel[i].y = 0.0f;
  }
  if (pos[i].z < fp.tankMin.z) {
    pos[i].z = fp.tankMin.z;
    vel[i].z = 0.0f;
  } else if (pos[i].z > fp.tankMax.z) {
    pos[i].z = fp.tankMax.z;
    vel[i].z = 0.0f;
  }
}

__device__ void addVelocityFromParticle(float3 ppos, int3 wpos,
                                        float3 sc_pnt_vel, float3 &vel,
                                        float3 &r, Component component) {
  ppos -= offsetGrid(component);

  // Only take into account particles within 2x2x2 grid.
  if (ppos.x >= wpos.x - 1 && ppos.x < wpos.x + 1 &&
      ppos.y >= wpos.y - 1 && ppos.y < wpos.y + 1 &&
      ppos.z >= wpos.z - 1 && ppos.z < wpos.z + 1) {
    float weight = (1.0f - abs(ppos.x - wpos.x)) *
                   (1.0f - abs(ppos.y - wpos.y)) *
                   (1.0f - abs(ppos.z - wpos.z));
    switch (component) {
      case Component::X:
        vel.x += sc_pnt_vel.x * weight;
        r.x += weight;
        break;
      case Component::Y:
        vel.y += sc_pnt_vel.y * weight;
        r.y += weight;
        break;
      case Component::Z:
        vel.z += sc_pnt_vel.z * weight;
        r.z += weight;
        break;
    }
  }
}

__device__ void getVelocityFromGrid(float3 ppos, int3 wpos, uint idx,
                                    float3 *gridvel, bool *valid, float3 *vel,
                                    Component component) {
  ppos -= offsetGrid(component);

  // Only take into account particles within this cell.
  if (int(ppos.x) == wpos.x && int(ppos.y) == wpos.y && int(ppos.z) == wpos.z) {
    float3 v = getVelocityFromGridCell(ppos - make_float3(wpos),
                                       gridvel, valid, component);
    switch (component) {
    case Component::X:
      vel[idx].x = v.x;
      break;
    case Component::Y:
      vel[idx].y = v.y;
      break;
    case Component::Z:
      vel[idx].z = v.z;
      break;
    }
  }
}

__device__ void getValidCells(VDBInfo *gvdb, int3 *cell, bool *valid,
                              Component component) {
  int3 offset =
      make_int3(component == Component::X, component == Component::Y,
                component == Component::Z);

  for (int i = 0; i < 8; i++) {
    // TODO: This could be more efficient?
    CellType c1 = (CellType)surf3Dread<uchar>(gvdb->volOut[CHAN_CELL_TYPE],
                                              cell[i].x * sizeof(uchar),
                                              cell[i].y, cell[i].z);
    CellType c2 = (CellType)surf3Dread<uchar>(
        gvdb->volOut[CHAN_CELL_TYPE], (cell[i].x - offset.x) * sizeof(uchar),
        cell[i].y - offset.y, cell[i].z - offset.z);
    valid[i] = !(c1 == CellType::Air && c2 == CellType::Air);
  }
}

__global__ void transferToGrid(VDBInfo *gvdb, int num_sc, int *sc_nid,
                               int *sc_cnt, int *sc_off, int3 *sc_pos,
                               float3 *sc_pnt_pos, float3 *sc_pnt_vel) {
  GVDB_VOXSUBCELL

  float3 vel = make_float3(0.0f, 0.0f, 0.0f);
  float3 r = make_float3(0.0f, 0.0f, 0.0f);

  for (int j = 0; j < sc_cnt[sc_id]; j++) {
    float3 ppos = sc_pnt_pos[sc_off[sc_id] + j];

    addVelocityFromParticle(ppos, wpos, sc_pnt_vel[sc_off[sc_id] + j], vel, r,
                            Component::X);
    addVelocityFromParticle(ppos, wpos, sc_pnt_vel[sc_off[sc_id] + j], vel, r,
                            Component::Y);
    addVelocityFromParticle(ppos, wpos, sc_pnt_vel[sc_off[sc_id] + j], vel, r,
                            Component::Z);
  }

  if (r.x > 0.0f) vel.x /= r.x;
  if (r.y > 0.0f) vel.y /= r.y;
  if (r.z > 0.0f) vel.z /= r.z;

  surf3Dwrite(make_float4(vel), gvdb->volOut[CHAN_VELOCITY],
              vox.x * sizeof(float4), vox.y, vox.z);
}

__global__ void transferFromGrid(VDBInfo *gvdb, int num_sc, int *sc_nid,
                                 int *sc_cnt, int *sc_off, int3 *sc_pos,
                                 float3 *sc_pnt_pos, uint *sc_pnt_clr,
                                 float3 *vel) {
  GVDB_VOXSUBCELL
  
  int3 cell[8];
  getNeighborCellIndices(vox, cell);

  // Velocities from each corner.
  float3 gridvel[8];
  for (int i = 0; i < 8; i++) {
    gridvel[i] = fxyz(surf3Dread<float4>(gvdb->volOut[CHAN_VELOCITY],
                                         cell[i].x * sizeof(float4), cell[i].y,
                                         cell[i].z));
  }

  bool validX[8];
  bool validY[8];
  bool validZ[8];
  getValidCells(gvdb, cell, validX, Component::X);
  getValidCells(gvdb, cell, validY, Component::Y);
  getValidCells(gvdb, cell, validZ, Component::Z);

  for (int j = 0; j < sc_cnt[sc_id]; j++) {
    float3 ppos = sc_pnt_pos[sc_off[sc_id] + j];
    uint idx = sc_pnt_clr[sc_off[sc_id] + j];

    getVelocityFromGrid(ppos, wpos, idx, gridvel, validX, vel, Component::X);
    getVelocityFromGrid(ppos, wpos, idx, gridvel, validY, vel, Component::Y);
    getVelocityFromGrid(ppos, wpos, idx, gridvel, validZ, vel, Component::Z);
  }
}

__global__ void applyGravity(VDBInfo *gvdb, int num_sc, int *sc_nid,
                             int3 *sc_pos) {
  GVDB_VOXSUBCELL

  float3 val = fxyz(surf3Dread<float4>(gvdb->volOut[CHAN_VELOCITY],
                                       vox.x * sizeof(float4), vox.y, vox.z));
  val += fp.gravity * fp.dt;

  if (wpos.x == 1 || wpos.x == fp.gridres.x - 1) val.x = 0.0f;
  if (wpos.y == 1 || wpos.y == fp.gridres.y - 1) val.y = 0.0f;
  if (wpos.z == 1 || wpos.z == fp.gridres.z - 1) val.z = 0.0f;

  surf3Dwrite(make_float4(val), gvdb->volOut[CHAN_VELOCITY],
              vox.x * sizeof(float4), vox.y, vox.z);
}

__global__ void markCells(VDBInfo *gvdb, int num_sc, int *sc_nid, int *sc_cnt,
                          int *sc_off, int3 *sc_pos, float3 *sc_pnt_pos) {
  GVDB_VOXSUBCELL

  for (int j = 0; j < sc_cnt[sc_id]; j++) {
    int3 ppos = make_int3(sc_pnt_pos[sc_off[sc_id] + j]);

    if (ppos.x == wpos.x && ppos.y == wpos.y && ppos.z == wpos.z) {
      surf3Dwrite((uchar) CellType::Fluid, gvdb->volOut[CHAN_CELL_TYPE],
                  vox.x * sizeof(uchar), vox.y, vox.z);
      return;
    }
  }
#ifdef WARM_START
  surf3Dwrite(0.0f, gvdb->volOut[CHAN_PRESSURE], vox.x * sizeof(float), vox.y,
              vox.z);
#endif
}

__global__ void computeDivergence(VDBInfo *gvdb, int num_sc, int *sc_nid,
                                  int3 *sc_pos) {
  GVDB_VOXSUBCELL

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
  GVDB_VOXSUBCELL

  // Air cells have zero pressure.
  if (surf3Dread<uchar>(gvdb->volOut[CHAN_CELL_TYPE], vox.x * sizeof(uchar),
                        vox.y, vox.z) == CellType::Air) {
    surf3Dwrite(0.0f, gvdb->volOut[p_tmp_chan], vox.x * sizeof(float), vox.y,
                vox.z);
    return;
  }
  
  // Boundary cells have zero pressure.
  if (wpos.x == 0 || wpos.x == fp.gridres.x - 1 || wpos.y == 0 ||
      wpos.y == fp.gridres.y - 1 || wpos.z == 0 || wpos.z == fp.gridres.z - 1) {
    surf3Dwrite(0.0f, gvdb->volOut[p_tmp_chan], vox.x * sizeof(float), vox.y,
                vox.z);
    return;
  }

  float s_sum = float(wpos.x != 1) + float(wpos.x != fp.gridres.x - 2) +
                float(wpos.y != 1) + float(wpos.y != fp.gridres.y - 2) +
                float(wpos.z != 1) + float(wpos.z != fp.gridres.z - 2);

  if (s_sum == 0.0f) return;

  float div = surf3Dread<float>(gvdb->volOut[CHAN_DIVERGENCE],
                                vox.x * sizeof(float), vox.y, vox.z);

  // Neighbor cells' pressures. Assumes boundary pressures are kept zero.
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

  float lastPressure = surf3Dread<float>(gvdb->volOut[p_chan],
                                         vox.x * sizeof(float), vox.y, vox.z);
  float residual = (s_sum * lastPressure - p_sum) / fp.h_sq + fp.c * div;
  surf3Dwrite(residual, gvdb->volOut[CHAN_RESIDUAL], vox.x * sizeof(float),
              vox.y, vox.z);

  float pressure = (p_sum - (fp.h_sq * fp.c) * div) / s_sum;
  surf3Dwrite(pressure, gvdb->volOut[p_tmp_chan], vox.x * sizeof(float), vox.y,
              vox.z);
}

__global__ void applyPressure(VDBInfo *gvdb, int num_sc, int *sc_nid, int3 *sc_pos) {
  GVDB_VOXSUBCELL

  float dt_div_rho_0_h = fp.dt / (fp.density * fp.h);

  float3 vel = fxyz(surf3Dread<float4>(gvdb->volOut[CHAN_VELOCITY],
                                      vox.x * sizeof(float4), vox.y, vox.z));

  float p = surf3Dread<float>(gvdb->volOut[CHAN_PRESSURE],
                              vox.x * sizeof(float), vox.y, vox.z);

  if (wpos.x == 1 || wpos.x == fp.gridres.x - 1) vel.x = 0.0f;
  else
    vel.x -=
        (p - surf3Dread<float>(gvdb->volOut[CHAN_PRESSURE],
                               (vox.x - 1) * sizeof(float), vox.y, vox.z)) *
        dt_div_rho_0_h;
  if (wpos.y == 1 || wpos.y == fp.gridres.y - 1) vel.y = 0.0f;
  else
    vel.y -= (p - surf3Dread<float>(gvdb->volOut[CHAN_PRESSURE],
                                    vox.x * sizeof(float), vox.y - 1, vox.z)) *
             dt_div_rho_0_h;
  if (wpos.z == 1 || wpos.z == fp.gridres.z - 1) vel.z = 0.0f;
  else
    vel.z -= (p - surf3Dread<float>(gvdb->volOut[CHAN_PRESSURE],
                                    vox.x * sizeof(float), vox.y, vox.z - 1)) *
             dt_div_rho_0_h;

  surf3Dwrite(make_float4(vel), gvdb->volOut[CHAN_VELOCITY],
              vox.x * sizeof(float4), vox.y, vox.z);
}
