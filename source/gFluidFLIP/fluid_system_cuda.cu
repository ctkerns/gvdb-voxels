// Christopher Kerns 2025

#include "fluid_system_cuda.cuh"

#include "fluid_params.h"

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

  // TODO: Why is this boundary so small?
  if (pos[i].x < fp.h) {
    pos[i].x = fp.h;
    vel[i].x = 0.0f;
  } else if (pos[i].x > (fp.gridres.x - 2) * fp.h) {
    pos[i].x = (fp.gridres.x - 2) * fp.h;
    vel[i].x = 0.0f;
  }
  if (pos[i].y < fp.h) {
    pos[i].y = fp.h;
    vel[i].y = 0.0f;
  } else if (pos[i].y > (fp.gridres.y - 2) * fp.h) {
    pos[i].y = (fp.gridres.y - 2) * fp.h;
    vel[i].y = 0.0f;
  }
  if (pos[i].z < fp.h) {
    pos[i].z = fp.h;
    vel[i].z = 0.0f;
  } else if (pos[i].z > (fp.gridres.z - 2) * fp.h) {
    pos[i].z = (fp.gridres.z - 2) * fp.h;
    vel[i].z = 0.0f;
  }
}

__global__ void transferFromGrid(VDBInfo *gvdb, int num_sc, int3 component,
                                 int *sc_nid, int *sc_cnt, int *sc_off,
                                 int3 *sc_pos, float3 *sc_pnt_pos,
                                 uint *sc_pnt_clr, float3 *vel) {
  int sc_id = blockIdx.x;
  if (sc_id >= num_sc) return;

  int3 wpos;
  wpos.x = sc_pos[sc_id].x + int(threadIdx.x);
  wpos.y = sc_pos[sc_id].y + int(threadIdx.y);
  wpos.z = sc_pos[sc_id].z + int(threadIdx.z);

  VDBNode *node = getNode(gvdb, 0, sc_nid[sc_id]);
	float3 vmin = make_float3(node->mPos);
  int3 vox = node->mValue +
             make_int3(wpos.x - vmin.x, wpos.y - vmin.y, wpos.z - vmin.z);

  float3 gridVel =
      fxyz(surf3Dread<float4>(gvdb->volOut[1], vox.x * sizeof(float4), vox.y, vox.z));

  for (int j = 0; j < sc_cnt[sc_id]; j++) {
    float3 ppos = sc_pnt_pos[sc_off[sc_id] + j];
    uint idx = sc_pnt_clr[sc_off[sc_id] + j];

    // Only take into account particles within this cell.
    if (ppos.x >= wpos.x && ppos.x < wpos.x + 1 &&
        ppos.y >= wpos.y && ppos.y < wpos.y + 1 &&
        ppos.z >= wpos.z && ppos.z < wpos.z + 1) {
      vel[idx] = gridVel;
    }
  }
}

__global__ void transferToGrid(VDBInfo *gvdb, int num_sc, int3 component,
                               int *sc_nid, int *sc_cnt, int *sc_off,
                               int3 *sc_pos, float3 *sc_pnt_pos,
                               float3 *sc_pnt_vel) {
  int sc_id = blockIdx.x;
  if (sc_id >= num_sc) return;

  int3 wpos;
  wpos.x = sc_pos[sc_id].x + int(threadIdx.x);
  wpos.y = sc_pos[sc_id].y + int(threadIdx.y);
  wpos.z = sc_pos[sc_id].z + int(threadIdx.z);

  VDBNode *node = getNode(gvdb, 0, sc_nid[sc_id]);
	float3 vmin = make_float3(node->mPos);
  int3 vox = node->mValue +
             make_int3(wpos.x - vmin.x, wpos.y - vmin.y, wpos.z - vmin.z);

  float3 val = make_float3(0.0f, 0.0f, 0.0f);
  int num = 0;

  for (int j = 0; j < sc_cnt[sc_id]; j++) {
    float3 ppos = sc_pnt_pos[sc_off[sc_id] + j];

    // Only take into account particles within this cell.
    if (ppos.x >= wpos.x && ppos.x < wpos.x + 1 &&
        ppos.y >= wpos.y && ppos.y < wpos.y + 1 &&
        ppos.z >= wpos.z && ppos.z < wpos.z + 1) {
      val += sc_pnt_vel[sc_off[sc_id] + j];
      num++;
    }
  }

  val /= float(num);

  surf3Dwrite(make_float4(val), gvdb->volOut[1], vox.x * sizeof(float4), vox.y,
              vox.z);
}
