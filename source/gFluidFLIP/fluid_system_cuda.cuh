// Christopher Kerns 2025

#include <cuda_math.cuh>

#define CUDA_PATHWAY
#include <cuda_gvdb_scene.cuh>
#include <cuda_gvdb_nodes.cuh>
#include <cuda_gvdb_geom.cuh>
#include <cuda_gvdb_operators.cuh>

#include "fluid_utils.h"

#define GVDB_VOXSUBCELL \
  int sc_id = blockIdx.x * fp.subcellPerBlock + (threadIdx.x / fp.subcell); \
  if (sc_id >= num_sc) return; \
  \
  int3 idx = \
      make_int3(threadIdx.x % fp.subcell, threadIdx.y, threadIdx.z); \
  int3 wpos = sc_pos[sc_id] + idx; /* World voxel position. */ \
  VDBNode *node = getNode(gvdb, 0, sc_nid[sc_id]); \
  int3 vox = node->mValue + (wpos - node->mPos); /* Atlas index of voxel. */

extern "C" {
  __global__ void integrateParticles(float3 *pos, float3 *vel);
  __global__ void handleParticleCollision(float3 *pos, float3 *vel);
  __global__ void transferToGrid(VDBInfo *gvdb, int num_sc, int *sc_nid,
                                 int *sc_cnt, int *sc_off, int3 *sc_pos,
                                 float3 *sc_pnt_pos, float3 *sc_pnt_vel);
  __global__ void transferFromGrid(VDBInfo *gvdb, int num_sc, int *sc_nid,
                                   int *sc_cnt, int *sc_off, int3 *sc_pos,
                                   float3 *sc_pnt_pos, uint *sc_pnt_clr,
                                   float3 *vel);
  __global__ void applyGravity(VDBInfo *gvdb, int num_sc, int *sc_nid,
                               int3 *sc_pos);
  __global__ void markCells(VDBInfo *gvdb, int num_sc, int *sc_nid, int *sc_cnt,
                            int *sc_off, int3 *sc_pos, float3 *sc_pnt_pos);
  __global__ void computeDivergence(VDBInfo *gvdb, int num_sc, int *sc_nid,
                               int3 *sc_pos);
  __global__ void solveJacobi(VDBInfo *gvdb, int num_sc, int p_chan,
                              int p_tmp_chan, int *sc_nid, int3 *sc_pos);
  __global__ void applyPressure(VDBInfo *gvdb, int num_sc, int *sc_nid,
                                int3 *sc_pos);
}
