// Christopher Kerns 2025

#include <cuda_math.cuh>

#define CUDA_PATHWAY
#include <cuda_gvdb_scene.cuh>
#include <cuda_gvdb_nodes.cuh>
#include <cuda_gvdb_geom.cuh>
#include <cuda_gvdb_operators.cuh>

#include "fluid_utils.h"

extern "C" {
  __global__ void integrateParticles(float3 *pos, float3 *vel);
  __global__ void handleParticleCollision(float3 *pos, float3 *vel);
  __global__ void transferFromGrid(VDBInfo *gvdb, int num_sc,
                                   Component component, int *sc_nid,
                                   int *sc_cnt, int *sc_off, int3 *sc_pos,
                                   float3 *sc_pnt_pos, uint *sc_pnt_clr,
                                   float3 *vel);
  __global__ void transferToGrid(VDBInfo *gvdb, int num_sc, Component component,
                                 int *sc_nid, int *sc_cnt, int *sc_off,
                                 int3 *sc_pos, float3 *sc_pnt_pos,
                                 float3 *sc_pnt_vel);
  }
