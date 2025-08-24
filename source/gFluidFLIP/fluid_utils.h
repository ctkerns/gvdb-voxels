// Christopher Kerns 2025

#ifndef DEF_FLUID_UTILS
#define DEF_FLUID_UTILS

#include <cuda_math.cuh>

#include "fluid_params.h"

enum Component { X, Y, Z };
enum CellType { Air, Fluid, Solid };

__host__ __device__
inline void getCellWeights(float3 ppos, float (&w)[8]) {
  w[0] = (1.0f - ppos.x) * (1.0f - ppos.y) * (1.0f - ppos.z);
  w[1] = (1.0f - ppos.x) * (1.0f - ppos.y) * (ppos.z);
  w[2] = (1.0f - ppos.x) * (ppos.y) * (1.0f - ppos.z);
  w[3] = (1.0f - ppos.x) * (ppos.y) * (ppos.z);
  w[4] = (ppos.x) * (1.0f - ppos.y) * (1.0f - ppos.z);
  w[5] = (ppos.x) * (1.0f - ppos.y) * (ppos.z);
  w[6] = (ppos.x) * (ppos.y) * (1.0f - ppos.z);
  w[7] = (ppos.x) * (ppos.y) * (ppos.z);
}

__host__ __device__
inline void getNeighborCellIndices(int3 cellidx, int3 (&indices)[8]) {
  indices[0] = cellidx;
  indices[1] = cellidx + make_int3(0, 0, 1);
  indices[2] = cellidx + make_int3(0, 1, 0);
  indices[3] = cellidx + make_int3(0, 1, 1);
  indices[4] = cellidx + make_int3(1, 0, 0);
  indices[5] = cellidx + make_int3(1, 0, 1);
  indices[6] = cellidx + make_int3(1, 1, 0);
  indices[7] = cellidx + make_int3(1, 1, 1);
}

__host__ __device__
inline float3 offsetGrid(Component component) {
  switch (component) {
  case Component::X:
    return make_float3(0.0f, 0.5f, 0.5f);
  case Component::Y:
    return make_float3(0.5f, 0.0f, 0.5f);
  case Component::Z:
    return make_float3(0.5f, 0.5f, 0.0f);
  };
}

__host__ __device__
inline float3 getVelocityFromGridCell(
    float3 ppos, float3 vel[8], bool valid[8], Component component
) {
  float w[8];
  getCellWeights(ppos, w);

  float3 mask =
      make_float3(component == Component::X, component == Component::Y,
                  component == Component::Z);

  float3 vsum = make_float3(0.0f, 0.0f, 0.0f);
  float wsum = 0.0f;

  for (int i = 0; i < 8; i++) {
    if (valid[i]) {
      vsum += vel[i] * mask * w[i];
      wsum += w[i];
    }
  }

  return vsum / wsum;
}

#endif
