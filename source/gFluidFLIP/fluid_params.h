// Christopher Kerns 2025

#ifndef DEF_FLUID_PARAM
#define DEF_FLUID_PARAM

#include <cuda.h>
#include <curand.h>

typedef struct FluidParams {
  int3 gridres;
  int numpnts;
  float h;
  float dt;
  float3 gravity;
} FluidParams;

#endif
