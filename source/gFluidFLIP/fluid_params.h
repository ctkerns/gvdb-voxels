// Christopher Kerns 2025

#ifndef DEF_FLUID_PARAM
#define DEF_FLUID_PARAM

#include <cuda.h>
#include <curand.h>

#define CHAN_LEVEL_SET 0
#define CHAN_VELOCITY 1
#define CHAN_CELL_TYPE 2

typedef struct FluidParams {
  int3 gridres;
  int numpnts;
  float h;
  float dt;
  float3 gravity;
  float density;
  float radius;
} FluidParams;

#endif
