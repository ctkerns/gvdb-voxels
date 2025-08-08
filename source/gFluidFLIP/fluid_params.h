// Christopher Kerns 2025

#include <cuda.h>
#include <curand.h>

typedef struct FluidParams {
  int3 gridres;
  int numpnts;
  float h;
  float dt;
  float3 gravity;
} FluidParams;
