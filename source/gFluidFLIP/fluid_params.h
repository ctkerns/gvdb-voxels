// Christopher Kerns 2025

#ifndef DEF_FLUID_PARAM
#define DEF_FLUID_PARAM

#include <cuda.h>
#include <curand.h>

#define CHAN_LEVEL_SET 0
#define CHAN_VELOCITY 1
#define CHAN_CELL_TYPE 2
#define CHAN_DIVERGENCE 3
#define CHAN_PRESSURE 4
#define CHAN_PRESSURE_TMP 5

typedef struct FluidParams {
  int3 gridres;
  float3 tankMin;
  float3 tankMax;
  int subcell;
  int block;
  int subcellPerBlock;
  int numpnts;
  float h;
  float dt;
  float3 gravity;
  float density;
  float radius;
} FluidParams;

#endif
