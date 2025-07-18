#include <cuda.h>
#include <curand.h>
#include "gvdb_vec.h"
using namespace nvdb;

typedef struct FluidParams {
  int3 gridres;
  float h;
  float dt;
  float3 gravity;
} FluidParams;
