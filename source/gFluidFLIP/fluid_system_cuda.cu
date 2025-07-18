#include "fluid_system_cuda.cuh"

#include "fluid_params.h"

#include "cutil_math.h" // cutil32.lib

__constant__ FluidParams fp;

extern "C"  __global__ void integrateParticles(float3 *pos, float3 *vel) {
  uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

  vel[i] += fp.dt * fp.gravity;
  pos[i] += vel[i] * fp.dt;
}

__global__ void handleParticleCollision(float3 *pos, float3 *vel) {
  uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

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
