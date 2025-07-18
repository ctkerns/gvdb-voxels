extern "C" {
  __global__ void integrateParticles(float3 *pos, float3 *vel);
  __global__ void handleParticleCollision(float3 *pos, float3 *vel);
}
