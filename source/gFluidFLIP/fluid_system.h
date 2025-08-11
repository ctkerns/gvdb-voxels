// Christopher Kerns 2025

// GVDB library
#include "gvdb.h"
using namespace nvdb;

#include <vector>

#include "main.h"
#include "nv_gui.h"

#include "fluid_params.h"
#include "fluid_utils.h"

#define FUNC_INTEGRATE 0
#define FUNC_HANDLE_COLLISION 1
#define FUNC_TRANSFER_FROM_GRID 2
#define FUNC_TRANSFER_TO_GRID 3
#define FUNC_MAX 4

class FluidSystem {
private:
  // Particles.
  std::vector<Vector3DF> pos;
  std::vector<Vector3DF> vel;
  CUdeviceptr cu_pos;
  CUdeviceptr cu_vel;

  // Cells.
  std::vector<std::vector<std::vector<CellType>>> celltype;
  std::vector<std::vector<std::vector<Vector3DF>>> cellvel;
  std::vector<std::vector<std::vector<Vector3DF>>> r;
  std::vector<std::vector<std::vector<float>>> particleDensity;

  float particleRestDensity = 0.0f;

  // CUDA.
  CUmodule m_Module;
  CUfunction m_Func[FUNC_MAX];

  bool mbDebug = true;

public:
  FluidSystem();
  ~FluidSystem();

  void LoadKernel(int id, std::string kname);

  void setup();
  void run(VolumeGVDB &gvdb);

  std::vector<Vector3DF> getPoints() { return pos; }
  CUdeviceptr getPosGPU() { return cu_pos; }
  CUdeviceptr getVelGPU() { return cu_vel; }

  void integrateParticles();
  void handleParticleCollision();
  Vector3DF getVelocityFromGrid(Vector3DF pos, Component component);
  float addVelocityFromParticle(Vector3DF pos, Vector3DF vel,
                                Component component);
  void clearCells();
  void transferFromGrid();
  void transferToGrid();
  void solveIncompressibility();
  void updateParticleDensity();

  void transferToCUDA();
  void transferFromCUDA();
  void integrateParticlesCUDA();
  void handleParticleCollisionCUDA();
  void transferFromGridCUDA(VolumeGVDB &gvdb);
  void transferToGridCUDA(VolumeGVDB &gvdb);

  // Simulation parameters.
  const Vector3DI gridres = Vector3DI(30, 30, 30);
  const int solveIters = 200;
  const float overRelaxation = 1.9f;
  FluidParams fp;
  CUdeviceptr cu_fp;
  int subcell = 4;

  const int threadsPerBlock = 512;
  int numThreads;
  int numBlocks;
};

template <typename T> T clamp(T x, T min, T max) {
  return std::max(min, std::min(x, max));
}
