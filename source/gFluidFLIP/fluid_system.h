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
#define FUNC_MARK_CELLS 4
#define FUNC_APPLY_GRAVITY 5
#define FUNC_COMPUTE_DIVERGENCE 6
#define FUNC_SOLVE_JACOBI 7
#define FUNC_APPLY_PRESSURE 8
#define FUNC_MAX 9

// #define CPU_SIM
// #define COMPENSATE_DRIFT

#define CELLS_X 50
#define CELLS_Y 50
#define CELLS_Z 50

class FluidSystem {
private:
  // Particles.
  std::vector<Vector3DF> pos;
  std::vector<Vector3DF> vel;
  CUdeviceptr cu_pos;
  CUdeviceptr cu_vel;

  // Cells.
  static const size_t numcells = CELLS_X * CELLS_Y * CELLS_Z;
  std::vector<CellType> celltype;
  std::vector<Vector3DF> cellvel;
  std::vector<Vector3DF> r;
  std::vector<float> particleDensity;
  std::vector<float> divergence;
  std::vector<float> p;
  std::vector<float> p_tmp;

  inline int getCellIdx(int x, int y, int z) {
    return x * CELLS_Y * CELLS_Z + y * CELLS_Z + z;
  }
  inline int getCellIdx(Vector3DI idx) {
    return getCellIdx(idx.x, idx.y, idx.z);
  }
  inline int getCellIdx(int3 idx) {
    return getCellIdx(idx.x, idx.y, idx.z);
  }

  // CUDA.
  CUmodule m_Module;
  CUfunction m_Func[FUNC_MAX];

  bool mbDebug = true;
  int frame = 0;
  int exitFrame = -1; // -1 to disable.

public:
  FluidSystem();
  ~FluidSystem();

  void LoadKernel(int id, std::string kname);

  void setup();
  void run(VolumeGVDB &gvdb);

  int getNumPoints() { return pos.size(); }
  int getFrame() { return frame; }
  CUdeviceptr getPosGPU() { return cu_pos; }
  CUdeviceptr getVelGPU() { return cu_vel; }

  void integrateParticles();
  void handleParticleCollision();
  float addVelocityFromParticle(Vector3DF pos, Vector3DF vel,
                                Component component);
  Vector3DF getVelocityFromGrid(Vector3DF pos, Component component);
  void transferToGrid();
  void transferFromGrid();
  void updateCells();
  void updateParticleDensity();
  void computeDivergence();
  void solveGaussSeidel();
  void solveJacobi();
  void applyPressure();

  void transferToCUDA();
  void transferFromCUDA();
  void integrateParticlesCUDA();
  void handleParticleCollisionCUDA();
  void transferToGridCUDA(VolumeGVDB &gvdb);
  void transferFromGridCUDA(VolumeGVDB &gvdb);
  void updateCellsCUDA(VolumeGVDB &gvdb);
  void computeDivergenceCUDA(VolumeGVDB &gvdb);
  void solveJacobiCUDA(VolumeGVDB &gvdb);
  void applyPressureCUDA(VolumeGVDB &gvdb);
  float maxResidualGVDB(VolumeGVDB &gvdb);

  // Simulation parameters.
  const int solveIters = 200;
  const float overRelaxation = 1.9f;
  float particleRestDensity = 0.0f;
  FluidParams fp;
  CUdeviceptr cu_fp;

  // Tank parameters.
  Vector3DF fluidMin, fluidMax;
  Vector3DI tankPnts;

  const int threadsPerBlock = 512;
  int numThreads;
  int numBlocks;
};
