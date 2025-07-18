#include <vector>

#include "main.h"
#include "nv_gui.h"

#include "fluid_params.h"

enum CellType { Fluid, Solid, Air };
enum Component { X, Y, Z };

#define FUNC_INTEGRATE 0
#define FUNC_HANDLE_COLLISION 1
#define FUNC_MAX 2

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

  bool mbDebug = false;

public:
  FluidSystem();
  ~FluidSystem();

  void LoadKernel(int id, std::string kname);

  void setup();
  void run();

  std::vector<Vector3DF> getPoints() { return pos; }
  Vector3DI getCellIndex(Vector3DF pos);

  void integrateParticles();
  void handleParticleCollision();
  void getCellWeights(Vector3DF pos, Vector3DI idx, float (&w)[8]);
  Vector3DF offsetGrid(Vector3DF pos, Component component);
  void getNeighborCellIndices(Vector3DI idx, Vector3DI (&indices)[8]);
  Vector3DF getVelocityFromGrid(Vector3DF pos, Component component);
  float addVelocityFromParticle(Vector3DF pos, Vector3DF vel, Component component);
  void clearCells();
  void transferFromGrid();
  void transferToGrid();
  void solveIncompressibility();
  void updateParticleDensity();

  void transferToCUDA();
  void transferFromCUDA();
  void integrateParticlesCUDA();
  void handleParticleCollisionCUDA();

  // Simulation parameters.
  const Vector3DI gridres = Vector3DI(30, 30, 30);
  const int numpnts = (gridres.x - 2) * (gridres.y - 2) * (gridres.z - 2);
  const int solveIters = 200;
  const float overRelaxation = 1.9f;
  FluidParams fp;
  CUdeviceptr cu_fp;

  const int threadsPerBlock = 512;
  const int numThreads =
      (numpnts < threadsPerBlock) ? numpnts : threadsPerBlock;
  const int numBlocks = (numpnts % numThreads != 0) ? (numpnts / numThreads + 1)
                                                    : (numpnts / numThreads);
};

template <typename T> T clamp(T x, T min, T max) {
  return std::max(min, std::min(x, max));
}
