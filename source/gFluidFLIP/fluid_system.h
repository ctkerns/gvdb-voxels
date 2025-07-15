#include <vector>

#include "main.h"
#include "nv_gui.h"

enum CellType { Fluid, Solid, Air };
enum Component { X, Y, Z };

class FluidSystem {
private:
  // Particles.
  std::vector<Vector3DF> pos;
  std::vector<Vector3DF> vel;

  // Cells.
  std::vector<std::vector<std::vector<CellType>>> celltype;
  std::vector<std::vector<std::vector<Vector3DF>>> cellvel;
  std::vector<std::vector<std::vector<Vector3DF>>> r;
  std::vector<std::vector<std::vector<float>>> particleDensity;

  float particleRestDensity = 0.0f;

public:
  FluidSystem();
  ~FluidSystem();

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

  // Simulation parameters.
  const Vector3DI gridres = Vector3DI(30, 30, 30);
  const float h = 1.0f; // Cell size.
  const Vector3DF gravity = Vector3DF(0.0f, -9.81f, 0.0f);
  const float dt = 1.0f / 32.0f;
  const int solveIters = 200;
  const float overRelaxation = 1.9f;
};

template <typename T> T clamp(T x, T min, T max) {
  return std::max(min, std::min(x, max));
}

