// Christopher Kerns 2025

#include <cuda.h>	

#include "fluid_system.h"

bool cuCheck (CUresult launch_stat, char* method, char* apicall, char* arg, bool bDebug)
{
	CUresult kern_stat = CUDA_SUCCESS;

	if (bDebug) {
		kern_stat = cuCtxSynchronize();
	}
	if (kern_stat != CUDA_SUCCESS || launch_stat != CUDA_SUCCESS) {
		const char* launch_statmsg = "";
    const char* error_name = "";
		const char* kern_statmsg = "";
		cuGetErrorString(launch_stat, &launch_statmsg);
    cuGetErrorName(launch_stat, &error_name);
		cuGetErrorString(kern_stat, &kern_statmsg);
		nvprintf("FLUID SYSTEM, CUDA ERROR:\n");
		nvprintf("  Launch status: %s\n", launch_statmsg);
		nvprintf("  Error name: %s\n", error_name);
		nvprintf("  Kernel status: %s\n", kern_statmsg);
		nvprintf("  Caller: FluidSystem::%s\n", method);
		nvprintf("  Call:   %s\n", apicall);
		nvprintf("  Args:   %s\n", arg);

		if (bDebug) {
			nvprintf("  Generating assert to examine call stack.\n");
			assert(0);		// debug - trigger break (see call stack)
		}
		else {
			nverror();		// exit - return 0
		}
		return false;
	}
	return true;
}

FluidSystem::FluidSystem() {
	for (int n=0; n < FUNC_MAX; n++ ) m_Func[n] = (CUfunction) -1;

  fp.gridres = make_int3(30, 30, 30);
  fp.h = 1.0f;
  // fp.dt = 1.0f / (3.0f * 60.0f);
  fp.dt = 1.0f / 32.0f;
  fp.gravity = make_float3(0.0f, -9.8f, 0.0f);
  fp.numpnts = (fp.gridres.x - 2) * (fp.gridres.y - 2) * (fp.gridres.z - 2);

  numThreads = (fp.numpnts < threadsPerBlock) ? fp.numpnts : threadsPerBlock;
  numBlocks = (fp.numpnts % numThreads != 0) ? (fp.numpnts / numThreads + 1)
                                          : (fp.numpnts / numThreads);
}

FluidSystem::~FluidSystem() {}

void FluidSystem::LoadKernel(int id, std::string kname) {
  char cfn[512];
  strcpy(cfn, kname.c_str());

  if (m_Func[id] == (CUfunction)-1)
    cuCheck(cuModuleGetFunction(&m_Func[id], m_Module, cfn), "LoadKernel",
            "cuModuleGetFunction", cfn, mbDebug);
}

void FluidSystem::setup() {
  cuCheck(cuModuleLoad(&m_Module, "fluid_system_cuda.ptx"),
          "FluidSystem::setup", "cuModuleLoad", "fluid_system_cuda.ptx",
          mbDebug);

  // Initialize particles
  cuCheck(cuMemAlloc(&cu_pos, sizeof(Vector3DF)*fp.numpnts), "FluidSystem::setup",
          "cuMemAlloc", "cu_pos", mbDebug);
  cuCheck(cuMemAlloc(&cu_vel, sizeof(Vector3DF)*fp.numpnts), "FluidSystem::setup",
          "cuMemAlloc", "cu_vel", mbDebug);

  pos = std::vector<Vector3DF>(fp.numpnts);
  vel = std::vector<Vector3DF>(fp.numpnts, Vector3DF(0.0f, 0.0f, 0.0f));

  // Load parameters.
  size_t len = 0;
  cuCheck(cuModuleGetGlobal(&cu_fp, &len, m_Module, "fp"),
          "FluidSystem::setup", "cuModuleGetGlobal", "cu_fp", mbDebug);

  cuCheck(cuMemcpyHtoD(cu_fp, &fp, sizeof(FluidParams)), "FluidSystem::setup",
          "cuMemcpyHtoD", "cu_fp", mbDebug);

  LoadKernel(FUNC_INTEGRATE, "integrateParticles");
  LoadKernel(FUNC_HANDLE_COLLISION, "handleParticleCollision");
  LoadKernel(FUNC_TRANSFER_FROM_GRID, "transferFromGrid");
  LoadKernel(FUNC_TRANSFER_TO_GRID, "transferToGrid");

  int p = 0;
  for (int i = 1; i < fp.gridres.x - 1; i++) {
    for (int j = 1; j < fp.gridres.y - 1; j++) {
      for (int k = 1; k < fp.gridres.z - 1; k++) {
        pos[p++] = Vector3DF((i + fp.gridres.x)/2.0f, (j + fp.gridres.x)/2.0f, (k + fp.gridres.x)/2.0f)*fp.h;
        // pos[p++] = Vector3DF(i, j, k)*fp.h;
      }
    }
  }

  // Initialize cells.
  celltype.resize(fp.gridres.x);
  cellvel.resize(fp.gridres.x);
  r.resize(fp.gridres.x);
  particleDensity.resize(fp.gridres.x);
  for (int i = 0; i < fp.gridres.x; i++) {
    celltype[i].resize(fp.gridres.y);
    cellvel[i].resize(fp.gridres.y);
    r[i].resize(fp.gridres.y);
    particleDensity[i].resize(fp.gridres.y);

    for (int j = 0; j < fp.gridres.y; j++) {
      celltype[i][j].resize(fp.gridres.z);
      cellvel[i][j].resize(fp.gridres.z, Vector3DF(0.0f, 0.0f, 0.0f));
      r[i][j].resize(fp.gridres.z, Vector3DF(0.0f, 0.0f, 0.0f));
      particleDensity[i][j].resize(fp.gridres.z);

      for (int k = 0; k < fp.gridres.z; k++) {
        if (i == 0 || j == 0 || k == 0 || i == fp.gridres.x - 1 ||
            j == fp.gridres.y - 1 || k == fp.gridres.z - 1) {
          celltype[i][j][k] = CellType::Solid;
        }
      }
    }
  }
}

void FluidSystem::run(VolumeGVDB &gvdb) {
  transferToCUDA();
  // integrateParticles();
  // handleParticleCollision();

  integrateParticlesCUDA();
  handleParticleCollisionCUDA();

  transferToGridCUDA(gvdb);
  transferFromGridCUDA(gvdb);
  cuCtxSynchronize();

  transferFromCUDA();
  clearCells();
  transferToGrid();
  updateParticleDensity();
  solveIncompressibility();
  transferFromGrid();
}

Vector3DF FluidSystem::getVelocityFromGrid(Vector3DF pos, Component component) {
  float3 ppos = offsetGrid(fp, make_float3(pos.x, pos.y, pos.z), component);
  int3 cellidx = make_int3(ppos.x / fp.h, ppos.y / fp.h, ppos.z / fp.h);

  int3 cellIndices[8];
  getNeighborCellIndices(cellidx, cellIndices);

  // Velocities from each corner.
  float3 vel[8];
  for (int i = 0; i < 8; i++) {
    Vector3DF gridvel =
        cellvel[cellIndices[i].x][cellIndices[i].y][cellIndices[i].z];

    vel[i] = make_float3(gridvel.x, gridvel.y, gridvel.z);
  }

  int3 offsetCell =
      make_int3(component == Component::X, component == Component::Y,
                component == Component::Z);

  bool valid[8];
  for (int i = 0; i < 8; i++) {
    valid[i] =
        !(celltype[cellIndices[i].x][cellIndices[i].y][cellIndices[i].z] ==
              CellType::Air &&
          celltype[cellIndices[i].x - offsetCell.x]
                  [cellIndices[i].y - offsetCell.y]
                  [cellIndices[i].z - offsetCell.z] == CellType::Air);
  }

  float3 rvel = getVelocityFromGridCell(ppos - make_float3(cellidx), vel, valid,
                                        component);
  return Vector3DF(rvel.x, rvel.y, rvel.z);
}

float FluidSystem::addVelocityFromParticle(Vector3DF pos, Vector3DF vel,
                                           Component component) {
  float3 ppos = offsetGrid(fp, make_float3(pos.x, pos.y, pos.z), component);
  int3 cellidx = make_int3(ppos.x / fp.h, ppos.y / fp.h, ppos.z / fp.h);

  // Weights for each corner.
  float w[8];
  float3 pposc =
      make_float3(ppos.x - cellidx.x, ppos.y - cellidx.y, ppos.z - cellidx.z);
  getCellWeights(pposc, w);

  Vector3DF mask =
      Vector3DF(component == Component::X, component == Component::Y,
                component == Component::Z);

  int3 cellIndices[8];
  getNeighborCellIndices(*(int3*)(&cellidx), cellIndices);

  const int max = 8;
  for (int i=0; i < max; i++) {
    r[cellIndices[i].x][cellIndices[i].y][cellIndices[i].z] += mask * w[i];
    cellvel[cellIndices[i].x][cellIndices[i].y][cellIndices[i].z] += mask * vel * w[i];
  }
}

// Apply gravity and velocity.
void FluidSystem::integrateParticles() {
  for (int i = 0; i < pos.size(); i++) {
    pos[i] += vel[i] * fp.dt;
    vel[i] += *(Vector3DF*)(&fp.gravity) * fp.dt;
  }
}

// Make sure particles do not escape boundary.
void FluidSystem::handleParticleCollision() {
  for (int i = 0; i < pos.size(); i++) {
    if (pos[i].x < fp.h) {
      pos[i].x = fp.h;
      vel[i].x = 0.0f;
    } else if (pos[i].x >= (fp.gridres.x - 1) * fp.h) {
      pos[i].x = (fp.gridres.x - 1) * fp.h - 0.001f;
      vel[i].x = 0.0f;
    }
    if (pos[i].y < fp.h) {
      pos[i].y = fp.h;
      vel[i].y = 0.0f;
    } else if (pos[i].y >= (fp.gridres.y - 1) * fp.h) {
      pos[i].y = (fp.gridres.y - 1) * fp.h - 0.001f;
      vel[i].y = 0.0f;
    }
    if (pos[i].z < fp.h) {
      pos[i].z = fp.h;
      vel[i].z = 0.0f;
    } else if (pos[i].z >= (fp.gridres.z - 1) * fp.h) {
      pos[i].z = (fp.gridres.z - 1) * fp.h - 0.001f;
      vel[i].z = 0.0f;
    }
  }
}

void FluidSystem::clearCells() {
  // Set all fluid cells to air cells.
  for (int i = 0; i < fp.gridres.x; i++) {
    for (int j = 0; j < fp.gridres.y; j++) {
      for (int k = 0; k < fp.gridres.z; k++) {
        if (celltype[i][j][k] == CellType::Fluid) {
            celltype[i][j][k] = CellType::Air;
        }
      }
    }
  }

  // Set cells with particles to fluid cells.
  for (int i = 0; i < pos.size(); i++) {
    Vector3DI cellidx = pos[i] / fp.h;

    if (celltype[cellidx.x][cellidx.y][cellidx.z] == CellType::Air) {
      celltype[cellidx.x][cellidx.y][cellidx.z] = CellType::Fluid;
    }
  }
}

// Transfer velocities from grid to particle.
void FluidSystem::transferFromGrid() {
  for (int i = 0; i < pos.size(); i++) {
    vel[i] = getVelocityFromGrid(pos[i], Component::X) +
             getVelocityFromGrid(pos[i], Component::Y) +
             getVelocityFromGrid(pos[i], Component::Z);
  }
}

// Transfer velocities from particle to grid.
void FluidSystem::transferToGrid() {
  // Clear all grid velocities.
  for (int i = 0; i < fp.gridres.x; i++) {
    for (int j = 0; j < fp.gridres.y; j++) {
      for (int k = 0; k < fp.gridres.z; k++) {
        cellvel[i][j][k] = Vector3DF(0.0f, 0.0f, 0.0f);
        r[i][j][k] = Vector3DF(0.0f, 0.0f, 0.0f);
      }
    }
  }

  for (int i = 0; i < pos.size(); i++) {
    addVelocityFromParticle(pos[i], vel[i], Component::X);
    addVelocityFromParticle(pos[i], vel[i], Component::Y);
    addVelocityFromParticle(pos[i], vel[i], Component::Z);
  }

  for (int i = 0; i < fp.gridres.x; i++) {
    for (int j = 0; j < fp.gridres.y; j++) {
      for (int k = 0; k < fp.gridres.z; k++) {
        if (r[i][j][k].x > 0.0f)
            cellvel[i][j][k].x /= r[i][j][k].x;
        if (r[i][j][k].y > 0.0f)
            cellvel[i][j][k].y /= r[i][j][k].y;
        if (r[i][j][k].z > 0.0f)
            cellvel[i][j][k].z /= r[i][j][k].z;
      }
    }
  }
}
  
void FluidSystem::solveIncompressibility() {
  float maxDiv = 0.0f;
  for (int iter = 0; iter < solveIters; iter++) {
    for (int i = 1; i < fp.gridres.x - 1; i++) {
      for (int j = 1; j < fp.gridres.y - 1; j++) {
        for (int k = 1; k < fp.gridres.z - 1; k++) {
          if (celltype[i][j][k] != CellType::Fluid) continue;

          float sx0 = (float) celltype[i - 1][j][k] != CellType::Solid; 
          float sx1 = (float) celltype[i + 1][j][k] != CellType::Solid; 
          float sy0 = (float) celltype[i][j - 1][k] != CellType::Solid; 
          float sy1 = (float) celltype[i][j + 1][k] != CellType::Solid; 
          float sz0 = (float) celltype[i][j][k - 1] != CellType::Solid; 
          float sz1 = (float) celltype[i][j][k + 1] != CellType::Solid; 
          float s_sum = sx0 + sx1 + sy0 + sy1 + sz0 + sz1;

          if (s_sum == 0.0f) continue;

          float div = cellvel[i + 1][j][k].x - cellvel[i][j][k].x +
                      cellvel[i][j + 1][k].y - cellvel[i][j][k].y +
                      cellvel[i][j][k + 1].z - cellvel[i][j][k].z;

          if (particleRestDensity > 0.0f) {
            float compression = particleDensity[i][j][k] - particleRestDensity;

            if (compression > 0.0f) {
              float k = 1.0f;
              div = div - k * compression;
            }
          }
          
          float p_val = (-div / s_sum) * overRelaxation;
          cellvel[i][j][k].x -= sx0 * p_val;
          cellvel[i][j][k].y -= sy0 * p_val;
          cellvel[i][j][k].z -= sz0 * p_val;
          cellvel[i + 1][j][k].x += sx1 * p_val;
          cellvel[i][j + 1][k].y += sy1 * p_val;
          cellvel[i][j][k + 1].z += sz1 * p_val;

          if (iter == solveIters - 1) {
            if (div > maxDiv) {
              maxDiv = div;
            }
          }
        }
      }
    }
  }

  nvprintf("\n%f\n", maxDiv);
}

void FluidSystem::updateParticleDensity() {
  // Clear density of all particles.
  for (int i = 0; i < fp.gridres.x; i++) {
    for (int j = 0; j < fp.gridres.y; j++) {
      for (int k = 0; k < fp.gridres.z; k++) {
        particleDensity[i][j][k] = 0.0f;
      }
    }
  }

  // Add density to each cell from every particle.
  for (int i = 0; i < pos.size(); i++) {
    Vector3DF offsetpos = pos[i] - Vector3DF(fp.h/2.0f, fp.h/2.0f, fp.h/2.0f);
    Vector3DI cellidx = offsetpos / fp.h;
    float3 pposc = make_float3(offsetpos.x - cellidx.x, offsetpos.y - cellidx.y,
                               offsetpos.z - cellidx.z);
    float w[8];
    getCellWeights(pposc, w);

    int3 cellIndices[8];
    getNeighborCellIndices(*(int3*)(&cellidx), cellIndices);
    for (int i=0; i < 8; i++) {
      particleDensity[cellIndices[i].x][cellIndices[i].y][cellIndices[i].z] += w[i];
    }
  }

  // Set particle rest density to average particle density over fluid cells.
  if (particleRestDensity == 0.0f) {
    float sum = 0.0f;
    int numFluidCells = 0;

    for (int i = 0; i < fp.gridres.x; i++) {
      for (int j = 0; j < fp.gridres.y; j++) {
        for (int k = 0; k < fp.gridres.z; k++) {
          if (celltype[i][j][k] == CellType::Fluid) {
            sum += particleDensity[i][j][k];
            numFluidCells++;
          }
        }
      }
    }

    if (numFluidCells > 0) {
      particleRestDensity = sum / numFluidCells;
    }
  }
}

void FluidSystem::transferToCUDA() {
  cuCheck(cuMemcpyHtoD(cu_pos, pos.data(), fp.numpnts*sizeof(Vector3DF)), "transferToCUDA", "cuMemcpyHtoD", "cu_pos", mbDebug);
  cuCheck(cuMemcpyHtoD(cu_vel, vel.data(), fp.numpnts*sizeof(Vector3DF)), "transferToCUDA", "cuMemcpyHtoD", "cu_vel", mbDebug);
}

void FluidSystem::transferFromCUDA() {
  cuCheck(cuMemcpyDtoH(pos.data(), cu_pos, fp.numpnts*sizeof(Vector3DF)), "transferFromCUDA", "cuMemcpyDtoH", "cu_pos", mbDebug);
  cuCheck(cuMemcpyDtoH(vel.data(), cu_vel, fp.numpnts*sizeof(Vector3DF)), "transferFromCUDA", "cuMemcpyDtoH", "cu_vel", mbDebug);
}

void FluidSystem::integrateParticlesCUDA() {
  void* args[2] = {&cu_pos, &cu_vel};

  cuCheck(cuLaunchKernel(m_Func[FUNC_INTEGRATE], numBlocks, 1, 1, numThreads, 1, 1,
                         0, NULL, args, NULL),
          "IntegrateParticlesCUDA", "cuLaunch", "FUNC_INTEGRATE", mbDebug);
}

void FluidSystem::handleParticleCollisionCUDA() {
  void* args[2] = {&cu_pos, &cu_vel};

  cuCheck(cuLaunchKernel(m_Func[FUNC_HANDLE_COLLISION], numBlocks, 1, 1,
                         numThreads, 1, 1, 0, NULL, args, NULL),
          "handleParticleCollisionCUDA", "cuLaunch", "FUNC_HANDLE_COLLISION",
          mbDebug);
}

void FluidSystem::transferFromGridCUDA(VolumeGVDB &gvdb) {
  int num_brick = static_cast<int>(gvdb.mPool->getPoolUsedCnt(0, 0));
  if (num_brick == 0) return;

  CUdeviceptr cuVDBInfo = gvdb.getCUVDBInfo();
  int numSCell = static_cast<int>(pow(gvdb.getRes(0) / subcell, 3)) * num_brick;
  Component component;
  int pntlen = 0;

  void *args[10] = {&cuVDBInfo,
                    &numSCell,
                    &component,
                    &gvdb.getAux(AUX_SUBCELL_NID).gpu,
                    &gvdb.getAux(AUX_SUBCELL_CNT).gpu,
                    &gvdb.getAux(AUX_SUBCELL_PREFIXSUM).gpu,
                    &gvdb.getAux(AUX_SUBCELL_POS).gpu,
                    &gvdb.getAux(AUX_SUBCELL_PNT_POS).gpu,
                    &gvdb.getAux(AUX_SUBCELL_PNT_CLR).gpu,
                    &gvdb.getAux(AUX_PNTVEL).gpu};

  component = Component::X;
  gvdb.InsertPointsSubcell(subcell, fp.numpnts, 0.0f,
                           Vector3DF(0.0f, -0.5f, -0.5f), pntlen);
  cuCheck(
      cuLaunchKernel(m_Func[FUNC_TRANSFER_FROM_GRID], numSCell, 1, 1,
                     subcell, subcell, subcell, 0, NULL, args, NULL),
      "transferFromGridCUDA", "cuLaunch", "FUNC_TRANSFER_FROM_GRID", mbDebug);

  component = Component::Y;
  gvdb.InsertPointsSubcell(subcell, fp.numpnts, 0.0f,
                           Vector3DF(-0.5f, 0.0f, -0.5f), pntlen);
  cuCheck(
      cuLaunchKernel(m_Func[FUNC_TRANSFER_FROM_GRID], numSCell, 1, 1,
                     subcell, subcell, subcell, 0, NULL, args, NULL),
      "transferFromGridCUDA", "cuLaunch", "FUNC_TRANSFER_FROM_GRID", mbDebug);

  component = Component::Z;
  gvdb.InsertPointsSubcell(subcell, fp.numpnts, 0.0f,
                           Vector3DF(-0.5f, -0.5f, 0.0f), pntlen);
  cuCheck(
      cuLaunchKernel(m_Func[FUNC_TRANSFER_FROM_GRID], numSCell, 1, 1,
                     subcell, subcell, subcell, 0, NULL, args, NULL),
      "transferFromGridCUDA", "cuLaunch", "FUNC_TRANSFER_FROM_GRID", mbDebug);
}

void FluidSystem::transferToGridCUDA(VolumeGVDB &gvdb) {
  int num_brick = static_cast<int>(gvdb.mPool->getPoolUsedCnt(0, 0));
  if (num_brick == 0) return;

  CUdeviceptr cuVDBInfo = gvdb.getCUVDBInfo();
  int numSCell = static_cast<int>(pow(gvdb.getRes(0) / subcell, 3)) * num_brick;
  Component component;
  int pntlen = 0;
  float radius = 1.0f;

  void *args[9] = {&cuVDBInfo,
                    &numSCell,
                    &component,
                    &gvdb.getAux(AUX_SUBCELL_NID).gpu,
                    &gvdb.getAux(AUX_SUBCELL_CNT).gpu,
                    &gvdb.getAux(AUX_SUBCELL_PREFIXSUM).gpu,
                    &gvdb.getAux(AUX_SUBCELL_POS).gpu,
                    &gvdb.getAux(AUX_SUBCELL_PNT_POS).gpu,
                    &gvdb.getAux(AUX_SUBCELL_PNT_VEL).gpu};

  gvdb.ClearChannel(1);

  component = Component::X;
  gvdb.InsertPointsSubcell(subcell, fp.numpnts, radius,
                           Vector3DF(0.0f, -0.5f, -0.5f), pntlen);
  cuCheck(
      cuLaunchKernel(m_Func[FUNC_TRANSFER_TO_GRID], numSCell, 1, 1,
                     subcell, subcell, subcell, 0, NULL, args, NULL),
      "transferToGridCUDA", "cuLaunch", "FUNC_TRANSFER_TO_GRID", mbDebug);
  gvdb.UpdateApron(1);

  component = Component::Y;
  gvdb.InsertPointsSubcell(subcell, fp.numpnts, radius,
                           Vector3DF(-0.5f, 0.0f, -0.5f), pntlen);
  cuCheck(
      cuLaunchKernel(m_Func[FUNC_TRANSFER_TO_GRID], numSCell, 1, 1,
                     subcell, subcell, subcell, 0, NULL, args, NULL),
      "transferToGridCUDA", "cuLaunch", "FUNC_TRANSFER_TO_GRID", mbDebug);
  gvdb.UpdateApron(1);

  component = Component::Z;
  gvdb.InsertPointsSubcell(subcell, fp.numpnts, radius,
                           Vector3DF(-0.5f, -0.5f, 0.0f), pntlen);
  cuCheck(
      cuLaunchKernel(m_Func[FUNC_TRANSFER_TO_GRID], numSCell, 1, 1,
                     subcell, subcell, subcell, 0, NULL, args, NULL),
      "transferToGridCUDA", "cuLaunch", "FUNC_TRANSFER_TO_GRID", mbDebug);
  gvdb.UpdateApron(1);
}
