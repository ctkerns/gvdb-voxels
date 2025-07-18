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
  fp.dt = 1.0f / 32.0f;
  fp.gravity = make_float3(0.0f, -9.8f, 0.0f);
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
  cuCheck(cuMemAlloc(&cu_pos, sizeof(Vector3DF)*numpnts), "FluidSystem::setup",
          "cuMemAlloc", "cu_pos", mbDebug);
  cuCheck(cuMemAlloc(&cu_vel, sizeof(Vector3DF)*numpnts), "FluidSystem::setup",
          "cuMemAlloc", "cu_vel", mbDebug);

  pos = std::vector<Vector3DF>(numpnts);
  vel = std::vector<Vector3DF>(numpnts, Vector3DF(0.0f, 0.0f, 0.0f));

  // Load parameters.
  size_t len = 0;
  cuCheck(cuModuleGetGlobal(&cu_fp, &len, m_Module, "fp"),
          "FluidSystem::setup", "cuModuleGetGlobal", "cu_fp", mbDebug);

  cuCheck(cuMemcpyHtoD(cu_fp, &fp, sizeof(FluidParams)), "FluidSystem::setup",
          "cuMemcpyHtoD", "cu_fp", mbDebug);

  LoadKernel(FUNC_INTEGRATE, "integrateParticles");
  LoadKernel(FUNC_HANDLE_COLLISION, "handleParticleCollision");

  int p = 0;
  for (int i = 1; i < fp.gridres.x - 1; i++) {
    for (int j = 1; j < fp.gridres.y - 1; j++) {
      for (int k = 1; k < fp.gridres.z - 1; k++) {
        pos[p++] = Vector3DF((i + fp.gridres.x)/2.0f, (j + fp.gridres.x)/2.0f, (k + fp.gridres.x)/2.0f)*fp.h;
        // pos[p++] = Vector3DF(i, j, k)*h;
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
        if (i == 0 || j == 0 || k == 0 || i == fp.gridres.x - 1 || j == fp.gridres.y - 1
            || k == fp.gridres.z - 1) {
          celltype[i][j][k] = CellType::Solid;
        }
      }
    }
  }
}

void FluidSystem::run() {
  transferToCUDA();
  // integrateParticles();
  // handleParticleCollision();
  integrateParticlesCUDA();
  handleParticleCollisionCUDA();
  cuCtxSynchronize();
  transferFromCUDA();
  clearCells();
  transferToGrid();
  updateParticleDensity();
  solveIncompressibility();
  transferFromGrid();
}

// Get index of cell this particle is in.
Vector3DI FluidSystem::getCellIndex(Vector3DF pos) {
  int x = (int)clamp(pos.x / fp.h, 0.0f, (float)fp.gridres.x);
  int y = (int)clamp(pos.y / fp.h, 0.0f, (float)fp.gridres.y);
  int z = (int)clamp(pos.z / fp.h, 0.0f, (float)fp.gridres.z);

  return Vector3DI(x, y, z);
}

void FluidSystem::getCellWeights(Vector3DF pos, Vector3DI idx, float (&w)[8]) {
  Vector3DF posInCell = pos - idx * fp.h;

  w[0] = (1.0f - posInCell.x / fp.h) * (1.0f - posInCell.y / fp.h) * (1.0f - posInCell.z / fp.h);
  w[1] = (1.0f - posInCell.x / fp.h) * (1.0f - posInCell.y / fp.h) * (posInCell.z / fp.h);
  w[2] = (1.0f - posInCell.x / fp.h) * (posInCell.y / fp.h) * (1.0f - posInCell.z / fp.h);
  w[3] = (1.0f - posInCell.x / fp.h) * (posInCell.y / fp.h) * (posInCell.z / fp.h);
  w[4] = (posInCell.x / fp.h) * (1.0f - posInCell.y / fp.h) * (1.0f - posInCell.z / fp.h);
  w[5] = (posInCell.x / fp.h) * (1.0f - posInCell.y / fp.h) * (posInCell.z / fp.h);
  w[6] = (posInCell.x / fp.h) * (posInCell.y / fp.h) * (1.0f - posInCell.z / fp.h);
  w[7] = (posInCell.x / fp.h) * (posInCell.y / fp.h) * (posInCell.z / fp.h);
}

Vector3DF FluidSystem::offsetGrid(Vector3DF pos, Component component) {
  switch (component) {
  case Component::X:
    return pos - Vector3DF(0.0f, fp.h / 2.0f, fp.h / 2.0f);
  case Component::Y:
    return pos - Vector3DF(fp.h / 2.0f, 0.0f, fp.h / 2.0f);
  case Component::Z:
    return pos - Vector3DF(fp.h / 2.0f, fp.h / 2.0f, 0.0f);
  };
}

void FluidSystem::getNeighborCellIndices(Vector3DI idx, Vector3DI (&indices)[8]) {
  indices[0] = idx;
  indices[1] = idx + Vector3DI(0, 0, 1);
  indices[2] = idx + Vector3DI(0, 1, 0);
  indices[3] = idx + Vector3DI(0, 1, 1);
  indices[4] = idx + Vector3DI(1, 0, 0);
  indices[5] = idx + Vector3DI(1, 0, 1);
  indices[6] = idx + Vector3DI(1, 1, 0);
  indices[7] = idx + Vector3DI(1, 1, 1);
}

Vector3DF FluidSystem::getVelocityFromGrid(Vector3DF pos, Component component) {
  pos = offsetGrid(pos, component);
  Vector3DI cellidx = getCellIndex(pos);

  float w[8];
  getCellWeights(pos, cellidx, w);

  Vector3DI cellIndices[8];
  getNeighborCellIndices(cellidx, cellIndices);

  // Velocities from each corner.
  Vector3DF q[8];
  for (int i=0; i < 8; i++)
    q[i] = cellvel[cellIndices[i].x][cellIndices[i].y][cellIndices[i].z];

  Vector3DI offsetCell =
      Vector3DI(component == Component::X, component == Component::Y,
                component == Component::Z);

  for (int i = 0; i < 8; i++) {
    if (celltype[cellIndices[i].x][cellIndices[i].y][cellIndices[i].z] ==
            CellType::Air &&
        celltype[cellIndices[i].x + offsetCell.x]
                [cellIndices[i].y + offsetCell.y]
                [cellIndices[i].z + offsetCell.z] == CellType::Air) {
      w[i] = 0.0f;
    }
  }

  Vector3DF mask =
      Vector3DF(component == Component::X, component == Component::Y,
                component == Component::Z);

  Vector3DF qsum = Vector3DF(0.0f, 0.0f, 0.0f);
  float wsum = 0.0f;

  for (int i = 0; i < 8; i++) {
    qsum += q[i]*mask * w[i];
    wsum += w[i];
  }

  Vector3DF qp = qsum / wsum;
  return qp;
}

float FluidSystem::addVelocityFromParticle(Vector3DF pos, Vector3DF vel,
                                           Component component) {
  pos = offsetGrid(pos, component);
  Vector3DI cellidx = getCellIndex(pos);

  // Weights for each corner.
  float w[8];
  getCellWeights(pos, cellidx, w);

  Vector3DF mask =
      Vector3DF(component == Component::X, component == Component::Y,
                component == Component::Z);

  Vector3DI cellIndices[8];
  getNeighborCellIndices(cellidx, cellIndices);

  for (int i=0; i < 8; i++) {
    r[cellIndices[i].x][cellIndices[i].y][cellIndices[i].z] += mask * w[i];
    cellvel[cellIndices[i].x][cellIndices[i].y][cellIndices[i].z] += mask * vel * w[i];
  }
}

// Apply gravity and velocity.
void FluidSystem::integrateParticles() {
  for (int i = 0; i < pos.size(); i++) {
    vel[i] += *(Vector3DF*)(&fp.gravity) * fp.dt;
    pos[i] += vel[i] * fp.dt;
  }
}

// Make sure particles do not escape boundary.
void FluidSystem::handleParticleCollision() {
  // TODO: Why is this boundary so small?
  for (int i = 0; i < pos.size(); i++) {
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
    Vector3DI cellidx = getCellIndex(pos[i]);

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
    Vector3DI cellidx = getCellIndex(offsetpos);
    float w[8];
    getCellWeights(offsetpos, cellidx, w);

    Vector3DI cellIndices[8];
    getNeighborCellIndices(cellidx, cellIndices);
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
  cuCheck(cuMemcpyHtoD(cu_pos, pos.data(), numpnts*sizeof(Vector3DF)), "transferToCUDA", "cuMemcpyHtoD", "cu_pos", mbDebug);
  cuCheck(cuMemcpyHtoD(cu_vel, vel.data(), numpnts*sizeof(Vector3DF)), "transferToCUDA", "cuMemcpyHtoD", "cu_vel", mbDebug);
}

void FluidSystem::transferFromCUDA() {
  cuCheck(cuMemcpyDtoH(pos.data(), cu_pos, numpnts*sizeof(Vector3DF)), "transferFromCUDA", "cuMemcpyDtoH", "cu_pos", mbDebug);
  cuCheck(cuMemcpyDtoH(vel.data(), cu_vel, numpnts*sizeof(Vector3DF)), "transferFromCUDA", "cuMemcpyDtoH", "cu_vel", mbDebug);
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
