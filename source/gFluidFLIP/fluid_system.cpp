// Christopher Kerns 2025

#include <cuda.h>	

#include "fluid_system.h"
#include "fluid_params.h"
#include "fluid_utils.h"

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

Vector3DF lerp3(Vector3DF v1, Vector3DF v2, Vector3DF t) {
  return v1 * (Vector3DF(1.0f, 1.0f, 1.0f) - t) + v2 * t;
}

FluidSystem::FluidSystem() {
	for (int n=0; n < FUNC_MAX; n++ ) m_Func[n] = (CUfunction) -1;

  fp.gridres = make_int3(CELLS_X, CELLS_Y, CELLS_Z); // TODO: Why do we have to
                                                     // define here again?
  fp.h = 1.0;
  fp.dt = 1.0f / (12.0f * 30.0f);
  fp.gravity = make_float3(0.0f, -9.8f, 0.0f);
  fp.gravity *= 30.0f; // Unit scale.
  fp.numpnts = (fp.gridres.x) * (fp.gridres.y) * (fp.gridres.z);
  fp.density = 1000.0f;
  fp.radius = 0.5f;

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
  LoadKernel(FUNC_APPLY_GRAVITY, "applyGravity");
  LoadKernel(FUNC_MARK_CELLS, "markCells");
  LoadKernel(FUNC_COMPUTE_DIVERGENCE, "computeDivergence");
  LoadKernel(FUNC_SOLVE_JACOBI, "solveJacobi");
  LoadKernel(FUNC_APPLY_PRESSURE, "applyPressure");

  // Initialize particles
  cuCheck(cuMemAlloc(&cu_pos, sizeof(Vector3DF)*fp.numpnts), "FluidSystem::setup",
          "cuMemAlloc", "cu_pos", mbDebug);
  cuCheck(cuMemAlloc(&cu_vel, sizeof(Vector3DF)*fp.numpnts), "FluidSystem::setup",
          "cuMemAlloc", "cu_vel", mbDebug);

  pos = std::vector<Vector3DF>(fp.numpnts);
  vel = std::vector<Vector3DF>(fp.numpnts, Vector3DF(0.0f, 0.0f, 0.0f));

  Vector3DF minlerp(0.2f, 0.4f, 0.2f); // Adjust these to configure starting
  Vector3DF maxlerp(1.0f, 1.0f, 1.0f); // fluid.

  Vector3DF tankmin(fp.h + fp.radius, fp.h + fp.radius, fp.h + fp.radius);
  Vector3DF tankmax((fp.gridres.x - 1) * fp.h - fp.radius,
                    (fp.gridres.y - 1) * fp.h - fp.radius,
                    (fp.gridres.z - 1) * fp.h - fp.radius);
  Vector3DF fluidmin = lerp3(tankmin, tankmax, minlerp);
  Vector3DF fluidmax = lerp3(tankmin, tankmax, maxlerp);
  int point = 0;
  for (int i = 0; i < fp.gridres.x; i++) {
    for (int j = 0; j < fp.gridres.y; j++) {
      for (int k = 0; k < fp.gridres.z; k++) {
        Vector3DF lerp(float(i) / (fp.gridres.x - 1),
                       float(j) / (fp.gridres.y - 1),
                       float(k) / (fp.gridres.z - 1));
        pos[point++] = lerp3(fluidmin, fluidmax, lerp);
      }
    }
  }

  // Initialize cells.
  p.resize(numcells);
  p_tmp.resize(numcells);
  for (int i = 0; i < fp.gridres.x; i++) {
    for (int j = 0; j < fp.gridres.y; j++) {
      for (int k = 0; k < fp.gridres.z; k++) {
        if (i == 0 || j == 0 || k == 0 || i == fp.gridres.x - 1 ||
            j == fp.gridres.y - 1 || k == fp.gridres.z - 1) {
          celltype[getCellIdx(i, j, k)] = CellType::Solid;
        }
      }
    }
  }
}

void FluidSystem::run(VolumeGVDB &gvdb) {
  transferToCUDA();

#ifdef CPU_SIM
  integrateParticles();
  handleParticleCollision();
  transferToGrid();
  updateCells();
#ifdef COMPENSATE_DRIFT
  updateParticleDensity();
#endif
  computeDivergence();
  solveJacobi();
  applyPressure();
  transferFromGrid();
#else // GPU_SIM
  integrateParticlesCUDA();
  handleParticleCollisionCUDA();
  transferToGridCUDA(gvdb);
  updateCellsCUDA(gvdb);
  computeDivergenceCUDA(gvdb);
  solveJacobiCUDA(gvdb);
  applyPressureCUDA(gvdb);
  transferFromGridCUDA(gvdb);

  cuCtxSynchronize();
  transferFromCUDA(); // CPU readback. TODO: Can we avoid this?
#endif
}

// Apply gravity and velocity.
void FluidSystem::integrateParticles() {
  for (int i = 0; i < pos.size(); i++) {
    pos[i] += vel[i] * fp.dt;
  }
}

// Make sure particles do not escape boundary.
void FluidSystem::handleParticleCollision() {
  float min = fp.h + fp.radius;
  float maxX = (fp.gridres.x - 1) * fp.h - fp.radius;
  float maxY = (fp.gridres.y - 1) * fp.h - fp.radius;
  float maxZ = (fp.gridres.z - 1) * fp.h - fp.radius;

  for (int i = 0; i < pos.size(); i++) {
    if (pos[i].x < min) {
      pos[i].x = min;
      vel[i].x = 0.0f;
    } else if (pos[i].x >= maxX) {
      pos[i].x = maxX;
      vel[i].x = 0.0f;
    }
    if (pos[i].y < min) {
      pos[i].y = min;
      vel[i].y = 0.0f;
    } else if (pos[i].y >= maxY) {
      pos[i].y = maxY;
      vel[i].y = 0.0f;
    }
    if (pos[i].z < min) {
      pos[i].z = min;
      vel[i].z = 0.0f;
    } else if (pos[i].z >= maxZ) {
      pos[i].z = maxZ;
      vel[i].z = 0.0f;
    }
  }
}

Vector3DF FluidSystem::getVelocityFromGrid(Vector3DF pos, Component component) {
  float3 ppos = offsetGrid(fp, make_float3(pos.x, pos.y, pos.z), component);
  int3 cellidx = make_int3(ppos.x / fp.h, ppos.y / fp.h, ppos.z / fp.h);

  int3 cellIndices[8];
  getNeighborCellIndices(cellidx, cellIndices);

  // Velocities from each corner.
  float3 vel[8];
  for (int i = 0; i < 8; i++) {
    Vector3DF gridvel = cellvel[getCellIdx(cellIndices[i])];
    vel[i] = make_float3(gridvel.x, gridvel.y, gridvel.z);
  }

  int3 offsetCell =
      make_int3(component == Component::X, component == Component::Y,
                component == Component::Z);

  bool valid[8];
  for (int i = 0; i < 8; i++) {
    CellType c1 = celltype[getCellIdx(cellIndices[i])];
    CellType c2 = celltype[getCellIdx(cellIndices[i] - offsetCell)];
    valid[i] = !(c1 == CellType::Air && c2 == CellType::Air);
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
  getNeighborCellIndices(make_int3(cellidx.x, cellidx.y, cellidx.z),
                         cellIndices);

  const int max = 8;
  for (int i=0; i < max; i++) {
    r[getCellIdx(cellIndices[i])] += mask * w[i];
    cellvel[getCellIdx(cellIndices[i])] += mask * vel * w[i];
  }
}

// Transfer velocities from particle to grid.
void FluidSystem::transferToGrid() {
  // Clear the grid.
  for (int i = 0; i < numcells; i++) {
    cellvel[i] = Vector3DF(0.0f, 0.0f, 0.0f);
    r[i] = Vector3DF(0.0f, 0.0f, 0.0f);
    p[i] = 0.0f;
    if (celltype[i] == CellType::Fluid) {
      celltype[i] = CellType::Air;
    }
  }

  for (int i = 0; i < pos.size(); i++) {
    addVelocityFromParticle(pos[i], vel[i], Component::X);
    addVelocityFromParticle(pos[i], vel[i], Component::Y);
    addVelocityFromParticle(pos[i], vel[i], Component::Z);
  }

  for (int i = 0; i < numcells; i++) {
    if (r[i].x > 0.0f)
      cellvel[i].x /= r[i].x;
    if (r[i].y > 0.0f)
      cellvel[i].y /= r[i].y;
    if (r[i].z > 0.0f)
      cellvel[i].z /= r[i].z;
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

void FluidSystem::updateCells() {
  // Apply gravity and boundary conditions.
  for (int i = 1; i < fp.gridres.x; i++) {
    for (int j = 1; j < fp.gridres.y; j++) {
      for (int k = 1; k < fp.gridres.z; k++) {
        if (celltype[getCellIdx(i - 1, j, k)] == CellType::Solid ||
            celltype[getCellIdx(i, j, k)] == CellType::Solid) {
          cellvel[getCellIdx(i, j, k)].x = 0.0f;
        } else {
          cellvel[getCellIdx(i, j, k)].x += fp.gravity.x * fp.dt;
        }
        if (celltype[getCellIdx(i, j - 1, k)] == CellType::Solid ||
            celltype[getCellIdx(i, j, k)] == CellType::Solid) {
          cellvel[getCellIdx(i, j, k)].y = 0.0f;
        } else {
          cellvel[getCellIdx(i, j, k)].y += fp.gravity.y * fp.dt;
        }
        if (celltype[getCellIdx(i, j, k - 1)] == CellType::Solid ||
            celltype[getCellIdx(i, j, k)] == CellType::Solid) {
          cellvel[getCellIdx(i, j, k)].z = 0.0f;
        } else {
          cellvel[getCellIdx(i, j, k)].z += fp.gravity.z * fp.dt;
        }
      }
    }
  }

  // Mark cells with particles as fluid cells.
  for (int i = 0; i < pos.size(); i++) {
    Vector3DI cellidx = pos[i] / fp.h;

    if (celltype[getCellIdx(cellidx)] == CellType::Air) {
      celltype[getCellIdx(cellidx)] = CellType::Fluid;
    }
  }
}

void FluidSystem::updateParticleDensity() {
  for (int i = 0; i < numcells; i++) {
      particleDensity[i] = 0.0f;
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
    getNeighborCellIndices(make_int3(cellidx.x, cellidx.y, cellidx.z), cellIndices);
    for (int i=0; i < 8; i++) {
      particleDensity[getCellIdx(cellIndices[i])] += w[i];
    }
  }

  // Set particle rest density to average particle density over fluid cells.
  if (particleRestDensity == 0.0f) {
    float sum = 0.0f;
    int numFluidCells = 0;

    for (int i = 0; i < numcells; i++) {
      if (celltype[i] == CellType::Fluid) {
        sum += particleDensity[i];
        numFluidCells++;
      }
    }

    if (numFluidCells > 0) {
      particleRestDensity = sum / numFluidCells;
    }
  }
}

void FluidSystem::computeDivergence() {
  float maxDiv = 0.0f;
  for (int i = 1; i < fp.gridres.x - 1; i++) {
    for (int j = 1; j < fp.gridres.y - 1; j++) {
      for (int k = 1; k < fp.gridres.z - 1; k++) {
        if (celltype[getCellIdx(i, j, k)] != CellType::Fluid) continue;

        float div = (cellvel[getCellIdx(i + 1, j, k)].x -
                     cellvel[getCellIdx(i, j, k)].x +
                     cellvel[getCellIdx(i, j + 1, k)].y -
                     cellvel[getCellIdx(i, j, k)].y +
                     cellvel[getCellIdx(i, j, k + 1)].z -
                     cellvel[getCellIdx(i, j, k)].z) /
                    fp.h;

        // Compensate drift.
#ifdef COMPENSATE_DRIFT
        if (particleRestDensity > 0.0f) {
          float compression =
              particleDensity[getCellIdx(i, j, k)] - particleRestDensity;

          if (compression > 0.0f) {
            float k = 1.0f;
            div = div - k * compression;
          }
        }
#endif

        divergence[getCellIdx(i, j, k)] = div;

        // Calculate max divergence for debugging.
        if (div > maxDiv) maxDiv = div;
      }
    }
  }

  nvprintf("Max divergence: %f\n", maxDiv);
}

void FluidSystem::solveGaussSeidel() {
  float c = fp.density / fp.dt;
  float h_sq = fp.h * fp.h;

  float maxResidual = 0.0f;
  for (int iter = 0; iter < solveIters; iter++) {
    for (int i = 1; i < fp.gridres.x - 1; i++) {
      for (int j = 1; j < fp.gridres.y - 1; j++) {
        for (int k = 1; k < fp.gridres.z - 1; k++) {
          if (celltype[getCellIdx(i, j, k)] != CellType::Fluid) continue;

          float sx0 = (float) celltype[getCellIdx(i - 1, j, k)] != CellType::Solid; 
          float sx1 = (float) celltype[getCellIdx(i + 1, j, k)] != CellType::Solid; 
          float sy0 = (float) celltype[getCellIdx(i, j - 1, k)] != CellType::Solid; 
          float sy1 = (float) celltype[getCellIdx(i, j + 1, k)] != CellType::Solid; 
          float sz0 = (float) celltype[getCellIdx(i, j, k - 1)] != CellType::Solid; 
          float sz1 = (float) celltype[getCellIdx(i, j, k + 1)] != CellType::Solid; 
          float s_sum = sx0 + sx1 + sy0 + sy1 + sz0 + sz1;

          if (s_sum == 0.0f) continue;

          float div = divergence[getCellIdx(i, j, k)];

          // Accumulate neighbor cells pressures.
          float p_sum = 0.0f;
          p_sum += p[getCellIdx(i - 1, j, k)];
          p_sum += p[getCellIdx(i + 1, j, k)];
          p_sum += p[getCellIdx(i, j - 1, k)];
          p_sum += p[getCellIdx(i, j + 1, k)];
          p_sum += p[getCellIdx(i, j, k - 1)];
          p_sum += p[getCellIdx(i, j, k + 1)];

          float newval = (p_sum - (h_sq * c) * div) / s_sum;
          p[getCellIdx(i, j, k)] = (1.0f - overRelaxation) * p[getCellIdx(i, j, k)] + overRelaxation * newval;

          // Calculate max residual for debugging.
          if (iter == solveIters - 1) {
            float r = -c * div - (s_sum * p[getCellIdx(i, j, k)] - p_sum) / (h_sq);
            if (abs(r) > maxResidual) maxResidual = abs(r);
          }
        }
      }
    }
  }

  nvprintf("Max residual: %f\n", maxResidual);
}

void FluidSystem::solveJacobi() {
  float c = fp.density / fp.dt;
  float h_sq = fp.h * fp.h;

  for (int iter = 0; iter < solveIters; iter++) {
    for (int i = 1; i < fp.gridres.x - 1; i++) {
      for (int j = 1; j < fp.gridres.y - 1; j++) {
        for (int k = 1; k < fp.gridres.z - 1; k++) {
          if (celltype[getCellIdx(i, j, k)] != CellType::Fluid) {
            p_tmp[getCellIdx(i, j, k)] = 0.0f;
            continue;
          }

          float sx0 =
              (float)celltype[getCellIdx(i - 1, j, k)] != CellType::Solid;
          float sx1 =
              (float)celltype[getCellIdx(i + 1, j, k)] != CellType::Solid;
          float sy0 =
              (float)celltype[getCellIdx(i, j - 1, k)] != CellType::Solid;
          float sy1 =
              (float)celltype[getCellIdx(i, j + 1, k)] != CellType::Solid;
          float sz0 =
              (float)celltype[getCellIdx(i, j, k - 1)] != CellType::Solid;
          float sz1 =
              (float)celltype[getCellIdx(i, j, k + 1)] != CellType::Solid;
          float s_sum = sx0 + sx1 + sy0 + sy1 + sz0 + sz1;

          if (s_sum == 0.0f) continue;

          float div = divergence[getCellIdx(i, j, k)];

          // Neighbor cells' pressures.
          float p_sum = 0.0f;
          p_sum += p[getCellIdx(i - 1, j, k)];
          p_sum += p[getCellIdx(i + 1, j, k)];
          p_sum += p[getCellIdx(i, j - 1, k)];
          p_sum += p[getCellIdx(i, j + 1, k)];
          p_sum += p[getCellIdx(i, j, k - 1)];
          p_sum += p[getCellIdx(i, j, k + 1)];

          p_tmp[getCellIdx(i, j, k)] = (p_sum - (h_sq * c) * div) / s_sum;
        }
      }
    }
    p_tmp.swap(p);
  } 
}

void FluidSystem::applyPressure() {
  float dt_div_rho_0_h = fp.dt / (fp.density * fp.h);

  for (int i = 1; i < fp.gridres.x; i++) {
    for (int j = 1; j < fp.gridres.y; j++) {
      for (int k = 1; k < fp.gridres.z; k++) {
        if (celltype[getCellIdx(i - 1, j, k)] == CellType::Solid ||
            celltype[getCellIdx(i, j, k)] == CellType::Solid) {
          cellvel[getCellIdx(i, j, k)].x = 0.0f;
        } else {
          cellvel[getCellIdx(i, j, k)].x -=
              dt_div_rho_0_h *
              (p[getCellIdx(i, j, k)] - p[getCellIdx(i - 1, j, k)]);
        }
        if (celltype[getCellIdx(i, j - 1, k)] == CellType::Solid ||
            celltype[getCellIdx(i, j, k)] == CellType::Solid) {
          cellvel[getCellIdx(i, j, k)].y = 0.0f;
        } else {
          cellvel[getCellIdx(i, j, k)].y -=
              dt_div_rho_0_h *
              (p[getCellIdx(i, j, k)] - p[getCellIdx(i, j - 1, k)]);
        }
        if (celltype[getCellIdx(i, j, k - 1)] == CellType::Solid ||
            celltype[getCellIdx(i, j, k)] == CellType::Solid) {
          cellvel[getCellIdx(i, j, k)].z = 0.0f;
        } else {
          cellvel[getCellIdx(i, j, k)].z -=
              dt_div_rho_0_h *
              (p[getCellIdx(i, j, k)] - p[getCellIdx(i, j, k - 1)]);
        }
      }
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

  cuCheck(cuLaunchKernel(m_Func[FUNC_INTEGRATE], numBlocks, 1, 1, numThreads, 1,
                         1, 0, NULL, args, NULL),
          "IntegrateParticlesCUDA", "cuLaunch", "FUNC_INTEGRATE", mbDebug);
}

void FluidSystem::handleParticleCollisionCUDA() {
  void* args[2] = {&cu_pos, &cu_vel};

  cuCheck(cuLaunchKernel(m_Func[FUNC_HANDLE_COLLISION], numBlocks, 1, 1,
                         numThreads, 1, 1, 0, NULL, args, NULL),
          "handleParticleCollisionCUDA", "cuLaunch", "FUNC_HANDLE_COLLISION",
          mbDebug);
}

void FluidSystem::transferToGridCUDA(VolumeGVDB &gvdb) {
  int num_brick = static_cast<int>(gvdb.mPool->getPoolUsedCnt(0, 0));
  if (num_brick == 0) return;

  CUdeviceptr cuVDBInfo = gvdb.getCUVDBInfo();
  int numSCell = static_cast<int>(pow(gvdb.getRes(0) / subcell, 3)) * num_brick;
  int pntlen = 0;
  float radius = 0.5f;
  gvdb.InsertPointsSubcell(subcell, fp.numpnts, radius,
                           Vector3DF(0.0f, 0.0f, 0.0f), pntlen);

  Component component;
  void *args[9] = {&cuVDBInfo,
                    &numSCell,
                    &component,
                    &gvdb.getAux(AUX_SUBCELL_NID).gpu,
                    &gvdb.getAux(AUX_SUBCELL_CNT).gpu,
                    &gvdb.getAux(AUX_SUBCELL_PREFIXSUM).gpu,
                    &gvdb.getAux(AUX_SUBCELL_POS).gpu,
                    &gvdb.getAux(AUX_SUBCELL_PNT_POS).gpu,
                    &gvdb.getAux(AUX_SUBCELL_PNT_VEL).gpu};

  gvdb.ClearChannel(CHAN_VELOCITY);

  component = Component::X;
  cuCheck(cuLaunchKernel(m_Func[FUNC_TRANSFER_TO_GRID], numSCell, 1, 1, subcell,
                         subcell, subcell, 0, NULL, args, NULL),
          "transferToGridCUDA", "cuLaunch", "FUNC_TRANSFER_TO_GRID", mbDebug);

  component = Component::Y;
  cuCheck(cuLaunchKernel(m_Func[FUNC_TRANSFER_TO_GRID], numSCell, 1, 1, subcell,
                         subcell, subcell, 0, NULL, args, NULL),
          "transferToGridCUDA", "cuLaunch", "FUNC_TRANSFER_TO_GRID", mbDebug);

  component = Component::Z;
  cuCheck(cuLaunchKernel(m_Func[FUNC_TRANSFER_TO_GRID], numSCell, 1, 1, subcell,
                         subcell, subcell, 0, NULL, args, NULL),
          "transferToGridCUDA", "cuLaunch", "FUNC_TRANSFER_TO_GRID", mbDebug);
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
  cuCheck(cuLaunchKernel(m_Func[FUNC_TRANSFER_FROM_GRID], numSCell, 1, 1,
                         subcell, subcell, subcell, 0, NULL, args, NULL),
          "transferFromGridCUDA", "cuLaunch", "FUNC_TRANSFER_FROM_GRID",
          mbDebug);

  component = Component::Y;
  cuCheck(cuLaunchKernel(m_Func[FUNC_TRANSFER_FROM_GRID], numSCell, 1, 1,
                         subcell, subcell, subcell, 0, NULL, args, NULL),
          "transferFromGridCUDA", "cuLaunch", "FUNC_TRANSFER_FROM_GRID",
          mbDebug);

  component = Component::Z;
  cuCheck(cuLaunchKernel(m_Func[FUNC_TRANSFER_FROM_GRID], numSCell, 1, 1,
                         subcell, subcell, subcell, 0, NULL, args, NULL),
          "transferFromGridCUDA", "cuLaunch", "FUNC_TRANSFER_FROM_GRID",
          mbDebug);
}

void FluidSystem::updateCellsCUDA(VolumeGVDB &gvdb) {
  gvdb.ClearChannel(CHAN_CELL_TYPE);

  int num_brick = static_cast<int>(gvdb.mPool->getPoolUsedCnt(0, 0));
  if (num_brick == 0) return;

  CUdeviceptr cuVDBInfo = gvdb.getCUVDBInfo();
  int numSCell = static_cast<int>(pow(gvdb.getRes(0) / subcell, 3)) * num_brick;

  void *argsGravity[4] = {&cuVDBInfo, &numSCell,
                         &gvdb.getAux(AUX_SUBCELL_NID).gpu,
                         &gvdb.getAux(AUX_SUBCELL_POS).gpu};

  cuCheck(cuLaunchKernel(m_Func[FUNC_APPLY_GRAVITY], numSCell, 1, 1, subcell,
                         subcell, subcell, 0, NULL, argsGravity, NULL),
          "updateCellsCUDA", "cuLaunch", "FUNC_APPLY_GRAVITY", mbDebug);

  gvdb.UpdateApron(CHAN_VELOCITY);

  void *argsMark[7] = {&cuVDBInfo,
                    &numSCell,
                    &gvdb.getAux(AUX_SUBCELL_NID).gpu,
                    &gvdb.getAux(AUX_SUBCELL_CNT).gpu,
                    &gvdb.getAux(AUX_SUBCELL_PREFIXSUM).gpu,
                    &gvdb.getAux(AUX_SUBCELL_POS).gpu,
                    &gvdb.getAux(AUX_SUBCELL_PNT_POS).gpu};

  cuCheck(cuLaunchKernel(m_Func[FUNC_MARK_CELLS], numSCell, 1, 1, subcell,
                         subcell, subcell, 0, NULL, argsMark, NULL),
          "updateCellsCUDA", "cuLaunch", "FUNC_MARK_CELLS", mbDebug);

  gvdb.UpdateApron(CHAN_CELL_TYPE);
}

void FluidSystem::computeDivergenceCUDA(VolumeGVDB &gvdb) {
  gvdb.ClearChannel(CHAN_DIVERGENCE);

  int num_brick = static_cast<int>(gvdb.mPool->getPoolUsedCnt(0, 0));
  if (num_brick == 0) return;

  CUdeviceptr cuVDBInfo = gvdb.getCUVDBInfo();
  int numSCell = static_cast<int>(pow(gvdb.getRes(0) / subcell, 3)) * num_brick;

  void *args[4] = {&cuVDBInfo, &numSCell,
                         &gvdb.getAux(AUX_SUBCELL_NID).gpu,
                         &gvdb.getAux(AUX_SUBCELL_POS).gpu};

  cuCheck(cuLaunchKernel(m_Func[FUNC_COMPUTE_DIVERGENCE], numSCell, 1, 1,
                         subcell, subcell, subcell, 0, NULL, args, NULL),
          "computeDivergenceCUDA", "cuLaunch", "FUNC_COMPUTE_DIVERGENCE",
          mbDebug);
}

void FluidSystem::solveJacobiCUDA(VolumeGVDB &gvdb) {
  int num_brick = static_cast<int>(gvdb.mPool->getPoolUsedCnt(0, 0));
  if (num_brick == 0) return;

  CUdeviceptr cuVDBInfo = gvdb.getCUVDBInfo();
  int numSCell = static_cast<int>(pow(gvdb.getRes(0) / subcell, 3)) * num_brick;

  int p_chan = CHAN_PRESSURE;
  int p_tmp_chan = CHAN_PRESSURE_TMP;
  void *args[6] = {&cuVDBInfo, &numSCell, &p_chan, &p_tmp_chan,
                         &gvdb.getAux(AUX_SUBCELL_NID).gpu,
                         &gvdb.getAux(AUX_SUBCELL_POS).gpu};

  for (int i = 0; i < solveIters; i++) {
    cuCheck(cuLaunchKernel(m_Func[FUNC_SOLVE_JACOBI], numSCell, 1, 1, subcell,
                           subcell, subcell, 0, NULL, args, NULL),
            "solveJacobiCUDA", "cuLaunch", "FUNC_SOLVE_JACOBI", mbDebug);

    // Swap pressure buffers.
    p_chan = (p_chan == CHAN_PRESSURE) ? CHAN_PRESSURE_TMP : CHAN_PRESSURE;
    p_tmp_chan = (p_chan == CHAN_PRESSURE) ? CHAN_PRESSURE_TMP : CHAN_PRESSURE;

    gvdb.UpdateApron(p_chan);
  }
}

void FluidSystem::applyPressureCUDA(VolumeGVDB &gvdb) {
  int num_brick = static_cast<int>(gvdb.mPool->getPoolUsedCnt(0, 0));
  if (num_brick == 0) return;

  CUdeviceptr cuVDBInfo = gvdb.getCUVDBInfo();
  int numSCell = static_cast<int>(pow(gvdb.getRes(0) / subcell, 3)) * num_brick;

  void *args[4] = {&cuVDBInfo, &numSCell,
                         &gvdb.getAux(AUX_SUBCELL_NID).gpu,
                         &gvdb.getAux(AUX_SUBCELL_POS).gpu};

  cuCheck(cuLaunchKernel(m_Func[FUNC_APPLY_PRESSURE], numSCell, 1, 1,
                         subcell, subcell, subcell, 0, NULL, args, NULL),
          "applyPressureCUDA", "cuLaunch", "FUNC_APPLY_PRESSURE",
          mbDebug);

  gvdb.UpdateApron(CHAN_VELOCITY);
}
