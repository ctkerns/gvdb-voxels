#include "fluid_system.h"

FluidSystem::FluidSystem() { setup(); }

FluidSystem::~FluidSystem() {}

void FluidSystem::setup() {
  // Initialize particles
  int numpnts = (gridres.x - 2) * (gridres.y - 2) * (gridres.z - 2);

  pos = std::vector<Vector3DF>(numpnts);
  vel = std::vector<Vector3DF>(numpnts, Vector3DF(0.0f, 0.0f, 0.0f));

  int p = 0;
  for (int i = 1; i < gridres.x - 1; i++) {
    for (int j = 1; j < gridres.y - 1; j++) {
      for (int k = 1; k < gridres.z - 1; k++) {
        pos[p++] = Vector3DF((i + gridres.x)/2.0f, (j + gridres.x)/2.0f, (k + gridres.x)/2.0f)*h;
        // pos[p++] = Vector3DF(i, j, k)*h;
      }
    }
  }

  // Initialize cells.
  celltype.resize(gridres.x);
  cellvel.resize(gridres.x);
  r.resize(gridres.x);
  particleDensity.resize(gridres.x);
  for (int i = 0; i < gridres.x; i++) {
    celltype[i].resize(gridres.y);
    cellvel[i].resize(gridres.y);
    r[i].resize(gridres.y);
    particleDensity[i].resize(gridres.y);

    for (int j = 0; j < gridres.y; j++) {
      celltype[i][j].resize(gridres.z);
      cellvel[i][j].resize(gridres.z, Vector3DF(0.0f, 0.0f, 0.0f));
      r[i][j].resize(gridres.z, Vector3DF(0.0f, 0.0f, 0.0f));
      particleDensity[i][j].resize(gridres.z);

      for (int k = 0; k < gridres.z; k++) {
        if (i == 0 || j == 0 || k == 0 || i == gridres.x - 1 || j == gridres.y - 1
            || k == gridres.z - 1) {
          celltype[i][j][k] = CellType::Solid;
        }
      }
    }
  }
}

void FluidSystem::run() {
  integrateParticles();
  handleParticleCollision();
  clearCells();
  transferToGrid();
  updateParticleDensity();
  solveIncompressibility();
  transferFromGrid();
}

// Get index of cell this particle is in.
Vector3DI FluidSystem::getCellIndex(Vector3DF pos) {
  int x = (int)clamp(pos.x / h, 0.0f, (float)gridres.x);
  int y = (int)clamp(pos.y / h, 0.0f, (float)gridres.y);
  int z = (int)clamp(pos.z / h, 0.0f, (float)gridres.z);

  return Vector3DI(x, y, z);
}

void FluidSystem::getCellWeights(Vector3DF pos, Vector3DI idx, float (&w)[8]) {
  Vector3DF posInCell = pos - idx * h;

  w[0] = (1.0f - posInCell.x / h) * (1.0f - posInCell.y / h) * (1.0f - posInCell.z / h);
  w[1] = (1.0f - posInCell.x / h) * (1.0f - posInCell.y / h) * (posInCell.z / h);
  w[2] = (1.0f - posInCell.x / h) * (posInCell.y / h) * (1.0f - posInCell.z / h);
  w[3] = (1.0f - posInCell.x / h) * (posInCell.y / h) * (posInCell.z / h);
  w[4] = (posInCell.x / h) * (1.0f - posInCell.y / h) * (1.0f - posInCell.z / h);
  w[5] = (posInCell.x / h) * (1.0f - posInCell.y / h) * (posInCell.z / h);
  w[6] = (posInCell.x / h) * (posInCell.y / h) * (1.0f - posInCell.z / h);
  w[7] = (posInCell.x / h) * (posInCell.y / h) * (posInCell.z / h);
}

Vector3DF FluidSystem::offsetGrid(Vector3DF pos, Component component) {
  switch (component) {
  case Component::X:
    return pos - Vector3DF(0.0f, h / 2.0f, h / 2.0f);
  case Component::Y:
    return pos - Vector3DF(h / 2.0f, 0.0f, h / 2.0f);
  case Component::Z:
    return pos - Vector3DF(h / 2.0f, h / 2.0f, 0.0f);
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
    vel[i] += gravity * dt;
    pos[i] += vel[i] * dt;
  }
}

// Make sure particles do not escape boundary.
void FluidSystem::handleParticleCollision() {
  // TODO: Why is this boundary so small?
  for (int i = 0; i < pos.size(); i++) {
    if (pos[i].x < h) {
      pos[i].x = h;
      vel[i].x = 0.0f;
    } else if (pos[i].x > (gridres.x - 2) * h) {
      pos[i].x = (gridres.x - 2) * h;
      vel[i].x = 0.0f;
    }
    if (pos[i].y < h) {
      pos[i].y = h;
      vel[i].y = 0.0f;
    } else if (pos[i].y > (gridres.y - 2) * h) {
      pos[i].y = (gridres.y - 2) * h;
      vel[i].y = 0.0f;
    }
    if (pos[i].z < h) {
      pos[i].z = h;
      vel[i].z = 0.0f;
    } else if (pos[i].z > (gridres.z - 2) * h) {
      pos[i].z = (gridres.z - 2) * h;
      vel[i].z = 0.0f;
    }
  }
}

void FluidSystem::clearCells() {
  // Set all fluid cells to air cells.
  for (int i = 0; i < gridres.x; i++) {
    for (int j = 0; j < gridres.y; j++) {
      for (int k = 0; k < gridres.z; k++) {
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
  for (int i = 0; i < gridres.x; i++) {
    for (int j = 0; j < gridres.y; j++) {
      for (int k = 0; k < gridres.z; k++) {
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

  for (int i = 0; i < gridres.x; i++) {
    for (int j = 0; j < gridres.y; j++) {
      for (int k = 0; k < gridres.z; k++) {
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
    for (int i = 1; i < gridres.x - 1; i++) {
      for (int j = 1; j < gridres.y - 1; j++) {
        for (int k = 1; k < gridres.z - 1; k++) {
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
  for (int i = 0; i < gridres.x; i++) {
    for (int j = 0; j < gridres.y; j++) {
      for (int k = 0; k < gridres.z; k++) {
        particleDensity[i][j][k] = 0.0f;
      }
    }
  }

  // Add density to each cell from every particle.
  for (int i = 0; i < pos.size(); i++) {
    Vector3DF offsetpos = pos[i] - Vector3DF(h/2.0f, h/2.0f, h/2.0f);
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

    for (int i = 0; i < gridres.x; i++) {
      for (int j = 0; j < gridres.y; j++) {
        for (int k = 0; k < gridres.z; k++) {
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
