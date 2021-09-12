#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
#include "kernel.h"

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char* msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f

#define scene_scale 100.0f


int numObjects;
dim3 threadsPerBlock(blockSize);

glm::vec3* dev_pos;
glm::vec3* dev_vel1;
glm::vec3* dev_vel2;

int* dev_particleArrayIndices;
int* dev_particleGridIndices;

thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int* dev_gridCellStartIndices;
int* dev_gridCellEndIndices;

glm::vec3* dev_pos_coherent;
glm::vec3* dev_vel_coherent;

int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);

  return a;
}

__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
  thrust::default_random_engine rng(hash((int)(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3* arr, float scale) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    glm::vec3 rand = generateRandomVec3(time, index);
    arr[index] = rand * scale;
  }
}

void Boids::initSimulation(int N) {
  numObjects = N;
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  kernGenerateRandomPosArray << <fullBlocksPerGrid, blockSize >> > (1, numObjects, dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  gridCellWidth = 1.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");

  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");

  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");

  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");


  cudaMalloc((void**)&dev_pos_coherent, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos_coherent failed!");

  cudaMalloc((void**)&dev_vel_coherent, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos_coherent failed!");

  dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
  dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);

  cudaDeviceSynchronize();
}

__global__ void kernCopyPositionsToVBO(int N, glm::vec3* pos, float* vbo, float s_scale) {
  /**
  * Copy the boid positions into the VBO so that they can be drawn by OpenGL.
  */
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  if (index < N) {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3* vel, float* vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}

void Boids::copyBoidsToVBO(float* vbodptr_positions, float* vbodptr_velocities) {
  /**
  * Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
  */
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_vel1, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyBoidsToVBO failed!");

  cudaDeviceSynchronize();
}

__global__ void kernUpdatePos(int N, float dt, glm::vec3* pos, glm::vec3* vel) {
  // Update position by velocity
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  glm::vec3 thisPos = pos[index];
  thisPos += vel[index] * dt;

  // Wrap the boids around so we don't lose them
  thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
  thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
  thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

  thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
  thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
  thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

  pos[index] = thisPos;
}

__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
  // LOOK-2.1 Consider this method of computing a 1D index from a 3D grid index.
  // LOOK-2.3 Looking at this method, what would be the most memory efficient
  //          order for iterating over neighboring grid cells?
  //          for(x)
  //            for(y)
  //             for(z)? Or some other order?
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3* pos, int* arrayIndices, int* gridIndices) {
  // Both arrayIndices and gridIndices are length N (one element per boid)
  // and are later parallel sorted using the thrust library.
  // arrayIndices is initialized with values 0 to N.
  // gridIndices is initialized with the gridIndex of each boid.

  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (index < N) {
    arrayIndices[index] = index;

    glm::vec3 gridIndex = glm::floor((pos[index] - gridMin) * inverseCellWidth);
    gridIndices[index] = gridIndex3Dto1D(int(gridIndex.x), int(gridIndex.y), int(gridIndex.z), gridResolution);
  }
}

__global__ void kernResetIntBuffer(int N, int* intBuffer, int value) {
  // LOOK-2.1 Consider how this could be useful for indicating that a cell
  //          does not enclose any boids
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

__global__ void kernIdentifyCellStartEnd(int N, int* particleGridIndices,
  int* gridCellStartIndices, int* gridCellEndIndices) {
  // particleGridIndices has length N (one element per boid) and is sorted.
  // particleGridIndices and particleArrayIndices are parallel and the indices
  // have no significance other than that they match up with each other.

  // gridCell{Start,End}Indices have length equal to gridCellCount. The indices
  // of these arrays have significance. Each index corresponds to a value we
  // would get back from gridIndex3Dto1D. Each element will correspond to an
  // index in particleGridIndices.

  // So for every element of particleGridIndices, we simple compare it to the
  // previous element. If the elements are different, then that means that
  // we've identified a boundary between two cells. These indices then get
  // saved in gridCell{Start,End}Indices.
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  int currentIndex = particleGridIndices[index];
  int previousIndex = particleGridIndices[index - 1];

  if (index < N) {
    if (index == 0) {
      gridCellStartIndices[currentIndex] = index;
    } else {
      if (index == N - 1) {
        gridCellEndIndices[currentIndex] = index;
      }

      if (currentIndex != previousIndex) {
        gridCellEndIndices[previousIndex] = index - 1;
        gridCellStartIndices[currentIndex] = index;
      }
    }
  }
}

__global__ void kernSemiCoherent(
  int N, int* particleArrayIndices,
  glm::vec3* dev_pos, glm::vec3* dev_pos_coherent,
  glm::vec3* dev_vel1, glm::vec3* dev_vel_coherent) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    dev_pos_coherent[index] = dev_pos[particleArrayIndices[index]];
    dev_vel_coherent[index] = dev_vel1[particleArrayIndices[index]];
  }
}

__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3* pos, const glm::vec3* vel) {
  // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
  // Rule 2: boids try to stay a distance d away from each other
  // Rule 3: boids try to match the speed of surrounding boids

  glm::vec3 rule1Vec(0, 0, 0);
  glm::vec3 rule2Vec(0, 0, 0);
  glm::vec3 rule3Vec(0, 0, 0);
  int rule1N = 0;
  int rule3N = 0;

  for (int i = 0; i < N; i++) {
    if (i != iSelf) {
      glm::vec3 distanceVec = pos[i] - pos[iSelf];
      float distance = glm::length(distanceVec);

      if (distance < rule1Distance) {
        rule1Vec += pos[i];
        rule1N++;
      }

      if (distance < rule2Distance) {
        rule2Vec -= distanceVec;
      }

      if (distance < rule3Distance) {
        rule3Vec += vel[i];
        rule3N++;
      }
    }
  }

  glm::vec3 newVel = vel[iSelf];

  if (rule1N > 0) {
    rule1Vec /= rule1N;
    glm::vec3 rule1 = (rule1Vec - pos[iSelf]) * rule1Scale;
    newVel += rule1;
  }


  glm::vec3 rule2 = rule2Vec * rule2Scale;
  newVel += rule2;

  if (rule3N > 0) {
    rule3Vec /= rule3N;
    glm::vec3 rule3 = rule3Vec * rule3Scale;
    newVel += rule3;
  }

  if (glm::length(newVel) > maxSpeed) {
    return glm::normalize(newVel) * maxSpeed;
  }

  return newVel;
}

__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3* pos, glm::vec3* vel1, glm::vec3* vel2) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vel2[index] = computeVelocityChange(N, index, pos, vel1);
  }
}

__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int* gridCellStartIndices, int* gridCellEndIndices,
  int* particleArrayIndices,
  glm::vec3* pos, glm::vec3* vel1, glm::vec3* vel2) {

  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    glm::vec3 rule1Vec(0, 0, 0);
    glm::vec3 rule2Vec(0, 0, 0);
    glm::vec3 rule3Vec(0, 0, 0);
    int rule1N = 0;
    int rule3N = 0;

    glm::vec3 gridCornerIndex = glm::floor((pos[index] - gridMin) * inverseCellWidth);

    glm::vec3 potentialMax = gridCornerIndex + 1.f;
    glm::vec3 potentialMin = gridCornerIndex - 1.f;
    potentialMax.x = imin(potentialMax.x, gridResolution - 1);
    potentialMax.y = imin(potentialMax.y, gridResolution - 1);
    potentialMax.z = imin(potentialMax.z, gridResolution - 1);
    potentialMin.x = imax(potentialMin.x, 0);
    potentialMin.y = imax(potentialMin.y, 0);
    potentialMin.z = imax(potentialMin.z, 0);

    for (int z = int(potentialMin.z); z < int(potentialMax.z); z++) {
      for (int y = int(potentialMin.y); y < int(potentialMax.y); y++) {
        for (int x = int(potentialMin.x); x < int(potentialMax.x); x++) {
          int cellIndex = gridIndex3Dto1D(x, y, z, gridResolution);

          int startCellIndex = gridCellStartIndices[cellIndex];

          if (startCellIndex == -1) {
            continue;
          }

          int endCellIndex = gridCellEndIndices[cellIndex];

          for (int i = startCellIndex; i <= endCellIndex; i++) {
            glm::vec3 otherBoidPos = pos[particleArrayIndices[i]];
            glm::vec3 otherBoidVel = vel1[particleArrayIndices[i]];

            if (i != index) {
              glm::vec3 distanceVec = otherBoidPos - pos[index];
              float distance = glm::length(distanceVec);

              if (distance < rule1Distance) {
                rule1Vec += otherBoidPos;
                rule1N++;
              }

              if (distance < rule2Distance) {
                rule2Vec -= distanceVec;
              }

              if (distance < rule3Distance) {
                rule3Vec += otherBoidVel;
                rule3N++;
              }
            }
          }
        }
      }
    }

    glm::vec3 newVel = vel1[index];

    if (rule1N > 0) {
      rule1Vec /= rule1N;
      glm::vec3 rule1 = (rule1Vec - pos[index]) * rule1Scale;
      newVel += rule1;
    }

    glm::vec3 rule2 = rule2Vec * rule2Scale;
    newVel += rule2;

    if (rule3N > 0) {
      rule3Vec /= rule3N;
      glm::vec3 rule3 = rule3Vec * rule3Scale;
      newVel += rule3;
    }

    if (glm::length(newVel) > maxSpeed) {
      newVel = glm::normalize(newVel) * maxSpeed;
    }

    vel2[index] = newVel;
  }
}

__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int* gridCellStartIndices, int* gridCellEndIndices,
  glm::vec3* pos, glm::vec3* vel1, glm::vec3* vel2) {
  // TODO-2.3 - This should be very similar to kernUpdateVelNeighborSearchScattered,
  // except with one less level of indirection.
  // This should expect gridCellStartIndices and gridCellEndIndices to refer
  // directly to pos and vel1.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  //   DIFFERENCE: For best results, consider what order the cells should be
  //   checked in to maximize the memory benefits of reordering the boids data.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2


  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    glm::vec3 rule1Vec(0, 0, 0);
    glm::vec3 rule2Vec(0, 0, 0);
    glm::vec3 rule3Vec(0, 0, 0);
    int rule1N = 0;
    int rule3N = 0;

    glm::vec3 gridCornerIndex = glm::floor((pos[index] - gridMin) * inverseCellWidth);

    glm::vec3 potentialMax = gridCornerIndex + 1.f;
    glm::vec3 potentialMin = gridCornerIndex - 1.f;
    potentialMax.x = imin(potentialMax.x, gridResolution - 1);
    potentialMax.y = imin(potentialMax.y, gridResolution - 1);
    potentialMax.z = imin(potentialMax.z, gridResolution - 1);
    potentialMin.x = imax(potentialMin.x, 0);
    potentialMin.y = imax(potentialMin.y, 0);
    potentialMin.z = imax(potentialMin.z, 0);

    for (int z = potentialMin.z; z < potentialMax.z; z++) {
      for (int y = potentialMin.y; y < potentialMax.y; y++) {
        for (int x = potentialMin.x; x < potentialMax.x; x++) {
          int cellIndex = gridIndex3Dto1D(x, y, z, gridResolution);

          int startCellIndex = gridCellStartIndices[cellIndex];

          if (startCellIndex == -1) {
            continue;
          }

          int endCellIndex = gridCellEndIndices[cellIndex];

          for (int i = startCellIndex; i <= endCellIndex; i++) {
            glm::vec3 otherBoidPos = pos[i];
            glm::vec3 otherBoidVel = vel1[i];

            if (i != index) {
              glm::vec3 distanceVec = otherBoidPos - pos[index];
              float distance = glm::length(distanceVec);

              if (distance < rule1Distance) {
                rule1Vec += otherBoidPos;
                rule1N++;
              }

              if (distance < rule2Distance) {
                rule2Vec -= distanceVec;
              }

              if (distance < rule3Distance) {
                rule3Vec += otherBoidVel;
                rule3N++;
              }
            }
          }
        }
      }
    }

    glm::vec3 newVel = vel1[index];

    if (rule1N > 0) {
      rule1Vec /= rule1N;
      glm::vec3 rule1 = (rule1Vec - pos[index]) * rule1Scale;
      newVel += rule1;
    }

    glm::vec3 rule2 = rule2Vec * rule2Scale;
    newVel += rule2;

    if (rule3N > 0) {
      rule3Vec /= rule3N;
      glm::vec3 rule3 = rule3Vec * rule3Scale;
      newVel += rule3;
    }

    if (glm::length(newVel) > maxSpeed) {
      newVel = glm::normalize(newVel) * maxSpeed;
    }

    vel2[index] = newVel;
  }
}

void Boids::stepSimulationNaive(float dt) {
  int N = numObjects;
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  kernUpdateVelocityBruteForce << <fullBlocksPerGrid, blockSize >> > (N, dev_pos, dev_vel1, dev_vel2);
  checkCUDAErrorWithLine("kernUpdateVelocityBruteForce failed!");

  kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (N, dt, dev_pos, dev_vel2);
  checkCUDAErrorWithLine("kernUpdatePos failed!");

  glm::vec3* tmp = dev_vel1;
  dev_vel1 = dev_vel2;
  dev_vel2 = tmp;
}

void Boids::stepSimulationScatteredGrid(float dt) {
  int N = numObjects;
  dim3 fullBlocksPerGridBoids((N + blockSize - 1) / blockSize);
  dim3 fullBlocksPerGridCells((gridCellCount + blockSize - 1) / blockSize);

  // Initialize dev_particleArrayIndices and dev_particleGridIndices, which will be parallel sorted.
  kernComputeIndices << <fullBlocksPerGridBoids, blockSize >> > (
    N, gridSideCount, gridMinimum, gridInverseCellWidth,
    dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
  checkCUDAErrorWithLine("kernComputeIndices failed!");

  thrust::sort_by_key(
    dev_thrust_particleGridIndices,
    dev_thrust_particleGridIndices + N,
    dev_thrust_particleArrayIndices);

  kernResetIntBuffer << <fullBlocksPerGridCells, blockSize >> > (gridCellCount, dev_gridCellStartIndices, -1);
  kernResetIntBuffer << <fullBlocksPerGridCells, blockSize >> > (gridCellCount, dev_gridCellEndIndices, -1);
  checkCUDAErrorWithLine("kernResetIntBuffer failed!");

  // Populate dev_gridCell{Start,End}Indices.
  kernIdentifyCellStartEnd << <fullBlocksPerGridBoids, blockSize >> > (
    N, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
  checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

  kernUpdateVelNeighborSearchScattered << <fullBlocksPerGridBoids, blockSize >> > (
    N, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
    dev_gridCellStartIndices, dev_gridCellEndIndices, dev_particleArrayIndices,
    dev_pos, dev_vel1, dev_vel2);
  checkCUDAErrorWithLine("kernUpdateVelNeighborSearchScattered failed!");

  kernUpdatePos << <fullBlocksPerGridBoids, blockSize >> > (N, dt, dev_pos, dev_vel2);
  checkCUDAErrorWithLine("kernUpdatePos failed!");

  glm::vec3* tmp = dev_vel1;
  dev_vel1 = dev_vel2;
  dev_vel2 = tmp;
}

void Boids::stepSimulationCoherentGrid(float dt) {
  // TODO-2.3 - start by copying Boids::stepSimulationNaiveGrid
  // Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
  // In Parallel:
  // - Label each particle with its array index as well as its grid index.
  //   Use 2x width grids
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
  //   the particle data in the simulation array.
  //   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.

  int N = numObjects;
  dim3 fullBlocksPerGridBoids((N + blockSize - 1) / blockSize);
  dim3 fullBlocksPerGridCells((gridCellCount + blockSize - 1) / blockSize);

  // Initialize dev_particleArrayIndices and dev_particleGridIndices, which will be parallel sorted.
  kernComputeIndices << <fullBlocksPerGridBoids, blockSize >> > (
    N, gridSideCount, gridMinimum, gridInverseCellWidth,
    dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
  checkCUDAErrorWithLine("kernComputeIndices failed!");

  thrust::sort_by_key(
    dev_thrust_particleGridIndices,
    dev_thrust_particleGridIndices + N,
    dev_thrust_particleArrayIndices);

  kernResetIntBuffer << <fullBlocksPerGridCells, blockSize >> > (gridCellCount, dev_gridCellStartIndices, -1);
  kernResetIntBuffer << <fullBlocksPerGridCells, blockSize >> > (gridCellCount, dev_gridCellEndIndices, -1);
  checkCUDAErrorWithLine("kernResetIntBuffer failed!");

  // Populate dev_gridCell{Start,End}Indices.
  kernIdentifyCellStartEnd << <fullBlocksPerGridBoids, blockSize >> > (
    N, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
  checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

  // Preprocess to avoid chasing pointers
  kernSemiCoherent << <fullBlocksPerGridBoids, blockSize >> > (
    N, dev_particleArrayIndices,
    dev_pos, dev_pos_coherent,
    dev_vel1, dev_vel_coherent);
  checkCUDAErrorWithLine("kernSemiCoherent failed!");

  kernUpdateVelNeighborSearchCoherent << <fullBlocksPerGridBoids, blockSize >> > (
    N, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
    dev_gridCellStartIndices, dev_gridCellEndIndices,
    dev_pos_coherent, dev_vel_coherent, dev_vel2);
  checkCUDAErrorWithLine("kernUpdateVelNeighborSearchScattered failed!");

  glm::vec3* tmp = dev_pos;
  dev_pos = dev_pos_coherent;
  dev_pos_coherent = tmp;

  kernUpdatePos << <fullBlocksPerGridBoids, blockSize >> > (N, dt, dev_pos, dev_vel2);
  checkCUDAErrorWithLine("kernUpdatePos failed!");
  
  tmp = dev_vel1;
  dev_vel1 = dev_vel2;
  dev_vel2 = tmp;
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);

  // TODO-2.1 TODO-2.3 - Free any additional buffers here.
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_gridCellEndIndices);

  cudaFree(dev_pos_coherent);
  cudaFree(dev_vel_coherent);
}

void Boids::unitTest() {
  dim3 x(4);
  std::cout << x.x << x.y << x.z << std::endl;
  // LOOK-1.2 Feel free to write additional tests here.

  // test unstable sort
  int* dev_intKeys;
  int* dev_intValues;
  int N = 10;

  std::unique_ptr<int[]>intKeys{ new int[N] };
  std::unique_ptr<int[]>intValues{ new int[N] };

  intKeys[0] = 0; intValues[0] = 0;
  intKeys[1] = 1; intValues[1] = 1;
  intKeys[2] = 0; intValues[2] = 2;
  intKeys[3] = 3; intValues[3] = 3;
  intKeys[4] = 0; intValues[4] = 4;
  intKeys[5] = 2; intValues[5] = 5;
  intKeys[6] = 2; intValues[6] = 6;
  intKeys[7] = 0; intValues[7] = 7;
  intKeys[8] = 5; intValues[8] = 8;
  intKeys[9] = 6; intValues[9] = 9;

  cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

  cudaMalloc((void**)&dev_intValues, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  std::cout << "before unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // How to copy data to the GPU
  cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

  // Wrap device vectors in thrust iterators for use with thrust.
  thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
  thrust::device_ptr<int> dev_thrust_values(dev_intValues);
  // LOOK-2.1 Example for using thrust::sort_by_key
  thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

  // How to copy data back to the CPU side from the GPU
  cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("memcpy back failed!");

  std::cout << "after unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // cleanup
  cudaFree(dev_intKeys);
  cudaFree(dev_intValues);
  checkCUDAErrorWithLine("cudaFree failed!");
  return;
}
