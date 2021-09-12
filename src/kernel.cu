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

#define WIDTHMODE 1 // toggle between 1 and 2 for cell widths

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

// LOOK-1.2 Parameters for the boids algorithm.
// These worked well in our reference implementation.
#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);
int currBuff = 1;

// LOOK-1.2 - These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.
glm::vec3 *dev_pos1;
glm::vec3 *dev_pos2;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;

// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
int *dev_particleArrayIndices; // What index in dev_pos1 and dev_velX represents this particle?
int *dev_particleGridIndices; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

/**
* LOOK-1.2 - this is a typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
  thrust::default_random_engine rng(hash((int)(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* LOOK-1.2 - This is a basic CUDA kernel.
* CUDA kernel for generating boids with a specified mass randomly around the star.
*/
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 * arr, float scale) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    glm::vec3 rand = generateRandomVec3(time, index);
    arr[index].x = scale * rand.x;
    arr[index].y = scale * rand.y;
    arr[index].z = scale * rand.z;
  }
}

/**
* Initialize memory, update some globals
*/
void Boids::initSimulation(int N) {
  numObjects = N;
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  // LOOK-1.2 - This is basic CUDA memory management and error checking.
  // Don't forget to cudaFree in  Boids::endSimulation.
  cudaMalloc((void**)&dev_pos1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos1 failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  // LOOK-1.2 - This is a typical CUDA kernel invocation.
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
    dev_pos1, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  // LOOK-2.1 computing grid params
  if (WIDTHMODE == 1) {
      gridCellWidth = std::max(std::max(rule1Distance, rule2Distance), rule3Distance);

  }
  else if (WIDTHMODE == 2) {
      gridCellWidth = 2.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  }
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

  cudaMalloc((void**)&dev_pos2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos2 failed!");

  cudaDeviceSynchronize();
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  if (index < N) {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_pos1, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_vel1, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyBoidsToVBO failed!");

  cudaDeviceSynchronize();
}


/******************
* stepSimulation *
******************/

/**
* LOOK-1.2 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3* pos, const glm::vec3* vel) {

    glm::vec3 perceived_center = glm::vec3(0.f, 0.f, 0.f);
    glm::vec3 c = glm::vec3(0.f, 0.f, 0.f);
    glm::vec3 perceived_velocity = glm::vec3(0.f, 0.f, 0.f);
    glm::vec3 out = glm::vec3(0.f, 0.f, 0.f);
    float neighborCountR1 = 0.f, neighborCountR3 = 0.f;
    for (int i = 0; i < N; i++) {
        float distance = glm::distance(pos[iSelf], pos[i]);
        if (i != iSelf) {
            if (distance < rule1Distance) {
                perceived_center += pos[i];
                neighborCountR1 += 1.0;
            }
            if (distance < rule2Distance) {
                c -= (pos[i] - pos[iSelf]);
            }
            if (distance < rule3Distance) {
                perceived_velocity += vel[i];
                neighborCountR3 += 1.0;
            }
        }
    }

    if (neighborCountR1 > 0) out += (perceived_center / (float)neighborCountR1 - pos[iSelf]) * rule1Scale;
    out += c * rule2Scale;
    if (neighborCountR3 > 0) out += (perceived_velocity / (float)neighborCountR3) * rule3Scale;

    return vel[iSelf] + out;
}

/**
* Update velocity of each boid in parallel by checking every other boid in the simulation
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {
  // Compute a new velocity based on pos and vel1
  // Clamp the speed
  // Record the new velocity into vel2. Question: why NOT vel1?
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) return;

    glm::vec3 velocity = computeVelocityChange(N, index, pos, vel1);
    vel2[index] = glm::length(velocity) > maxSpeed ? glm::normalize(velocity) * maxSpeed : velocity;
}

/**
* LOOK-1.2 Since this is pretty trivial, we implemented it for you.
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel) {
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

// LOOK-2.1 Consider this method of computing a 1D index from a 3D grid index.
// LOOK-2.3 Looking at this method, what would be the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order?
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) return;
    glm::vec3 grid = glm::floor((pos[index] - gridMin) * inverseCellWidth);
    gridIndices[index] = gridIndex3Dto1D(grid.x, grid.y, grid.z, gridResolution); // - Label each boid with the index of its grid cell.
    indices[index] = index; // parallel array of integer indices as pointers to the actual boid data in pos and vel1/vel2
}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
  int *gridCellStartIndices, int *gridCellEndIndices) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) return;
    gridCellStartIndices[index] = -1, gridCellEndIndices[index] = -1;
    int pIndex = particleGridIndices[index];
    if (index > 0) {
        if (particleGridIndices[index - 1] != particleGridIndices[index]) {
            gridCellStartIndices[particleGridIndices[index]] = index;
        }
    }
    if (index < N - 1) {
        if (particleGridIndices[index + 1] != particleGridIndices[index]) {
            gridCellEndIndices[particleGridIndices[index]] = index;
        }
    }
}

__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) return;

    // Compute center to divide grid into octants
    glm::vec3 positionSelf = pos[index];
    glm::vec3 grid = glm::floor((positionSelf - gridMin) / cellWidth);
    glm::vec3 center = ((grid * cellWidth) + gridMin) + glm::vec3(cellWidth / 2.f, cellWidth / 2.f, cellWidth / 2.f);
    glm::vec3 norm = (positionSelf - center) / cellWidth; // distance from center to the boid normalized to be -0.5 to 0.5 on all axes

    // Determine neighbors to check based on octants
    glm::vec3 negBound = glm::vec3(0.f, 0.f, 0.f);
    glm::vec3 posBound = glm::vec3(0.f, 0.f, 0.f);
    negBound.x = (norm.x <= 0.0f && norm.x > -0.5f) ? 1.f : 0.f;
    negBound.y = (norm.y <= 0.0f && norm.y > -0.5f) ? 1.f : 0.f;
    negBound.z = (norm.z <= 0.0f && norm.z > -0.5f) ? 1.f : 0.f;
    posBound.x = (norm.x > 0.0f && norm.x < 0.5f) ? 1.f : 0.f;
    posBound.y = (norm.y > 0.0f && norm.y < 0.5f) ? 1.f : 0.f;
    posBound.z = (norm.z > 0.0f && norm.z < 0.5f) ? 1.f : 0.f;

    // Apply boid rules to compute new velocity using neighboring grids
    glm::vec3 velocity = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 perceived_center = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 c = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 perceived_velocity = glm::vec3(0.0f, 0.0f, 0.0f);
    float neighborCountR1 = 0.f, neighborCountR3 = 0.f;
    for (int z = grid.z - negBound.z; z <= grid.z + posBound.z; z++) {
        for (int y = grid.y - negBound.y; y <= grid.y + posBound.y; y++) {
            for (int x = grid.x - negBound.x; x <= grid.x + posBound.x; x++) {
                int neighborCell = gridIndex3Dto1D(x, y, z, gridResolution);
                if (gridCellStartIndices[neighborCell] == -1) continue;
                for (int currBoid = gridCellStartIndices[neighborCell]; currBoid <= gridCellEndIndices[neighborCell]; currBoid++) {
                    glm::vec3 positionNeighbor = pos[particleArrayIndices[currBoid]];
                    if (particleArrayIndices[currBoid] != index) {
                        float distance = glm::distance(positionNeighbor, positionSelf);
                        if (distance < rule1Distance) {
                            perceived_center += positionNeighbor;
                            neighborCountR1++;
                        }
                        if (distance < rule2Distance) {
                            c -= (positionNeighbor - positionSelf);
                        }
                        if (distance < rule3Distance) {
                            perceived_velocity += vel1[particleArrayIndices[currBoid]];
                            neighborCountR3++;
                        }
                    }
                }
            }
        }
    }

    // Combine rule velocities
    if (neighborCountR1 > 0) velocity += (perceived_center / (float)neighborCountR1 - positionSelf) * rule1Scale;
    velocity += c * rule2Scale;
    if (neighborCountR3 > 0) velocity += (perceived_velocity / (float)neighborCountR3) * rule3Scale;

    // Combine, clamp, and set
    glm::vec3 out = vel1[index] + velocity;
    vel2[index] = glm::length(out) > maxSpeed ? maxSpeed * glm::normalize(out) : out;
}

__global__ void kernRearrangeIndices(int N, const int *particleArrayIndices, const glm::vec3 *pos1, const glm::vec3 *vel1,
    glm::vec3 *pos2, glm::vec3 *vel2) {
    // 1. find the particleArrayIndex for this thread
    // 2. look up the pos + vel values for this index
    // 3. set this thread's index in other pos/vel arrays to be these values
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) return;
    int pIndex = particleArrayIndices[index];
    pos2[index] = pos1[pIndex];
    vel2[index] = vel1[pIndex];
}

__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {

    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) return;

    // Compute center to divide grid into octants
    glm::vec3 positionSelf = pos[index];
    glm::vec3 grid = glm::floor((positionSelf - gridMin) / cellWidth);
    glm::vec3 center = ((grid * cellWidth) + gridMin) + glm::vec3(cellWidth / 2.f, cellWidth / 2.f, cellWidth / 2.f);
    glm::vec3 norm = (positionSelf - center) / cellWidth; // distance from center to the boid normalized to be -0.5 to 0.5 on all axes

    // Determine neighbors to check based on octants
    glm::vec3 negBound = glm::vec3(0.f, 0.f, 0.f);
    glm::vec3 posBound = glm::vec3(0.f, 0.f, 0.f);
    negBound.x = (norm.x <= 0.0f && norm.x > -0.5f) ? 1.f : 0.f;
    negBound.y = (norm.y <= 0.0f && norm.y > -0.5f) ? 1.f : 0.f;
    negBound.z = (norm.z <= 0.0f && norm.z > -0.5f) ? 1.f : 0.f;
    posBound.x = (norm.x > 0.0f && norm.x < 0.5f) ? 1.f : 0.f;
    posBound.y = (norm.y > 0.0f && norm.y < 0.5f) ? 1.f : 0.f;
    posBound.z = (norm.z > 0.0f && norm.z < 0.5f) ? 1.f : 0.f;

    // Apply boid rules to compute new velocity using neighboring grids
    glm::vec3 velocity = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 perceived_center = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 c = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 perceived_velocity = glm::vec3(0.0f, 0.0f, 0.0f);
    float neighborCountR1 = 0.f, neighborCountR3 = 0.f;
    for (int z = grid.z - negBound.z; z <= grid.z + posBound.z; z++) {
        for (int y = grid.y - negBound.y; y <= grid.y + posBound.y; y++) {
            for (int x = grid.x - negBound.x; x <= grid.x + posBound.x; x++) {
                int neighborCell = gridIndex3Dto1D(x, y, z, gridResolution);
                if (gridCellStartIndices[neighborCell] == -1) continue;
                for (int currBoid = gridCellStartIndices[neighborCell]; currBoid <= gridCellEndIndices[neighborCell]; currBoid++) {
                    glm::vec3 positionNeighbor = pos[currBoid];
                    if (currBoid != index) {
                        float distance = glm::distance(positionNeighbor, positionSelf);
                        if (distance < rule1Distance) {
                            perceived_center += positionNeighbor;
                            neighborCountR1++;
                        }
                        if (distance < rule2Distance) {
                            c -= (positionNeighbor - positionSelf);
                        }
                        if (distance < rule3Distance) {
                            perceived_velocity += vel1[currBoid];
                            neighborCountR3++;
                        }
                    }
                }
            }
        }
    }

    // Combine rule velocities
    if (neighborCountR1 > 0) velocity += (perceived_center / (float)neighborCountR1 - positionSelf) * rule1Scale;
    velocity += c * rule2Scale;
    if (neighborCountR3 > 0) velocity += (perceived_velocity / (float)neighborCountR3) * rule3Scale;

    // Combine, clamp, and set
    glm::vec3 out = vel1[index] + velocity;
    vel2[index] = glm::length(out) > maxSpeed ? maxSpeed * glm::normalize(out) : out;
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
    if (currBuff == 1) {
        kernUpdateVelocityBruteForce <<<fullBlocksPerGrid, blockSize>>> (numObjects, dev_pos1, dev_vel1, dev_vel2);
        kernUpdatePos <<<fullBlocksPerGrid, blockSize>>> (numObjects, dt, dev_pos1, dev_vel1);
        currBuff = 2;
    }
    else if (currBuff == 2) {
        kernUpdateVelocityBruteForce <<<fullBlocksPerGrid, blockSize>>> (numObjects, dev_pos1, dev_vel2, dev_vel1);
        kernUpdatePos <<<fullBlocksPerGrid, blockSize>>> (numObjects, dt, dev_pos1, dev_vel2);
        currBuff = 1;
    }
}

void Boids::stepSimulationScatteredGrid(float dt) {
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
    kernComputeIndices << <fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount, gridMinimum,
        gridInverseCellWidth, dev_pos1, dev_particleArrayIndices, dev_particleGridIndices);

    dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
    dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);

    thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);

    kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_particleGridIndices, dev_gridCellStartIndices,
        dev_gridCellEndIndices);

    kernUpdateVelNeighborSearchScattered << <fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount, gridMinimum,
        gridInverseCellWidth, gridCellWidth, dev_gridCellStartIndices, dev_gridCellEndIndices, dev_particleArrayIndices,
        dev_pos1, dev_vel1, dev_vel2);
    kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_pos1, dev_vel1);

    std::swap(dev_vel1, dev_vel2);
}

int num = 0;

void Boids::stepSimulationCoherentGrid(float dt) {
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
    kernComputeIndices << <fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount, gridMinimum,
        gridInverseCellWidth, dev_pos1, dev_particleArrayIndices, dev_particleGridIndices);

    dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
    dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);
    thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);

    kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_particleGridIndices, dev_gridCellStartIndices,
        dev_gridCellEndIndices);

    kernRearrangeIndices << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_particleArrayIndices, dev_pos1, dev_vel1, dev_pos2, dev_vel2);
    kernUpdateVelNeighborSearchCoherent << <fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount, gridMinimum,
        gridInverseCellWidth, gridCellWidth, dev_gridCellStartIndices, dev_gridCellEndIndices, dev_pos2, dev_vel2, dev_vel1);
    kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_pos2, dev_vel2);
    
    std::swap(dev_pos2, dev_pos1);
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos1);
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_gridCellEndIndices);
  cudaFree(dev_pos2);
}

void Boids::unitTest() {
  // test unstable sort
  int *dev_intKeys;
  int *dev_intValues;
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
