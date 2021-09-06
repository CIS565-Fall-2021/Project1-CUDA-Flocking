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
#define blockSize 32 //16 //32 //64 //1024 //512 //256 //128

/* Start my own definition */
// Clamp simply on each component.
#define clampFunc(dst, min, max) dst = glm::clamp(dst, min, max)

// If true, use rule3 from Conard Parker's note, otherwise follow the instruction.
#define USE_RULE3_FROM_CONARD_PARKER 0

// If true, use stable sort, otherwise unstable sort.
#define USE_STABLE_SORT 0 //1

// If true, use half cell width and check 27 cells, otherwise check 8 cells.
#define USE_HALF_SIZE_OF_CELL 0 //1

// If true, for loop x->y->z, otherwise z->y->x.
#define FOR_LOOP_XYZ 0 //1

// If true, adjust the search area by cell width.
#define GRID_LOOPING_OPTIMIZATION 0

#define blockSizePerDim blockSize // 8
/* End my own definition */

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

// LOOK-1.2 - These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.
glm::vec3 *dev_pos;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;

// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.
glm::vec3* dev_pos_reshuffle;
glm::vec3* dev_vel_reshuffle;

// For efficient sorting and the uniform grid. These should always be parallel.
int *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int *dev_particleGridIndices; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// DONE-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.

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
  cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  // LOOK-1.2 - This is a typical CUDA kernel invocation.
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
    dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  // LOOK-2.1 computing grid params
#if !GRID_LOOPING_OPTIMIZATION
#if !USE_HALF_SIZE_OF_CELL
  gridCellWidth = 2.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
#else // USE_HALF_SIZE_OF_CELL
  gridCellWidth = std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
#endif // USE_HALF_SIZE_OF_CELL
#else //GRID_LOOPING_OPTIMIZATION
  gridCellWidth = std::min(std::min(rule1Distance, rule2Distance), rule3Distance); // Not appropriate?
#endif // GRID_LOOPING_OPTIMIZATION
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  // DONE-2.1 DONE-2.3 - Allocate additional buffers here.

  // Start implementation 2.1
  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int)); // boid index -> sorted pos + vel index
  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int)); // boid index -> grid cell index
  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int)); // grid cell index -> start boid index
  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int)); // grid cell index -> end boid index

  dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
  dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);
  // End implementation 2.1

  // Start implementation 2.3
  cudaMalloc((void**)&dev_pos_reshuffle, N * sizeof(glm::vec3));
  cudaMalloc((void**)&dev_vel_reshuffle, N * sizeof(glm::vec3));
  // End implementation 2.3

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

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

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
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
  // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
  // Rule 2: boids try to stay a distance d away from each other
  // Rule 3: boids try to match the speed of surrounding boids

  // return glm::vec3(0.0f, 0.0f, 0.0f);

  // Start implementation
  glm::vec3 dVel(0.f, 0.f, 0.f);
  
  glm::vec3 perceivedCenter(0.f, 0.f, 0.f);
  glm::vec3 collisionP(0.f, 0.f, 0.f);
  glm::vec3 perceivedVel(0.f, 0.f, 0.f);

  int rule1Neighbor = 0, rule3Neighbor = 0;

  for (int i = 0; i < N; ++i) {
    glm::vec3 diff = pos[i] - pos[iSelf];
    float distance = glm::length(diff);
    if (i != iSelf) {
      // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
      if (distance < rule1Distance) {
        perceivedCenter += pos[i];
        ++rule1Neighbor;
      }

      // Rule 2: boids try to stay a distance d away from each other
      if (distance < rule2Distance) {
        collisionP -= diff;
      }

      // Rule 3: boids try to match the speed of surrounding boids
      if (distance < rule3Distance) {
        perceivedVel += vel[i];
        ++rule3Neighbor;
      }
    }
  }

  if (rule1Neighbor > 0) {
    perceivedCenter /= rule1Neighbor;
    dVel += (perceivedCenter - pos[iSelf]) * rule1Scale;
  }
  
  dVel += collisionP * rule2Scale;

  if (rule3Neighbor > 0) {
    perceivedVel /= rule3Neighbor;
#if !USE_RULE3_FROM_CONARD_PARKER
    dVel += perceivedVel * rule3Scale; // From instruction
#else // USE_RULE3_FROM_CONARD_PARKER
    dVel += (perceivedVel - vel[iSelf]) * rule3Scale; // From Conard Parker's, maybe it runs much better than the instructions' now? 
#endif // USE_RULE3_FROM_CONARD_PARKER
  }
  // End implementation

  return dVel;
}

/**
* DONE-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {
  // Compute a new velocity based on pos and vel1
  // Clamp the speed
  // Record the new velocity into vel2. Question: why NOT vel1?

  // Answer: Because the new velocity is related to the velocities of the other boids. If you change vel1, it may change the result of the new velocity.
  
  // Start implementation
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }

  // Compute a new velocity based on pos and vel1
  glm::vec3 dVel = computeVelocityChange(N, index, pos, vel1);
  vel2[index] = vel1[index] + dVel;

  // Clamp the speed
  clampFunc(vel2[index], -maxSpeed, maxSpeed);

  // End implementation
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
    // DONE-2.1
    // - Label each boid with the index of its grid cell.
    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2

  // Start implementation
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= N) {
    return;
  }

  // Init sorted boid index
  indices[index] = index;

  glm::vec3 posLocal = pos[index] - gridMin;
  glm::vec3 idxLocal = posLocal * inverseCellWidth;
  int x = idxLocal.x, y = idxLocal.y, z = idxLocal.z;
  int indexCell = gridIndex3Dto1D(x, y, z, gridResolution);

  gridIndices[index] = indexCell;
  // End implementation
}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

__global__ void kernIdentifyCellStartEnd(int N, int* particleGridIndices,
  int* gridCellStartIndices, int* gridCellEndIndices) {
  // DONE-2.1
  // Identify the start point of each cell in the gridIndices array.
  // This is basically a parallel unrolling of a loop that goes
  // "this index doesn't match the one before it, must be a new cell!"

  // Start implementation
  int indexCell = (blockIdx.x * blockDim.x) + threadIdx.x;
  // Can we use CPU global variable in CUDA device scpoe?
  //if (index >= gridCellCount) {
  //  return;
  //}

  for (int i = 0; i < N; ++i) {
    if (particleGridIndices[i] == indexCell && gridCellStartIndices[indexCell] == -1) {
      gridCellStartIndices[indexCell] = i;
    }
    if (particleGridIndices[i] > indexCell && gridCellStartIndices[indexCell] != -1) {
      gridCellEndIndices[indexCell] = i;
      break;
    }
  }
  // End implementation
}

__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // DONE-2.1 - Update a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2

  // Start implementation
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= N) {
    return;
  }

  int boidIndex = particleArrayIndices[index];

  // 2.1.1 Identify the cell
  glm::vec3 posLocal = pos[boidIndex] - gridMin;
  glm::vec3 idxLocal = posLocal * inverseCellWidth;
  //int x = idxLocal.x, y = idxLocal.y, z = idxLocal.z;
  //int indexCell = gridIndex3Dto1D(x, y, z, gridResolution);

  // 2.1.2 Identify which cells may contain neighbors.
#if !GRID_LOOPING_OPTIMIZATION
#if !USE_HALF_SIZE_OF_CELL
  int minX = idxLocal.x - 0.5f, minY = idxLocal.y - 0.5f, minZ = idxLocal.z - 0.5f;
  int maxX = idxLocal.x + 0.5f, maxY = idxLocal.y + 0.5f, maxZ = idxLocal.z + 0.5f;
#else // USE_HALF_SIZE_OF_CELL
  int minX = idxLocal.x - 1, minY = idxLocal.y - 1, minZ = idxLocal.z - 1;
  int maxX = idxLocal.x + 1, maxY = idxLocal.y + 1, maxZ = idxLocal.z + 1;
#endif // USE_HALF_SIZE_OF_CELL
#else // GRID_LOOPING_OPTIMIZATION
  float searchRadius = imax(rule1Distance, imax(rule2Distance, rule3Distance));
  float searchDiffIdx = searchRadius * inverseCellWidth;
  int minX = idxLocal.x - searchDiffIdx, minY = idxLocal.y - searchDiffIdx, minZ = idxLocal.z - searchDiffIdx;
  int maxX = idxLocal.x + searchDiffIdx, maxY = idxLocal.y + searchDiffIdx, maxZ = idxLocal.z + searchDiffIdx;
#endif // GRID_LOOPING_OPTIMIZATION

  minX = imax(minX, 0);
  maxX = imin(maxX, gridResolution - 1);
  minY = imax(minY, 0);
  maxY = imin(maxY, gridResolution - 1);
  minZ = imax(minZ, 0);
  maxZ = imin(maxZ, gridResolution - 1);

  // 2.1.3 For each cell, ...
  glm::vec3 dVel(0.f, 0.f, 0.f);

  glm::vec3 perceivedCenter(0.f, 0.f, 0.f);
  glm::vec3 collisionP(0.f, 0.f, 0.f);
  glm::vec3 perceivedVel(0.f, 0.f, 0.f);

  int rule1Neighbor = 0, rule3Neighbor = 0;

#if !FOR_LOOP_XYZ
  for (int z1 = minZ; z1 <= maxZ; ++z1) {
    for (int y1 = minY; y1 <= maxY; ++y1) {
      for (int x1 = minX; x1 <= maxX; ++x1) {
#else // FOR_LOOP_XYZ
  for (int x1 = minX; x1 <= maxX; ++x1) {
    for (int y1 = minY; y1 <= maxY; ++y1) {
      for (int z1 = minZ; z1 <= maxZ; ++z1) {
#endif // FOR_LOOP_XYZ
        int indexC1 = gridIndex3Dto1D(x1, y1, z1, gridResolution);
        int startIndex = gridCellStartIndices[indexC1];
        int endIndex = gridCellEndIndices[indexC1];
        if (startIndex == -1) {
          continue;
        }

        // 2.1.4 Acces each boid in the cell and compute velocity change
        for (int i = startIndex; i < endIndex; ++i) {
          int bi1 = particleArrayIndices[i];
          
          glm::vec3 diff = pos[bi1] - pos[boidIndex];
          float distance = glm::length(diff);
          if (bi1 != boidIndex) {
            // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
            if (distance < rule1Distance) {
              perceivedCenter += pos[bi1];
              ++rule1Neighbor;
            }

            // Rule 2: boids try to stay a distance d away from each other
            if (distance < rule2Distance) {
              collisionP -= diff;
            }

            // Rule 3: boids try to match the speed of surrounding boids
            if (distance < rule3Distance) {
              perceivedVel += vel1[bi1];
              ++rule3Neighbor;
            }
          }
        }
      }
    }
  }


  if (rule1Neighbor > 0) {
    perceivedCenter /= rule1Neighbor;
    dVel += (perceivedCenter - pos[boidIndex]) * rule1Scale;
  }

  dVel += collisionP * rule2Scale;

  if (rule3Neighbor > 0) {
    perceivedVel /= rule3Neighbor;
#if !USE_RULE3_FROM_CONARD_PARKER
    dVel += perceivedVel * rule3Scale; // From instruction
#else // USE_RULE3_FROM_CONARD_PARKER
    dVel += (perceivedVel - vel1[boidIndex]) * rule3Scale; // From Conard Parker's, maybe it runs much better than the instructions' now? 
#endif // USE_RULE3_FROM_CONARD_PARKER
  }

  vel2[boidIndex] = vel1[boidIndex] + dVel;

  // 2.1.5 Clamp the speed
  clampFunc(vel2[boidIndex], -maxSpeed, maxSpeed);

  // End implemetation
}

__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // DONE-2.3 - This should be very similar to kernUpdateVelNeighborSearchScattered,
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

  // Start implementation
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= N) {
    return;
  }

  // 2.3.1 Identify the cell
  glm::vec3 posLocal = pos[index] - gridMin;
  glm::vec3 idxLocal = posLocal * inverseCellWidth;
  //int x = idxLocal.x, y = idxLocal.y, z = idxLocal.z;
  //int indexCell = gridIndex3Dto1D(x, y, z, gridResolution);

  // 2.3.2 Identify which cells may contain neighbors.
#if !GRID_LOOPING_OPTIMIZATION
#if !USE_HALF_SIZE_OF_CELL
  int minX = idxLocal.x - 0.5f, minY = idxLocal.y - 0.5f, minZ = idxLocal.z - 0.5f;
  int maxX = idxLocal.x + 0.5f, maxY = idxLocal.y + 0.5f, maxZ = idxLocal.z + 0.5f;
#else // USE_HALF_SIZE_OF_CELL
  int minX = idxLocal.x - 1, minY = idxLocal.y - 1, minZ = idxLocal.z - 1;
  int maxX = idxLocal.x + 1, maxY = idxLocal.y + 1, maxZ = idxLocal.z + 1;
#endif // USE_HALF_SIZE_OF_CELL
#else // GRID_LOOPING_OPTIMIZATION
  float searchRadius = imax(rule1Distance, imax(rule2Distance, rule3Distance));
  float searchDiffIdx = searchRadius * inverseCellWidth;
  int minX = idxLocal.x - searchDiffIdx, minY = idxLocal.y - searchDiffIdx, minZ = idxLocal.z - searchDiffIdx;
  int maxX = idxLocal.x + searchDiffIdx, maxY = idxLocal.y + searchDiffIdx, maxZ = idxLocal.z + searchDiffIdx;
#endif // GRID_LOOPING_OPTIMIZATION

  minX = imax(minX, 0);
  maxX = imin(maxX, gridResolution - 1);
  minY = imax(minY, 0);
  maxY = imin(maxY, gridResolution - 1);
  minZ = imax(minZ, 0);
  maxZ = imin(maxZ, gridResolution - 1);

  // 2.3.3 For each cell, ...
  // Pay attention to the access order of cells for memory benefits
  glm::vec3 dVel(0.f, 0.f, 0.f);

  glm::vec3 perceivedCenter(0.f, 0.f, 0.f);
  glm::vec3 collisionP(0.f, 0.f, 0.f);
  glm::vec3 perceivedVel(0.f, 0.f, 0.f);

  int rule1Neighbor = 0, rule3Neighbor = 0;

#if !FOR_LOOP_XYZ
  for (int z1 = minZ; z1 <= maxZ; ++z1) {
    for (int y1 = minY; y1 <= maxY; ++y1) {
      for (int x1 = minX; x1 <= maxX; ++x1) {
#else // FOR_LOOP_XYZ
  for (int x1 = minX; x1 <= maxX; ++x1) {
    for (int y1 = minY; y1 <= maxY; ++y1) {
      for (int z1 = minZ; z1 <= maxZ; ++z1) {
#endif // FOR_LOOP_XYZ
        int indexC1 = gridIndex3Dto1D(x1, y1, z1, gridResolution);
        int startIndex = gridCellStartIndices[indexC1];
        int endIndex = gridCellEndIndices[indexC1];
        if (startIndex == -1) {
          continue;
        }

        // 2.3.4 Acces each boid in the cell and compute velocity change
        for (int i = startIndex; i < endIndex; ++i) {
          glm::vec3 diff = pos[i] - pos[index];
          float distance = glm::length(diff);
          if (i != index) {
            // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
            if (distance < rule1Distance) {
              perceivedCenter += pos[i];
              ++rule1Neighbor;
            }

            // Rule 2: boids try to stay a distance d away from each other
            if (distance < rule2Distance) {
              collisionP -= diff;
            }

            // Rule 3: boids try to match the speed of surrounding boids
            if (distance < rule3Distance) {
              perceivedVel += vel1[i];
              ++rule3Neighbor;
            }
          }
        }
      }
    }
  }


  if (rule1Neighbor > 0) {
    perceivedCenter /= rule1Neighbor;
    dVel += (perceivedCenter - pos[index]) * rule1Scale;
  }

  dVel += collisionP * rule2Scale;

  if (rule3Neighbor > 0) {
    perceivedVel /= rule3Neighbor;
#if !USE_RULE3_FROM_CONARD_PARKER
    dVel += perceivedVel * rule3Scale; // From instruction
#else // USE_RULE3_FROM_CONARD_PARKER
    dVel += (perceivedVel - vel1[boidIndex]) * rule3Scale; // From Conard Parker's, maybe it runs much better than the instructions' now? 
#endif // USE_RULE3_FROM_CONARD_PARKER
  }

  vel2[index] = vel1[index] + dVel;

  // 2.3.5 Clamp the speed
  clampFunc(vel2[index], -maxSpeed, maxSpeed);

  // End implemetation
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
  // DONE-1.2 - use the kernels you wrote to step the simulation forward in time.
  // DONE-1.2 ping-pong the velocity buffers

  // Start implementation
  // 1.2.1
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
  kernUpdateVelocityBruteForce<<<fullBlocksPerGrid, threadsPerBlock>>>(numObjects, dev_pos, dev_vel1, dev_vel2);
  kernUpdatePos<<<fullBlocksPerGrid, threadsPerBlock>>>(numObjects, dt, dev_pos, dev_vel2);

  // 1.2.2
  cudaMemcpy(dev_vel1, dev_vel2, numObjects * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
  
  // End implementation
}

void Boids::stepSimulationScatteredGrid(float dt) {
  // DONE-2.1
  // Uniform Grid Neighbor search using Thrust sort.
  // In Parallel:
  // - label each particle with its array index as well as its grid index.
  //   Use 2x width grids.
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed

  // Start implementation
  dim3 fullBlocksPerGridForBoid((numObjects + blockSize - 1) / blockSize);

  dim3 threadsPerBlock1DForCell((gridCellCount + blockSize - 1) / blockSize);

  int fullBlocksPerDimForCell = (gridSideCount + blockSizePerDim - 1) / blockSizePerDim;
  //dim3 threadsPerBlock3DForCell(blockSizePerDim, blockSizePerDim, blockSizePerDim);
  //dim3 fullBlocksPerGrid3DForCell(fullBlocksPerDimForCell, fullBlocksPerDimForCell, fullBlocksPerDimForCell);

  // 2.1.1 Label each particle
  kernComputeIndices<<<fullBlocksPerGridForBoid, threadsPerBlock>>>(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);

  // 2.1.2 Key sort (Key = boid index)
#if !USE_STABLE_SORT
  thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices); // Unstable sort
#else // USE_STABLE_SORT
  thrust::stable_sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices); // Stable sort, for comparison
#endif // USE_STABLE_SORT

  // 2.1.3 Find start and end
  kernResetIntBuffer<<<threadsPerBlock1DForCell, threadsPerBlock>>>(gridCellCount, dev_gridCellStartIndices, -1);
  kernResetIntBuffer<<<threadsPerBlock1DForCell, threadsPerBlock>>>(gridCellCount, dev_gridCellEndIndices, gridCellCount);
  kernIdentifyCellStartEnd<<<threadsPerBlock1DForCell, threadsPerBlock>>>(numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);

  // 2.1.4 Update velocity using neighbor search
  kernUpdateVelNeighborSearchScattered<<<fullBlocksPerGridForBoid, threadsPerBlock>>>(
    numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth, 
    dev_gridCellStartIndices, dev_gridCellEndIndices, dev_particleArrayIndices, dev_pos, dev_vel1, dev_vel2);

  // 2.1.5 Update positions
  kernUpdatePos<<<fullBlocksPerGridForBoid, threadsPerBlock>>>(numObjects, dt, dev_pos, dev_vel2);

  // 2.1.6 Ping-pong buffers
  cudaMemcpy(dev_vel1, dev_vel2, numObjects * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
  // End implementation
}

/**
* Reshuffle `src` to `dst`, using `key`.
*/
__global__ void kernReshuffle(int N, glm::vec3* dst, const glm::vec3* src, const int* key) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= N) {
    return;
  }
  int dstIdx = key[index];
  dst[index] = src[dstIdx];
}

void Boids::stepSimulationCoherentGrid(float dt) {
  // DONE-2.3 - start by copying Boids::stepSimulationNaiveGrid
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

  // Start implementation
  dim3 fullBlocksPerGridForBoid((numObjects + blockSize - 1) / blockSize);

  dim3 threadsPerBlock1DForCell((gridCellCount + blockSize - 1) / blockSize);

  int fullBlocksPerDimForCell = (gridSideCount + blockSizePerDim - 1) / blockSizePerDim;

  // 2.3.1 Label each particle
  kernComputeIndices<<<fullBlocksPerGridForBoid, threadsPerBlock>>>(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);

  // 2.3.2 Key sort (Key = boid index)
#if !USE_STABLE_SORT
  thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices); // Unstable sort
#else // USE_STABLE_SORT
  thrust::stable_sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices); // Stable sort, for comparison
#endif // USE_STABLE_SORT

  // 2.3.3 Find start and end
  kernResetIntBuffer<<<threadsPerBlock1DForCell, threadsPerBlock>>>(gridCellCount, dev_gridCellStartIndices, -1);
  kernResetIntBuffer<<<threadsPerBlock1DForCell, threadsPerBlock>>>(gridCellCount, dev_gridCellEndIndices, gridCellCount);
  kernIdentifyCellStartEnd<<<threadsPerBlock1DForCell, threadsPerBlock>>>(numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);

  // 2.3.4 Reshuffle pos+vel
  kernReshuffle<<<fullBlocksPerGridForBoid, threadsPerBlock>>>(numObjects, dev_pos_reshuffle, dev_pos, dev_particleArrayIndices);
  kernReshuffle<<<fullBlocksPerGridForBoid, threadsPerBlock>>>(numObjects, dev_vel_reshuffle, dev_vel1, dev_particleArrayIndices);

  // 2.3.5 Update velocity using neighbor search (vel2 is reshuffled)
  kernUpdateVelNeighborSearchCoherent<<<fullBlocksPerGridForBoid, threadsPerBlock>>>(
    numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth, 
    dev_gridCellStartIndices, dev_gridCellEndIndices, dev_pos_reshuffle, dev_vel_reshuffle, dev_vel2);

  // 2.3.6 Update positions
  kernUpdatePos<<<fullBlocksPerGridForBoid, threadsPerBlock>>>(numObjects, dt, dev_pos_reshuffle, dev_vel2);

  // 2.3.7 Ping-pong buffers
  cudaMemcpy(dev_vel1, dev_vel2, numObjects * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
  cudaMemcpy(dev_pos, dev_pos_reshuffle, numObjects * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
  // End implementation
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);

  // DONE-2.1 DONE-2.3 - Free any additional buffers here.

  // Start implementation 2.3
  cudaFree(dev_pos_reshuffle);
  cudaFree(dev_vel_reshuffle);
  // End implementation 2.3

  // Start implementation 2.1
  cudaFree(dev_gridCellEndIndices);
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_particleGridIndices);
  cudaFree(dev_particleArrayIndices);
  // End implementation 2.1
}

void Boids::unitTest() {
  // LOOK-1.2 Feel free to write additional tests here.

  int blockForBoid = (numObjects + blockSize - 1) / blockSize;
  int blockForCell = (gridCellCount + blockSizePerDim - 1) / blockSizePerDim;

  std::cout << "numObjects = " << numObjects;
  std::cout << "\ngridSize: " << gridSideCount << "^3 = " << gridCellCount;
  std::cout << "\nblockForBoid = " << blockForBoid << ", totalThreadForBoid = " << blockForBoid * blockSize;
  std::cout << "\nblockForCell = " << blockForCell << ", totalThreadForCell = " << blockForCell * blockSizePerDim;
  std::cout << std::endl;

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
