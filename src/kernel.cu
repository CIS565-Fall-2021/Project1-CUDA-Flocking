#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include <cassert>
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

/**
* Return the distance between two boids
*/
__host__ __device__ float distanceBoid(const glm::vec3& firstBoid, const glm::vec3& secondBoid)
{
    return glm::length(firstBoid - secondBoid);
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 1024

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

// For efficient sorting and the uniform grid. These should always be parallel.
int *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int *dev_particleGridIndices; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.
glm::vec3* dev_pos_new;
glm::vec3* dev_vel1_new;

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
  gridCellWidth = 2.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  // TODO-2.1 TODO-2.3 - Allocate additional buffers here.
  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");

  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");

  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");

  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");

  cudaMalloc((void**)&dev_pos_new, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_vel1_new, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  dev_thrust_particleArrayIndices =
      thrust::device_pointer_cast(dev_particleArrayIndices);
  dev_thrust_particleGridIndices =
      thrust::device_pointer_cast(dev_particleGridIndices);

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
*  Rule 1: boids fly towards their local perceived center of mass, which 
*  excludes themselves
*/
__device__ glm::vec3 computeCohesion(int N, int iSelf, const glm::vec3* pos, 
                                     const glm::vec3* vel) 
{
    glm::vec3 perceived_center{0.0f, 0.0f, 0.0f};
    int number_of_neighbors{ 0 };

    for (int i = 0; i < N; i++)
    {
        if ((i != iSelf) && (distanceBoid(pos[iSelf], pos[i]) < rule1Distance))
        {
            number_of_neighbors++;
            perceived_center += pos[i];
        }
    }
    if (number_of_neighbors == 0)
    {
        // doesn't affect this boid if there are no neighbors
        return glm::vec3(0, 0, 0);
    }
    perceived_center /= number_of_neighbors;
    return (perceived_center - pos[iSelf]) * rule1Scale;
}

/**
*  Rule 2: boids try to stay a distance d away from each other
*/
__device__ glm::vec3 computeSeparation(int N, int iSelf, const glm::vec3* pos,
                                       const glm::vec3* vel)
{
    glm::vec3 c{ 0.0f, 0.0f, 0.0f };
    for (int i = 0; i < N; i++)
    {
        if ((i != iSelf) && (distanceBoid(pos[iSelf], pos[i]) < rule2Distance))
        {
             c -= (pos[i] - pos[iSelf]);
        }
    }
    return c * rule2Scale;
}

/**
*  Rule 3: boids try to match the speed of surrounding boids
*/
__device__ glm::vec3 computeAlignment(int N, int iSelf, const glm::vec3* pos,
                                      const glm::vec3* vel)
{
    glm::vec3 perceived_velocity{ 0.0f, 0.0f, 0.0f };
    int number_of_neighbors{ 0 };

    for (int i = 0; i < N; i++)
    {
        if ((i != iSelf) && (distanceBoid(pos[iSelf], pos[i]) < rule3Distance))
        {
            number_of_neighbors++;
            perceived_velocity += vel[i];
        }
    }
    if (number_of_neighbors == 0)
    {
        // doesn't affect this boid if there are no neighbors
        return glm::vec3(0.0f, 0.0f, 0.0f);
    }
    perceived_velocity /= number_of_neighbors;
    return perceived_velocity * rule3Scale;
}

/**
* LOOK-1.2 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, 
                                           const glm::vec3 *vel) 
{
  return vel[iSelf] + computeAlignment(N, iSelf, pos, vel) + 
                      computeSeparation(N, iSelf, pos, vel) +
                      computeCohesion(N, iSelf, pos, vel);
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
* Note: GLM 0.9. 2 introduces CUDA compiler support allowing programmer to 
* use GLM inside a CUDA Kernel.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
                                             glm::vec3 *vel1, glm::vec3 *vel2) 
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }
    // Compute a new velocity based on pos and vel1
    glm::vec3 newVelocity = computeVelocityChange(N, index, pos, vel1);

    glm::vec3 newDir = glm::normalize(newVelocity);
    float newSpeed = glm::length(newVelocity);

    // Clamp the speed 
    if (newSpeed >= maxSpeed)
    {
         newVelocity = newDir * maxSpeed;
    }
    // Record the new velocity into vel2. Question: why NOT vel1?
    vel2[index] = newVelocity;
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

__device__ int get1DCellIndex(int index, int gridResolution,
    glm::vec3 gridMin, float inverseCellWidth, glm::vec3* pos)
{
    int iX = glm::floor((pos[index].x - gridMin.x) * inverseCellWidth);
    int iY = glm::floor((pos[index].y - gridMin.y) * inverseCellWidth);
    int iZ = glm::floor((pos[index].z - gridMin.z) * inverseCellWidth);
    return gridIndex3Dto1D(iX, iY, iZ, gridResolution);
}

__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) 
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }
    // TODO-2.1
    // - Label each boid with the index of its grid cell.
    gridIndices[index] = get1DCellIndex(index, gridResolution, gridMin,
        inverseCellWidth, pos);
    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2
    indices[index] = index;
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
  // TODO-2.1
  // Identify the start point of each cell in the gridIndices array.
  // This is basically a parallel unrolling of a loop that goes
  // "this index doesn't match the one before it, must be a new cell!"
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }
    int cellNumIndex = particleGridIndices[index];
    // Last element in particleGridIndices
    if (index == N - 1)
    {
        gridCellEndIndices[cellNumIndex] = index;
    }
    // first element in particleGridIndices
    if (index == 0)
    {
        gridCellStartIndices[cellNumIndex] = index;
    }

    int indexBefore = index - 1;
    if (indexBefore >= 0)
    {
        int cellNumIndexBefore = particleGridIndices[indexBefore];
        if (cellNumIndexBefore != cellNumIndex)
        {
            gridCellEndIndices[cellNumIndexBefore] = indexBefore;
            gridCellStartIndices[cellNumIndex] = index;
        }
    }
}

__device__ void computeContributionFromThisCellCoherentApproach(int x, int y, int z, int index,
    glm::vec3& perceived_velocity, glm::vec3& perceived_center, int gridResolution,
    glm::vec3& c, int& number_of_neighbors_velocity, int& number_of_neighbors_center,
    int* gridCellStartIndices, int* gridCellEndIndices, glm::vec3* pos, glm::vec3* vel1)
{
    // don't need to check cells that are nonexistent
    if (x < 0 || x >= gridResolution || y < 0 || y >= gridResolution || z < 0 || z >= gridResolution)
    {
        return;
    }

    int cellIndex = gridIndex3Dto1D(x, y, z, gridResolution);
    int startIndex = gridCellStartIndices[cellIndex];
    int endIndex = gridCellEndIndices[cellIndex];

    // - For each cell, read the start/end indices in the boid pointer array.
    if ((startIndex != -1) && (endIndex != -1))
    {
        for (int i = startIndex; i <= endIndex; i++)
        {
            if (i != index)
            {
                float distanceThisBoidAndNeig = distanceBoid(pos[index], pos[i]);
                if (distanceThisBoidAndNeig < rule3Distance)
                {
                    number_of_neighbors_velocity++;
                    perceived_velocity += vel1[i];
                }

                if (distanceThisBoidAndNeig < rule2Distance)
                {
                    c -= (pos[i] - pos[index]);
                }

                if (distanceThisBoidAndNeig < rule1Distance)
                {
                    number_of_neighbors_center++;
                    perceived_center += pos[i];
                }
            }
        }
    }
}

__device__ void computeContributionFromThisCell(int x, int y, int z, int index,
    glm::vec3& perceived_velocity, glm::vec3& perceived_center, int gridResolution,
    glm::vec3& c, int& number_of_neighbors_velocity, int& number_of_neighbors_center,
    int* gridCellStartIndices, int* gridCellEndIndices,
    int* particleArrayIndices, glm::vec3* pos, glm::vec3* vel1)
{
    // don't need to check cells that are nonexistent
    if (x < 0 || x >= gridResolution || y < 0 || y >= gridResolution || z < 0 || z >= gridResolution)
    {
        return;
    }

    int cellIndex = gridIndex3Dto1D(x, y, z, gridResolution);
    int startIndex = gridCellStartIndices[cellIndex];
    int endIndex = gridCellEndIndices[cellIndex];

    // - For each cell, read the start/end indices in the boid pointer array.
    if ((startIndex != -1) && (endIndex != -1))
    {
        for (int i = startIndex; i <= endIndex; i++)
        {
            int particleIndex = particleArrayIndices[i];
            if (particleIndex != index)
            {
                float distanceThisBoidAndNeig = distanceBoid(pos[index], pos[particleIndex]);
                if (distanceThisBoidAndNeig < rule3Distance)
                {
                    number_of_neighbors_velocity++;
                    perceived_velocity += vel1[particleIndex];
                }

                if (distanceThisBoidAndNeig < rule2Distance)
                {
                    c -= (pos[particleIndex] - pos[index]);
                }

                if (distanceThisBoidAndNeig < rule1Distance)
                {
                    number_of_neighbors_center++;
                    perceived_center += pos[particleIndex];
                }
            }
        }
    }
}

__global__ void kernUpdateVelNeighborSearchScattered27Cells(
    int N, int gridResolution, glm::vec3 gridMin,
    float inverseCellWidth, float cellWidth,
    int* gridCellStartIndices, int* gridCellEndIndices,
    int* particleArrayIndices, glm::vec3* pos, glm::vec3* vel1, glm::vec3* vel2)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }

    // TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
    // the number of boids that need to be checked.
    // - Identify the grid cell that this particle is in

    // - Identify which cells may contain neighbors. This isn't always 8.

      // Probelm: you have only examined one cell, you may also to need to examine neighboring cells
      // return gridIndex3Dto1D(iX, iY, iZ, gridResolution);

    int iX = glm::floor((pos[index].x - gridMin.x) * inverseCellWidth);
    int iY = glm::floor((pos[index].y - gridMin.y) * inverseCellWidth);
    int iZ = glm::floor((pos[index].z - gridMin.z) * inverseCellWidth);

    glm::vec3 perceived_velocity{ 0.0f, 0.0f, 0.0f };
    glm::vec3 perceived_center{ 0.0f, 0.0f, 0.0f };
    glm::vec3 c{ 0.0f, 0.0f, 0.0f };
    int number_of_neighbors_velocity{ 0 };
    int number_of_neighbors_center{ 0 };

    for (int i = iZ - 1; i <= iZ + 1; i++)
    {
        for (int j = iY - 1; j <= iY + 1; j++)
        {
            for (int k = iX - 1; k <= iX + 1; k++)
            {
                // going through 27 cells is not necessary (NEED CHANGE!)
                int cellIndex = gridIndex3Dto1D(k, j, i, gridResolution);
                int startIndex = gridCellStartIndices[cellIndex];
                int endIndex = gridCellEndIndices[cellIndex];

                // - For each cell, read the start/end indices in the boid pointer array.
                if ((startIndex != -1) && (endIndex != -1))
                {
                    for (int i = startIndex; i <= endIndex; i++)
                    {
                        int particleIndex = particleArrayIndices[i];
                        if (particleIndex != index)
                        {
                            float distanceThisBoidAndNeig = distanceBoid(pos[index], pos[particleIndex]);
                            if (distanceThisBoidAndNeig < rule3Distance)
                            {
                                number_of_neighbors_velocity++;
                                perceived_velocity += vel1[particleIndex];
                            }

                            if (distanceThisBoidAndNeig < rule2Distance)
                            {
                                c -= (pos[particleIndex] - pos[index]);
                            }

                            if (distanceThisBoidAndNeig < rule1Distance)
                            {
                                number_of_neighbors_center++;
                                perceived_center += pos[particleIndex];
                            }
                        }
                    }

                }
            }
        }
    }

    glm::vec3 alignmentVelocity{ 0.0f, 0.0f, 0.0f };
    glm::vec3 separationVelocity{ 0.0f, 0.0f, 0.0f };
    glm::vec3 cohesionVelocity{ 0.0f, 0.0f, 0.0f };

    if (number_of_neighbors_center != 0)
    {
        perceived_center /= number_of_neighbors_center;
        cohesionVelocity = (perceived_center - pos[index]) * rule1Scale;
    }

    if (number_of_neighbors_velocity != 0)
    {
        perceived_velocity /= number_of_neighbors_velocity;
        alignmentVelocity = perceived_velocity * rule3Scale;
    }

    separationVelocity = c * rule2Scale;

    // - Access each boid in the cell and compute velocity change from
    //   the boids rules, if this boid is within the neighborhood distance.
    glm::vec3 newVelocity = vel1[index] + alignmentVelocity
        + separationVelocity + cohesionVelocity;

    glm::vec3 newDir = glm::normalize(newVelocity);
    float newSpeed = glm::length(newVelocity);

    // Clamp the speed 
    if (newSpeed >= maxSpeed)
    {
        newVelocity = newDir * maxSpeed;
    }
    // - Clamp the speed change before putting the new speed in vel2
    vel2[index] = newVelocity;
}

__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices, glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) 
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }

  // TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
  // - Identify the grid cell that this particle is in
    
  // - Identify which cells may contain neighbors. This isn't always 8.

    // Probelm: you have only examined one cell, you may also to need to examine neighboring cells
    int iX = glm::floor((pos[index].x - gridMin.x) * inverseCellWidth);
    int iY = glm::floor((pos[index].y - gridMin.y) * inverseCellWidth);
    int iZ = glm::floor((pos[index].z - gridMin.z) * inverseCellWidth);

    int cubeCornerX = iX * cellWidth + gridMin.x;
    int cubeCornerY = iY * cellWidth + gridMin.y;
    int cubeCornerZ = iZ * cellWidth + gridMin.z;

    float cellMiddleLineX = cubeCornerX + cellWidth / 2.0f;
    float cellMiddleLineY = cubeCornerY + cellWidth / 2.0f;
    float cellMiddleLineZ = cubeCornerZ + cellWidth / 2.0f;

    glm::vec3 perceived_velocity{ 0.0f, 0.0f, 0.0f };
    glm::vec3 perceived_center{ 0.0f, 0.0f, 0.0f };
    glm::vec3 c{ 0.0f, 0.0f, 0.0f };
    int number_of_neighbors_velocity{ 0 };
    int number_of_neighbors_center{ 0 };


    bool cellInLeftSideX = pos[index].x < cellMiddleLineX;
    bool cellInUpperPartY = pos[index].y < cellMiddleLineY;
    bool cellInLowerPartZ = pos[index].z < cellMiddleLineX;

    // Always check the cell the particle is in 
    computeContributionFromThisCell(iX, iY, iZ, index,
        perceived_velocity, perceived_center, gridResolution,
        c, number_of_neighbors_velocity, number_of_neighbors_center,
        gridCellStartIndices, gridCellEndIndices, particleArrayIndices,
        pos, vel1);
    if (cellInLowerPartZ)
    {
        computeContributionFromThisCell(iX, iY, iZ - 1, index,
            perceived_velocity, perceived_center, gridResolution,
            c, number_of_neighbors_velocity, number_of_neighbors_center,
            gridCellStartIndices, gridCellEndIndices, particleArrayIndices,
            pos, vel1);
    }
    else {
        computeContributionFromThisCell(iX, iY, iZ + 1, index,
            perceived_velocity, perceived_center, gridResolution,
            c, number_of_neighbors_velocity, number_of_neighbors_center,
            gridCellStartIndices, gridCellEndIndices, particleArrayIndices,
            pos, vel1);
    }


    if (cellInLeftSideX)
    {
        computeContributionFromThisCell(iX - 1, iY, iZ, index,
            perceived_velocity, perceived_center, gridResolution,
            c, number_of_neighbors_velocity, number_of_neighbors_center,
            gridCellStartIndices, gridCellEndIndices, particleArrayIndices,
            pos, vel1);
        if (cellInLowerPartZ)
        {
            computeContributionFromThisCell(iX - 1, iY, iZ - 1, index,
                perceived_velocity, perceived_center, gridResolution,
                c, number_of_neighbors_velocity, number_of_neighbors_center,
                gridCellStartIndices, gridCellEndIndices, particleArrayIndices,
                pos, vel1);
        }
        else {
            computeContributionFromThisCell(iX - 1, iY, iZ + 1, index,
                perceived_velocity, perceived_center, gridResolution,
                c, number_of_neighbors_velocity, number_of_neighbors_center,
                gridCellStartIndices, gridCellEndIndices, particleArrayIndices,
                pos, vel1);
        }
    }
    else {
        computeContributionFromThisCell(iX + 1, iY, iZ, index,
            perceived_velocity, perceived_center, gridResolution,
            c, number_of_neighbors_velocity, number_of_neighbors_center,
            gridCellStartIndices, gridCellEndIndices, particleArrayIndices,
            pos, vel1);
        if (cellInLowerPartZ)
        {
            computeContributionFromThisCell(iX + 1, iY, iZ - 1, index,
                perceived_velocity, perceived_center, gridResolution,
                c, number_of_neighbors_velocity, number_of_neighbors_center,
                gridCellStartIndices, gridCellEndIndices, particleArrayIndices,
                pos, vel1);
        }
        else {
            computeContributionFromThisCell(iX + 1, iY, iZ + 1, index,
                perceived_velocity, perceived_center, gridResolution,
                c, number_of_neighbors_velocity, number_of_neighbors_center,
                gridCellStartIndices, gridCellEndIndices, particleArrayIndices,
                pos, vel1);
        }
    }

    if (cellInUpperPartY)
    {
        computeContributionFromThisCell(iX, iY - 1, iZ, index,
            perceived_velocity, perceived_center, gridResolution,
            c, number_of_neighbors_velocity, number_of_neighbors_center,
            gridCellStartIndices, gridCellEndIndices, particleArrayIndices,
            pos, vel1);
        if (cellInLowerPartZ)
        {
            computeContributionFromThisCell(iX, iY - 1, iZ - 1, index,
                perceived_velocity, perceived_center, gridResolution,
                c, number_of_neighbors_velocity, number_of_neighbors_center,
                gridCellStartIndices, gridCellEndIndices, particleArrayIndices,
                pos, vel1);
        }
        else {
            computeContributionFromThisCell(iX, iY - 1, iZ + 1, index,
                perceived_velocity, perceived_center, gridResolution,
                c, number_of_neighbors_velocity, number_of_neighbors_center,
                gridCellStartIndices, gridCellEndIndices, particleArrayIndices,
                pos, vel1);
        }
    }
    else {
        computeContributionFromThisCell(iX, iY + 1, iZ, index,
            perceived_velocity, perceived_center, gridResolution,
            c, number_of_neighbors_velocity, number_of_neighbors_center,
            gridCellStartIndices, gridCellEndIndices, particleArrayIndices,
            pos, vel1);
        if (cellInLowerPartZ)
        {
            computeContributionFromThisCell(iX, iY + 1, iZ - 1, index,
                perceived_velocity, perceived_center, gridResolution,
                c, number_of_neighbors_velocity, number_of_neighbors_center,
                gridCellStartIndices, gridCellEndIndices, particleArrayIndices,
                pos, vel1);
        }
        else {
            computeContributionFromThisCell(iX, iY + 1, iZ + 1, index,
                perceived_velocity, perceived_center, gridResolution,
                c, number_of_neighbors_velocity, number_of_neighbors_center,
                gridCellStartIndices, gridCellEndIndices, particleArrayIndices,
                pos, vel1);
        }
    }

    if (cellInLeftSideX && cellInUpperPartY)
    {
        computeContributionFromThisCell(iX - 1, iY - 1, iZ, index,
            perceived_velocity, perceived_center, gridResolution,
            c, number_of_neighbors_velocity, number_of_neighbors_center,
            gridCellStartIndices, gridCellEndIndices, particleArrayIndices,
            pos, vel1);
        if (cellInLowerPartZ)
        {
            computeContributionFromThisCell(iX - 1, iY - 1, iZ - 1, index,
                perceived_velocity, perceived_center, gridResolution,
                c, number_of_neighbors_velocity, number_of_neighbors_center,
                gridCellStartIndices, gridCellEndIndices, particleArrayIndices,
                pos, vel1);
        }
        else {
            computeContributionFromThisCell(iX - 1, iY - 1, iZ + 1, index,
                perceived_velocity, perceived_center, gridResolution,
                c, number_of_neighbors_velocity, number_of_neighbors_center,
                gridCellStartIndices, gridCellEndIndices, particleArrayIndices,
                pos, vel1);
        }
    }
    else if (cellInLeftSideX && !cellInUpperPartY)
    {
        computeContributionFromThisCell(iX - 1, iY + 1, iZ, index,
            perceived_velocity, perceived_center, gridResolution,
            c, number_of_neighbors_velocity, number_of_neighbors_center,
            gridCellStartIndices, gridCellEndIndices, particleArrayIndices,
            pos, vel1);
        if (cellInLowerPartZ)
        {
            computeContributionFromThisCell(iX - 1, iY + 1, iZ - 1, index,
                perceived_velocity, perceived_center, gridResolution,
                c, number_of_neighbors_velocity, number_of_neighbors_center,
                gridCellStartIndices, gridCellEndIndices, particleArrayIndices,
                pos, vel1);
        }
        else {
            computeContributionFromThisCell(iX - 1, iY + 1, iZ + 1, index,
                perceived_velocity, perceived_center, gridResolution,
                c, number_of_neighbors_velocity, number_of_neighbors_center,
                gridCellStartIndices, gridCellEndIndices, particleArrayIndices,
                pos, vel1);
        }
    }
    else if (!cellInLeftSideX && cellInUpperPartY)
    {
        computeContributionFromThisCell(iX + 1, iY - 1, iZ, index,
            perceived_velocity, perceived_center, gridResolution,
            c, number_of_neighbors_velocity, number_of_neighbors_center,
            gridCellStartIndices, gridCellEndIndices, particleArrayIndices,
            pos, vel1);
        if (cellInLowerPartZ)
        {
            computeContributionFromThisCell(iX + 1, iY - 1, iZ - 1, index,
                perceived_velocity, perceived_center, gridResolution,
                c, number_of_neighbors_velocity, number_of_neighbors_center,
                gridCellStartIndices, gridCellEndIndices, particleArrayIndices,
                pos, vel1);
        }
        else {
            computeContributionFromThisCell(iX + 1, iY - 1, iZ + 1, index,
                perceived_velocity, perceived_center, gridResolution,
                c, number_of_neighbors_velocity, number_of_neighbors_center,
                gridCellStartIndices, gridCellEndIndices, particleArrayIndices,
                pos, vel1);
        }
    }
    else {
        computeContributionFromThisCell(iX + 1, iY + 1, iZ, index,
            perceived_velocity, perceived_center, gridResolution,
            c, number_of_neighbors_velocity, number_of_neighbors_center,
            gridCellStartIndices, gridCellEndIndices, particleArrayIndices,
            pos, vel1);
        if (cellInLowerPartZ)
        {
            computeContributionFromThisCell(iX + 1, iY + 1, iZ - 1, index,
                perceived_velocity, perceived_center, gridResolution,
                c, number_of_neighbors_velocity, number_of_neighbors_center,
                gridCellStartIndices, gridCellEndIndices, particleArrayIndices,
                pos, vel1);
        }
        else {
            computeContributionFromThisCell(iX + 1, iY + 1, iZ + 1, index,
                perceived_velocity, perceived_center, gridResolution,
                c, number_of_neighbors_velocity, number_of_neighbors_center,
                gridCellStartIndices, gridCellEndIndices, particleArrayIndices,
                pos, vel1);
        }
    }


    glm::vec3 alignmentVelocity{ 0.0f, 0.0f, 0.0f };
    glm::vec3 separationVelocity{ 0.0f, 0.0f, 0.0f };
    glm::vec3 cohesionVelocity{ 0.0f, 0.0f, 0.0f };

    if (number_of_neighbors_center != 0)
    {
        perceived_center /= number_of_neighbors_center;
        cohesionVelocity = (perceived_center - pos[index]) * rule1Scale;
    }

    if (number_of_neighbors_velocity != 0)
    {
        perceived_velocity /= number_of_neighbors_velocity;
        alignmentVelocity = perceived_velocity * rule3Scale;
    }
   
    separationVelocity = c * rule2Scale;
    
    // - Access each boid in the cell and compute velocity change from
    //   the boids rules, if this boid is within the neighborhood distance.
    glm::vec3 newVelocity = vel1[index] + alignmentVelocity
        + separationVelocity + cohesionVelocity;

    glm::vec3 newDir = glm::normalize(newVelocity);
    float newSpeed = glm::length(newVelocity);

    // Clamp the speed 
    if (newSpeed >= maxSpeed)
    {
        newVelocity = newDir * maxSpeed;
    }
    // - Clamp the speed change before putting the new speed in vel2
    vel2[index] = newVelocity;
}

__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
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
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }

    int iX = glm::floor((pos[index].x - gridMin.x) * inverseCellWidth);
    int iY = glm::floor((pos[index].y - gridMin.y) * inverseCellWidth);
    int iZ = glm::floor((pos[index].z - gridMin.z) * inverseCellWidth);

    int cubeCornerX = iX * cellWidth + gridMin.x;
    int cubeCornerY = iY * cellWidth + gridMin.y;
    int cubeCornerZ = iZ * cellWidth + gridMin.z;

    float cellMiddleLineX = cubeCornerX + cellWidth / 2.0f;
    float cellMiddleLineY = cubeCornerY + cellWidth / 2.0f;
    float cellMiddleLineZ = cubeCornerZ + cellWidth / 2.0f;

    glm::vec3 perceived_velocity{ 0.0f, 0.0f, 0.0f };
    glm::vec3 perceived_center{ 0.0f, 0.0f, 0.0f };
    glm::vec3 c{ 0.0f, 0.0f, 0.0f };
    int number_of_neighbors_velocity{ 0 };
    int number_of_neighbors_center{ 0 };


    bool cellInLeftSideX = pos[index].x < cellMiddleLineX;
    bool cellInUpperPartY = pos[index].y < cellMiddleLineY;
    bool cellInLowerPartZ = pos[index].z < cellMiddleLineX;

    // Always check the cell the particle is in 
    computeContributionFromThisCellCoherentApproach(iX, iY, iZ, index,
        perceived_velocity, perceived_center, gridResolution,
        c, number_of_neighbors_velocity, number_of_neighbors_center,
        gridCellStartIndices, gridCellEndIndices,
        pos, vel1);
    if (cellInLowerPartZ)
    {
        computeContributionFromThisCellCoherentApproach(iX, iY, iZ - 1, index,
            perceived_velocity, perceived_center, gridResolution,
            c, number_of_neighbors_velocity, number_of_neighbors_center,
            gridCellStartIndices, gridCellEndIndices,
            pos, vel1);
    }
    else {
        computeContributionFromThisCellCoherentApproach(iX, iY, iZ + 1, index,
            perceived_velocity, perceived_center, gridResolution,
            c, number_of_neighbors_velocity, number_of_neighbors_center,
            gridCellStartIndices, gridCellEndIndices,
            pos, vel1);
    }


    if (cellInLeftSideX)
    {
        computeContributionFromThisCellCoherentApproach(iX - 1, iY, iZ, index,
            perceived_velocity, perceived_center, gridResolution,
            c, number_of_neighbors_velocity, number_of_neighbors_center,
            gridCellStartIndices, gridCellEndIndices,
            pos, vel1);
        if (cellInLowerPartZ)
        {
            computeContributionFromThisCellCoherentApproach(iX - 1, iY, iZ - 1, index,
                perceived_velocity, perceived_center, gridResolution,
                c, number_of_neighbors_velocity, number_of_neighbors_center,
                gridCellStartIndices, gridCellEndIndices,
                pos, vel1);
        }
        else {
            computeContributionFromThisCellCoherentApproach(iX - 1, iY, iZ + 1, index,
                perceived_velocity, perceived_center, gridResolution,
                c, number_of_neighbors_velocity, number_of_neighbors_center,
                gridCellStartIndices, gridCellEndIndices,
                pos, vel1);
        }
    }
    else {
        computeContributionFromThisCellCoherentApproach(iX + 1, iY, iZ, index,
            perceived_velocity, perceived_center, gridResolution,
            c, number_of_neighbors_velocity, number_of_neighbors_center,
            gridCellStartIndices, gridCellEndIndices,
            pos, vel1);
        if (cellInLowerPartZ)
        {
            computeContributionFromThisCellCoherentApproach(iX + 1, iY, iZ - 1, index,
                perceived_velocity, perceived_center, gridResolution,
                c, number_of_neighbors_velocity, number_of_neighbors_center,
                gridCellStartIndices, gridCellEndIndices,
                pos, vel1);
        }
        else {
            computeContributionFromThisCellCoherentApproach(iX + 1, iY, iZ + 1, index,
                perceived_velocity, perceived_center, gridResolution,
                c, number_of_neighbors_velocity, number_of_neighbors_center,
                gridCellStartIndices, gridCellEndIndices,
                pos, vel1);
        }
    }

    if (cellInUpperPartY)
    {
        computeContributionFromThisCellCoherentApproach(iX, iY - 1, iZ, index,
            perceived_velocity, perceived_center, gridResolution,
            c, number_of_neighbors_velocity, number_of_neighbors_center,
            gridCellStartIndices, gridCellEndIndices,
            pos, vel1);
        if (cellInLowerPartZ)
        {
            computeContributionFromThisCellCoherentApproach(iX, iY - 1, iZ - 1, index,
                perceived_velocity, perceived_center, gridResolution,
                c, number_of_neighbors_velocity, number_of_neighbors_center,
                gridCellStartIndices, gridCellEndIndices,
                pos, vel1);
        }
        else {
            computeContributionFromThisCellCoherentApproach(iX, iY - 1, iZ + 1, index,
                perceived_velocity, perceived_center, gridResolution,
                c, number_of_neighbors_velocity, number_of_neighbors_center,
                gridCellStartIndices, gridCellEndIndices,
                pos, vel1);
        }
    }
    else {
        computeContributionFromThisCellCoherentApproach(iX, iY + 1, iZ, index,
            perceived_velocity, perceived_center, gridResolution,
            c, number_of_neighbors_velocity, number_of_neighbors_center,
            gridCellStartIndices, gridCellEndIndices,
            pos, vel1);
        if (cellInLowerPartZ)
        {
            computeContributionFromThisCellCoherentApproach(iX, iY + 1, iZ - 1, index,
                perceived_velocity, perceived_center, gridResolution,
                c, number_of_neighbors_velocity, number_of_neighbors_center,
                gridCellStartIndices, gridCellEndIndices,
                pos, vel1);
        }
        else {
            computeContributionFromThisCellCoherentApproach(iX, iY + 1, iZ + 1, index,
                perceived_velocity, perceived_center, gridResolution,
                c, number_of_neighbors_velocity, number_of_neighbors_center,
                gridCellStartIndices, gridCellEndIndices,
                pos, vel1);
        }
    }

    if (cellInLeftSideX && cellInUpperPartY)
    {
        computeContributionFromThisCellCoherentApproach(iX - 1, iY - 1, iZ, index,
            perceived_velocity, perceived_center, gridResolution,
            c, number_of_neighbors_velocity, number_of_neighbors_center,
            gridCellStartIndices, gridCellEndIndices,
            pos, vel1);
        if (cellInLowerPartZ)
        {
            computeContributionFromThisCellCoherentApproach(iX - 1, iY - 1, iZ - 1, index,
                perceived_velocity, perceived_center, gridResolution,
                c, number_of_neighbors_velocity, number_of_neighbors_center,
                gridCellStartIndices, gridCellEndIndices,
                pos, vel1);
        }
        else {
            computeContributionFromThisCellCoherentApproach(iX - 1, iY - 1, iZ + 1, index,
                perceived_velocity, perceived_center, gridResolution,
                c, number_of_neighbors_velocity, number_of_neighbors_center,
                gridCellStartIndices, gridCellEndIndices,
                pos, vel1);
        }
    }
    else if (cellInLeftSideX && !cellInUpperPartY)
    {
        computeContributionFromThisCellCoherentApproach(iX - 1, iY + 1, iZ, index,
            perceived_velocity, perceived_center, gridResolution,
            c, number_of_neighbors_velocity, number_of_neighbors_center,
            gridCellStartIndices, gridCellEndIndices,
            pos, vel1);
        if (cellInLowerPartZ)
        {
            computeContributionFromThisCellCoherentApproach(iX - 1, iY + 1, iZ - 1, index,
                perceived_velocity, perceived_center, gridResolution,
                c, number_of_neighbors_velocity, number_of_neighbors_center,
                gridCellStartIndices, gridCellEndIndices,
                pos, vel1);
        }
        else {
            computeContributionFromThisCellCoherentApproach(iX - 1, iY + 1, iZ + 1, index,
                perceived_velocity, perceived_center, gridResolution,
                c, number_of_neighbors_velocity, number_of_neighbors_center,
                gridCellStartIndices, gridCellEndIndices,
                pos, vel1);
        }
    }
    else if (!cellInLeftSideX && cellInUpperPartY)
    {
        computeContributionFromThisCellCoherentApproach(iX + 1, iY - 1, iZ, index,
            perceived_velocity, perceived_center, gridResolution,
            c, number_of_neighbors_velocity, number_of_neighbors_center,
            gridCellStartIndices, gridCellEndIndices,
            pos, vel1);
        if (cellInLowerPartZ)
        {
            computeContributionFromThisCellCoherentApproach(iX + 1, iY - 1, iZ - 1, index,
                perceived_velocity, perceived_center, gridResolution,
                c, number_of_neighbors_velocity, number_of_neighbors_center,
                gridCellStartIndices, gridCellEndIndices,
                pos, vel1);
        }
        else {
            computeContributionFromThisCellCoherentApproach(iX + 1, iY - 1, iZ + 1, index,
                perceived_velocity, perceived_center, gridResolution,
                c, number_of_neighbors_velocity, number_of_neighbors_center,
                gridCellStartIndices, gridCellEndIndices,
                pos, vel1);
        }
    }
    else {
        computeContributionFromThisCellCoherentApproach(iX + 1, iY + 1, iZ, index,
            perceived_velocity, perceived_center, gridResolution,
            c, number_of_neighbors_velocity, number_of_neighbors_center,
            gridCellStartIndices, gridCellEndIndices,
            pos, vel1);
        if (cellInLowerPartZ)
        {
            computeContributionFromThisCellCoherentApproach(iX + 1, iY + 1, iZ - 1, index,
                perceived_velocity, perceived_center, gridResolution, 
                c, number_of_neighbors_velocity, number_of_neighbors_center,
                gridCellStartIndices, gridCellEndIndices,
                pos, vel1);
        }
        else {
            computeContributionFromThisCellCoherentApproach(iX + 1, iY + 1, iZ + 1, index,
                perceived_velocity, perceived_center, gridResolution, 
                c, number_of_neighbors_velocity, number_of_neighbors_center,
                gridCellStartIndices, gridCellEndIndices,
                pos, vel1);
        }
    }


    glm::vec3 alignmentVelocity{ 0.0f, 0.0f, 0.0f };
    glm::vec3 separationVelocity{ 0.0f, 0.0f, 0.0f };
    glm::vec3 cohesionVelocity{ 0.0f, 0.0f, 0.0f };

    if (number_of_neighbors_center != 0)
    {
        perceived_center /= number_of_neighbors_center;
        cohesionVelocity = (perceived_center - pos[index]) * rule1Scale;
    }

    if (number_of_neighbors_velocity != 0)
    {
        perceived_velocity /= number_of_neighbors_velocity;
        alignmentVelocity = perceived_velocity * rule3Scale;
    }

    separationVelocity = c * rule2Scale;

    // - Access each boid in the cell and compute velocity change from
    //   the boids rules, if this boid is within the neighborhood distance.
    glm::vec3 newVelocity = vel1[index] + alignmentVelocity
        + separationVelocity + cohesionVelocity;

    glm::vec3 newDir = glm::normalize(newVelocity);
    float newSpeed = glm::length(newVelocity);

    // Clamp the speed 
    if (newSpeed >= maxSpeed)
    {
        newVelocity = newDir * maxSpeed;
    }
    // - Clamp the speed change before putting the new speed in vel2
    vel2[index] = newVelocity;
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
    assert(numObjects >= 0);

    // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
  // Boids examine neighboring boids to determine new celocity
    kernUpdateVelocityBruteForce <<<fullBlocksPerGrid, blockSize >>> (numObjects, 
                                                                      dev_pos,
                                                                      dev_vel1,
                                                                      dev_vel2);
  // Boids update position based on velocity and change in time
    kernUpdatePos <<<fullBlocksPerGrid, blockSize >>> (numObjects, dt,
                                                       dev_pos, dev_vel1);
    // TODO-1.2 ping-pong the velocity buffers
    glm::vec3* temp = dev_vel2;
    dev_vel2 = dev_vel1;
    dev_vel1 = temp;
}

void Boids::stepSimulationScatteredGrid(float dt) {
    assert(numObjects >= 0);
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
  // TODO-2.1
  // Uniform Grid Neighbor search using Thrust sort.
  // In Parallel:
  // - label each particle with its array index as well as its grid index.
  //   Use 2x width grids.
    kernComputeIndices <<<fullBlocksPerGrid, blockSize >>> (numObjects,
                                                            gridSideCount,
                                                            gridMinimum,
                                                            gridInverseCellWidth,
                                                            dev_pos, 
                                                            dev_particleArrayIndices,
                                                            dev_particleGridIndices);


    
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
    thrust::sort_by_key(dev_thrust_particleGridIndices,
        dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);


  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
   
    // -1 indicating that a cell does not enclose any boids
    dim3 fullBlocksPerGridCellArrays((gridCellCount + blockSize - 1) / blockSize);
    kernResetIntBuffer <<<fullBlocksPerGridCellArrays, blockSize >>> (gridCellCount,
        dev_gridCellStartIndices, -1);
    kernResetIntBuffer << <fullBlocksPerGridCellArrays, blockSize >> > (gridCellCount,
        dev_gridCellEndIndices, -1);
    kernIdentifyCellStartEnd <<<fullBlocksPerGrid, blockSize >>> (numObjects, 
        dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);


  // - Perform velocity updates using neighbor search
    kernUpdateVelNeighborSearchScattered <<<fullBlocksPerGrid, blockSize >>> (numObjects,
        gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth, 
        dev_gridCellStartIndices, dev_gridCellEndIndices, 
        dev_particleArrayIndices, dev_pos, dev_vel1, dev_vel2);

  // - Update positions
    kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt,
        dev_pos, dev_vel1);

  // - Ping-pong buffers as needed
    glm::vec3* temp = dev_vel2;
    dev_vel2 = dev_vel1;
    dev_vel1 = temp;
}

__global__ void kernReshuffle(
    int N,  int* particleArrayIndices, glm::vec3* pos, glm::vec3* pos_new, glm::vec3* vel, glm::vec3* vel_new)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }
    int particleIndex = particleArrayIndices[index];
    pos_new[index] = pos[particleIndex];
    vel_new[index] = vel[particleIndex];
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
  // - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
    kernComputeIndices << <fullBlocksPerGrid, blockSize >> > (numObjects,
        gridSideCount,
        gridMinimum,
        gridInverseCellWidth,
        dev_pos,
        dev_particleArrayIndices,
        dev_particleGridIndices);
    thrust::sort_by_key(dev_thrust_particleGridIndices,
        dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);
    dim3 fullBlocksPerGridCellArrays((gridCellCount + blockSize - 1) / blockSize);
    kernResetIntBuffer << <fullBlocksPerGridCellArrays, blockSize >> > (gridCellCount,
        dev_gridCellStartIndices, -1);
    kernResetIntBuffer << <fullBlocksPerGridCellArrays, blockSize >> > (gridCellCount,
        dev_gridCellEndIndices, -1);
    kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> > (numObjects,
        dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
    kernReshuffle <<<fullBlocksPerGrid, blockSize >>> (numObjects, 
        dev_particleArrayIndices, dev_pos, dev_pos_new, dev_vel1, dev_vel1_new);
    kernUpdateVelNeighborSearchCoherent << <fullBlocksPerGrid, blockSize >> > (numObjects,
         gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
        dev_gridCellStartIndices, dev_gridCellEndIndices,
        dev_pos_new, dev_vel1_new, dev_vel2);

    // - Update positions
    kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt,
        dev_pos_new, dev_vel1_new);

    // - Ping-pong buffers as needed
    glm::vec3* temp = dev_pos;
    dev_pos = dev_pos_new;
    dev_pos_new = temp;

    temp = dev_vel1;
    dev_vel1 = dev_vel1_new;
    dev_vel1_new = temp;

    temp = dev_vel2;
    dev_vel2 = dev_vel1;
    dev_vel1 = temp;
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
  cudaFree(dev_pos_new);
  cudaFree(dev_vel1_new);
}

void Boids::LabelingBoidWithGridCellIndexUnitTest()
{
    static bool runOnce = false;
    if (!runOnce) {
        runOnce = true;
        std::unique_ptr<int[]>testArray{ new int[numObjects] };
        // How to copy data back to the CPU side from the GPU
        cudaMemcpy(testArray.get(), dev_particleGridIndices, sizeof(int) * numObjects,
            cudaMemcpyDeviceToHost);

        std::cout << "dev_particleGridIndices: " << std::endl;
        for (int i = 0; i < numObjects; i++) {
            std::cout << "[" << i << "]: " << testArray[i] << '\n';
        }
    }
}

void Boids::LabelingBoidWithIndexUnitTest()
{
    static bool runOnce = false;
    if (!runOnce) {
        runOnce = true;
        std::unique_ptr<int[]>testArray{ new int[numObjects] };
        // How to copy data back to the CPU side from the GPU
        cudaMemcpy(testArray.get(), dev_particleArrayIndices, sizeof(int) * numObjects,
            cudaMemcpyDeviceToHost);

        std::cout << "dev_particleArrayIndices: " << std::endl;
        for (int i = 0; i < numObjects; i++) {
            std::cout << "[" << i << "]: " << testArray[i] << '\n';
        }
    }
}

void Boids::SortingUnitTest()
{
    static bool runOnce = false;
    if (!runOnce) {
        runOnce = true;

        std::unique_ptr<int[]>intKeys{ new int[numObjects] };
        std::unique_ptr<int[]>intValues{ new int[numObjects] };

        cudaMemcpy(intKeys.get(), dev_particleGridIndices, sizeof(int) * numObjects,
            cudaMemcpyDeviceToHost);
        cudaMemcpy(intValues.get(), dev_particleArrayIndices, sizeof(int) * numObjects,
            cudaMemcpyDeviceToHost);
        checkCUDAErrorWithLine("memcpy back failed!");

        std::cout << "after unstable sort: " << std::endl;
        for (int i = 0; i < numObjects; i++) {
            std::cout << "  dev_particleGridIndices: " << intKeys[i];
            std::cout << " dev_particleArrayIndices: " << intValues[i] << std::endl;
        }
    }
}

void Boids::StartEndUnitTest()
{
    static bool runOnce = false;
    if (!runOnce) {
        runOnce = true;

        std::unique_ptr<int[]>intKeys{ new int[gridCellCount] };
        std::unique_ptr<int[]>intValues{ new int[gridCellCount] };

        cudaMemcpy(intKeys.get(), dev_gridCellStartIndices, sizeof(int) * gridCellCount,
            cudaMemcpyDeviceToHost);
        cudaMemcpy(intValues.get(), dev_gridCellEndIndices, sizeof(int) * gridCellCount,
            cudaMemcpyDeviceToHost);
        checkCUDAErrorWithLine("memcpy back failed!");

        std::cout << "Start and end arrays: " << std::endl;
        for (int i = 0; i < gridCellCount; i++) {
            std::cout << "Cell: " << i << '\n';
            std::cout << "  dev_gridCellStartIndices: " << intKeys[i];
            std::cout << " dev_gridCellEndIndices: " << intValues[i] << std::endl;
        }
    }
}

void Boids::PrintCellStats()
{
    static bool runOnce = false;
    if (!runOnce) {
        runOnce = true;
        std::cout << "gridCellCount: " << gridCellCount << '\n';
        std::cout << "gridSideCount: " << gridSideCount << '\n';
        std::cout << "gridCellWidth: " << gridCellWidth << '\n';
        std::cout << "gridInverseCellWidth: " << gridInverseCellWidth << '\n';
        std::cout << "gridMinimum: " << gridMinimum.x << ", "
            << gridMinimum.y << ", " << gridMinimum.z << '\n';
    }
}

void Boids::unitTest() {
  // LOOK-1.2 Feel free to write additional tests here.
#if 0
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
#endif
}
