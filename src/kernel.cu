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
#define blockSize 128
//#define blockSize 512
//#define blockSize 1024

// LOOK-1.2 Parameters for the boids algorithm.
// These worked well in our reference implementation.
#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

//#define rule1Distance 25.0f
//#define rule2Distance 9.0f
//#define rule3Distance 25.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f

/*! Size of the starting area in simulation space. */
#define scene_scale 200.0f

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

// for 2.3
thrust::device_ptr<int> dev_thrust_pos;
thrust::device_ptr<int> dev_thrust_vel1;
thrust::device_ptr<int> dev_thrust_vel2;
int* dev_particleGridIndices_back;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.

// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;
float dev_time_ms = 0.0;
float dev_cell_scale;

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
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects, dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  // Initialize with random velocity
  //kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(2, numObjects, dev_vel1, maxSpeed * 0.5);
  //checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

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

  cudaMalloc((void**)&dev_particleGridIndices_back, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices_back failed!");

  cudaMalloc((void**)&dev_time_ms, sizeof(float));
  checkCUDAErrorWithLine("cudaMalloc dev_time_ms failed!");
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

    glm::vec3 return_velocity = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 this_pos = pos[iSelf];
    glm::vec3 center_of_mass = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 close_pos_sum = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 close_boid_vel = glm::vec3(0.0f, 0.0f, 0.0f);
    float rule1_boid_count = 0.0;
    float rule3_boid_count = 0.0;
    
    for (int boid_ind = 0; boid_ind < N; boid_ind++) {
        if (boid_ind != iSelf) {
            glm::vec3 boid_pos = pos[boid_ind];
            float distance = glm::length(boid_pos - this_pos);

            // rule 1
            if (distance < rule1Distance) {
                rule1_boid_count = rule1_boid_count + 1.0;
                center_of_mass = center_of_mass + boid_pos;
            }

            // rule 2
            if (distance < rule2Distance) {
                close_pos_sum = close_pos_sum - (boid_pos - this_pos);
            }

            // rule 3
            if (distance < rule3Distance) {
                rule3_boid_count = rule3_boid_count + 1.0;
                close_boid_vel = close_boid_vel + vel[boid_ind];
            }
        }
    }
    if (rule1_boid_count != 0) {
        center_of_mass = center_of_mass / rule1_boid_count;
        return_velocity = return_velocity + (center_of_mass - this_pos) * rule1Scale;
    }
    return_velocity = return_velocity + close_pos_sum * rule2Scale;
    if (rule3_boid_count != 0) {
        close_boid_vel = close_boid_vel / rule3_boid_count;
        return_velocity = return_velocity + close_boid_vel * rule3Scale;
    }
    return_velocity = return_velocity + vel[iSelf];
    if (glm::length(return_velocity) > 1) {
        return_velocity = glm::normalize(return_velocity) * maxSpeed;
    }
    return return_velocity;
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos, glm::vec3* vel1, glm::vec3* vel2) {
  // Compute a new velocity based on pos and vel1
  // Clamp the speed
  // Record the new velocity into vel2. Question: why NOT vel1?

    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }
    vel2[index] = computeVelocityChange(N, index, pos, vel1);
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
  glm::vec3 this_pos = pos[index];
  this_pos += vel[index] * dt;

  // Wrap the boids around so we don't lose them
  this_pos.x = this_pos.x < -scene_scale ? scene_scale : this_pos.x;
  this_pos.y = this_pos.y < -scene_scale ? scene_scale : this_pos.y;
  this_pos.z = this_pos.z < -scene_scale ? scene_scale : this_pos.z;

  this_pos.x = this_pos.x > scene_scale ? -scene_scale : this_pos.x;
  this_pos.y = this_pos.y > scene_scale ? -scene_scale : this_pos.y;
  this_pos.z = this_pos.z > scene_scale ? -scene_scale : this_pos.z;

  pos[index] = this_pos;
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
    // TODO-2.1
    // - Label each boid with the index of its grid cell.
    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index > N) {
        return;
    }
    glm::vec3 this_pos_from_edge = pos[index] - gridMin;
    indices[index] = index;
    gridIndices[index] = gridIndex3Dto1D(this_pos_from_edge.x * inverseCellWidth,
        this_pos_from_edge.y * inverseCellWidth,
        this_pos_from_edge.z * inverseCellWidth, gridResolution);
}

__global__ void kernComputeIndices2(int N, int gridResolution,
    glm::vec3 gridMin, float inverseCellWidth,
    glm::vec3* pos, int* gridIndices) {
    // TODO-2.1
    // - Label each boid with the index of its grid cell.
    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index > N) {
        return;
    }
    glm::vec3 this_pos_from_edge = pos[index] - gridMin;
    gridIndices[index] = gridIndex3Dto1D(this_pos_from_edge.x * inverseCellWidth,
        this_pos_from_edge.y * inverseCellWidth,
        this_pos_from_edge.z * inverseCellWidth, gridResolution);
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
    int grid_ind = particleGridIndices[index];
    if (index == 0) {
        gridCellStartIndices[grid_ind] = index;
    }
    else {
        int pre_grid_ind = particleGridIndices[index - 1];
        if (grid_ind != pre_grid_ind) {
            gridCellEndIndices[pre_grid_ind] = index;
            gridCellStartIndices[grid_ind] = index;
        }
    }
}

__device__ int* find_8_neighbors(float inverseCellWidth, int gridResolution, glm::vec3 this_pos_from_edge) {
    // find the neighbors
    float probe_distance = imax(imax(rule1Distance, rule2Distance), rule3Distance);
    int self_xyzs[3];
    int neighbor_xyzs[3] = { -1 };
    int neighbor_grid_inds[8] = { -1 };
    self_xyzs[0] = this_pos_from_edge.x * inverseCellWidth;
    self_xyzs[1] = this_pos_from_edge.y * inverseCellWidth;
    self_xyzs[2] = this_pos_from_edge.z * inverseCellWidth;

    // find the xyz to combine for the 8 neighbors
    int probe_grid_ind = (this_pos_from_edge.x + probe_distance) * inverseCellWidth; // +x
    if (probe_grid_ind != self_xyzs[0] && probe_grid_ind >= 0 && probe_grid_ind < gridResolution) {
        neighbor_xyzs[0] = probe_grid_ind;
    }
    else {
        probe_grid_ind = (this_pos_from_edge.x - probe_distance) * inverseCellWidth; // -x
        if (probe_grid_ind != self_xyzs[0] && probe_grid_ind >= 0 && probe_grid_ind < gridResolution) {
            neighbor_xyzs[0] = probe_grid_ind;
        }
    }

    probe_grid_ind = (this_pos_from_edge.y + probe_distance) * inverseCellWidth; // +y
    if (probe_grid_ind != self_xyzs[1] && probe_grid_ind >= 0 && probe_grid_ind < gridResolution) {
        neighbor_xyzs[1] = probe_grid_ind;
    }
    else {
        probe_grid_ind = (this_pos_from_edge.y - probe_distance) * inverseCellWidth; // -y
        if (probe_grid_ind != self_xyzs[1] && probe_grid_ind >= 0 && probe_grid_ind < gridResolution) {
            neighbor_xyzs[1] = probe_grid_ind;
        }
    }

    probe_grid_ind = (this_pos_from_edge.z + probe_distance) * inverseCellWidth; // +z
    if (probe_grid_ind != self_xyzs[2] && probe_grid_ind >= 0 && probe_grid_ind < gridResolution) {
        neighbor_xyzs[2] = probe_grid_ind;
    }
    else {
        probe_grid_ind = (this_pos_from_edge.z - probe_distance) * inverseCellWidth; // -z
        if (probe_grid_ind != self_xyzs[2] && probe_grid_ind >= 0 && probe_grid_ind < gridResolution) {
            neighbor_xyzs[2] = probe_grid_ind;
        }
    }

    // assemble the 8 neighbors
    neighbor_grid_inds[0] = gridIndex3Dto1D(self_xyzs[0], self_xyzs[1], self_xyzs[2], gridResolution);
    if (neighbor_xyzs[0] != -1) {
        neighbor_grid_inds[1] = gridIndex3Dto1D(neighbor_xyzs[0], self_xyzs[1], self_xyzs[2], gridResolution);

        if (neighbor_xyzs[1] != -1) {
            neighbor_grid_inds[2] = gridIndex3Dto1D(neighbor_xyzs[0], neighbor_xyzs[1], self_xyzs[2], gridResolution);
            if (neighbor_xyzs[2] != -1) {
                neighbor_grid_inds[4] = gridIndex3Dto1D(neighbor_xyzs[0], neighbor_xyzs[1], neighbor_xyzs[2], gridResolution);
            }
        }

        if (neighbor_xyzs[2] != -1) {
            neighbor_grid_inds[3] = gridIndex3Dto1D(neighbor_xyzs[0], self_xyzs[1], neighbor_xyzs[2], gridResolution);
        }
    }
    if (neighbor_xyzs[1] != -1) {
        neighbor_grid_inds[5] = gridIndex3Dto1D(self_xyzs[0], neighbor_xyzs[1], self_xyzs[2], gridResolution);
        if (neighbor_xyzs[2] != -1) {
            neighbor_grid_inds[6] = gridIndex3Dto1D(self_xyzs[0], neighbor_xyzs[1], neighbor_xyzs[2], gridResolution);
        }
    }
    if (neighbor_xyzs[2] != -1) {
        neighbor_grid_inds[7] = gridIndex3Dto1D(self_xyzs[0], self_xyzs[1], neighbor_xyzs[2], gridResolution);
    }
    return neighbor_grid_inds;
}

__device__ int* find_27_neighbors(float inverseCellWidth, int gridResolution, glm::vec3 this_pos_from_edge) {
    // find the neighbors
    float probe_distance = imax(imax(rule1Distance, rule2Distance), rule3Distance);
    int self_xyzs[3];
    int neighbor_xyzs[6] = { -1 };
    int neighbor_grid_inds[27] = { -1 };
    self_xyzs[0] = this_pos_from_edge.x * inverseCellWidth;
    self_xyzs[1] = this_pos_from_edge.y * inverseCellWidth;
    self_xyzs[2] = this_pos_from_edge.z * inverseCellWidth;

    // find the xyz to combine for the 27 neighbors
    int probe_grid_ind = (this_pos_from_edge.x + probe_distance) * inverseCellWidth; // +x
    if (probe_grid_ind != self_xyzs[0] && probe_grid_ind >= 0 && probe_grid_ind < gridResolution) {
        neighbor_xyzs[0] = probe_grid_ind;
    }
    probe_grid_ind = (this_pos_from_edge.x - probe_distance) * inverseCellWidth; // -x
    if (probe_grid_ind != self_xyzs[0] && probe_grid_ind >= 0 && probe_grid_ind < gridResolution) {
        neighbor_xyzs[1] = probe_grid_ind;
    }

    probe_grid_ind = (this_pos_from_edge.y + probe_distance) * inverseCellWidth; // +y
    if (probe_grid_ind != self_xyzs[1] && probe_grid_ind >= 0 && probe_grid_ind < gridResolution) {
        neighbor_xyzs[2] = probe_grid_ind;
    }
    probe_grid_ind = (this_pos_from_edge.y - probe_distance) * inverseCellWidth; // -y
    if (probe_grid_ind != self_xyzs[1] && probe_grid_ind >= 0 && probe_grid_ind < gridResolution) {
        neighbor_xyzs[3] = probe_grid_ind;
    }

    probe_grid_ind = (this_pos_from_edge.z + probe_distance) * inverseCellWidth; // +z
    if (probe_grid_ind != self_xyzs[2] && probe_grid_ind >= 0 && probe_grid_ind < gridResolution) {
        neighbor_xyzs[4] = probe_grid_ind;
    }
    probe_grid_ind = (this_pos_from_edge.z - probe_distance) * inverseCellWidth; // -z
    if (probe_grid_ind != self_xyzs[2] && probe_grid_ind >= 0 && probe_grid_ind < gridResolution) {
        neighbor_xyzs[5] = probe_grid_ind;
    }

    // assemble the 27 neighbors
    neighbor_grid_inds[0] = gridIndex3Dto1D(self_xyzs[0], self_xyzs[1], self_xyzs[2], gridResolution);
    if (neighbor_xyzs[0] != -1) { // 9 in +x
        neighbor_grid_inds[1] = gridIndex3Dto1D(neighbor_xyzs[0], self_xyzs[1], self_xyzs[2], gridResolution);

        if (neighbor_xyzs[2] != -1) {
            neighbor_grid_inds[2] = gridIndex3Dto1D(neighbor_xyzs[0], neighbor_xyzs[2], self_xyzs[2], gridResolution);
            if (neighbor_xyzs[4] != -1) {
                neighbor_grid_inds[3] = gridIndex3Dto1D(neighbor_xyzs[0], neighbor_xyzs[2], neighbor_xyzs[4], gridResolution);
            }
            if (neighbor_xyzs[5] != -1) {
                neighbor_grid_inds[4] = gridIndex3Dto1D(neighbor_xyzs[0], neighbor_xyzs[2], neighbor_xyzs[5], gridResolution);
            }
        }

        if (neighbor_xyzs[3] != -1) {
            neighbor_grid_inds[5] = gridIndex3Dto1D(neighbor_xyzs[0], neighbor_xyzs[3], self_xyzs[2], gridResolution);
            if (neighbor_xyzs[4] != -1) {
                neighbor_grid_inds[6] = gridIndex3Dto1D(neighbor_xyzs[0], neighbor_xyzs[3], neighbor_xyzs[4], gridResolution);
            }
            if (neighbor_xyzs[5] != -1) {
                neighbor_grid_inds[7] = gridIndex3Dto1D(neighbor_xyzs[0], neighbor_xyzs[3], neighbor_xyzs[5], gridResolution);
            }
        }

        if (neighbor_xyzs[4] != -1) {
            neighbor_grid_inds[8] = gridIndex3Dto1D(neighbor_xyzs[0], self_xyzs[1], neighbor_xyzs[4], gridResolution);
        }
        if (neighbor_xyzs[5] != -1) {
            neighbor_grid_inds[9] = gridIndex3Dto1D(neighbor_xyzs[0], self_xyzs[1], neighbor_xyzs[5], gridResolution);
        }
    }

    if (neighbor_xyzs[1] != -1) { // 9 in -x
        neighbor_grid_inds[10] = gridIndex3Dto1D(neighbor_xyzs[1], self_xyzs[1], self_xyzs[2], gridResolution);

        if (neighbor_xyzs[2] != -1) {
            neighbor_grid_inds[11] = gridIndex3Dto1D(neighbor_xyzs[1], neighbor_xyzs[2], self_xyzs[2], gridResolution);
            if (neighbor_xyzs[4] != -1) {
                neighbor_grid_inds[12] = gridIndex3Dto1D(neighbor_xyzs[1], neighbor_xyzs[2], neighbor_xyzs[4], gridResolution);
            }
            if (neighbor_xyzs[5] != -1) {
                neighbor_grid_inds[13] = gridIndex3Dto1D(neighbor_xyzs[1], neighbor_xyzs[2], neighbor_xyzs[5], gridResolution);
            }
        }

        if (neighbor_xyzs[3] != -1) {
            neighbor_grid_inds[14] = gridIndex3Dto1D(neighbor_xyzs[1], neighbor_xyzs[3], self_xyzs[2], gridResolution);
            if (neighbor_xyzs[4] != -1) {
                neighbor_grid_inds[15] = gridIndex3Dto1D(neighbor_xyzs[1], neighbor_xyzs[3], neighbor_xyzs[4], gridResolution);
            }
            if (neighbor_xyzs[5] != -1) {
                neighbor_grid_inds[16] = gridIndex3Dto1D(neighbor_xyzs[1], neighbor_xyzs[3], neighbor_xyzs[5], gridResolution);
            }
        }

        if (neighbor_xyzs[4] != -1) {
            neighbor_grid_inds[17] = gridIndex3Dto1D(neighbor_xyzs[1], self_xyzs[1], neighbor_xyzs[4], gridResolution);
        }
        if (neighbor_xyzs[5] != -1) {
            neighbor_grid_inds[18] = gridIndex3Dto1D(neighbor_xyzs[1], self_xyzs[1], neighbor_xyzs[5], gridResolution);
        }
    }

    if (neighbor_xyzs[2] != -1) { // 3 in +y
        neighbor_grid_inds[19] = gridIndex3Dto1D(self_xyzs[0], neighbor_xyzs[2], self_xyzs[2], gridResolution);
        if (neighbor_xyzs[4] != -1) {
            neighbor_grid_inds[20] = gridIndex3Dto1D(self_xyzs[0], neighbor_xyzs[2], neighbor_xyzs[4], gridResolution);
        }
        if (neighbor_xyzs[5] != -1) {
            neighbor_grid_inds[21] = gridIndex3Dto1D(self_xyzs[0], neighbor_xyzs[2], neighbor_xyzs[5], gridResolution);
        }
    }

    if (neighbor_xyzs[3] != -1) { // 3 in -y
        neighbor_grid_inds[22] = gridIndex3Dto1D(self_xyzs[0], neighbor_xyzs[3], self_xyzs[2], gridResolution);
        if (neighbor_xyzs[4] != -1) {
            neighbor_grid_inds[23] = gridIndex3Dto1D(self_xyzs[0], neighbor_xyzs[3], neighbor_xyzs[4], gridResolution);
        }
        if (neighbor_xyzs[5] != -1) {
            neighbor_grid_inds[24] = gridIndex3Dto1D(self_xyzs[0], neighbor_xyzs[3], neighbor_xyzs[5], gridResolution);
        }
    }

    if (neighbor_xyzs[4] != -1) { // 1 in +z
        neighbor_grid_inds[25] = gridIndex3Dto1D(self_xyzs[0], self_xyzs[1], neighbor_xyzs[4], gridResolution);
    }
    if (neighbor_xyzs[5] != -1) { // 1 in -z
        neighbor_grid_inds[26] = gridIndex3Dto1D(self_xyzs[0], self_xyzs[1], neighbor_xyzs[5], gridResolution);
    }

    return neighbor_grid_inds;
}

__global__ void kernUpdateVelNeighborSearchScattered(
    int N, int gridResolution, glm::vec3 gridMin,
    float inverseCellWidth, float cellWidth,
    int* gridCellStartIndices, int* gridCellEndIndices,
    int* particleArrayIndices,
    glm::vec3* pos, glm::vec3* vel1, glm::vec3* vel2) {
    // TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
    // the number of boids that need to be checked.
    // - Identify the grid cell that this particle is in
    // - Identify which cells may contain neighbors. This isn't always 8.
    // - For each cell, read the start/end indices in the boid pointer array.
    // - Access each boid in the cell and compute velocity change from
    //   the boids rules, if this boid is within the neighborhood distance.
    // - Clamp the speed change before putting the new speed in vel2

    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }
    glm::vec3 this_pos = pos[index];
    glm::vec3 this_pos_from_edge = this_pos - gridMin;
    int* neighbor_grid_inds;
    neighbor_grid_inds = find_8_neighbors(inverseCellWidth, gridResolution, this_pos_from_edge);
    //neighbor_grid_inds = find_27_neighbors(inverseCellWidth, gridResolution, this_pos_from_edge);
    

    // find all velcities
    glm::vec3 return_velocity = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 center_of_mass = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 close_pos_sum = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 close_boid_vel = glm::vec3(0.0f, 0.0f, 0.0f);
    float rule1_boid_count = 0.0;
    float rule3_boid_count = 0.0;
    for (int neighbor_grid_list_ind = 0; neighbor_grid_list_ind < 8; neighbor_grid_list_ind++) {
        int neighbor_grid_ind = neighbor_grid_inds[neighbor_grid_list_ind];
        if (neighbor_grid_ind != -1) {
            for (int boid_in_grid_ind = gridCellStartIndices[neighbor_grid_ind];
                boid_in_grid_ind < gridCellEndIndices[neighbor_grid_ind]; boid_in_grid_ind++) {
                int boid_ind = particleArrayIndices[boid_in_grid_ind];
                if (boid_ind != index) {
                    glm::vec3 boid_pos = pos[boid_ind];
                    float distance = glm::length(boid_pos - this_pos);

                    // rule 1
                    if (distance < rule1Distance) {
                        rule1_boid_count = rule1_boid_count + 1.0;
                        center_of_mass = center_of_mass + boid_pos;
                    }

                    // rule 2
                    if (distance < rule2Distance) {
                        close_pos_sum = close_pos_sum - (boid_pos - this_pos);
                    }

                    // rule 3
                    if (distance < rule3Distance) {
                        rule3_boid_count = rule3_boid_count + 1.0;
                        close_boid_vel = close_boid_vel + vel1[boid_ind];
                    }
                }
            }
        }
    }
    if (rule1_boid_count != 0) {
        center_of_mass = center_of_mass / rule1_boid_count;
        return_velocity = return_velocity + (center_of_mass - this_pos) * rule1Scale;
    }
    return_velocity = return_velocity + close_pos_sum * rule2Scale;
    if (rule3_boid_count != 0) {
        close_boid_vel = close_boid_vel / rule3_boid_count;
        return_velocity = return_velocity + close_boid_vel * rule3Scale;
    }
    return_velocity = return_velocity + vel1[index];
    if (glm::length(return_velocity) > 1) {
        return_velocity = glm::normalize(return_velocity) * maxSpeed;
    }
    vel2[index] = return_velocity;
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
    glm::vec3 this_pos = pos[index];
    glm::vec3 this_pos_from_edge = this_pos - gridMin;
    int* neighbor_grid_inds;
    neighbor_grid_inds = find_8_neighbors(inverseCellWidth, gridResolution, this_pos_from_edge);
    //neighbor_grid_inds = find_27_neighbors(inverseCellWidth, gridResolution, this_pos_from_edge);

    // find all velcities
    glm::vec3 return_velocity = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 center_of_mass = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 close_pos_sum = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 close_boid_vel = glm::vec3(0.0f, 0.0f, 0.0f);
    float rule1_boid_count = 0.0;
    float rule3_boid_count = 0.0;
    for (int neighbor_grid_list_ind = 0; neighbor_grid_list_ind < 8; neighbor_grid_list_ind++) {
        int neighbor_grid_ind = neighbor_grid_inds[neighbor_grid_list_ind];
        if (neighbor_grid_ind != -1) {
            for (int boid_in_grid_ind = gridCellStartIndices[neighbor_grid_ind];
                boid_in_grid_ind < gridCellEndIndices[neighbor_grid_ind]; boid_in_grid_ind++) {
                if (boid_in_grid_ind != index) {
                    glm::vec3 boid_pos = pos[boid_in_grid_ind];
                    float distance = glm::length(boid_pos - this_pos);

                    // rule 1
                    if (distance < rule1Distance) {
                        rule1_boid_count = rule1_boid_count + 1.0;
                        center_of_mass = center_of_mass + boid_pos;
                    }

                    // rule 2
                    if (distance < rule2Distance) {
                        close_pos_sum = close_pos_sum - (boid_pos - this_pos);
                    }

                    // rule 3
                    if (distance < rule3Distance) {
                        rule3_boid_count = rule3_boid_count + 1.0;
                        close_boid_vel = close_boid_vel + vel1[boid_in_grid_ind];
                    }
                }
            }
        }
    }
    if (rule1_boid_count != 0) {
        center_of_mass = center_of_mass / rule1_boid_count;
        return_velocity = return_velocity + (center_of_mass - this_pos) * rule1Scale;
    }
    return_velocity = return_velocity + close_pos_sum * rule2Scale;
    if (rule3_boid_count != 0) {
        close_boid_vel = close_boid_vel / rule3_boid_count;
        return_velocity = return_velocity + close_boid_vel * rule3Scale;
    }
    return_velocity = return_velocity + vel1[index];
    if (glm::length(return_velocity) > 1) {
        return_velocity = glm::normalize(return_velocity) * maxSpeed;
    }
    vel2[index] = return_velocity;
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
  // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
  // TODO-1.2 ping-pong the velocity buffers
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

    kernUpdateVelocityBruteForce << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_pos, dev_vel1, dev_vel2);
    checkCUDAErrorWithLine("kernUpdateVelocityBruteForce failed!");

    cudaMemcpy(dev_vel1, dev_vel2, sizeof(glm::vec3) * numObjects, cudaMemcpyDeviceToDevice);

    kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_pos, dev_vel1);
    checkCUDAErrorWithLine("kernUpdatePos failed!");

}

void Boids::stepSimulationScatteredGrid(float dt) {
  // TODO-2.1
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

    // compute grid_ind for each boid
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
    kernComputeIndices << <fullBlocksPerGrid, blockSize >> > (
        numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
    checkCUDAErrorWithLine("kernUpdateVelocityBruteForce failed!");

    // sort the grid_ind-boid_ind by grid_ind
    thrust::device_ptr<int> dev_thrust_particleArrayIndices(dev_particleArrayIndices);
    thrust::device_ptr<int> dev_thrust_particleGridIndices(dev_particleGridIndices);
    thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);
    
    // find boid in each grid cell
    kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> > (
        numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
    
    // update pos and vels
    kernUpdateVelNeighborSearchScattered << <fullBlocksPerGrid, blockSize >> > (
        numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
        dev_gridCellStartIndices, dev_gridCellEndIndices, dev_particleArrayIndices,
        dev_pos, dev_vel1, dev_vel2);
    checkCUDAErrorWithLine("kernUpdateVelNeighborSearchScattered failed!");

    cudaMemcpy(dev_vel1, dev_vel2, sizeof(glm::vec3) * numObjects, cudaMemcpyDeviceToDevice);

    kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_pos, dev_vel1);
    checkCUDAErrorWithLine("kernUpdatePos failed!");

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
    //cudaEvent_t start, stop;
    //cudaEventCreate(&start);
    //cudaEventCreate(&stop);
    //cudaEventRecord(start);

    // compute grid_ind for each boid
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
    kernComputeIndices2 << <fullBlocksPerGrid, blockSize >> > (
        numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos, dev_particleGridIndices);
    checkCUDAErrorWithLine("kernUpdateVelocityBruteForce failed!");

    // sort the grid_ind-boid_ind by grid_ind
    thrust::device_ptr<glm::vec3> dev_thrust_pos(dev_pos);
    thrust::device_ptr<glm::vec3> dev_thrust_vel1(dev_vel1);
    thrust::device_ptr<glm::vec3> dev_thrust_vel2(dev_vel2);
    thrust::device_ptr<int> dev_thrust_particleGridIndices(dev_particleGridIndices);
    cudaMemcpy(dev_particleGridIndices_back, dev_particleGridIndices, sizeof(int) * numObjects, cudaMemcpyDeviceToDevice);
    thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_pos);
    cudaMemcpy(dev_particleGridIndices, dev_particleGridIndices_back, sizeof(int) * numObjects, cudaMemcpyDeviceToDevice);
    thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_vel1);
    cudaMemcpy(dev_particleGridIndices, dev_particleGridIndices_back, sizeof(int) * numObjects, cudaMemcpyDeviceToDevice);
    thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_vel2);

    // find boid in each grid cell
    kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> > (
        numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);

    // update pos and vels
    kernUpdateVelNeighborSearchCoherent << <fullBlocksPerGrid, blockSize >> > (
        numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
        dev_gridCellStartIndices, dev_gridCellEndIndices,
        dev_pos, dev_vel1, dev_vel2);
    checkCUDAErrorWithLine("kernUpdateVelNeighborSearchScattered failed!");

    cudaMemcpy(dev_vel1, dev_vel2, sizeof(glm::vec3) * numObjects, cudaMemcpyDeviceToDevice);

    kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_pos, dev_vel1);
    checkCUDAErrorWithLine("kernUpdatePos failed!");

    //cudaEventRecord(stop);
    //cudaEventSynchronize(stop);
    //float milliseconds = 0;
    //cudaEventElapsedTime(&milliseconds, start, stop);
    //dev_time_ms = dev_time_ms * 0.99 + milliseconds * 0.01;
    //printf("%.6f /n", dev_time_ms);
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel1);
  cudaFree(dev_pos);

  // TODO-2.1 TODO-2.3 - Free any additional buffers here.
}

void Boids::unitTest() {
  // LOOK-1.2 Feel free to write additional tests here.

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
