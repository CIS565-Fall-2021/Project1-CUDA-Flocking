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
glm::vec3 *dev_sortedPos;
glm::vec3 *dev_sortedVel1;
thrust::device_ptr<glm::vec3> dev_thrust_vel2;

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
  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));

  dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
  dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);

  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));

  cudaMalloc((void**)&dev_sortedPos, N * sizeof(glm::vec3));
  cudaMalloc((void**)&dev_sortedVel1, N * sizeof(glm::vec3));

  dev_thrust_vel2 = thrust::device_ptr<glm::vec3>(dev_vel2);

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
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) 
{
	glm::vec3 center = glm::vec3(0.f, 0.f, 0.f);
	glm::vec3 separate = glm::vec3(0.f, 0.f, 0.f);
	glm::vec3 cohesion = glm::vec3(0.f, 0.f, 0.f);
	int neighborCount = 0;

	glm::vec3 boidVel = vel[iSelf];
	const glm::vec3 &boidPos = pos[iSelf];

	// iterate through all of the boids
	for (int i = 0; i < N; i++) {
		if (i == iSelf) continue;
		const glm::vec3 &neighborPos = pos[i];
		const glm::vec3 &neighborVel = vel[i];

		// if boid is in this boid's neighborhood, perform flocking calcs
		float distance = glm::distance(neighborPos, boidPos);
		if (distance < rule1Distance) {
			center += neighborPos;
			neighborCount++;

			if (distance < rule2Distance) {
				separate -= (neighborPos - boidPos);
			}

			cohesion += neighborVel;
		}
	}
	if (neighborCount > 0) {
		center /= neighborCount;
		boidVel += (center - boidPos) * rule1Scale;
		boidVel += cohesion * rule3Scale;
	}
	boidVel += separate * rule2Scale;

	return boidVel;
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
	glm::vec3 *vel1, glm::vec3 *vel2) 
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N) {
		return;
	}
	// Compute a new velocity based on pos and vel1
	glm::vec3 newVel = computeVelocityChange(N, index, pos, vel1);

	// Clamp the speed
	float speed = glm::length(newVel);
	if (speed > maxSpeed) {
		newVel = (newVel / speed) * maxSpeed;
	}

	// Record the new velocity into vel2. Question: why NOT vel1? 
	  // ANS: we still need previous velocities to compute flocking velocities for other boids
	vel2[index] = newVel;
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
  glm::vec3 *pos, int *indices, int *gridIndices) 
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N) {
		return;
	}
	// label each boid with the index of its grid cell
	glm::vec3 boidPos = pos[index];
	int gridIndex_x = floor((boidPos.x - gridMin.x) * inverseCellWidth);
	int gridIndex_y = floor((boidPos.y - gridMin.y) * inverseCellWidth);
	int gridIndex_z = floor((boidPos.z - gridMin.z) * inverseCellWidth);
	int gridIndex = gridIndex3Dto1D(gridIndex_x, gridIndex_y, gridIndex_z, gridResolution);
	gridIndices[index] = gridIndex;

    // Set up a parallel array of integer indices as pointers to the actual
    // boid data in pos and vel1/vel2
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

// Identify the start and end points of each cell
__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
	int *gridCellStartIndices, int *gridCellEndIndices) 
{
	// get thread index
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= N) return;

	// get the current cell and the next cell (if > length of array, will be -1)
	int gridCell = particleGridIndices[index];
	int gridCellAfter = index + 1 >= N ? -1 : particleGridIndices[index + 1];

	// if first in gridIndices array, mark start of cell
	if (index == 0) {
		gridCellStartIndices[gridCell] = index;
	}
	// otherwise, if the current cell does not equal the cell next in the array,
	// mark the end index of the current cell and the start index of the next
	else if (gridCell != gridCellAfter) {
		gridCellEndIndices[gridCell] = index;
		if (gridCellAfter != -1) {
			gridCellStartIndices[gridCellAfter] = index + 1;
		}
	}
}

__device__ glm::vec3 computeVelocityChangeGrid(int x_offset, int y_offset, int z_offset, int gridCell,
												   const int *startIndices, const int *endIndices, 
												   int iSelf, const glm::vec3 *pos, const glm::vec3 *vel,
												   const int *particleArrayIndices, int gridResolution) {
	glm::vec3 center = glm::vec3(0.f, 0.f, 0.f);
	glm::vec3 separate = glm::vec3(0.f, 0.f, 0.f);
	glm::vec3 cohesion = glm::vec3(0.f, 0.f, 0.f);
	int neighborCount = 0;

	glm::vec3 boidVel = vel[iSelf];
	const glm::vec3 &boidPos = pos[iSelf];

	int gridCellCount = gridResolution * gridResolution * gridResolution;

	// there are 8 potential neighbor cells
	for (int i = 0; i < 8; i++) {
		// find neighbor cell from offset
		// following calculations set up to go through all combinations of offsets
		int cell_x = i < 4 ? 0 : x_offset;
		int cell_y = i % 2 == 0 ? y_offset : 0;
		int cell_z = i > 1 && i < 6 ? z_offset : 0;
		int neighborCell = gridCell + gridIndex3Dto1D(cell_x, cell_y, cell_z, gridResolution);

		// if neighbor cell exists, iterate over boids inside
		if (neighborCell >= 0 && neighborCell < gridCellCount) {
			int startIndex = startIndices[neighborCell];
			int endIndex = endIndices[neighborCell];

			for (int j = startIndex; j < endIndex + 1; j++) {
				if (j == iSelf) continue;
				int neighborBoid = particleArrayIndices[j];
				const glm::vec3 &neighborPos = pos[neighborBoid];
				const glm::vec3 &neighborVel = vel[neighborBoid];

				// if neighbor boid is in neighborhood, perform flocking calcs
				float distance = glm::distance(neighborPos, boidPos);
				if (distance < rule1Distance) {
					center += neighborPos;
					neighborCount++;

					if (distance < rule2Distance) {
						separate -= (neighborPos - boidPos);
					}
					cohesion += neighborVel;
				}
			}
		}
	}
	if (neighborCount > 0) {
		center /= neighborCount;
		boidVel += (center - boidPos) * rule1Scale;
		boidVel += cohesion * rule3Scale;
	}
	boidVel += separate * rule2Scale;

	return boidVel;
}

// Update a boid's velocity using the uniform grid to reduce
// the number of boids that need to be checked.
__global__ void kernUpdateVelNeighborSearchScattered(
	int N, int gridResolution, glm::vec3 gridMin,
	float inverseCellWidth, float cellWidth,
	int *gridCellStartIndices, int *gridCellEndIndices,
	int *particleArrayIndices, 
	glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) 
{
	// get thread index
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= N) return;

	int iSelf = particleArrayIndices[index];
	glm::vec3 boidPos = pos[iSelf];
	int boidPos_x = boidPos.x;

	// Identify the grid cell that this particle is in
	int gridCell_x = floor((boidPos.x - gridMin.x) * inverseCellWidth);
	int gridCell_y = floor((boidPos.y - gridMin.y) * inverseCellWidth);
	int gridCell_z = floor((boidPos.z - gridMin.z) * inverseCellWidth);
	int gridCell = gridIndex3Dto1D(gridCell_x, gridCell_y, gridCell_z, gridResolution);

	// get offsets from current cell to find 8 potential neighbor cells
	float neighborhood = cellWidth / 2.f;
	// based on quadrant boid is in, cell offset will be -1 or 1 in each dimension
	int x_offset = boidPos.x - gridCell_x < neighborhood ? -1 : 1;
	int y_offset = boidPos.y - gridCell_y < neighborhood ? -1 : 1;
	int z_offset = boidPos.z - gridCell_z < neighborhood ? -1 : 1;

	// compute the velocity change given all potential neighboring cells
	glm::vec3 newVel = computeVelocityChangeGrid(x_offset, y_offset, z_offset, gridCell, gridCellStartIndices, gridCellEndIndices,
												 iSelf, pos, vel1, particleArrayIndices, gridResolution);
	// clamp the speed
	float speed = glm::length(newVel);
	if (speed > maxSpeed) {
		newVel = (newVel / speed) * maxSpeed;
	}

	vel2[iSelf] = newVel;
}

__device__ glm::vec3 computeVelocityChangeCoherent(
	int x_start, int y_start, int z_start, int gridCell,
	const int *startIndices, const int *endIndices,
	int iSelf, const glm::vec3 *pos, const glm::vec3 *vel,
	int gridResolution) {
	glm::vec3 center = glm::vec3(0.f, 0.f, 0.f);
	glm::vec3 separate = glm::vec3(0.f, 0.f, 0.f);
	glm::vec3 cohesion = glm::vec3(0.f, 0.f, 0.f);
	int neighborCount = 0;

	glm::vec3 boidVel = vel[iSelf];
	const glm::vec3 &boidPos = pos[iSelf];

	int gridCellCount = gridResolution * gridResolution * gridResolution;

	// iterate through 8 potential neighbor cells
	// x, y, z serve as offsets to current grid cell
	for (int z = z_start; z < z_start + 2; z++) {
		for (int y = y_start; y < y_start + 2; y++) {
			for (int x = x_start; x < x_start + 2; x++) {
				int neighborCell = gridCell + gridIndex3Dto1D(x, y, z, gridResolution);

				// if neighbor cell exists, iterate over boids inside
				if (neighborCell >= 0 && neighborCell < gridCellCount) {
					int startIndex = startIndices[neighborCell];
					int endIndex = endIndices[neighborCell];

					for (int j = startIndex; j < endIndex + 1; j++) {
						if (j == iSelf) continue;
						const glm::vec3 &neighborPos = pos[j];
						const glm::vec3 &neighborVel = vel[j];

						// if neighbor boid is in neighborhood, perform flocking calcs
						float distance = glm::distance(neighborPos, boidPos);
						if (distance < rule1Distance) {
							center += neighborPos;
							neighborCount++;

							if (distance < rule2Distance) {
								separate -= (neighborPos - boidPos);
							}
							cohesion += neighborVel;
						}
					}
				}
			}
		}
	}

	if (neighborCount > 0) {
		center /= neighborCount;
		boidVel += (center - boidPos) * rule1Scale;
		boidVel += cohesion * rule3Scale;
	}
	boidVel += separate * rule2Scale;

	return boidVel;
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

  // - For each cell, read the start/end indices in the boid pointer array.
  //   DIFFERENCE: For best results, consider what order the cells should be
  //   checked in to maximize the memory benefits of reordering the boids data.
 

	// get thread index
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= N) return;

	glm::vec3 boidPos = pos[index];

	// Identify the grid cell that this particle is in
	int gridCell_x = floor((boidPos.x - gridMin.x) * inverseCellWidth);
	int gridCell_y = floor((boidPos.y - gridMin.y) * inverseCellWidth);
	int gridCell_z = floor((boidPos.z - gridMin.z) * inverseCellWidth);
	int gridCell = gridIndex3Dto1D(gridCell_x, gridCell_y, gridCell_z, gridResolution);

	// get offsets from current cell to find 8 potential neighbor cells
	float neighborhood = cellWidth / 2.f;
	// based on quadrant boid is in, cell start will be the cur cell_dim offset either by -1 or 0
	int x_start = boidPos.x - gridCell_x < neighborhood ? -1 : 0; 
	int y_start = boidPos.y - gridCell_y < neighborhood ? -1 : 0;
	int z_start = boidPos.z - gridCell_z < neighborhood ? -1 : 0;

	// compute the velocity change given all potential neighboring cells
	glm::vec3 newVel = computeVelocityChangeCoherent(x_start, y_start, z_start, gridCell, gridCellStartIndices, gridCellEndIndices,
													 index, pos, vel1, gridResolution);
	// clamp the speed
	float speed = glm::length(newVel);
	if (speed > maxSpeed) {
		newVel = (newVel / speed) * maxSpeed;
	}

	float newVel_x = newVel.x;
	float newVel_y = newVel.y;
	float newVel_z = newVel.z;
	vel2[index] = newVel;
}

__global__ void kernSortPosVelArrays(int *particleArrayIndices, glm::vec3 *sortedPos, glm::vec3 *sortedVel1,
									 const glm::vec3 *pos, const glm::vec3 *vel1) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int boid = particleArrayIndices[index];
	sortedPos[index] = pos[boid];
	sortedVel1[index] = vel1[boid];
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
	// use the kernels you wrote to step the simulation forward in time.
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
	kernUpdateVelocityBruteForce<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_pos, dev_vel1, dev_vel2);
	kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel2);

	// ping-pong the velocity buffers
	glm::vec3 *tmp = dev_vel1;
	dev_vel1 = dev_vel2;
	dev_vel2 = tmp;
}

// Uniform Grid Neighbor search using Thrust sort.
void Boids::stepSimulationScatteredGrid(float dt) {
  
  // label each particle with its array index as well as its grid index.
  // TODO? Use 2x width grids.
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
	kernComputeIndices<<<fullBlocksPerGrid, blockSize>>>(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth,
														 dev_pos, dev_particleArrayIndices, dev_particleGridIndices);

  // Unstable key sort using Thrust
	thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, 
			            dev_thrust_particleArrayIndices);

	// fill all cells with -1 (designates empty cell)
	dim3 blockNumForCells((gridCellCount + blockSize - 1) / blockSize);
	dim3 blockSize3(blockSize, blockSize, blockSize);
	kernResetIntBuffer<<<blockNumForCells, blockSize>>>(gridCellCount, dev_gridCellStartIndices, -1);
	kernResetIntBuffer<<<blockNumForCells, blockSize>>>(gridCellCount, dev_gridCellEndIndices, -1);

	// Naively unroll the loop for finding the start and end indices of each
	// cell's data pointers in the array of boid indices
	kernIdentifyCellStartEnd<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_particleGridIndices,
															   dev_gridCellStartIndices, dev_gridCellEndIndices);
	// Perform velocity updates using neighbor search
	kernUpdateVelNeighborSearchScattered <<<fullBlocksPerGrid, blockSize >>>
		(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
		 dev_gridCellStartIndices, dev_gridCellEndIndices, dev_particleArrayIndices, dev_pos, dev_vel1, dev_vel2);

  // Update positions
	kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel2);

  // Ping-pong buffers as needed
	glm::vec3 *tmp = dev_vel1;
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

  // label each particle with its array index as well as its grid index.
  // TODO? Use 2x width grids.
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
	kernComputeIndices << <fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth,
		dev_pos, dev_particleArrayIndices, dev_particleGridIndices);

	// Unstable key sort using Thrust
	thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects,
		dev_thrust_particleArrayIndices);

	// fill all cells with -1 (designates empty cell)
	dim3 blockNumForCells((gridCellCount + blockSize - 1) / blockSize);
	dim3 blockSize3(blockSize, blockSize, blockSize);
	kernResetIntBuffer << <blockNumForCells, blockSize >> > (gridCellCount, dev_gridCellStartIndices, -1);
	kernResetIntBuffer << <blockNumForCells, blockSize >> > (gridCellCount, dev_gridCellEndIndices, -1);

	// Naively unroll the loop for finding the start and end indices of each
	// cell's data pointers in the array of boid indices
	kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_particleGridIndices,
		dev_gridCellStartIndices, dev_gridCellEndIndices);

	// sort pos and vel1 into sorted arrays
	kernSortPosVelArrays <<<fullBlocksPerGrid, blockSize >> > (dev_particleArrayIndices, dev_sortedPos, 
															   dev_sortedVel1, dev_pos, dev_vel1);

	// Perform velocity updates using neighbor search
	kernUpdateVelNeighborSearchCoherent << <fullBlocksPerGrid, blockSize >> >
		(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth, dev_gridCellStartIndices, 
		 dev_gridCellEndIndices, dev_sortedPos, dev_sortedVel1, dev_vel2);

	// sort vel2 so that it's in boid order again
	thrust::sort_by_key(dev_thrust_particleArrayIndices, dev_thrust_particleArrayIndices + numObjects, dev_thrust_vel2);

	// Update positions
	kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_pos, dev_vel2);

	// Ping-pong buffers as needed
	glm::vec3 *tmp = dev_vel1;
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
