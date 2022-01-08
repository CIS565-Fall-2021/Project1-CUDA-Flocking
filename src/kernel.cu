#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
#include "kernel.h"
#include "cVec.h"


#define GRID_LOOP 0
#define ALWAYS_27 0
/* always checking 27 cells only enabled if GRID_LOOP is 0*/

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/* Check for CUDA errors; print and exit if there was a problem */
void checkCUDAError(const char *msg, int line = -1)
{
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		if (line >= 0) {
			fprintf(stderr, "Line %d: ", line);
		}
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

/* better max and min functions*/
template <typename T>
__host__ __device__ T max(T v) {
	return v;
}

template <typename T, typename... U>
__host__ __device__ T max(T v1, T v2, U ... vs) {
	return max(v1 > v2 ? v1 : v2, vs...);
}

template <typename T>
__host__ __device__ T min(T v) {
	return v;
}

template <typename T, typename... U>
__host__ __device__ T min(T v1, T v2, U ... vs) {
	return max(v1 < v2 ? v1 : v2, vs...);
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define block_size 128

// LOOK-1.2 Parameters for the boids algorithm.
// These worked well in our reference implementation.
#define rule1_dist 5.0f
#define rule2_dist 3.0f
#define rule3_dist 5.0f

#define rule1_scale 0.01f
#define rule2_scale 0.1f
#define rule3_scale 0.1f

#define max_speed 1.0f

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int num_boids;
dim3 threadsPerBlock(block_size);


using glm::vec3;

// LOOK-1.2 - These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.
cu::cPtr<vec3> dv_pos;
cu::cPtr<vec3> dv_vel1;
cu::cPtr<vec3> dv_vel2;

// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
cu::cPtr<int> dv_particle_grid_indices; // What grid cell is this particle in? (grid cell index)
cu::cPtr<int> dv_particle_array_indices; // What index in dv_pos and dev_velX represents this particle? (boid index)

thrust::device_ptr<int> dv_thrust_particle_array_indices;
thrust::device_ptr<int> dv_thrust_particle_grid_indices;

cu::cPtr<int> dv_gridcell_start_indices; // What part of dev_particleArrayIndices belongs
cu::cPtr<int> dv_gridcell_end_indices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells:
cu::cPtr<vec3> dv_pos2; /* we already have two dv_vel so we can use the second as the rearranged version */

// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
vec3 gridMinimum;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a)
{
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
__host__ __device__ vec3 generateRandomVec3(float time, int index)
{
	thrust::default_random_engine rng(hash((int)(index * time)));
	thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

	return vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}


/**
* LOOK-1.2 - This is a basic CUDA kernel.
* CUDA kernel for generating boids with a specified mass randomly around the star.
*/
__global__ void kernGenerateRandomPosArray(int time, int N, vec3 *arr, float scale)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		vec3 rand = generateRandomVec3(time, index);
		arr[index] = scale * rand;
	}
}

/* Initialize memory, update some globals */
void Boids::initSimulation(int N)
{
	num_boids = N;
	dim3 fullBlocksPerGrid((N + block_size - 1) / block_size);

	// LOOK-1.2 - This is basic CUDA memory management and error checking.
	dv_pos = cu::make<vec3>(N);
	dv_vel1 = cu::make<vec3>(N);
	dv_vel2 = cu::make<vec3>(N);

	// LOOK-1.2 - This is a typical CUDA kernel invocation.
	kernGenerateRandomPosArray<<<fullBlocksPerGrid, block_size>>>(1, num_boids, dv_pos.get(), scene_scale);
	checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

	// LOOK-2.1 computing grid params
	gridCellWidth = 2.0f * max(rule1_dist, rule2_dist, rule3_dist); //std::max(std::max(rule1_dist, rule2_dist), rule3_dist);
	int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
	gridSideCount = 2 * halfSideCount;

	gridCellCount = gridSideCount * gridSideCount * gridSideCount;
	gridInverseCellWidth = 1.0f / gridCellWidth;
	float halfGridWidth = gridCellWidth * halfSideCount;
	gridMinimum.x -= halfGridWidth;
	gridMinimum.y -= halfGridWidth;
	gridMinimum.z -= halfGridWidth;

	// TODO-2.1 TODO-2.3 - Allocate additional buffers here.
	
	dv_particle_grid_indices = cu::make<int>(N);
	dv_particle_array_indices = cu::make<int>(N);

	dv_gridcell_start_indices = cu::make<int>(gridCellCount);
	dv_gridcell_end_indices = cu::make<int>(gridCellCount);

	dv_thrust_particle_grid_indices = thrust::device_ptr<int>(dv_particle_grid_indices.get());
	dv_thrust_particle_array_indices = thrust::device_ptr<int>(dv_particle_array_indices.get());

	dv_pos2 = cu::make<vec3>(N);

	cudaDeviceSynchronize();
}


/******************
* copyBoidsToVBO *
******************/

/* Copy the boid positions into the VBO so that they can be drawn by OpenGL */
__global__ void kernCopyPositionsToVBO(int N, const vec3 *pos, float *vbo, float s_scale)
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	float c_scale = -1.0f / s_scale;

	if (index < N) {
		vbo[4 * index + 0] = pos[index].x * c_scale;
		vbo[4 * index + 1] = pos[index].y * c_scale;
		vbo[4 * index + 2] = pos[index].z * c_scale;
		vbo[4 * index + 3] = 1.0f;
	}
}

__global__ void kernCopyVelocitiesToVBO(int N, const vec3 *vel, float *vbo, float s_scale)
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if (index < N) {
		vbo[4 * index + 0] = vel[index].x + 0.3f;
		vbo[4 * index + 1] = vel[index].y + 0.3f;
		vbo[4 * index + 2] = vel[index].z + 0.3f;
		vbo[4 * index + 3] = 1.0f;
	}
}

/* Wrapper for call to the kernCopyboidsToVBO CUDA kernel */
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities)
{
	dim3 fullBlocksPerGrid((num_boids + block_size - 1) / block_size);

	kernCopyPositionsToVBO <<<fullBlocksPerGrid, block_size>>>(num_boids, dv_pos.get(), vbodptr_positions, scene_scale);
	kernCopyVelocitiesToVBO <<<fullBlocksPerGrid, block_size>>>(num_boids, dv_vel1.get(), vbodptr_velocities, scene_scale);

	checkCUDAErrorWithLine("copyBoidsToVBO failed!");

	cudaDeviceSynchronize();
}


/******************
* stepSimulation *
******************/

/* p is position of this boid, boid_pos, boid_vel are of neighbour we*/
__device__ __forceinline__ void apply_rules(const vec3 &p, const vec3 &boid_pos, const vec3 &boid_vel,
	vec3 *perceived_center, vec3 *perceived_vel, vec3 *c,
	int *neighbour_count_p, int *neighbour_count_v)
{
	float len = glm::distance(boid_pos, p);

	// Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
	if (len < rule1_dist)  {
		*perceived_center += boid_pos;
		(*neighbour_count_p)++;
	}

	// Rule 2: boids try to stay a distance d away from each other
	if (len < rule2_dist)
		*c -= (boid_pos - p);

	// Rule 3: boids try to match the speed of surrounding boids
	if (len < rule3_dist) {
		*perceived_vel += boid_vel;
		(*neighbour_count_v)++;
	}
}

__device__ __forceinline__ vec3 out_vel(const vec3 &p, vec3 v, const vec3 &perceived_center, const vec3 &perceived_vel,
	const vec3 &c, int neighbour_count_p, int neighbour_count_v)
{
	if (neighbour_count_p > 0)
		v += (perceived_center / (float) neighbour_count_p - p) * rule1_scale;
	v += c * rule2_scale;
	if (neighbour_count_v > 0)
		v += perceived_vel / (float) neighbour_count_v * rule3_scale;

	return v * max_speed / max(max_speed, glm::length(v)); /* clamp to max_speed */
}


/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
* Compute the new velocity on the body with index `idx` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__global__ void kern_update_vel_brute_force(int N, const vec3 *pos, const vec3 *vel1, vec3 *vel2)
{
	int idx= threadIdx.x + (blockIdx.x * blockDim.x);

	vec3 v = vel1[idx];
	vec3 p = pos[idx];

	// Compute a new velocity based on pos and vel1

	vec3 perceived_center(0.0f);
	vec3 perceived_vel(0.0f);
	int neighbour_count_p = 0, neighbour_count_v = 0;
	vec3 c(0.0f);

	for (int i = 0; i < N; i++) {
		if (i != idx)
			apply_rules(p, pos[i], vel1[i], &perceived_center, &perceived_vel, &c, &neighbour_count_p, &neighbour_count_v);
	}

	vel2[idx] = out_vel(p, v, perceived_center, perceived_vel, c, neighbour_count_p,  neighbour_count_v);
}

/**
* LOOK-1.2 Since this is pretty trivial, we implemented it for you.
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, vec3 *pos, const vec3 *vel)
{
	// Update position by velocity
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N)
		return;

	vec3 thisPos = pos[index] + vel[index] * dt;

	// Wrap the boids around so we don't lose them
	thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x > scene_scale ? -scene_scale : thisPos.x;
	thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y > scene_scale ? -scene_scale : thisPos.y;
	thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z > scene_scale ? -scene_scale : thisPos.z;

	pos[index] = thisPos;
}

// LOOK-2.1 Consider this method of computing a 1D index from a 3D grid index.
// LOOK-2.3 Looking at this method, what would be the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order?
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution)
{
	return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(int N, int gridResolution,
	vec3 gridMin, float inverseCellWidth,
	const vec3 *pos, int * __restrict__ indices, int * __restrict__ gridIndices) {
		// TODO-2.1
		// - Label each boid with the index of its grid cell.
		// - Set up a parallel array of integer indices as pointers to the actual
		//   boid data in pos and vel1/vel2
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= N)
		return;
	
	vec3 offset = (pos[index] - gridMin) * inverseCellWidth;
	gridIndices[index] = gridIndex3Dto1D(offset.x, offset.y, offset.z, gridResolution);
	indices[index] = index;
}


__global__ void kernIdentifyCellStartEnd(int N, const int *particleGridIndices,
	int *__restrict__ gridCellStartIndices, int *__restrict__ gridCellEndIndices) {
	// TODO-2.1
	// Identify the start point of each cell in the gridIndices array.
	// This is basically a parallel unrolling of a loop that goes
	// "this index doesn't match the one before it, must be a new cell!"
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= N)
		return;

	if (index == 0)
		gridCellStartIndices[particleGridIndices[0]] = 0;
	else if (particleGridIndices[index] != particleGridIndices[index-1])
		gridCellStartIndices[particleGridIndices[index]] = index;
	if (index == N-1)
		gridCellEndIndices[particleGridIndices[N-1]] = N-1;
	else if (particleGridIndices[index] != particleGridIndices[index+1])
		gridCellEndIndices[particleGridIndices[index]] = index;

}

/* boid_search_apply iterates over the boids of the neighbouring cells and executes function apply_rules_b2 with
 * their indices: in the case of the scattered grid, this is the indices that go into particleGridIndices
 * in the case of coherent grid, the indices are directly the indices of the boids in the pos and vel arrays */
/* F is a function that takes in the neighbour boid index/particleArrayIndex index as determined by this function */
template <typename F>
__device__ __forceinline__ void boid_search_apply(const vec3 &p, int gridResolution, vec3 gridMin, float inverseCellWidth,
	const int *gridCellStartIndices, const int *gridCellEndIndices, F apply_rules_b2)
{
	// - Identify the grid cell that this particle is in
	// - Identify which cells may contain neighbors. This isn't always 8.

#if GRID_LOOP
	float dist = max(rule1_dist, rule2_dist, rule3_dist) * inverseCellWidth;
	vec3 minv = (p - gridMin) * inverseCellWidth - dist; /* grid looping optimization */
	vec3 maxv = (p - gridMin) * inverseCellWidth + dist;
#else
 #if ALWAYS_27
	vec3 minv = (p - gridMin) * inverseCellWidth - 1.0f; /* always checks 27 squares */
	vec3 maxv = (p - gridMin) * inverseCellWidth + 1.0f;

 #else
	vec3 minv = (p - gridMin) * inverseCellWidth - 0.5f; /* always checks 8 squares */
	vec3 maxv = (p - gridMin) * inverseCellWidth + 0.5f;
 #endif
#endif

	dim3 mincoords = dim3(max(0, (int) minv.x), max(0, (int) minv.y), max(0, (int) minv.z));
	dim3 maxcoords = dim3(min(gridResolution - 1, (int) maxv.x), min(gridResolution - 1, (int) maxv.y), min(gridResolution - 1, (int) maxv.z));

	for (int z = mincoords.z; z <= maxcoords.z; z++) {
		for (int y = mincoords.y; y <= maxcoords.y; y++) {
			for (int x = mincoords.x; x <= maxcoords.x; x++) {
				// - For each cell, read the start/end indices in the boid pointer array.
				int start = gridCellStartIndices[gridIndex3Dto1D(x, y, z, gridResolution)];
				int end = gridCellEndIndices[gridIndex3Dto1D(x, y, z, gridResolution)];
				if (start == -1)
					continue;

				// - Access each boid in the cell and compute velocity change from
				//   the boids rules, if this boid is within the neighborhood distance.
				for (int i = start; i <= end; i++) {
					apply_rules_b2(i);
				}
			}
		}
	}

}

__global__ void kernUpdateVelNeighborSearchScattered(
	int N, int gridResolution, vec3 gridMin,
	float inverseCellWidth,
	const int *gridCellStartIndices, const int *gridCellEndIndices,
	const int *particleArrayIndices,
	const vec3 *pos, const vec3 *vel1, vec3 *vel2) {
	// TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
	// the number of boids that need to be checked.

	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (idx >= N)
		return;

	vec3 p = pos[idx];
	vec3 v = vel1[idx];

	vec3 perceived_center(0.0f);
	vec3 perceived_vel(0.0f);
	int neighbour_count_p = 0, neighbour_count_v = 0;
	vec3 c(0.0f);


	boid_search_apply(p, gridResolution, gridMin, inverseCellWidth, gridCellStartIndices, gridCellEndIndices,
		[&] (int i) {
			int b2 = particleArrayIndices[i];
			if (idx != b2)
				apply_rules(p, pos[b2], vel1[b2], &perceived_center, &perceived_vel, &c, &neighbour_count_p, &neighbour_count_v);
		});


	vel2[idx] = out_vel(p, v, perceived_center, perceived_vel, c, neighbour_count_p,  neighbour_count_v);
}

__global__ void kernUpdateVelNeighborSearchCoherent(
	int N, int gridResolution, vec3 gridMin,
	float inverseCellWidth,
	const int *gridCellStartIndices, const int *gridCellEndIndices,
	const vec3 *pos, const vec3 *vel1, vec3 *vel2) {

	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (idx >= N)
		return;

	vec3 p = pos[idx];
	vec3 v = vel1[idx];

	vec3 perceived_center(0.0f);
	vec3 perceived_vel(0.0f);
	int neighbour_count_p = 0, neighbour_count_v = 0;
	vec3 c(0.0f);

	boid_search_apply(p, gridResolution, gridMin, inverseCellWidth, gridCellStartIndices, gridCellEndIndices,
		[&] (int b2) {
			if (idx != b2)
				apply_rules(p, pos[b2], vel1[b2], &perceived_center, &perceived_vel, &c, &neighbour_count_p, &neighbour_count_v);
		});

	vel2[idx] = out_vel(p, v, perceived_center, perceived_vel, c, neighbour_count_p,  neighbour_count_v);
}


/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt)
{
	// TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
	// TODO-1.2 ping-pong the velocity buffers
	
	int blocks_per_grid = (num_boids + block_size - 1) / block_size;

	kernUpdatePos<<<blocks_per_grid, block_size>>>(num_boids, dt, dv_pos.get(), dv_vel1.get());
	checkCUDAErrorWithLine("kernUpdatePos failed!");

	kern_update_vel_brute_force<<<blocks_per_grid, block_size>>>(num_boids, dv_pos.get(), dv_vel1.get(), dv_vel2.get());
	checkCUDAErrorWithLine("kern_update_vel_brute_force failed!");

	std::swap(dv_vel1, dv_vel2);
}

void Boids::stepSimulationScatteredGrid(float dt)
{
	int blocks_per_grid = (num_boids + block_size - 1) / block_size;
	
	// - label each particle with its array index as well as its grid index.
	kernComputeIndices<<<blocks_per_grid, block_size>>>(num_boids, gridSideCount, gridMinimum,
		gridInverseCellWidth, dv_pos.get(), dv_particle_array_indices.get(), dv_particle_grid_indices.get());
	checkCUDAErrorWithLine("kernComputeIndices failed!");

	// - Unstable key sort using Thrust
	thrust::sort_by_key(dv_thrust_particle_grid_indices, dv_thrust_particle_grid_indices + num_boids, dv_thrust_particle_array_indices);

	// - Naively unroll the loop for finding the start and end indices of each
	//   cell's data pointers in the array of boid indices
	cu::set(dv_gridcell_start_indices, -1, gridCellCount);

	kernIdentifyCellStartEnd<<<blocks_per_grid, block_size>>>(num_boids, dv_particle_grid_indices.get(),
		dv_gridcell_start_indices.get(), dv_gridcell_end_indices.get());
	checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

	// - Perform velocity updates using neighbor search
	kernUpdateVelNeighborSearchScattered<<<blocks_per_grid, block_size>>>(num_boids, gridSideCount, gridMinimum, gridInverseCellWidth,
		dv_gridcell_start_indices.get(), dv_gridcell_end_indices.get(), dv_particle_array_indices.get(),
		dv_pos.get(), dv_vel1.get(), dv_vel2.get());
	checkCUDAErrorWithLine("kernUpdateVelNeighborSearchScattered failed!");

	// - Update positions
	kernUpdatePos<<<blocks_per_grid, block_size>>>(num_boids, dt, dv_pos.get(), dv_vel2.get());
	checkCUDAErrorWithLine("kernUpdatePos failed!");
	
	// - Ping-pong buffers
	std::swap(dv_vel1, dv_vel2);
}


__global__ void kern_rearrange_boid_data(int N, const int* indices, const vec3* pos, const vec3* vel,
	vec3* __restrict__ pos2, vec3* __restrict__ vel2)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index >= N)
		return;

	pos2[index] = pos[indices[index]];
	vel2[index] = vel[indices[index]];
}

void Boids::stepSimulationCoherentGrid(float dt)
{
	// TODO-2.3 - start by copying Boids::stepSimulationNaiveGrid
	// Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
	// In Parallel:

	int blocks_per_grid = (num_boids + block_size - 1) / block_size;

	// - Label each particle with its array index as well as its grid index.
	kernComputeIndices<<<blocks_per_grid, block_size>>>(num_boids, gridSideCount, gridMinimum,
		gridInverseCellWidth, dv_pos.get(), dv_particle_array_indices.get(), dv_particle_grid_indices.get());
	checkCUDAErrorWithLine("kernComputeIndices failed!");

	// - Unstable key sort using Thrust
	thrust::sort_by_key(dv_thrust_particle_grid_indices, dv_thrust_particle_grid_indices + num_boids, dv_thrust_particle_array_indices);

	// - Naively unroll the loop for finding the start and end indices of each
	//   cell's data pointers in the array of boid indices
	cu::set(dv_gridcell_start_indices, -1, gridCellCount);

	kernIdentifyCellStartEnd<<<blocks_per_grid, block_size>>>(num_boids, dv_particle_grid_indices.get(),
		dv_gridcell_start_indices.get(), dv_gridcell_end_indices.get());
	checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

	// - use the rearranged array index buffer to reshuffle all the particle data in the simulation array.
	kern_rearrange_boid_data<<<blocks_per_grid, block_size>>>(num_boids, dv_particle_array_indices.get(), dv_pos.get(), dv_vel1.get(), dv_pos2.get(), dv_vel2.get());
	checkCUDAErrorWithLine("kern_rearrange_boid_data failed!");

	// - Perform velocity updates using neighbor search
	kernUpdateVelNeighborSearchCoherent<<<blocks_per_grid, block_size>>>(num_boids, gridSideCount, gridMinimum, gridInverseCellWidth,
		dv_gridcell_start_indices.get(), dv_gridcell_end_indices.get(), dv_pos2.get(), dv_vel2.get(), dv_vel1.get());
	checkCUDAErrorWithLine("kernUpdateVelNeighborSearchCoherent failed!");

	// - Update positions
	kernUpdatePos<<<blocks_per_grid, block_size>>>(num_boids, dt, dv_pos2.get(), dv_vel1.get());
	checkCUDAErrorWithLine("kernUpdatePos failed!");

	// - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.
	std::swap(dv_pos, dv_pos2); /* vel1 contains the final velocities so only need to update pos */
	
}

void Boids::endSimulation()
{
	cu::del(dv_pos);
	cu::del(dv_vel1);
	cu::del(dv_vel2);

	// TODO-2.1 TODO-2.3 - Free any additional buffers here.
	cu::del(dv_particle_grid_indices);
	cu::del(dv_particle_array_indices);
	cu::del(dv_gridcell_start_indices);
	cu::del(dv_gridcell_end_indices);

	cu::del(dv_pos2);
}

void Boids::unitTest()
{
	// LOOK-1.2 Feel free to write additional tests here.

	// test unstable sort
	int N = 10;
	cu::cVec<int> dev_intKeys(N);
	cu::cVec<int> dev_intValues(N);

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


	dim3 fullBlocksPerGrid((N + block_size - 1) / block_size);

	std::cout << "before unstable sort: " << std::endl;
	for (int i = 0; i < N; i++) {
		std::cout << "  key: " << intKeys[i];
		std::cout << " value: " << intValues[i] << std::endl;
	}

	// How to copy data to the GPU
	cu::copy(dev_intKeys.ptr(), intKeys.get(), N);
	cu::copy(dev_intValues.ptr(), intValues.get(), N);

	// Wrap device vectors in thrust iterators for use with thrust.
	thrust::device_ptr<int> dev_thrust_keys(dev_intKeys.get());
	thrust::device_ptr<int> dev_thrust_values(dev_intValues.get());
	// LOOK-2.1 Example for using thrust::sort_by_key
	thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

	// How to copy data back to the CPU side from the GPU
	cu::copy(intKeys.get(), dev_intKeys.ptr(), N);
	cu::copy(intValues.get(), dev_intValues.ptr(), N);

	std::cout << "after unstable sort: " << std::endl;
	for (int i = 0; i < N; i++) {
		std::cout << "  key: " << intKeys[i];
		std::cout << " value: " << intValues[i] << std::endl;
	}

	return;
}
