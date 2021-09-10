#pragma once

#include <stdio.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <cuda.h>
#include <cmath>
#include <vector>

/********************************************
 * Start Performance Analysis Configuration *
 ********************************************/

#define PFM_ANA_VISUALIZE 1
#define PFM_ANA_UNIFORM_GRID 1
#define PFM_ANA_COHERENT_GRID 1

#define PFM_ANA_NUM_OBJECTS 5000
//1000000 //500000 //300000 //150000
//100000 //60000 //20000 //5000 //1000

#define PFM_ANA_blockSize 128 //32 //64 //1024 //512 //256 //128

// If true, use rule3 from Conard Parker's note, otherwise follow the instruction.
#define USE_RULE3_FROM_CONARD_PARKER 0

// If true, use stable sort, otherwise unstable sort.
#define USE_STABLE_SORT 0 //1

// If true, use half cell width and check 27 cells, otherwise check 8 cells.
#define USE_HALF_SIZE_OF_CELL 1 //1

// If true, for loop x->y->z, otherwise z->y->x.
#define FOR_LOOP_XYZ 0 //1

// If true, identify start and end by binary search (parallel by grid cells), otherwise in constant time (parallel by boids). 
#define IDENTIFY_START_END_BY_BINARY_SEARCH 0 //0

// If true, use shared memory optimization
#define USE_SHARED_MEMORY 0 

// If true, adjust the search area by cell width.
#define GRID_LOOPING_OPTIMIZATION 1
#define GRID_LOOPING_WIDTH 5.0f //5.0f //4.0f //3.0f //2.5f //6.0f //8.0f //10.0f

/******************************************
 * End Performance Analysis Configuration *
 ******************************************/

namespace Boids {
    void initSimulation(int N);
    void stepSimulationNaive(float dt);
    void stepSimulationScatteredGrid(float dt);
    void stepSimulationCoherentGrid(float dt);
    void copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities);

    void endSimulation();
    void unitTest();
}
