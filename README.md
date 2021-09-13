**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Jiyu Huang
  * [LinkedIn](https://www.linkedin.com/in/jiyu-huang-0123)
* Tested on: Windows 10, i7-6700 @ 3.41GHz 16GB, Quadro P1000 4096MB (Town 062 Lab)

# Overview

![GIF](/images/ScreenCapture.gif)

This project simulates boid flocking on GPU using Cuda, where the boid particles follow the following rules:

1. cohesion - boids move towards the perceived center of mass of their neighbors
2. separation - boids avoid getting to close to their neighbors
3. alignment - boids generally try to move with the same direction and speed as their neighbors

The simulation is implemented using three methods:

1. **Naive Neighbor Search**: at every timestep, a boid looks at every other boid in the simulation to determine whether count as a neighbor and affects its velocity.
2. **Uniform Grid**: by creating a uniform spatial grid, we only need to search for neighbor boids among relevant grid cells (8 by default implementation), instead of the entire boid population.
3. **Coherent Grid**: like uniform grid, except we reduce random accessing by reordering the arrays for boid position and velocity.

## Performance Analysis

For performance analysis, framerates are observed over 20 seconds of execution.

### Number of Boids and FPS

![graph1](/images/graph1.png)
![graph2](/images/graph2.png)

As seen from the graphs, framerate declines as the number of boids increases. Intuitively this makes sense since as the number for boids increases, the number of neighbors each boid has increases.

### Block Size (Number of Blocks) and FPS

![graph3](/images/graph3.png)

Changing the block size and the number of blocks doesn't affect framerate, since it mainly affects how threads are distributed to SMs and not much else.

### Performance Gain from Coherent Uniform Grid

From the graphs we can see that there's an improvement in performance with the more coherent uniform grid. As previously touched upon, this is because the reordering of the arrays make for conntiguous access to the data, reducing random accesses and improving performance.

### Cell Width and FPS

Using the default configuration, changing the cell width from twice the maximum neighborhood distance (checking 8 relevant grid cells) to maximum neighborhood distance (checking 27 relevant grid cells) does not change framerate too much. This is likely because maximum distance is rather small (5 when scene_scale = 100). When we increase the neighborhood distance, smaller cell width starts to improve performance. As an extreme example, when the distance is 50, the smaller cell width performances at 700fps while the larger default cell width performs at 360fps.

This is due to the fact that when we decrease cell width, we are checking smaller volumes for each cell, which means that the checking volume gets more granular, and the total volume that needs to be checked decreases ($27\times1^2<8\times2^2$)

## Extra Credit

### Grid Looping Optimization

instead of hard-coding a designated search area, the search area is based on grid cells that have any aspect of them within maximum neighborhood distance. This avoids the need for hard-coding specific number of cells to check when changing cell widths.

To toggle on or off the feature, simply change the value of GRID_LOOPING_OPTIMIZATION to 1 or 0 in line 9 of src/kernel.cu
