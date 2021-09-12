**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Ashley Alexander-Lee
  * [LinkedIn](linkedin.com/in/asalexanderlee), [Personal Website](https://asalexanderlee.myportfolio.com/)
* Tested on: Windows 10, i7-6700 @ 3.40GHz 16GB, Quadro P1000 (Moore 100A Lab)

### Project Description

![Flocking Simulation GIF](/images/coherent.gif)

This project involves creating a flocking simulation, where "boids", or particles, behave according to three rules:

1. **cohesion** - boids move towards the perceived center of mass of their neighbors
2. **separation** - boids avoid getting to close to their neighbors
3. **alignment** - boids generally try to move with the same direction and speed as their neighbors

These three rules help dictate the velocity of the boids at each timestep. The 3D flocking logic is based off of this [2D implementation in Processing](http://studio.sketchpad.cc/sp/pad/view/ro.9cbgCRcgbPOI6/rev.23). 

I've implemented this simulation using three different methods:

1. **NAIVE** - Each boid checks every other boid in the simulation against its neighborhood to compute its resulting velocity.

2. **UNIFORM GRID** - Each boid is pre-processed into a grid cell that has the width of a neighborhood * 2. The boid just needs to check the boids in 8 surrounding cells against its neighborhood to compute its resulting velocity. Four arrays are used to keep track of this information:
   * `particleGridIndices`: each boid's corresponding cell; sorted with `particleArrayIndices` so that the cells are contiguous
   * `particleArrayIndices`: the original boid indices, pointers to a boid's position and velocity -- sorted using `particleGridIndices` as the key
   * `cellStartIndices`: the start index of each cell
   * `cellEndIndices`: the end index of each cell

3. **COHERENT GRID** - Like above, except that the boid positions and velocities are sorted so that the cells are contiguous in memory. 

### Performance Analysis

#### Boids vs. FPS
I've charted below how the FPS declines as the numbers of boids increases (with the visualization on and off).

![How Boids Affect FPS](/images/howBoidsAffectFPS1.png)
![How Boids Affect FPS](/images/howBoidsAffectFPS2.png)

As you can see, the FPS declines exponentially as the number of boids increases. There is also a significant difference in performance between the naive method and the grid methods. This makes sense, since the naive method runs in `O(n^2)` time, while the grid method runs in `O(n*m)` time, where m refers to some subset of boids (the worst case could be `O(n^2)` if all boids are in the same cells). Finally, you can observe the slight performance improvement in the coherent grid method over the uniform grid method, likely due to the cell data being contiguous in memory. 

#### Block Size vs. FPS
You can see a mild falloff in FPS as the block size increases, but not a significant one. The **block size** affects how many threads run together, and is guaranteed to run on the same SM. It makes sense that there is no significant performance change, since none of the threads rely on each other. So, it doesn't matter whether 1024 threads run as 32 blocks of 32 threads or 1 block of 1024 threads.

| Block Size | FPS |
| ---------- |---- |
| 32 | 1720 |
| 64 | 1750 |
| 128 | 1741 |
| 256 | 1737 |
| 512 | 1735 |
| 1024 | 1723 |

#### Question Responses

**Q: For each implementation how does changing the number of boids affect performance? Why do you think this is?**

As you can see in the charts I included above, the framerate decreases exponentially as the number of boids increases. This makes sense, since for each boid, you have to check a number of neighbors. As the number of boids increases, there are more boids to factor into each boid's velocity calculation. It also makes sense that the framerate plunges more quickly for the naive method, since it runs in `O(n^2)` time, while the grid methods (though technically running in `O(n^2)` worst-case time) run in approximately `O(n * 8(n / numGridCells))`. 

**Q: For each implementation, how does changing the block count and block size affect performance? Why do you think this is?**

I explain this briefly above, but I didn't notice much change in the FPS as the block size changed. I believe this is because none of the threads depend on each other, therefore it doesn't matter how they are separated into blocks. 

**Q: For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?**

I did -- the performance was better, as expected. I imagine the difference is due to the fact that global memory access is slow, and the sorting of the positions and velocities for the coherent uniform grid leads to more contiguous memory accesses. This means that more global memory is likely to be read at once, saving time.

**Q: Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not? Be careful: it is insufficient (and possibly incorrect) to say that 27-cell is slower simply because there are more cells to check!**

Interestingly enough, changing the cell width and checking 27 neighbors improved performance slightly. If you break it down, you can see (in the chart below) that decreasing the cell width to just the neighborhood distance greatly increases performance, while changing the number of cells checked from 8 to 27 greatly decreases the performance. The cell width change likely improved performance so much because there were fewer neighbors to check for each boid. Also, increasing the number of cells checked from 8 to 27 likely decreased performance so much due to the increased number of boids each boid has to check. It makes sense that the two changes would even out in performance, since the number of boids the cell width change removed was probably fairly even with the number the 27-cell check added.

| Cell Width | Cell Width / 2 | 8-Cell Check | 27-Cell Check | Cell Width / 2 + 27-Cell Check |
| ---------- | -------------- | ------------ | ------------- | ------------------------------ |
| 1740 FPS   | 1912 FPS       | 1740 FPS     | 1330 FPS      | 1756 FPS                       |
