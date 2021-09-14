**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Wayne Wu
  * [LinkedIn](https://www.linkedin.com/in/wayne-wu/), [Personal Website](https://www.wuwayne.com/)
* Tested on: Windows 10, AMD Ryzen 5 5600X @ 3.70GHz 32GB, RTX 3070 8GB (personal)

## Background

This project aims to introduce GPU programming in CUDA with boids flocking simulation.
The simulation is based on the well established method developed by [Craig Reynolds](http://www.red3d.com/cwr/boids/).

Three different approaches to the solution were implemented and analyzed for performance:
1. **Brute-Force/Naive** approach that iterates over all boids when looking for the nearest neighbors.
2. **Scattered uniform grid** approach that spatially divide all boids into grids for quick nearest neighbors look-up.
3. **Semi-coherent uniform grid** approach that is based on Method 2, but with a more coherent memory access to the boids data.

## Screenshots

![](images/boidsScreenshot.png)

![](images/boidsAnimation1.gif)
![](images/boidsAnimation2.gif)

## Performance Analysis

Figure 1: Number of Boids vs. Different Methods (Visualize ON)

![](images/fpsVisualizeOn.png)

Figure 2: Number of Boids vs. Different Methods (Visualize OFF)

![](images/fpsVisualizeOff.png)

Figure 3: Number of Blocks vs. Different Methods (N = 100000)

![](images/blocksTest.png)

Figure 4: Number of Blocks vs. Brute Force Method (N = 100000)

![](images/blocksTestBF.png)

## Questions

**For each implementation, how does changing the number of boids affect performance? Why do you think this is?**

For all implementations, increasing the number of boids decreases the average FPS.
This is expected given that as we scale up the number, it will exceed the number of threads that can be run in parallel at one time.
When we turn off the visualization, we can see that it increases the performance at lower number of boids.
However, it does not matter as much when we increase the number of boids since at that point the computation for OpenGL draws 
is too insignificant for the overall performance cost.

**For each implementation, how does changing the block count and block size affect performance? Why do you think this is?**

The block count and block size do affect the performance. As shown in Figure 3, when the block size is small (e.g. less than 64), the performance is poorer. However, once it passes a threshold, the average FPS stays the same with increasing block size, until it very slightly decreases again at the largest block size possible, e.g. 1024, as shown in Figure 4. At low block size count, the total number of threads that can be run in parallel is less than the number of parallel executions required (i.e. the number of boids), hence the slow down. Once it reaches the threshold where parallelism is exhausted, there will be no additional performance gain. The very slight decrease in performance at very large block size suggests that there is an optimal block size value for best performance.

**For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?**

The coherent uniform grid significantly improved the performance which is unexpected coming from traditional CPU-based programming. To have a coherent uniform grid, two new buffers (i.e. dev_coherentPos and dev_coherentVel*) and a new kernel (i.e. kernRearrangeBoidData) were introduced which I initially thought would be more costly than removing the need to access the arrayIndices buffer. This turned out to be false, which proves that GPU is 
very costly at accessing global memory.

**Note that dev_coherentVel can most likely just be dev_vel1 or dev_vel2, thus removing the need for a new buffer.*

**Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not?**

It did not significantly affect the performance.
While it is true that we're traverseing more cells, each cell is now smaller as the cell width is the search radius instead of, previously, two times the search radius. As such, on average, the number of points inside each cell will now be smaller, and thus the overall iteration count can vary depending on the distribution of boids.