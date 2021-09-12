<p align="center">
  <img src="images/logo.png" alt="Logo" width="150" height="150">
  <h2 align="center">Author: (Charles) Zixin Zhang</h2>
  <p align="center">
    A flocking simulation based on the <strong>Reynolds Boids algorithm</strong>
  </p>
</p>

--- 
## About The Project

A flocking simulation based on the <strong>Reynolds Boids algorithm</strong>, along with two levels of optimization: a <strong>uniform grid</strong>, and a <strong>uniform grid with semi-coherent memory access</strong>.

--- 
## Highlight

<p align="center">
  <img src="images/outSideCube.gif" alt="Outside Cube Pic" width="640" height="360">
  <img src="images/insideCube.gif" alt="Inside Cube Pic" width="640" height="360">
</p>


Stats: 
- Coherent uniform grid approach
- CPU: i7-10700F @ 2.90GHz
- GPU: SM 8.6 NVIDIA GeForce RTX 3080
- Number of Boids: 12 million 
- Average FPS: ~40

Note: For the first picture, I use a larger timestep to speed up the simulation to better observe the overall trend whereas the second picture uses a smaller time step to better observe the movement of the particles (it also looks really cool :sunglasses:).

--- 

## Performance Analysis

In this project, we investigate 3 approaches to implement the Reynolds Boids algorithm:

1. Naive approach has each boid check every other boid in the simulation. 
2. Uniform grid approach culls unnecessary neighbor checks using a data structure called a uniform spatial grid. 
3. Coherent uniform gird approach improves upon the second approach by cutting one level of indirection when accessing the boids' data.

---
To validate our optimizations, we plot the framerate change with increasing number of boids for these 3 approaches. Note that the below experiment has ```scene_scale=100.0f``` because it will affect FPS based on the number particles in the scene. Additionally, we consider 30~60 FPS to be an acceptable framerate. 

<img src="images/naive.png">

<img src="images/uniform.png">

<img src="images/coherent.png">

Based on the above 3 plots, we conclude that there is approximately **x10** efficiency improvement (in terms of the number boids the method can handle) per step going from the naive approach to the coherent uniform grid approach. For example, the naive approach can handle tens of thousands of particles whereas the coherent grid approach can handle millions of particles with ease. Our optimization works.  

---
We also plot framerate change with increasing block size to investigate the effect of block size on the efficiency of the algorithm: 


For each implementation, how does changing the block count and block size affect performance? Why do you think this is?

---
In this implementation, the cell width of the uniform grid is hardcoded to be twice the neighborhood distance. Therefore, we can get away with at most 8 neighbor cell checks. However, if we change the cell width to be the neighborhood distance, 27 neighboring cells will need to be checked. 

Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not? Be careful: it is insufficient (and possibly incorrect) to say that 27-cell is slower simply because there are more cells to check!

