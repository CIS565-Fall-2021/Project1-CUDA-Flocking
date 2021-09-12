**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Ashley Alexander-Lee
  * [LinkedIn](linkedin.com/in/asalexanderlee), [Personal Website](https://asalexanderlee.myportfolio.com/)
* Tested on: Windows 10, i7-6700 @ 3.40GHz 16GB, Quadro P1000 (Moore 100A Lab)

### Project Description

![Flocking Simulation](/images/coherent.png) ![Flocking Simulation GIF](/images/coherent.gif)

This project involves creating a flocking simulation, where "boids", or particles, behave according to three rules:

1. cohesion - boids move towards the perceived center of mass of their neighbors
2. separation - boids avoid getting to close to their neighbors
3. alignment - boids generally try to move with the same direction and speed as their neighbors

These three rules help dictate the velocity of the boids at each timestep. The 3D flocking logic is based off of this [2D implementation in Processing](http://studio.sketchpad.cc/sp/pad/view/ro.9cbgCRcgbPOI6/rev.23). 

I've implemented this simulation using three different methods:

`NAIVE` Each boid checks every other boid in the simulation against its neighborhood to compute its resulting velocity.
`UNIFORM GRID` Each boid is pre-processed into a grid cell that has the width of a neighborhood * 2. The boid just needs to check the boids in 8 surrounding cells against its neighborhood to compute its resulting velocity. Four arrays are used to keep track of this information:
* `particleGridIndices`: each boid's corresponding cell; sorted with particleArrayIndices so that the cells are contiguous
* `particleArrayIndices`: the original boid indices, pointers to a boid's position and velocity -- sorted using particleGridIndices as the key
* `cellStartIndices`: the start index of each cell
* `cellEndIndices`: the end index of each cell
`COHERENT GRID` Like above, except that he boid positions and velocities are sorted so that the cells are contiguous in memory. 

