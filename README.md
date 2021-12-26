**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Kaan Erdogmus
  * [LinkedIn](https://linkedin.com/in/kaanberk), [personal website](https://kaan9.github.io)
* Tested on: Windows 10, i7-8850H @ 2.592GHz 24GB, Quadro P1000

![Coherent Uniform Flocking](visuals/coherent_overview.gif)

## Project Overview

This project is a flocking simulation that implements the
[Reynold Boids](http://www.vergenet.net/~conrad/boids/pseudocode.html)
algorithm on the GPU using CUDA.

For each boid, its velocity is adjusted at each time-step based on 3 rules:
1. Cohesion: the boid moves towards the center of its neighbours.
2. Separation: the boid tries to maintain a minimum distance from other boids
3. Alignment: the boid tries maintain a similar velocity (including direction) as its neighbours.


## Features Implemented
* Naive Neighbour Search
	* every boid iterates over every boid to calculate rule values
* Uniform-grid based neighbour search
	* space is split into cubic cells and each boid only looks at nearby cells
* Coherent uniform-grid based neighbour search
	* in addition to the above, position and velocity buffers are rearranged before accesses to reduce memory
	indirection.
	* Grid-loop optimization: only checks adjacent cells that are close enough for one of the 3 rules


## Performance Analysis

![FPS measurements](visuals/fps.png)

