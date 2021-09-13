**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Lindsay Smith
  * [LinkedIn](https://www.linkedin.com/in/lindsay-j-smith/), [personal website](https://lindsays-portfolio-d6aa5d.webflow.io/)
* Tested on: Windows 10, i7-11800H 144Hz 16GB RAM, GeForce RTX 3060 512GB SSD (Personal Laptop)

# Questions

### For each implementation, how does changing the number of boids affect performance? Why do you think this is?
For every implementation, the performance gets worse as the number of boids increases. This is because when you have more boids you will also
have more threads running at a time. We will eventually get to a point where there are more threads needed than are possible to run at a time.
The more boids we have the more work our computers are going to have to do to maintain all of them.

### For each implementation, how does changing the block count and block size affect performance? Why do you think this is?
The block count and size do not appear to affect performance significantly. I think this is because the block size and count do not have a
direction relation to the number of threads that are able to run at a time. Although the configuration of the threads may change, the 
perfomance is not really impacted because the same number are still able to run in parallel.

### For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?
I have not gotten the coherent uniform grid to work yet.

### Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not?
I don't think that changing the cell width affects the performance because even though there are more cells to check, the distance between them is smaller.
This means each boid will be compared to fewer neighbors in a cell, and the outcome will be relatively similar to as if there were 8 cells being checked.

