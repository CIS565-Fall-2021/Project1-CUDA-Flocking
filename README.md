**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Richard Chen
  * [LinkedIn](https://www.linkedin.com/in/richardrlchen)
* Tested on: (TODO) Windows 11, i7-10875H @ 2.3GHz 16GB, RTX 2060 MAX-Q 6GB (Personal Computer)

## Overview
This project involved computing and rendering flocks of boids all on the GPU. 
First was a naive O(n^2) approach that involved pairwise checking all the boids. 
Next, a uniform grid data structure was employed so that only close boids would be checked, reducing the amount of math needed. Lastly, the uniform grid was improved by rearranging the buffers on the GPU rather than adding a layer of indirection. This should greatly improve memory access times.  

## Videos and Images
100,000 Boids  
<br>  
<img src="images/recording1.gif">    
  
10,000 Boids    
<br>  
<img src="images/recording2.gif">

### Naive Implementation
Naive barely handles 50k Boids
<br>
<img src="images/naiveIsSlow.png">

### Uniform Grid


## Performance
Visualize On
<br>
<img src="images/Visualize On.svg">

Visualize Off
<br>
<img src="images/Visualize Off.svg">

* As the number of boids increases, the naive approach does not scale
* At lower boid numbers, the coherent approach incurs overhead from reshuffling arrays but with more boids, the memory indirection time saved overcomes this

## Questions
* Increasing the boids increases the number of computations needed. With the naive 
implmentation acting on every pair, it is O(n^2) while for the spatial grid based
implementations, the repelling behavior at close distances means that we should not 
hit the strict n choose 2 case. 
* For each implementation, how does changing the block count and block size
affect performance? Why do you think this is?
* The coherent grid trades some extra copies and assigns to avoid reading from slow memory and needing to refresh the cache. With a small number of boids, the overhead outweighs the time saved but large boid numbers is where it shines
* Did changing cell width and checking 27 vs 8 neighboring cells affect performance?
Why or why not? Be careful: it is insufficient (and possibly incorrect) to say
that 27-cell is slower simply because there are more cells to check!


### (TODO: Your README)

Include screenshots, analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)
