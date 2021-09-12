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

### Uniform Grid

### Coherent Uniform Grid

### (TODO: Your README)

Include screenshots, analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)
