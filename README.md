**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Paul (San) Jewell
  * [LinkedIn](https://www.linkedin.com/in/paul-jewell-2aba7379), [work website](
    https://www.biociphers.org/paul-jewell-lab-member), [personal website](https://gitlab.com/inklabapp), [twitter](https://twitter.com/inklabapp), etc.
* Tested on: (TODO) Linux pop-os 5.11.0-7614-generic, i7-9750H CPU @ 2.60GHz 32GB, GeForce GTX 1650 Mobile / Max-Q 4GB 


- For each implementation, how does changing the number of boids affect performance? Why do you think this is?

| Number of Boids | ~FPS Brute Force | ~FPS Scatter Grid | ~FPS Coherent Grid |   |
|-----------------|-------------|--------------|---------------|---|
| 1000            | 116         | 22           |   --            |   |
| 2000            | 59          | 9            |   --           |   |
| 3000            | 33          | 287          | 329           |   |
| 4000            | 22          | 268          | 294           |   |
| 5000            | 14          | 146          | 176           |   |
| 10000           | 3           | 88           | 114           |   |
| 13000           | 1.6         | 61           | 89            |   |
| 15000           | 1           | 55           | 75            |   |
| 20000           | 0.7         | 41           | 58            |   |

In all cases in general, increasing the number of boids results in worse performance. 
This is simply because evaluating one frame of motion required more calculations 
to process, and is expected. 

There is an anomaly with the low boid counts with the second and third implementations,
where an unsolved error with the thrust sort took place, but I would expect these 
trends to continue otherwise. 

![boidstats](images/boidstats.png)

- For each implementation, how does changing the block count and block size affect performance? Why do you think this is?

| Block Size | Scatter FPS (5000 boids) |   |
|------------|--------------------------|---|
| 1          | 65                       |   |
| 2          | 85                       |   |
| 4          | 103                      |   |
| 8          | 122                      |   |
| 16         | 138                      |   |
| 32         | 164                      |   |
| 64         | 164                      |   |
| 128        | 161                      |   |
| 256        | 155                      |   |
| 512        | 153                      |   |
| 1024       | < out of memory >          |   |


![blockstats](images/blockstats.png)
    
It seems there is a certain ideal block size somewhere around 32 on my system, which 
yields a very good speedup. After this increasing the size seems to have saturated
the memory, and some thrashing may be occurring, though it's not terrible. I'm guessing 
it's a similar concept to multithreading in general. There will always be a bottleneck
in bandwidth somewhere. It seems 32 blocks is a good rule of thumb. I'm not sure
how variable this will be between problems / systems. 

- For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?

From the earlier plot, we can see there are minor performance improvements using the uniform
grid over the scattered one. This highlights the importance of trying to keep your memory 
accesses (from GPU main memory) as close together as you are able, to maximize the 
data read at a time in to the cache. (Because, this will be greater than just one byte at
a time!)

- Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not? Be careful: it is insufficient (and possibly incorrect) to say that 27-cell is slower simply because there are more cells to check!

|                                      | FPS |   |
|--------------------------------------|-----|---|
| Baseline (Double width) (2744 cells) | 162 |   |
| Single Width (17576 cells)           | 3   |   |
| Quad Width (512 cells)               | 106 |   |
| 8x (64 cells)                        | 40  |   |
| ~~~                                  |     |   |
| Baseline With max 8 neighbors        | 162 |   |
| Max 26 neighbors                     | 116 |   |

Changing cell width is a balancing exercise. The performance here is tied to the max 
rule distances used, so your mileage may vary. The takeaway is that your widths should 
generally be around double the rule distance as indicated, because otherwise you will be 
checking distance to extra boids which have no chance of falling within the rule. This
problem gets worse as you make the cells larger and larger. 

Single width just doesn't make any sense because you will miss most boid interactions. 

Checking all neighbor cells in all directions is a bit slower. This is because, while
there may be additional distances to check for boids in those 'farther away' cells, 
Based on out cell sizing, we won't ever pass the distance threshold for rules
when the boids are in the far cells, so it always equated to wasted work.  