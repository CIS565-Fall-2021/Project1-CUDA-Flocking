**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

Dear Grader!

As you may not know, I registered late for the class(Actually registered on the last day of registration), and I negotiated with Professor Shehzan about due dates via email and  my Project 1 due date is Sept 22. However as you can see, I was late for about 26 hours. I know there are 4 late days without penalty but I only want to use ONE on this project. I didn't mean to waste a late day only for the extra 2 hours. You can deduct points for being late for one day (hopefully 2 hours!)

Thanks

* Xiao Wei
  * (TODO) [LinkedIn](), [personal website](), [twitter](), etc.
* Tested on:  Windows 10, i9-9900k @ 3.6GHz 16.0GB, RTX 2080 SUPER 16GB

### (TODO: Your README)

Simulation gif


![p1-6](https://user-images.githubusercontent.com/66859615/134622482-b29722aa-0751-4b32-9732-9a65009cc15f.gif)

![p1-5](https://user-images.githubusercontent.com/66859615/134622494-2c6ffde0-e230-4ce1-b757-a03d36d04777.gif)



![20210924054902](https://user-images.githubusercontent.com/66859615/134626140-86cab152-370f-4cfa-9b1c-79ab40f0faa6.jpg)

![20210924055435](https://user-images.githubusercontent.com/66859615/134626143-63d35c54-4bd2-4be4-816c-46a207ff12b4.jpg)

![20210924061425](https://user-images.githubusercontent.com/66859615/134627047-6a582d78-b49f-45cf-bb48-43236121419a.jpg)

## For each implementation, how does changing the number of boids affect performance? Why do you think this is?
Performance drops as we increase the number of boids. Each boid in the flocking takes resource(thread) and computational resource #to get their velocity and postion. More boids, more work

## For each implementation, how does changing the block count and block size affect performance? Why do you think this is?
Changing block size actually does not change performance drastically. Acquiring more block only needs simple operation

##  For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?
Yes, the performance improved. I expected the outcome since we don't bother to access dev_particleArrayIndices 

## Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not? Be careful: it is insufficient (and possibly incorrect) to say that 27-cell is slower simply because there are more cells to check!

27 Cells actually is faster somehow, I guess the smaller 27 cells give more granularity for paralleling.
