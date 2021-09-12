from matplotlib import pyplot as plt

# number of boids
num_boids = [5000, 10000, 15000, 20000, 30000, 50000, 100000, 200000, 500000]

# simulation w/o visualization
fps_naive_boids = [430, 212, 138, 100, 51, 20, 5.5, 1.5, 0.2]
fps_scattered_boids = [1300, 1180, 920, 840, 600, 350, 167, 70, 5]
fps_coherent_boids = [1880, 1540, 1460, 1260, 750, 540, 230, 105, 25]

# simulation w/ visualization
fps_naive_boids_viz = [411, 206, 138, 100.2, 50.8, 20, 5.6, 1.5, 0.2]
fps_scattered_boids_viz = [1108, 1000, 830, 756, 565, 310, 155, 65.5, 5.2]
fps_coherent_boids_viz = [1218, 1065, 1000, 905, 630, 480, 220, 118, 27]

plt.figure()
plt.plot(num_boids, fps_naive_boids, '-b',
         num_boids, fps_scattered_boids, '.-b',
         num_boids, fps_coherent_boids, '*-b',
         num_boids, fps_naive_boids_viz, '--r',
         num_boids, fps_scattered_boids_viz, '.--r',
         num_boids, fps_coherent_boids_viz, '*--r')
plt.legend(['Naive w/o sim', 'Scattered w/o sim', 'Coherent w/o sim',
            'naive w/ sim', 'Scattered w/ sim', 'Coherent w/ sim'])
plt.xlabel('Num of Boids')
plt.ylabel('FPS')
plt.title('FPS vs. Num of Boids Graph')
plt.show()

# Block size (num_boids = 500000)
block_size = [128, 256, 512, 1024]
fps_coherent_block_size_viz = [24, 26, 30, 40]
plt.figure()
plt.plot(block_size, fps_coherent_block_size_viz, '.-')
plt.xlabel('Block size')
plt.ylabel('FPS')
plt.title('FPS vs. Block size Graph')
plt.show()
