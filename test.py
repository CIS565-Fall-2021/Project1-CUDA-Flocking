# with visualization
coherentBoids = np.linspace(1000000, 2500000, 4)
coherentFPS = [95, 50, 30, 19]
# without visualization
coherentBoidsV = np.linspace(1000000, 2500000, 4)
coherentFPSV = [108, 53, 31, 20]

coherentFig, coherentAxes = plt.subplots()

coherentAxes.plot(coherentBoids, coherentFPS, label="With Visualization", marker='o', markerfacecolor="yellow", markeredgecolor="green")
coherentAxes.plot(coherentBoidsV, coherentFPSV, label="Without Visualization", marker='o', markerfacecolor="yellow", markeredgecolor="green")
coherentAxes.yaxis.set_ticks(np.arange(0, 110, 10))
coherentAxes.set_xlabel('Number of Boids') # Notice the use of set_ to begin methods
coherentAxes.set_ylabel('FPS')
coherentAxes.set_title('coherent Approach')
coherentAxes.axhline(y=30, color='r', linestyle='--',alpha=0.5)
coherentAxes.axhline(y=60, color='g', linestyle='--',alpha=0.5)
coherentAxes.legend()