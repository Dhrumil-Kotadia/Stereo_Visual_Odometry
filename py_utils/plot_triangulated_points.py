import numpy as np
import matplotlib.pyplot as plt

feature_data = np.loadtxt("/media/storage/lost+found/projects/cpp/opencv_test/build/triangulated_points.txt", delimiter=",")
f_x, f_z = feature_data[:, 0], feature_data[:, 1]
vehicle_data = np.loadtxt("/media/storage/lost+found/projects/cpp/opencv_test/build/translation.txt", delimiter=",")
v_x, v_z = vehicle_data[:,0], vehicle_data[:,1]

# Code to have plt ion and update after each iteration of a loop. Read 1 from translation and 50 from triangulated points
plt.ion()
plt.show()
for i in range(7):
    plt.scatter(v_x[i], -v_z[i], s=15, c='b')    
    for j in range(200):
        plt.scatter(f_x[j+(i*50)], f_z[j+(i*50)], c='r', s=0.5)
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.title("Triangulated 3D Points (X-Z View)")
    plt.xlim([-5, 5])
    plt.ylim([-0.5, 3])
    # plt.grid()
    plt.pause(0.1)
    plt.savefig("/media/storage/lost+found/projects/cpp/opencv_test/plot" + str(i) + ".png")
plt.ioff()
plt.show()