import numpy as np
import matplotlib.pyplot as plt

feature_data = np.loadtxt("outputs/triangulated_points.txt", delimiter=",")
f_x, f_y, f_z = feature_data[:, 0], feature_data[:,1], feature_data[:,2]
vehicle_data = np.loadtxt("outputs/translation.txt", delimiter=",")
v_x, v_y, v_z = vehicle_data[:,0], vehicle_data[:,1], vehicle_data[:,2]

# # 3D plot of the triangulated points and camera poses
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(v_x, v_y, -v_z, c='b', label='Camera Poses')
# ax.scatter(f_x, f_y, f_z, c='r', s=0.5, label='Triangulated Points')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_xlim([-10, 60])
# ax.set_ylim([-10, 60])
# ax.set_zlim(-0.5, 100)
# plt.title("Triangulated 3D Points")
# plt.legend()
# plt.show()

# Code to have plt ion and update after each iteration of a loop. Read 1 from translation and 50 from triangulated points
# plt.ion()
# plt.show()
for i in range(26):
    # if i%2 == 0:
    #     continue
    plt.scatter(v_x[i], -v_z[i], s=15, c='b')    
    for j in range(100):
        plt.scatter(f_x[j+(i*50)], f_z[j+(i*50)], c='r', s=0.5)
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.title("Triangulated 3D Points (X-Z View)")
    plt.xlim([-5, 100])
    plt.ylim([-0.5, 100])
    # plt.grid()
    plt.pause(0.1)
    plt.savefig("/media/storage/lost+found/projects/cpp/stereo_visual_odometry/update_plot_" + str(i) + ".png")
# plt.ioff()
# plt.show()