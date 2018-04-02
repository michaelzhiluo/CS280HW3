import numpy as np
import cv2
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
#from mpl_toolkits.mplot3d import Axes3D
#from IPython import embed

'''
Homework 2: 3D reconstruction from two Views
This function takes as input the name of the image pairs (i.e. 'house' or
'library') and returns the 3D points as well as the camera matrices...but
some functions are missing.
NOTES
(1) The code has been written so that it can be easily understood. It has 
not been written for efficiency.
(2) Don't make changes to this main function since I will run my
reconstruct_3d.m and not yours. I only want from you the missing
functions and they should be able to run without crashing with my
reconstruct_3d.m
(3) Keep the names of the missing functions as they are defined here,
otherwise things will crash
'''

VISUALIZE = False

def fundamental_matrix(matches):
    unnorm_img1 = np.vstack((matches[:,0], matches[:,1], np.ones((matches.shape[0], ))))
    unnorm_img2 = np.vstack((matches[:,2], matches[:,3], np.ones((matches.shape[0], ))))
    # Calculate Matrixes T1 and T2 for normalization
    T1_mean = np.mean(unnorm_img1, axis =1)
    T1_std = np.sqrt(np.var(unnorm_img1, axis = 1))
    T2_mean = np.mean(unnorm_img2, axis =1)
    T2_std = np.sqrt(np.var(unnorm_img2, axis = 1))

    T1 = np.array([[1.0/T1_std[0], 0, -T1_mean[0]/T1_std[0]], [0, 1.0/T1_std[1], -T1_mean[1]/T1_std[1]], [0,0,1]])
    T2 = np.array([[1.0/T2_std[0], 0, -T2_mean[0]/T2_std[0]], [0, 1.0/T2_std[1], -T2_mean[1]/T2_std[1]], [0,0,1]])

    # Normalized Points
    norm_img1 = T1.dot(unnorm_img1).T
    norm_img2 = T2.dot(unnorm_img2).T

    A = np.array([norm_img1[:,0]*norm_img2[:,0], 
                  norm_img1[:,1]*norm_img2[:,0],
                  norm_img2[:,0],
                  norm_img1[:,0]*norm_img2[:,1],
                  norm_img1[:,1]*norm_img2[:,1],
                  norm_img2[:,1],
                  norm_img1[:,0],
                  norm_img1[:,1],
                  np.ones((matches.shape[0]))]).T
    
    # Rayleigh's Quotient Optimization Equation (Choose smallest eigenvalue with corresponding eigenvector in V)
    U,S,V = np.linalg.svd(A.T.dot(A))
    F = V[V.shape[0]-1]
    F = F.reshape((3,3))

    # Reducing Rank by taking SVD
    U,S,V = np.linalg.svd(F)
    S[2] = 0
    F = U.dot(np.diag(S)).dot(V)

    # Denormalize
    F = T2.T.dot(F).dot(T1)

    #  Calculate Residual
    residual =0.0
    for i in range(0, matches.shape[0]):
        x1 = unnorm_img1[:,i].reshape((3, 1))
        x2 = unnorm_img2[:,i].reshape((3, 1))
        residual+= (np.abs(np.asscalar(x2.T.dot(F).dot(x1)))/np.linalg.norm(F.dot(x1)))**2
        residual+= (np.abs(np.asscalar(x1.T.dot(F.T).dot(x2)))/np.linalg.norm(F.T.dot(x2)))**2

    residual /= (2*matches.shape[0])
    return F, residual

def find_rotation_translation(E):
    U,S,V = np.linalg.svd(E)
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    T_list = [U[:,2], -U[:,2]]
    R_list = [-U.dot(W).dot(V), -U.dot(W.T).dot(V)]
    
    # Make sure the det(R)=1, !=-1
    for i in R_list:
        print(np.linalg.det(i))
    #print(T_list)
    #print(R_list)

    return R_list, T_list

def find_3d_points(P1, P2, matches):
    print(P1)
    print(P1.shape)
    print(P2)
    print(P2.shape)
    
    return (1,1)


def reconstruct_3d(name):
    # ------- Load images, K matrices and matches -----
    data_dir = "../data/{}".format(name)

    # images
    I1 = cv2.imread(f"{data_dir}/{name}1.jpg")
    I2 = cv2.imread(f"{data_dir}/{name}2.jpg")
    # of shape (H,W,C)

    # K matrices
    K1 = scipy.io.loadmat(f"{data_dir}/{name}1_K.mat")["K"]
    K2 = scipy.io.loadmat(f"{data_dir}/{name}2_K.mat")["K"]

    # corresponding points
    lines = open(f"{data_dir}/{name}_matches.txt").readlines()
    matches = np.array([list(map(float, line.split())) for line in lines])

    # this is a N x 4 where:
    # matches(i,1:2) is a point (w,h) in the first image
    # matches(i,3:4) is the corresponding point in the second image

    if VISUALIZE:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.imshow(np.concatenate([I1, I2], axis=1))
        plt.plot(matches[:, 0], matches[:, 1], "+r")
        plt.plot(matches[:, 2] + I1.shape[1], matches[:, 3], "+r")
        for i in range(matches.shape[0]):
            line = Line2D([matches[i, 0], matches[i, 2] + I1.shape[1]], [matches[i, 1], matches[i, 3]], linewidth=1,
                          color="r")
            ax.add_line(line)
        plt.show()

    ## -------------------------------------------------------------------------
    ## --------- Find fundamental matrix --------------------------------------

    # F        : the 3x3 fundamental matrix,
    # res_err  : mean squared distance between points in the two images and their
    # their corresponding epipolar lines

    (F, res_err) = fundamental_matrix(matches)  # <------------------------------------- You write this one!
    print(f"Residual in F = {res_err}")
    E = K2.T @ F @ K1

    ## -------------------------------------------------------------------------
    ## ---------- Rotation and translation of camera 2 ------------------------

    # R : cell array with the possible rotation matrices of second camera
    # t : cell array of the possible translation vectors of second camera
    (R, t) = find_rotation_translation(E)  # <------------------------------------- You write this one!

    # Find R2 and t2 from R,t such that largest number of points lie in front
    # of the image planes of the two cameras
    P1 = K1 @ np.concatenate([np.identity(3), np.zeros((3, 1))], axis=1)

    # the number of points in front of the image planes for all combinations
    num_points = np.zeros([len(t), len(R)])
    errs = np.full([len(t), len(R)], np.inf)

    for ti in range(len(t)):
        t2 = t[ti]
        for ri in range(len(R)):
            R2 = R[ri]
            P2 = K2 @ np.concatenate([R2, t2[:, np.newaxis]], axis=1)
            (points_3d, errs[ti,ri]) = find_3d_points(P1, P2, matches) #<---------------------- You write this one!
            Z1 = points_3d[:,2]
            Z2 = (points_3d @ R2[2,:].T + t2[2])
            num_points[ti,ri] = np.sum(np.logical_and(Z1>0,Z2>0))
    (ti,ri) = np.where(num_points==np.max(num_points))
    j = 0 # pick one out the best combinations
    print(f"Reconstruction error = {errs[ti[j],ri[j]]}")

    t2 = t[ti[j]]
    R2 = R[ri[j]]
    P2 = K2 @ np.concatenate([R2, t2[:, np.newaxis]], axis=1)

    # % compute the 3D points with the final P2
    points, _ = find_3d_points() # <---------------------------------------------- You have already written this one!

    ## -------- plot points and centers of cameras ----------------------------

    plot_3d() #<-------------------------------------------------------------- You write this one!


# Main
reconstruct_3d("house")

