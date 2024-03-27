import numpy as np
pose_1 = np.array([[ 1.21138737e-01],
 [ 1.49942614e-01],
 [-3.20847765e-03],
 [-5.88464755e-05],
 [-4.63529530e-03],
 [ 9.12948342e-04]])
pose_2 = np.array([[0.12019571],
 [0.15165014],
 [0.00308858],
 [0.00052319],
 [0.0038989 ],
 [0.00015186]])
pose_3 = np.array([[ 0.1002221 ],
 [ 0.13202037],
 [-0.04188217],
 [-0.0020545 ],
 [-0.00182309],
 [-0.00270874]])
pose_4 = np.array([[0.        ],
 [0.        ],
 [0.        ],
 [0.        ],
 [0.        ],
 [3.14159265]])
pose_5 = np.array([[0.12888246],
 [0.14396619],
 [0.02176822],
 [0.01710431],
 [0.00075967],
 [0.00908928]])

poses = np.array([pose_1,pose_2,pose_3,pose_4,pose_5])
# print(np.mean(poses,axis=0,))
# print(np.median(poses,axis=0))
# print(np.std(poses,axis=0))

# def reject_outliers(data, m = 2.):
#     d = np.abs(data - np.median(data,axis=0))
#     mdev = m * np.median(d,axis=0)
#     # s = d/mdev if mdev.any() else np.zeros(len(d))

#     for i in range(5):
#         dist = data[:][:][i] - np.median(data,axis=0)
#         if np.mean(dist) > np.mean(mdev):
#             data[:][:][i] = np.array([None,None,None,None,None,None]).reshape((6,1))
#     return data

# print(reject_outliers(poses))
poses_toavg = np.array(poses)
weights = np.ones_like(poses_toavg)
# Normalize weights to sum up to 1
weights /= np.sum(weights, axis=0)

# Compute the weighted sum of arrays
weighted_pose = np.sum(poses_toavg * weights, axis=0)
print(weighted_pose)
print(np.mean(poses_toavg, axis=0))


