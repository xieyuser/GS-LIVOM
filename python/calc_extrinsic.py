import numpy as np
import numpy.linalg as npl

til = np.eye(4)
til[:3, :3] = np.array([0.999926, -0.007594, -0.009503, 0.007560, 0.999965, -0.003615, 0.009530,  0.003543, 0.999948]).reshape((3, 3))
til[:3, 3] = np.array([0.070478, -0.006242, 0.107762])
print(til)

tcl = np.eye(4)

tcl[:3, :3] = np.array([[0, 1, 0, 0, 0, -1, -1, 0, 0]]).reshape((3,3))

tcl[:3, 3] = np.array([0.2, -0.15, -0.1])

print(tcl)

tic = til @ npl.inv(tcl)
print(tic[:3, :3].reshape([1, -1]).tolist())
print(tic[:3, 3].reshape([1, 3]).tolist())
