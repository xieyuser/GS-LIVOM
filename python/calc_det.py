import numpy as np
import numpy.linalg as npl

mat = np.array([   0.000663,  -0.219112,  0.394736,  -0.317582,  -0.000145,  0.002445,  -0.280991,  0.207006,  -0.000211,  0.000506,  -0.000017,  -0.809130,  0.000262,  -0.000687,  27.270924,  22.065695,]).reshape((4, 4))
print(mat)

print(npl.det(mat))
print(npl.inv(mat))
