from scipy.spatial.distance import directed_hausdorff
import numpy as np

from metrics import *

u = np.array([(1.0, 0.0),
              (0.0, 1.0),
              (-1.0, 0.0),
              (0.0, -1.0)])
v = np.array([(2.0, 0.0),
              (0.0, 2.0),
              (-2.0, 0.0),
              (0.0, -4.0)])

print(directed_hausdorff(v,u))
print(directed_hausdorff(u, v))
print(dis_from_a2b(v,u))