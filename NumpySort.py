
import pandas as pd
import numpy as np


x=np.dtype([('name','S10'),('perc',float)])
y=np.array([('andrew',89),('kevin',89.2)],dtype=x)

print(np.sort(y,order='perc'))