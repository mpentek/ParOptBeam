import numpy as np


load_vector_for_my = {'a': np.array([-34878.00949995, -74260.9535405 , -99245.18311065, -34848.31069424]), 'b': np.array([-2482713.19890521,  1632947.26127197,  1454922.31834314,
        1820851.10738201]), 'g': np.array([ 26963381.7442002 ,   4406617.60109066,   5230943.04789097,
       -37128264.67161786]), 'y': np.array([1327414.79101693, 2967515.60415245, 3462513.91188062,
       1815035.58076857]), 'z': np.array([ 8244668.72809955, 16992557.14038   , 14091191.58333793,
        5757028.38122367])}


load_vector_for_mz = {'a': np.array([-34878.00949995, -74260.9535405 , -99245.18311065, -34848.31069424]), 'b': np.array([-305217.49140565,   27967.43511704,  141076.06064487,
        120992.54725119]), 'g': np.array([ 27878292.30461538,   4633964.68002086,   5505757.67123321,
       -38370980.79423772]), 'y': np.array([ 5790624.74048647, 15272233.83974417, 16090263.20718289,
        8648315.43530375]), 'z': np.array([13958.22971689, 31845.36669953, 13441.44511817,  2850.91745158])}


level = [0,60,120,180]

# single force*lever arm
my = sum(np.multiply(level,-load_vector_for_my['z']))
mz = sum(np.multiply(level,load_vector_for_mz['y']))

My= sum(load_vector_for_my['b'])
Mz= sum(load_vector_for_mz['g'])

scale = 1e+8
res_mate_my = -3744335519.55552244
result_my =   -3741852806.356598/scale
print ('dif my:', round(result_my*scale - res_mate_my))
res_mate_mz = 4403509427.46293259
result_mz =   4383530194.541412/scale
print ('dif mz:', round(result_mz*scale - res_mate_mz))


print ('My expected:', round((My+my)/scale,2), 'is JZ', round(result_my,2), 'is MP', round(res_mate_my/scale,2), '*1e+8')
print ('Mz expected:', round((Mz+mz)/scale,2), 'is JZ', round(result_mz,2), 'is MP', round(res_mate_mz/scale,2), '*1e+8')