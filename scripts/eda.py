import numpy as np
import pandas as pd
import sys, os, copy, palettable
from utilities.utilityFunctions import unpickle_object, pickle_object
from pathlib import Path
from MutualInfo.library import *
from MutualInfo.constants import *
from scipy.signal import convolve
import multiprocessing as mp
from joblib import Parallel, delayed

num_cores = mp.cpu_count()
print("Number of processors: ", mp.cpu_count())

# processed_list = Parallel(n_jobs=num_cores)(delayed(my_function(i,parameters) 
# inputs = myList

# num_mcmc_draws = 100000
# blur_sds = [10**(ii) for ii in np.linspace(-2, 1, 31)]
# x_size = 51
# x_grid = np.linspace(-np.pi, np.pi, x_size)
# kernel_grid = np.linspace(-np.pi, np.pi, 2*x_size)
# # scales = [1, 5, 10, 100, 1000]
# scales = [5, 10, 100, 1000]
# subset_bounds = [-np.pi/2, np.pi/2]

# for scale in scales: 
#     print(f'scale={scale}')
#     x0 = np.array(list(zip(x_grid, scale+scale*np.cos(x_grid))))
#     x1 = copy.deepcopy(x0)
#     idx = np.argwhere((x1[:,0]>-np.pi/2))[0][0]
#     x1[idx+3, 1] = scale+scale*0.8
#     xs = [x0, x1]
#     results = []
    # for blur_sd in blur_sds:
        # print(f'blur sd = {blur_sd}')
        # results.append(mcmc(xs, kernel_grid, num_mcmc_draws, blur_sd, subset_bounds))

#     pickle_object(
#         os.path.join(RESULTS_DIR, f'cosine_alteredCosine_scale{scale}.pkl'),
#         results
#     )

num_mcmc_draws = 100000
blur_sds = [10**(ii) for ii in np.linspace(-2, 1, 31)]
x_size = 51
x_grid = np.linspace(-np.pi, np.pi, x_size)
kernel_grid = np.linspace(-np.pi, np.pi, 2*x_size)
subset_bounds = [-np.pi/2, np.pi/2]
scales = [5, 10, 100, 1000]

for scale in scales: 
    print(f'scale={scale}')
    x0 = np.array(list(zip(x_grid, scale+scale*np.cos(x_grid))))
    x1 = copy.deepcopy(x0)
    idx = np.argwhere((x1[:,0]>-np.pi/2))[0][0]
    x1[idx+3, 1] = scale+scale*0.8
    x2 = np.array(list(zip(x_grid, np.repeat(scale, len(x_grid)))))
    x3 = np.array(list(zip(
        x_grid, scale+scale*np.sin(2*x_grid)
    )))
    xs = [x0, x1, x2, x3]

    results = []
    for blur_sd in blur_sds:
        print(f'blur sd = {blur_sd}')
        results.append(mcmc(xs, kernel_grid, num_mcmc_draws, blur_sd, subset_bounds=[-np.pi/2, np.pi/2]))

    pickle_object(
        os.path.join(RESULTS_DIR, f'cosine_alteredCosine_constant_sine_scale{scale}.pkl'),
        results
    )