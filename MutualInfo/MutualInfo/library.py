#%%
import numpy as np
import pandas as pd
from scipy.stats import poisson, norm
from scipy.special import logsumexp
from utilities.plotUtils import *
from scipy.signal import convolve
from pathlib import Path
import os, sys, palettable

FONT_SIZE = 15

def primary_unit_of_computation(y, x_blurred):
    """Computes 
    """
    return np.sum(np.multiply(y, np.log(x_blurred)) - x_blurred)

def pointwise_mutual_information(y, x_blurred, prior):
    """
    """
    term1 = primary_unit_of_computation(y, x_blurred)
    term2 = logsumexp(prior['blurred'].apply(lambda x: primary_unit_of_computation(y, x)))
    # print(term1, term2)
    return (term1 + np.log(prior.shape[0]) - term2)

class Prior1D:
    def __init__(self, xs):
        """Define the priors true images, i.e., the Xs
        """
        self.xs = {}
        for ii, x in enumerate(xs):
            self.xs[ii] = x
            if np.any(x[:,1]<0):
                print(x[x[:,1]<0])
                assert 0==1
        
    def evaluate_function_on_grid(self, fcn, grid):
        """Evaluate a function, f, on a grid. Return numpy array of [[x, f(x)]]
        """
        val = np.zeros((len(grid),2))
        val[:,0] = grid
        val[:,1] = [fcn(x) for x in grid]
        return val
    
    def gaussian_blur_kernel(self, blur_sd, grid):
        """Create Gaussian blurring kernel. 
        """
        self.blur_sd = blur_sd
        gaussian_kernel = np.exp(-np.power(grid,2)/(self.blur_sd**2))
        normalization_constant = np.sum(gaussian_kernel)
        assert normalization_constant!=0, 'normalization constant is zero!'
        self.blur_kernel = gaussian_kernel/normalization_constant
        
#     def blur_xs(self, grid):
    def blur_xs(self):
        """Blur the Xs. 
        """
        self.xs_blurred = {}
#         self.xs_discretized = {}
        for key, val in self.xs.items():
#             x = self.evaluate_function_on_grid(val, grid)
#             self.xs_discretized[key] = x
#             self.xs_blurred[key] = convolve(
#                 val, self.blur_kernel,
#                 mode='same',
#             )
            self.xs_blurred[key] = np.array(list(zip(
                val[:,0],
                convolve(
                    val[:,1], self.blur_kernel,
                    mode='same',
                )
            )))
    
    def subset_blurred_xs(self, lower_bound, upper_bound):
        """Subset true Xs AND blurred Xs to be on some support.
        I implemented this because of boundary effects. I.e., we might
        want to define the Xs on a larger support at first, blur, and then
        subset so that boundary effects aren't as bad.
        """
        for key, val in self.xs_blurred.items():
            # subset blurred Xs
            tmp = self.xs_blurred[key]
            tmp = tmp[(tmp[:,0]>=lower_bound) & (tmp[:,0]<=upper_bound)]
            self.xs_blurred[key] = tmp
            
            # subset true Xs
            tmp = self.xs[key]
            tmp = tmp[(tmp[:,0]>=lower_bound) & (tmp[:,0]<=upper_bound)]
            self.xs[key] = tmp
            
    
    def plot_blur_kernel(self, grid):
        """Plot blurring kernel.
        """
        fig, axs = fig_setup(1,1)
        axs[0].plot(
            grid, self.blur_kernel,
            label='blur kernel'
        )
        axs[0].set_title(f'blur sd = {self.blur_sd}')
        finalize(
            axs,
            fontsize=FONT_SIZE,
        )
    
    def plot_xs(self, noisy=False, clrs=palettable.cartocolors.qualitative.Bold_3.mpl_colors):
        """Plot the true Xs along with their blurred versions and (optional)
        noisy+blurred realizations.
        """
        # clrs = palettable.colorbrewer.qualitative.Dark2_3.mpl_colors
        ncolms = 2
        nrows = int(np.ceil(len(self.xs.keys())/ncolms))
        fig, axs = fig_setup(nrows, ncolms)
        for ii in range(len(self.xs.keys())):
#             print(ii)
            # set color scheme
            axs[ii].set_prop_cycle(
                'color', 
                clrs,
            )

            # plot true Xs
            axs[ii].plot(
                self.xs[ii][:,0], self.xs[ii][:,1],
                marker='x',
                label=f'x{ii}',
            )
            
            # plot blurred x
            axs[ii].plot(
                self.xs_blurred[ii][:,0], self.xs_blurred[ii][:,1],
                marker='.',
                label=f'blurred x{ii}',
            )
            if noisy==True:
                # plot example blurred+noisy x
                axs[ii].plot(
                    np.random.poisson(self.xs_blurred[ii]),
                    label=f'blurred+noisy x{ii}',
                )
            
        finalize(
            axs,
            fontsize=FONT_SIZE,
        )
        
    def create_prior_dataframe(self):
        """Create dataframe that is used by mcmc_mutual_information method.
        The dataframe is the prior and has columns 'prob'=probability and 
        'blurred'=blurred Xs
        """
        prior_df = {
            'prob': [],
#                 'unblurred': [],
            'blurred': [],
        #     'discretized': [],
        }
        for val in self.xs_blurred.values():
            prior_df['prob'].append(1/len(self.xs_blurred.keys()))
            prior_df['blurred'].append(val[:,1])
        self.prior_df = pd.DataFrame(prior_df)

    def plot_mcmc_trace(self):
        """Plot running cumulative mean of pointwise mutual informations
        across the MCMC draws. 
        """
        fig, axs = fig_setup(1, 1)
#             cumulative_mean = np.zeros(len(self.mutual_infos))
        cumulative_mean = np.divide(
            np.cumsum(self.mutual_infos), np.arange(1,len(self.mutual_infos)+1)
        )
        axs[0].plot(
            cumulative_mean,
#             label='mutual information'
        )
        axs[0].set_ylabel('mutual information\ncumulative mean')
        axs[0].set_xlabel('iteration')
        finalize(
            axs,
            fontsize=FONT_SIZE,
        )

    def mcmc_mutual_information(self, num_mcmc_draws):
        """Compute mutual information. 
        """
        # draw from prior
        xs_blurred = np.random.choice(
            # np.arange(0, prior.shape[0]), 
            self.prior_df['blurred'],
            size=num_mcmc_draws, 
            p=self.prior_df['prob'],
        )

        self.mutual_infos = np.zeros(num_mcmc_draws) # array where PMIs are saved
        for ii, x_blurred in enumerate(xs_blurred):
            # add noise to blurred X
            y = np.random.poisson(
                x_blurred
            )
            # compute PMI
            mutual_info = pointwise_mutual_information(y, x_blurred, self.prior_df)
            self.mutual_infos[ii] = mutual_info
        self.mutual_info = np.mean(self.mutual_infos) # approximate MI is average PMI

        
def mcmc(xs, kernel_grid, num_mcmc_draws, blur_sd, subset_bounds=False):
    """Computes the (approximate) mutual information.
    Given set of true images, i.e., the Xs, a grid on which to evaluate
    the kernel, the blur standard deviation, the number of MCMC draws,
    and the bounds (optional) to subset the blurred Xs/ 
    """
    prior = Prior1D(xs)
    prior.gaussian_blur_kernel(blur_sd, kernel_grid)
    prior.blur_xs()
    if subset_bounds!=False:
        prior.subset_blurred_xs(subset_bounds[0], subset_bounds[1])
    prior.create_prior_dataframe()
#     print(prior.prior_df.head(2))
    prior.mcmc_mutual_information(num_mcmc_draws)
    return [blur_sd, prior]

####################################
####################################
# OLD CODE

# def log_prob_of_y(y, prior):
#     return np.log(np.sum(prior.apply
#     (lambda row: row['prob']*prob_of_y_given_x(y, row['blurred'][:,1]), axis=1)))

# def prob_of_y_given_x(y, x):
#     return np.prod(poisson.pmf(y, x))
    
# def log_prob_of_y_given_x(y, x):
#     return np.sum(np.log(poisson.pmf(y, x)))

# def evaluate_function_on_grid(fcn, grid):
#     val = np.zeros((len(grid),2))
#     val[:,0] = grid
#     val[:,1] = [fcn(x) for x in grid]
#     return val

# def blur_operator(fcn, blur_sd, grid):
#     x_blurred = np.zeros((len(grid), 2))
#     x_blurred[:,0] = grid
#     x = np.array([fcn(tmp) for tmp in grid])
#     x_blurred[:,1] = gaussian_filter1d(
#         x, blur_sd,
#         mode='reflect',
#     )
#     return x_blurred

# def mcmc_mutual_info(prior, num_mcmc_samples):
#     # compute mutual information
#     xs_blurred = np.random.choice(
#         # np.arange(0, prior.shape[0]), 
#         prior['blurred'],
#         size=num_mcmc_samples, 
#         p=prior['prob'],
#     )
#     mutual_infos = np.zeros(num_mcmc_samples)
#     for ii, x_blurred in enumerate(xs_blurred):
#         y = np.random.poisson(
#             x_blurred
#         )
#         mutual_info = pointwise_mutual_information(y, x_blurred, prior)
#         mutual_infos[ii] = mutual_info
#     # assert mutual_info>0, 'mutual information should be greater than 0'
#     return mutual_infos

# def plot_prior(prior):
#     # fig, axs = fig_setup(prior.shape[0], 1)
#     fig, axs = fig_setup(1,1)
#     kk = 0
#     for ii, row in prior.iterrows():
#         # ii = 0
#         # if 'unblurred' in prior.columns:
#             # axs[ii].plot(
#             #     row['unblurred'][:,0], row['unblurred'][:,1],
#             #     label='unblurred'
#             # )
#         if 'discretized' in prior.columns:
#             axs[kk].plot(
#                 row['discretized'][:,0], row['discretized'][:,1],
#                 label='discretized'
#             )
#         if 'blurred' in prior.columns:
#             axs[kk].plot(
#                 row['blurred'][:,0], row['blurred'][:,1],
#                 label=f'blurred fcn {ii}'
#             )
#             noisy = np.random.poisson(row['blurred'][:,1])
#             axs[kk].plot(
#                 row['blurred'][:,0], noisy,
#                 label=f'noisy fcn {ii}'
#             )
#             # axs[ii].set_ylabel(f'prior fcn {ii+1} of {prior.shape[0]}')
#     # axs[0].set_title(
#     #     f'sd={blur_sd}',
#     #     fontsize=FONT_SIZE,
#     # )
#     finalize(
#         axs,
#         fontsize=FONT_SIZE,
#     )
#     plt.show()
#     return (fig, axs)

# def hat(x):
#     if abs(x)>1:
#         return 0
#     else:
#         return 1
