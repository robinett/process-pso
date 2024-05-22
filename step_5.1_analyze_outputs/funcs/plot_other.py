import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
from statsmodels.formula.api import ols
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap
import mpl_scatter_density

class plot_other:
    def histogram(self,vals,save_dir,save_name,x_label='',y_label='',
                  bin_width=np.nan,num_bins=np.nan):
        if np.isnan(bin_width) == False and np.isnan(num_bins) == False:
            raise Exception(
                'cannot specify both bin_width and num_bins!'
            )
        save_name = os.path.join(
            save_dir,save_name
        )
        num_nans = np.shape(np.where(np.isnan(vals) == True))[1]
        if num_nans == len(vals):
            return
        vals = np.array(vals)
        if np.isnan(bin_width) == True and np.isnan(num_bins) == False:
            bin_width = (np.nanmax(vals) - np.nanmin(vals))/num_bins
        elif np.isnan(bin_width) == True and np.isnan(num_bins) == True:
            bin_width = (np.nanmax(vals) - np.nanmin(vals))/10
        plt.hist(
            vals,
            bins = np.arange(
                np.nanmin(vals),
                np.nanmax(vals) + bin_width,
                bin_width
            )
        )
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.savefig(save_name)
        plt.close()
    def point_density(self,x,y,plots_dir,save_name,x_label,y_label,
                      best_fit_line=False,dot_size=0.5):
        # can't have nans
        x_nan_idx = np.where(np.isnan(x) == True)[0]
        y_nan_idx = np.where(np.isnan(y) == True)[0]
        all_nan_idx = np.unique(
            np.concatenate((x_nan_idx,y_nan_idx),0)
        )
        x = np.delete(x,all_nan_idx)
        y = np.delete(y,all_nan_idx)
        if best_fit_line:
            # get the best fit line
            data = pd.DataFrame(
                {
                    'x':x,
                    'y':y
                }
            )
            model = ols(
                "y ~ x",
                data
            ).fit()
            b = model._results.params[0]
            m = model._results.params[1]
            r2 = model.rsquared
            min_x = np.nanmin(x)
            max_x = np.nanmax(x)
            step = (max_x - min_x)/100
            line_x = np.arange(min_x,max_x,step)
            line_y = m*line_x + b
        # calculate the point density
        white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
            (0, '#ffffff'),
            (1e-20, '#440053'),
            (0.2, '#404388'),
            (0.4, '#2a788e'),
            (0.6, '#21a784'),
            (0.8, '#78d151'),
            (1, '#fde624'),
        ], N=256)
        fig = plt.figure()
        gs = fig.add_gridspec(2,3,width_ratios=[1,7,2],height_ratios=[7,1])
        ax0 = fig.add_subplot(gs[0,0])
        ax1 = fig.add_subplot(gs[0,1],projection='scatter_density')
        ax2 = fig.add_subplot(gs[1,0])
        ax3 = fig.add_subplot(gs[1,1])
        ax4 = fig.add_subplot(gs[:,2])
        density = ax1.scatter_density(x,y,cmap=white_viridis)
        fig.colorbar(density,ax=ax4,label = 'points per pixel')
        #plt.xlabel(x_label)
        #plt.ylabel(y_label)
        #if best_fit_line:
        #    ax.plot(line_x,line_y,label='best fit line',c='k')
        #    fig.annotate(
        #        'r = {:.2f}'.format(r2),
        #        xy=(0.05,0.95),xycoords='axes fraction'
        #    )
        #    ax.legend(loc='lower right')
        #    ax.annotate(
        #        'm = {:.2f}'.format(m),
        #        xy=(0.85,0.90),xycoords='axes fraction'
        #    )
        #    ax.annotate(
        #        'b = {:.2f}'.format(b),
        #        xy=(0.85,0.95),xycoords='axes fraction'
        #    )
        fig.savefig(
            os.path.join(
                plots_dir,
                save_name
            )
        )
        plt.close(fig)
    def scatter(self,x,y,plots_dir,save_name,x_label,y_label,
                best_fit_line=False,dot_size=1,one_to_one_line=False,
                xlim=[np.nan],ylim=[np.nan]):
        # can't do best fit or one to one line if all nan
        x_nan_idx = np.where(np.isnan(x) == True)[0]
        num_x_nan = len(x_nan_idx)
        x_nan_idx = np.where(np.isnan(x) == True)[0]
        num_x_nan = len(x_nan_idx)
        num_vals = len(x)
        if num_x_nan == num_vals or num_x_nan == num_vals:
            best_fit_line = False
            one_to_one_line = False
        plt.figure()
        plt.scatter(x,y,s=dot_size)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if best_fit_line:
            # get the best fit line
            data = pd.DataFrame(
                {
                    'x':x,
                    'y':y
                }
            )
            model = ols(
                "y ~ x",
                data
            ).fit()
            b = model._results.params[0]
            m = model._results.params[1]
            r2 = model.rsquared
            min_x = np.nanmin(x)
            max_x = np.nanmax(x)
            step = (max_x - min_x)/100
            line_x = np.arange(min_x,max_x,step)
            line_y = m*line_x + b
            plt.plot(line_x,line_y,label='best fit line',c='k')
            plt.annotate(
                'r2 = {:.2f}'.format(r2),
                xy=(0.05,0.95),xycoords='axes fraction'
            )
            plt.annotate(
                'm = {:.2f}'.format(m),
                xy=(0.85,0.90),xycoords='axes fraction'
            )
            plt.annotate(
                'b = {:.2f}'.format(b),
                xy=(0.85,0.95),xycoords='axes fraction'
            )
            plt.legend(loc='lower left')
        if one_to_one_line:
            min_x = np.nanmin(x)
            min_y = np.nanmin(y)
            min_all = np.min([min_x,min_y])
            max_x = np.nanmax(x)
            max_y = np.nanmax(y)
            max_all = np.nanmax([max_x,max_y])
            step = (max_all - min_all)/100
            vals = np.arange(min_all,max_all,step)
            plt.plot(vals,vals,label = 'one_to_one_line')
            plt.legend()
        if np.isnan(xlim[0]) == False:
            plt.xlim(xlim[0],xlim[1])
        if np.isnan(ylim[0]) == False:
            plt.ylim(ylim[0],ylim[1])
        plt.savefig(
            os.path.join(
                plots_dir,
                save_name
            )
        )
        plt.close()
    def iteration_plot(self,vals,vals_name,plots_dir,save_name):
        # each row is a line
        # each column is a point on that line
        x = np.arange(len(vals)) + 1
        plt.plot(x,vals)
        plt.xlabel('iteration number')
        plt.ylabel(vals_name)
        plt.savefig(
            os.path.join(
                plots_dir,
                save_name
            )
        )
        plt.close()






















