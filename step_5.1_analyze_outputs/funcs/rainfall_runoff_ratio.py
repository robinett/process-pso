import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
from plot_watersheds import plot_wat


class rainfall_runoff:
    def __init__(self,rainfall_obs,strm_obs,plots_dir,save_name):
        ratio,ratio_inv = self.get_ratio(rainfall_obs,strm_obs,save_name)
        watersheds = list(strm_obs.columns)
        self.ratio_scatter(ratio,watersheds,plots_dir,'standard')
        self.ratio_scatter(ratio_inv,watersheds,plots_dir,'inv')
        self.ratio_map(ratio,watersheds,plots_dir,'standard')
        self.ratio_map(ratio_inv,watersheds,plots_dir,'inv')
    def get_ratio(self,rainfall_obs,strm_obs,save_name):
        '''
        Function that calculates the percent of rainfall that leaves a basin as
        strm.
        Inputs:
            rainfall_obs: pandas df of rainfall obs, where index is time,
            columns are the basins
            strm_obs: pandas df of strm obs, where index is time, columns
            are basins. should be the same size as rainfall_obs
        Ouputs:
            ratio_df: pandas df that is the ratio at each basin
        '''
        basin_avg_rain = np.array(rainfall_obs.mean(axis=0))
        basin_avg_strm = np.array(strm_obs.mean(axis=0))
        ratio = basin_avg_strm/basin_avg_rain
        ratio_inv = basin_avg_rain/basin_avg_strm
        basins = list(rainfall_obs.columns)
        ratio_df = pd.DataFrame(columns=basins)
        ratio_df.loc['strm_over_rainfall'] = ratio
        ratio_df.loc['rainfall_over_strm'] = ratio_inv
        ratio_df.index.name = 'ratio'
        ratio_df.to_csv(save_name)
        return [ratio,ratio_inv]
    def ratio_scatter(self,ratio_scatter,watersheds,plots_dir,ratio_type):
        x_vals = np.arange(len(watersheds))
        if ratio_type == 'standard':
            save_name = os.path.join(
                plots_dir,
                'runoff_over_rainfall_scatter.png'
            )
        elif ratio_type == 'inv':
            save_name = os.path.join(
                plots_dir,
                'rainfall_over_runoff_scatter.png'
            )
        plt.figure(figsize=(25,5))
        plt.scatter(x_vals,ratio_scatter)
        if ratio_type == 'standard':
            plt.yscale('log')
        plt.xticks(ticks=x_vals,labels=watersheds,rotation=90)
        plt.savefig(save_name,bbox_inches='tight')
        plt.close()
    def ratio_map(self,ratio,watersheds,plots_dir,ratio_type):
        pw = plot_wat()
        if ratio_type == 'standard':
            name = 'runoff_over_rainfall'
            log_bool = True
        elif ratio_type == 'inv':
            name = 'rainfall_over_runoff'
            log_bool = False
        avg_val = -9999
        pw.plot_map(
            name,watersheds,ratio,avg_val,plots_dir,log=log_bool
        )
