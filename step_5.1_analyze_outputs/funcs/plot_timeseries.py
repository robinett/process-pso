import numpy as np
import matplotlib.pyplot as plt
import os
import sys

class timeseries:
    def plot_one_var(self,dfs,names,var,units,name,plots_dir,
                     locations=[np.nan],
                     start='all',end='all',
                     small_preds=[np.nan],obs_idx=np.nan,
                     colors = [0]):
        '''
        Function that plots timeseries
        Inputs:
            dfs: list of the different dfs to be plotted
            names: list of the different experiments names
            var: which variable from the df should be plotted?
            units: what are the units of the variable?
            time_window: if specfied, the time window to be plotted
            (inclusive). If not specified, all times are plotted.
            locations: the pixels/basins to plot. If not specified, all are
            plotted.
            small_preds: make the prediction lines smaller than the
            colors: list of colors of the different lines
        Outputs:
            nothing
        '''
        ex_df = dfs[0][var]
        if np.isnan(locations)[0] == True:
            locations = list(ex_df.columns)
        if np.isnan(small_preds)[0] == True:
            small_preds = np.repeat(
                1,len(dfs)
            )
        colors_given = True
        if colors[0] == 0:
            colors = plt.cm.get_cmap('rainbow',len(dfs))
            colors_given = False
        for l,loc in enumerate(locations):
            print(
                'plotting watershed timeseries for {} at watershed {}'.format(
                    var,loc
                )
            )
            plt.figure()
            for d,df in enumerate(dfs):
                if np.isnan(obs_idx) == False and obs_idx == d:
                    this_color = 'k'
                elif colors_given == True:
                    this_color = colors[d]
                else:
                    this_color = colors(d)
                this_var = df[var][loc]
                if start != 'none':
                    this_var = this_var.loc[start:]
                if end != 'none':
                    this_var = this_var.loc[:end]
                times = this_var.index
                plt.plot(
                    times,this_var,linewidth=small_preds[d],
                    label=names[d],c=this_color
                )
            plt.xlabel('date')
            plt.ylabel(
                '{var} ({unit})'.format(
                    var=var,unit=units
                )
            )
            plt.legend()
            save_name = os.path.join(
                plots_dir,
                name+'_{var}_{loc}.png'.format(
                    var=var,loc=loc
                )
            )
            plt.savefig(save_name,bbox_inches='tight',dpi=300)
            plt.close()

