import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import datetime

class timeseries:
    def plot_one_var(self,dfs,names,var,units,name,plots_dir,
                     locations=[np.nan],
                     start='all',end='all',
                     small_preds=[np.nan],obs_idx=np.nan,
                     colors=[0],back_times=[[[0]]],back_colors=np.nan,
                     figsize=[0]):
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
            if figsize[0] == 0:
                plt.figure()
            else:
                plt.figure(figsize=figsize)
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
            # let's add background if desired
            if back_times[0][0][0] != 0:
                this_back_times = back_times[l][:][:]
                num_blocks,none = np.shape(this_back_times)
                for b in range(num_blocks):
                    plt.axvspan(
                        this_back_times[b][0],this_back_times[b][1],
                        color=back_colors[l][b],alpha=0.3
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
            plt.savefig(save_name,bbox_inches='tight',dpi=400)
            plt.close()
    def plot_two_var(self,dfs_one,names_one,var_one,units_one,
                     df_two,name_two,var_two,units_two,
                     name,plots_dir,
                     locations=[np.nan],
                     start='all',end='all',
                     small_preds=[np.nan],obs_idx=np.nan,
                     colors=[0],back_times=[[[0]]],back_colors=np.nan,
                     figsize=[0],size_two=1,yearly_ticks=False):
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
        ex_df = dfs_one[0][var_one]
        if np.isnan(locations)[0] == True:
            locations = list(ex_df.columns)
        if np.isnan(small_preds)[0] == True:
            small_preds = np.repeat(
                1,len(dfs_one)
            )
        colors_given = True
        if colors[0] == 0:
            colors = plt.cm.get_cmap('rainbow',len(dfs))
            colors_given = False
        for l,loc in enumerate(locations):
            print(
                'plotting watershed timeseries for {} at watershed {}'.format(
                    var_one,loc
                )
            )
            if figsize[0] == 0:
                fig,ax1 = plt.subplots()
            else:
                fig,ax1 = plt.subplots(figsize=figsize)
            ax2 = ax1.twinx()
            for d,df in enumerate(dfs_one):
                if np.isnan(obs_idx) == False and obs_idx == d:
                    this_color = 'k'
                elif colors_given == True:
                    this_color = colors[d]
                else:
                    this_color = colors(d)
                this_var = df[var_one][loc]
                if start != 'none':
                    this_var = this_var.loc[start:]
                if end != 'none':
                    this_var = this_var.loc[:end]
                times = this_var.index
                ax1.plot(
                    times,this_var,linewidth=small_preds[d],
                    label=names_one[d],c=this_color
                )
            # let's plot the second variable
            this_data = df_two[var_two][loc]
            this_data_inv = this_data*-1
            data_min = this_data_inv.min()
            y_min = data_min*4
            times = list(this_data.index)
            ax2.bar(
                times,this_data_inv,label=name_two,width=size_two
            )
            ax2.set_ylim(y_min,0)
            ticks2 = ax2.get_yticks()
            ax2.set_yticklabels([int(abs(tick)) for tick in ticks2])
            if yearly_ticks:
                times = this_data.index
                unique_years = times.year.unique()
                years_dt = [0 for y in range(len(unique_years))]
                years_name = [0 for y in range(len(unique_years))]
                for u,un in enumerate(unique_years):
                    years_dt[u] = datetime.date(un,1,1)
                    years_name[u] = un
                ax1.set_xticks(years_dt,years_name)
            # let's add background if desired
            if back_times[0][0][0] != 0:
                this_back_times = back_times[l][:][:]
                num_blocks,none = np.shape(this_back_times)
                for b in range(num_blocks):
                    ax1.axvspan(
                        this_back_times[b][0],this_back_times[b][1],
                        color=back_colors[l][b],alpha=0.3
                    )
            ax1.set_xlabel('date')
            ax1.set_ylabel(
                '{var} ({unit})'.format(
                    var=var_one,unit=units_one
                )
            )
            ax2.set_ylabel(
                '{var} ({unit})'.format(
                    var=var_two,unit=units_two
                )
            )
            ax1.legend(loc='lower right')
            ax2.legend(loc='upper right')
            save_name = os.path.join(
                plots_dir,
                name+'_{var1}_and_{var2}_{loc}.png'.format(
                    var1=var_one,var2=var_two,loc=loc
                )
            )
            plt.savefig(save_name,bbox_inches='tight',dpi=400)
            plt.close()

