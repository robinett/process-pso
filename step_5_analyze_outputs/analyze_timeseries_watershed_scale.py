import pickle
import sys
import pandas as pd
import datetime
import copy
import numpy as np
from dateutil import relativedelta
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import netCDF4 as nc
from scipy.stats import linregress
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable

class analyze_watershed:
    def __init__(self):
        pass
    def get_tiles(self,tile_fname):
        # get the tiles that we are going to run as defined in step 1
        tiles = pd.read_csv(tile_fname,header=None)
        # turn this into an np array
        tiles = np.array(tiles).astype(int)
        # make a nice np array
        tiles = tiles.T
        tiles = tiles[0]
        # save to self
        self.tiles = tiles
        # index for this tile would start at 0, so subtract 1 to get index
        tiles_idx = tiles - 1
        self.tiles_idx = tiles_idx
        return tiles
    def get_intersection(self,int_dir):
        with open(int_dir,'rb') as f:
            out = pickle.load(f)
        return out
    def get_model_preds_watershed(self,start,end,exp_names,catch_timeseries,
                                  streamflow_timeseries,
                                  fluxcom_timeseries,intersection_info,
                                  get_metf=False):
        '''
        function that gets RMSE dict at the monthly/watershed level
        we will do this for both streamflow as well as LE
        '''
        # one edit to fluxcom timeseries for simplicity
        fluxcom_timeseries = fluxcom_timeseries.set_index('time')
        # get the lits of watersheds as strings to be used as the dict keys for
        # intersection_info
        watersheds_str = list(intersection_info.keys())
        # get the list of watersheds as integers
        watersheds = [int(k) for k in intersection_info.keys()]
        # for each watershed let's get the monthly streamflow and LE:
            # from waterwatch/fluxcom respectively
            # from default
            # from step 1 pso
            # from final step pso
        # get what will be our columns for dfs
        cols = copy.deepcopy(watersheds)
        cols.append('all')
        # get the information that we need and create the arrays that will hold
        # the final watershed information
        num_watersheds = len(watersheds)
        delta = relativedelta.relativedelta(
            end+datetime.timedelta(days=1),start
        )
        num_months = delta.years*12 + delta.months
        # placeholders for everything that we need to extract and convert
        default_all_strm = np.zeros((num_months,num_watersheds))
        default_all_le = np.zeros((num_months,num_watersheds))
        default_all_runoff = np.zeros((num_months,num_watersheds))
        default_all_baseflow = np.zeros((num_months,num_watersheds))
        fluxcom_all_le = np.zeros((num_months,num_watersheds))
        pso_init_all_strm = np.zeros((num_months,num_watersheds))
        pso_init_all_le = np.zeros((num_months,num_watersheds))
        pso_final_all_strm = np.zeros((num_months,num_watersheds))
        pso_final_all_le = np.zeros((num_months,num_watersheds))
        if get_metf:
            default_all_rainfsnowf = np.zeros((num_months,num_watersheds))
        for w,wat in enumerate(watersheds):
            # get the waterwatch streamflow for this watershed
            this_strm_data = streamflow_timeseries[str(wat)]
            # get the tiles in this watershed
            this_tiles = intersection_info[watersheds_str[w]][0]
            # get the percent of each of these tiles in this watershed
            this_perc = intersection_info[watersheds_str[w]][1]
            # get the fluxcom-defined le for each of these tiles
            fluxcom_le = fluxcom_timeseries[this_tiles]
            # get the default runoff and le for each of these tiles and average
            # runoff to monthly resolution and correct units
            # let's do this for default first
            default_timeseries = catch_timeseries[exp_names[0]]
            default_runoff = default_timeseries['runoff']
            default_baseflow = default_timeseries['baseflow']
            default_strm = default_runoff + default_baseflow
            default_runoff = default_runoff[this_tiles]
            default_baseflow = default_baseflow[this_tiles]
            default_strm = default_strm[this_tiles]
            default_le = default_timeseries['le']
            default_le = default_le[this_tiles]
            #print('default_runoff')
            #print(default_runoff)
            #print('default_baseflow')
            #print(default_baseflow)
            #sys.exit()
            # for default met forcing
            if get_metf:
                # for incoming precip
                default_rainfsnowf = default_timeseries['rainfsnowf']
                default_rainfsnowf = default_rainfsnowf[this_tiles]
            # let's do this for pso init
            pso_init_timeseries = catch_timeseries[exp_names[1]]
            pso_init_runoff = pso_init_timeseries['runoff']
            pso_init_baseflow = pso_init_timeseries['baseflow']
            pso_init_strm = pso_init_runoff + pso_init_baseflow
            pso_init_strm = pso_init_strm[this_tiles]
            pso_init_le = pso_init_timeseries['le']
            pso_init_le = pso_init_le[this_tiles]
            # and dofor pso final
            pso_final_timeseries = catch_timeseries[exp_names[2]]
            pso_final_runoff = pso_final_timeseries['runoff']
            pso_final_baseflow = pso_final_timeseries['baseflow']
            pso_final_strm = pso_final_runoff + pso_final_baseflow
            pso_final_strm = pso_final_strm[this_tiles]
            pso_final_le = pso_final_timeseries['le']
            pso_final_le = pso_final_le[this_tiles]
            # model-generated runoff is currently in kg/m2/s. we need it in mm
            # for the whole month
            # so let's start by changing each of these to totals for that day
            # and then we can add each day to get the monthly total
            default_strm = default_strm*86400 # now in kg/m2/day
            default_runoff = default_runoff*86400 # now in kg/m2/day
            default_baseflow = default_baseflow*86400 # now in kg/m2/day
            pso_init_strm = pso_init_strm*86400 # now in kg/m2/day
            pso_final_strm = pso_final_strm*86400 # now in kg/m2/day
            if get_metf:
                default_rainfsnowf = default_rainfsnowf*86400 # now in kg/m2/day
            # and one kg/m2 is equal to 1mm of water
            # so now we are good
            # and average to monthly temporal resolution
            # create the monthly resolution dataframe
            cols = list(this_tiles)
            cols.append('date')
            # for default
            default_strm_mon = pd.DataFrame(columns=cols)
            default_strm_mon = default_strm_mon.set_index('date')
            default_runoff_mon = pd.DataFrame(columns=cols)
            default_runoff_mon = default_runoff_mon.set_index('date')
            default_baseflow_mon = pd.DataFrame(columns=cols)
            default_baseflow_mon = default_baseflow_mon.set_index('date')
            default_le_mon = pd.DataFrame(columns=cols)
            default_le_mon = default_le_mon.set_index('date')
            # for default met forcing variables
            if get_metf:
                # total incoming precip
                default_rainfsnowf_mon = pd.DataFrame(columns=cols)
                default_rainfsnowf_mon = default_rainfsnowf_mon.set_index('date')
            # for pso init
            pso_init_strm_mon = pd.DataFrame(columns=cols)
            pso_init_strm_mon = pso_init_strm_mon.set_index('date')
            pso_init_le_mon = pd.DataFrame(columns=cols)
            pso_init_le_mon = pso_init_le_mon.set_index('date')
            # for pso final
            pso_final_strm_mon = pd.DataFrame(columns=cols)
            pso_final_strm_mon = pso_final_strm_mon.set_index('date')
            pso_final_le_mon = pd.DataFrame(columns=cols)
            pso_final_le_mon = pso_final_le_mon.set_index('date')
            # for fluxcom
            fluxcom_le_mon = pd.DataFrame(columns=cols)
            fluxcom_le_mon = fluxcom_le_mon.set_index('date')
            curr = copy.deepcopy(start)
            first_day = curr.strftime('%Y-%m-%d')
            last_month = curr.strftime('%m')
            while curr <= (end + datetime.timedelta(days=1)):
                this_date_fmt = curr.strftime('%Y-%m-%d')
                this_month = curr.strftime('%m')
                if this_month != last_month:
                    last_day = curr + datetime.timedelta(days=-1)
                    last_day_str = last_day.strftime('%Y-%m-%d')
                    # for default stream data
                    this_month_default_strm_data = default_strm.loc[
                        first_day:last_day_str
                    ]
                    this_month_default_strm_data_avg = this_month_default_strm_data.sum()
                    default_strm_mon.loc[last_day.strftime('%Y%m')] = (
                        this_month_default_strm_data_avg
                    )
                    # for default runoff
                    this_month_default_runoff_data = default_runoff.loc[
                        first_day:last_day_str
                    ]
                    this_month_default_runoff_data_avg = this_month_default_runoff_data.sum()
                    default_runoff_mon.loc[last_day.strftime('%Y%m')] = (
                        this_month_default_runoff_data_avg
                    )
                    # for default baseflow
                    this_month_default_baseflow_data = default_baseflow.loc[
                        first_day:last_day_str
                    ]
                    this_month_default_baseflow_data_avg = this_month_default_baseflow_data.sum()
                    default_baseflow_mon.loc[last_day.strftime('%Y%m')] = (
                        this_month_default_baseflow_data_avg
                    )
                    # for default le data
                    this_month_default_le_data = default_le.loc[
                        first_day:last_day_str
                    ]
                    this_month_default_le_data_avg = (
                        this_month_default_le_data.mean()
                    )
                    default_le_mon.loc[last_day.strftime('%Y%m')] = (
                        this_month_default_le_data_avg
                    )
                    # for default met force
                    if get_metf:
                        this_month_default_rainfsnowf_data = default_rainfsnowf.loc[
                            first_day:last_day_str
                        ]
                        this_month_default_rainfsnowf_avg = (
                            this_month_default_rainfsnowf_data.sum()
                        )
                        default_rainfsnowf_mon.loc[last_day.strftime('%Y%m')] = (
                            this_month_default_rainfsnowf_avg
                        )
                    # for pso_init stream data
                    this_month_pso_init_strm_data = pso_init_strm.loc[
                        first_day:last_day_str
                    ]
                    this_month_pso_init_strm_data_avg = this_month_pso_init_strm_data.sum()
                    pso_init_strm_mon.loc[last_day.strftime('%Y%m')] = (
                        this_month_pso_init_strm_data_avg
                    )
                    # for pso_init le data
                    this_month_pso_init_le_data = pso_init_le.loc[
                        first_day:last_day_str
                    ]
                    this_month_pso_init_le_data_avg = (
                        this_month_pso_init_le_data.mean()
                    )
                    pso_init_le_mon.loc[last_day.strftime('%Y%m')] = (
                        this_month_pso_init_le_data_avg
                    )
                    # for pso_final stream data
                    this_month_pso_final_strm_data = pso_final_strm.loc[
                        first_day:last_day_str
                    ]
                    this_month_pso_final_strm_data_avg = this_month_pso_final_strm_data.sum()
                    pso_final_strm_mon.loc[last_day.strftime('%Y%m')] = (
                        this_month_pso_final_strm_data_avg
                    )
                    # for pso_final le data
                    this_month_pso_final_le_data = pso_final_le.loc[
                        first_day:last_day_str
                    ]
                    this_month_pso_final_le_data_avg = (
                        this_month_pso_final_le_data.mean()
                    )
                    pso_final_le_mon.loc[last_day.strftime('%Y%m')] = (
                        this_month_pso_final_le_data_avg
                    )
                    # for fluxcom le data
                    this_month_fluxcom_le_data = fluxcom_le.loc[
                        first_day:last_day_str
                    ]
                    this_month_fluxcom_le_data_avg = (
                        this_month_fluxcom_le_data.mean()
                    )
                    fluxcom_le_mon.loc[last_day.strftime('%Y%m')] = (
                        this_month_fluxcom_le_data_avg
                    )
                    # update last month holder
                    first_day = curr.strftime('%Y-%m-%d')
                    last_month = curr.strftime('%m')
                # update curr and mvoe to next step
                curr += datetime.timedelta(days=1)
            # get the weighted average over the whole watershed and add to
            # array
            # for default streamflow
            default_strm_mon_np = np.array(default_strm_mon)
            default_strm_mon_avg = np.average(
                default_strm_mon_np,axis=1,weights=this_perc
            )
            default_all_strm[:,w] = default_strm_mon_avg
            # for default runoff
            default_runoff_mon_np = np.array(default_runoff_mon)
            default_runoff_mon_avg = np.average(
                default_runoff_mon_np,axis=1,weights=this_perc
            )
            default_all_runoff[:,w] = default_runoff_mon_avg
            # for default baseflow
            default_baseflow_mon_np = np.array(default_baseflow_mon)
            default_baseflow_mon_avg = np.average(
                default_baseflow_mon_np,axis=1,weights=this_perc
            )
            default_all_baseflow[:,w] = default_baseflow_mon_avg
            # for default metf
            if get_metf:
                default_rainfsnowf_mon_np = np.array(default_rainfsnowf_mon)
                default_rainfsnowf_mon_avg = np.average(
                    default_rainfsnowf_mon_np,axis=1,weights=this_perc
                )
                default_all_rainfsnowf[:,w] = default_rainfsnowf_mon_avg
            # for default le
            default_le_mon_np = np.array(default_le_mon)
            default_le_mon_avg = np.average(
                default_le_mon_np,axis=1,weights=this_perc
            )
            default_all_le[:,w] = default_le_mon_avg
            # for pso_init streamflow
            pso_init_strm_mon_np = np.array(pso_init_strm_mon)
            pso_init_strm_mon_avg = np.average(
                pso_init_strm_mon_np,axis=1,weights=this_perc
            )
            pso_init_all_strm[:,w] = pso_init_strm_mon_avg
            # for pso_init le
            pso_init_le_mon_np = np.array(pso_init_le_mon)
            pso_init_le_mon_avg = np.average(
                pso_init_le_mon_np,axis=1,weights=this_perc
            )
            pso_init_all_le[:,w] = pso_init_le_mon_avg
            # for pso_final streamflow
            pso_final_strm_mon_np = np.array(pso_final_strm_mon)
            pso_final_strm_mon_avg = np.average(
                pso_final_strm_mon_np,axis=1,weights=this_perc
            )
            pso_final_all_strm[:,w] = pso_final_strm_mon_avg
            # for pso_final le
            pso_final_le_mon_np = np.array(pso_final_le_mon)
            pso_final_le_mon_avg = np.average(
                pso_final_le_mon_np,axis=1,weights=this_perc
            )
            pso_final_all_le[:,w] = pso_final_le_mon_avg
            # for fluxcom le
            # let's get rid of nan's and replace with the average value
            # across all pixels for that month. we
            # don't want to cancel out the entire watershed because one pixels
            # has an nan value in the fluxcom dataset
            months = list(fluxcom_le_mon.index)
            for m,mon in enumerate(months):
                this_mon_vals = np.array(fluxcom_le_mon.loc[mon])
                this_nan_idx = np.where(np.isnan(this_mon_vals) == True)
                this_mon_avg = np.nanmean(this_mon_vals)
                this_mon_vals[this_nan_idx] = this_mon_avg
                fluxcom_le_mon.loc[mon] = this_mon_vals
            fluxcom_le_mon_np = np.array(fluxcom_le_mon)
            fluxcom_le_mon_avg = np.average(
                fluxcom_le_mon_np,axis=1,weights=this_perc
            )
            fluxcom_all_le[:,w] = fluxcom_le_mon_avg
        times = list(streamflow_timeseries.index)
        # for default stream
        default_strm_df = pd.DataFrame(
            default_all_strm,index=times,columns=watersheds
        )
        default_strm_df.index.name = 'date'
        # for default runoff
        default_runoff_df = pd.DataFrame(
            default_all_runoff,index=times,columns=watersheds
        )
        default_runoff_df.index.name = 'date'
        # for default baseflow
        default_baseflow_df = pd.DataFrame(
            default_all_baseflow,index=times,columns=watersheds
        )
        default_baseflow_df.index.name = 'date'
        # for default le
        default_le_df = pd.DataFrame(
            default_all_le,index=times,columns=watersheds
        )
        default_le_df.index.name = 'date'
        # met rainfsnowf
        if get_metf:
            default_rainfsnowf_df = pd.DataFrame(
                default_all_rainfsnowf,index=times,columns=watersheds
            )
            default_rainfsnowf_df.index.name = 'date'
        # for pso_init stream
        pso_init_strm_df = pd.DataFrame(
            pso_init_all_strm,index=times,columns=watersheds
        )
        pso_init_strm_df.index.name = 'date'
        # for pso_init le
        pso_init_le_df = pd.DataFrame(
            pso_init_all_le,index=times,columns=watersheds
        )
        pso_init_le_df.index.name = 'date'
        # for pso_final stream
        pso_final_strm_df = pd.DataFrame(
            pso_final_all_strm,index=times,columns=watersheds
        )
        pso_final_strm_df.index.name = 'date'
        # for pso_final le
        pso_final_le_df = pd.DataFrame(
            pso_final_all_le,index=times,columns=watersheds
        )
        pso_final_le_df.index.name = 'date'
        # for fluxcom le
        fluxcom_le_df = pd.DataFrame(
            fluxcom_all_le,index=times,columns=watersheds
        )
        fluxcom_le_df.index.name = 'date'
        outs = {
            'default_strm':default_strm_df,
            'default_runoff':default_runoff_df,
            'default_baseflow':default_baseflow_df,
            'defualt_le':default_le_df,
            'fluxcom_le':fluxcom_le_df,
            'waterwatch_strm':streamflow_timeseries,
            'pso_init_strm':pso_init_strm_df,
            'pso_init_le':pso_init_le_df,
            'pso_final_strm':pso_final_strm_df,
            'pso_final_le':pso_final_strm_df,
        }
        if get_metf:
            outs['default_rainfsnowf'] = default_rainfsnowf_df
        return outs
    def plot_streamflow_timeseries(self,outs,plots_dir,exp_names):
        default_strm = outs['default_strm']
        pso_init_strm = outs['pso_init_strm']
        pso_final_strm = outs['pso_final_strm']
        waterwatch_strm = outs['waterwatch_strm']
        watersheds = list(default_strm.columns)
        for w,wat in enumerate(watersheds):
            this_default_strm = default_strm[wat]
            this_waterwatch_strm = waterwatch_strm[str(wat)]
            this_pso_init_strm = pso_init_strm[wat]
            this_pso_final_strm = pso_final_strm[wat]
            dates = list(default_strm.index)
            dates_dtm = [
                datetime.datetime.strptime(str(d),'%Y%m') for d in dates
            ]
            this_exp = exp_names[2]
            savename = os.path.join(
                plots_dir,'streamflow_timeseries_{}_{}.png'.format(this_exp,wat)
            )
            plt.figure()
            plt.plot(
                dates_dtm,this_default_strm,label='Default Catchment-CN',
                c='r'
            )
            plt.plot(
                dates_dtm,this_waterwatch_strm,label='WaterWatch',
                c='k'
            )
            plt.plot(
                dates_dtm,this_pso_init_strm,label='PSO Init Catchment-CN',
                c='y'
            )
            plt.plot(
                dates_dtm,this_pso_final_strm,label='PSO Final Catchment-CN',
                c='g'
            )
            plt.legend()
            plt.xlabel('Date')
            plt.ylabel('Streamflow (mm/month)')
            plt.savefig(savename,dpi=300,bbox_inches='tight')
            plt.close()
        # let's do the same thing for streamflow, runoff, baseflow just for the
        # default catchment-CN case
        default_strm = outs['default_strm']
        default_runoff = outs['default_runoff']
        default_baseflow = outs['default_baseflow']
        waterwatch_strm = outs['waterwatch_strm']
        watersheds = list(default_strm.columns)
        for w,wat in enumerate(watersheds):
            this_default_strm = default_strm[wat]
            this_default_runoff = default_runoff[wat]
            this_default_baseflow = default_baseflow[wat]
            this_waterwatch_strm = waterwatch_strm[str(wat)]
            dates = list(default_strm.index)
            dates_dtm = [
                datetime.datetime.strptime(str(d),'%Y%m') for d in dates
            ]
            this_exp = exp_names[0]
            savename = os.path.join(
                plots_dir,'streamflow_runoff_baseflow_timeseries_{}_{}.png'.format(
                    this_exp,wat
                )
            )
            plt.figure()
            plt.plot(
                dates_dtm,this_default_strm,label='Catchment-CN Streamflow',
                c='r'
            )
            plt.plot(
                dates_dtm,this_waterwatch_strm,label='WaterWatch Streamflow',
                c='k'
            )
            plt.plot(
                dates_dtm,this_default_runoff,label='Catchment-CN Runoff',
                c='y'
            )
            plt.plot(
                dates_dtm,this_default_baseflow,label='Catchment-CN Baseflow',
                c='g'
            )
            plt.legend()
            plt.xlabel('Date')
            plt.ylabel('mm/month')
            plt.savefig(savename,dpi=300,bbox_inches='tight')
            plt.close()
    def get_rmse_dict(self,outs,lai_fname,intersection_info,all_tiles):
        # let's start by analyzing stream data
        default_strm = outs['default_strm']
        waterwatch_strm = outs['waterwatch_strm']
        pso_init_strm = outs['pso_init_strm']
        pso_final_strm = outs['pso_final_strm']
        # let's start by returning avg streamflow, streamflow rmse, and lai 
        # values

        # get the lits of watersheds as strings to be used as the dict keys for
        # intersection_info
        watersheds_str = list(intersection_info.keys())
        # get the list of watersheds as integers
        watersheds = [int(k) for k in intersection_info.keys()]
        # set up the df that we are going to add everything to
        cols = copy.deepcopy(watersheds)
        cols.append('all')
        default_df = pd.DataFrame(columns=cols)
        pso_init_df = pd.DataFrame(columns=cols)
        pso_final_df = pd.DataFrame(columns=cols)
        waterwatch_df = pd.DataFrame(columns=cols)
        # lets start by getting the avg lai for each watershed
        lai_ds = nc.Dataset(lai_fname)
        lai_vals = np.array(lai_ds['lai'])
        avg_lai = np.zeros(len(cols))
        for w,wat in enumerate(watersheds):
            # get the tiles in this watershed
            this_tiles = intersection_info[watersheds_str[w]][0]
            # get the percent of each of these tiles in this watershed
            this_perc = intersection_info[watersheds_str[w]][1]
            tiles_idx = np.zeros(0,dtype=np.int64)
            for t,ti in enumerate(this_tiles):
                this_ti_idx = int(np.where(all_tiles == ti)[0][0])
                tiles_idx = np.append(tiles_idx,this_ti_idx)
            # get the average lai for this watershed
            this_lai = lai_vals[tiles_idx]
            this_avg_lai = np.average(this_lai,weights=this_perc)
            avg_lai[w] = this_avg_lai
        avg_lai_all = np.mean(avg_lai)
        avg_lai[-1] = avg_lai_all
        default_df.loc['lai'] = avg_lai
        # let's get average streamflow
        # for the default senario
        default_strm_avg = np.array(default_strm.mean())
        default_strm_avg = np.append(
            default_strm_avg,default_strm_avg.mean())
        default_df.loc['strm'] = default_strm_avg
        # for the pso_init senario
        pso_init_strm_avg = np.array(pso_init_strm.mean())
        pso_init_strm_avg = np.append(
            pso_init_strm_avg,pso_init_strm_avg.mean())
        pso_init_df.loc['strm'] = pso_init_strm_avg
        # for the pso_final senario
        pso_final_strm_avg = np.array(pso_final_strm.mean())
        pso_final_strm_avg = np.append(
            pso_final_strm_avg,pso_final_strm_avg.mean())
        pso_final_df.loc['strm'] = pso_final_strm_avg
        # finally for waterwatch
        waterwatch_strm_avg = np.array(waterwatch_strm.mean())
        waterwatch_strm_avg = np.append(
            waterwatch_strm_avg,waterwatch_strm_avg.mean())
        waterwatch_df.loc['strm'] = waterwatch_strm_avg
        # let's get the rmse for the different senarios
        catch_cols = list(default_strm.columns)
        waterwatch_strm_int = waterwatch_strm.set_axis(catch_cols,axis=1)
        # caluclate rmse for default catchment-cn
        default_diff = np.array(default_strm - waterwatch_strm_int)
        default_diff_ave = np.mean(default_diff,axis=0)
        default_diff_ave_avg = np.mean(default_diff_ave)
        default_diff_ave = np.append(default_diff_ave,default_diff_ave_avg)
        default_df.loc['strm_diff'] = default_diff_ave
        default_se = default_diff**2
        default_mse = np.mean(default_se,axis=0)
        default_rmse = np.sqrt(default_mse)
        default_rmse_all = np.mean(default_rmse)
        default_rmse = np.append(default_rmse,default_rmse_all)
        default_df.loc['strm_rmse'] = default_rmse
        # calculate r2 for default catchment-CN
        default_r2 = np.zeros(len(catch_cols))
        pso_r2 = np.zeros(len(catch_cols))
        default_corr = np.zeros(len(catch_cols))
        pso_corr = np.zeros(len(catch_cols))
        default_ubrmse = np.zeros(len(catch_cols))
        pso_ubrmse = np.zeros(len(catch_cols))
        default_nse = np.zeros(len(catch_cols))
        pso_nse = np.zeros(len(catch_cols))
        for c,col in enumerate(catch_cols):
            this_default = default_strm[col]
            this_pso = pso_final_strm[col]
            this_waterwatch = waterwatch_strm_int[col]
            # calculate the r2 for default
            this_rss = np.sum(
                np.square(
                    this_waterwatch - this_default
                )
            )
            this_avg_y = np.average(this_waterwatch)
            this_tss = np.sum(
                np.square(
                    this_waterwatch - this_avg_y
                )
            )
            this_default_r2 = 1 - this_rss/this_tss
            # calculate the r2 for pso
            this_rss = np.sum(
                np.square(
                    this_waterwatch - this_pso
                )
            )
            this_avg_y = np.average(this_waterwatch)
            this_tss = np.sum(
                np.square(
                    this_waterwatch - this_avg_y
                )
            )
            this_pso_r2 = 1 - this_rss/this_tss
            # calcaulte the default correlation coefficient
            this_avg_x = np.mean(this_default)
            numerator = np.sum(
                (this_default - this_avg_x)*
                (this_waterwatch - this_avg_y)
            )
            denominator_1 = np.sqrt(
                np.sum(
                    np.square(
                        this_default - this_avg_x
                    )
                )
            )
            denominator_2 = np.sqrt(
                np.sum(
                    np.square(
                        this_waterwatch - this_avg_y
                    )
                )
            )
            denominator = denominator_1*denominator_2
            this_default_corr = numerator/denominator
            # calcaulte the pso correlation coefficient
            this_avg_x = np.mean(this_pso)
            numerator = np.sum(
                (this_pso - this_avg_x)*
                (this_waterwatch - this_avg_y)
            )
            denominator_1 = np.sqrt(
                np.sum(
                    np.square(
                        this_pso - this_avg_x
                    )
                )
            )
            denominator_2 = np.sqrt(
                np.sum(
                    np.square(
                        this_waterwatch - this_avg_y
                    )
                )
            )
            denominator = denominator_1*denominator_2
            this_pso_corr = numerator/denominator
            # calculate default ubrmse
            this_default_ubrmse = np.sqrt(
                (
                    (
                        (
                            this_default -
                            this_avg_x
                        ) - (
                            this_waterwatch -
                            this_avg_y
                        )
                    )**2
                ).mean()
            )
            # calculate pso ubrmse
            this_pso_ubrmse = np.sqrt(
                (
                    (
                        (
                            this_pso -
                            this_avg_x
                        ) - (
                            this_waterwatch -
                            this_avg_y
                        )
                    )**2
                ).mean()
            )
            # calculate the default nse
            nse_numerator = np.sum(
                np.square(
                    this_waterwatch - this_default
                )
            )
            nse_denom = np.sum(
                np.square(
                    this_waterwatch - this_avg_y
                )
            )
            this_default_nse = 1 - nse_numerator/nse_denom
            # calculate the pso nse
            nse_numerator = np.sum(
                np.square(
                    this_waterwatch - this_pso
                )
            )
            nse_denom = np.sum(
                np.square(
                    this_waterwatch - this_avg_y
                )
            )
            this_pso_nse = 1 - nse_numerator/nse_denom
            # add to running error metrics
            default_r2[c] = this_default_r2
            pso_r2[c] = this_pso_r2
            default_corr[c] = this_default_corr
            pso_corr[c] = this_pso_corr
            default_ubrmse[c] = this_default_ubrmse
            pso_ubrmse[c] = this_pso_ubrmse
            default_nse[c] = this_default_nse
            pso_nse[c] = this_pso_nse
        # we're going to need some np arrays for this
        waterwatch_strm_int_np = np.array(waterwatch_strm_int)
        default_strm_np = np.array(default_strm)
        pso_final_strm_np = np.array(pso_final_strm)
        # calculate the r2 for default all
        all_rss = np.sum(
            np.square(
                np.array(waterwatch_strm_int_np - default_strm_np)
            )
        )
        all_avg_y = np.average(waterwatch_strm_int_np)
        all_tss = np.sum(
            np.square(
                np.array(waterwatch_strm - all_avg_y)
            )
        )
        default_all_r2 = 1 - all_rss/all_tss
        default_r2 = np.append(default_r2,default_all_r2)
        # calculate the r2 for pso all
        all_rss = np.sum(
            np.square(
                np.array(waterwatch_strm_int_np - pso_final_strm_np)
            )
        )
        all_avg_y = np.average(waterwatch_strm_int_np)
        all_tss = np.sum(
            np.square(
                np.array(waterwatch_strm - all_avg_y)
            )
        )
        pso_all_r2 = 1 - all_rss/all_tss
        pso_r2 = np.append(pso_r2,pso_all_r2)
        # calcaulte the default correlation coefficient
        all_avg_x = np.mean(default_strm_np)
        numerator = np.sum(
            (default_strm_np - all_avg_x)*
            (waterwatch_strm_int_np - all_avg_y)
        )
        denominator_1 = np.sqrt(
            np.sum(
                np.square(
                    default_strm_np - all_avg_x
                )
            )
        )
        denominator_2 = np.sqrt(
            np.sum(
                np.square(
                    waterwatch_strm_int_np - all_avg_y
                )
            )
        )
        denominator = denominator_1*denominator_2
        all_default_corr = numerator/denominator
        default_corr = np.append(default_corr,all_default_corr)
        # calcaulte the pso correlation coefficient
        all_avg_x = np.mean(pso_final_strm_np)
        numerator = np.sum(
            (pso_final_strm_np - all_avg_x)*
            (waterwatch_strm_int_np - all_avg_y)
        )
        denominator_1 = np.sqrt(
            np.sum(
                np.square(
                    pso_final_strm_np - all_avg_x
                )
            )
        )
        denominator_2 = np.sqrt(
            np.sum(
                np.square(
                    waterwatch_strm_int_np - all_avg_y
                )
            )
        )
        denominator = denominator_1*denominator_2
        all_pso_corr = numerator/denominator
        pso_corr = np.append(pso_corr,all_pso_corr)
        # calculate default ubrmse
        all_default_ubrmse = np.sqrt(
            (
                (
                    (
                        default_strm_np -
                        all_avg_x
                    ) - (
                        waterwatch_strm_int_np -
                        all_avg_y
                    )
                )**2
            ).mean()
        )
        default_ubrmse = np.append(default_ubrmse,all_default_ubrmse)
        # calculate pso ubrmse
        all_pso_ubrmse = np.sqrt(
            (
                (
                    (
                        pso_final_strm_np -
                        all_avg_x
                    ) - (
                        waterwatch_strm_int_np -
                        all_avg_y
                    )
                )**2
            ).mean()
        )
        pso_ubrmse = np.append(pso_ubrmse,all_pso_ubrmse)
        # calculate the default nse
        nse_numerator = np.sum(
            np.square(
                waterwatch_strm_int_np - default_strm_np
            )
        )
        nse_denom = np.sum(
            np.square(
                waterwatch_strm_int_np - all_avg_y
            )
        )
        all_default_nse = 1 - nse_numerator/nse_denom
        default_nse = np.append(default_nse,all_default_nse)
        # calculate the pso nse
        nse_numerator = np.sum(
            np.square(
                waterwatch_strm_int_np - pso_final_strm_np
            )
        )
        nse_denom = np.sum(
            np.square(
                waterwatch_strm_int_np - all_avg_y
            )
        )
        all_pso_nse = 1 - nse_numerator/nse_denom
        pso_nse = np.append(pso_nse,all_pso_nse)
        # lets add all these new metrics to the dataframes
        default_df.loc['strm_r2'] = default_r2
        default_df.loc['strm_corr'] = default_corr
        default_df.loc['strm_ubrmse'] = default_ubrmse
        default_df.loc['strm_nse'] = default_nse
        pso_final_df.loc['strm_r2'] = pso_r2
        pso_final_df.loc['strm_corr'] = pso_corr
        pso_final_df.loc['strm_ubrmse'] = pso_ubrmse
        pso_final_df.loc['strm_nse'] = pso_nse
        # caluclate rmse for pso_init catchment-cn
        pso_init_diff = np.array(pso_init_strm - waterwatch_strm_int)
        pso_init_se = pso_init_diff**2
        pso_init_mse = np.mean(pso_init_se,axis=0)
        pso_init_rmse = np.sqrt(pso_init_mse)
        pso_init_rmse_all = np.mean(pso_init_rmse)
        pso_init_rmse = np.append(pso_init_rmse,pso_init_rmse_all)
        pso_init_df.loc['strm_rmse'] = pso_init_rmse
        # caluclate rmse for pso_final catchment-cn
        pso_final_diff = np.array(pso_final_strm - waterwatch_strm_int)
        pso_final_se = pso_final_diff**2
        pso_final_mse = np.mean(pso_final_se,axis=0)
        pso_final_rmse = np.sqrt(pso_final_mse)
        pso_final_rmse_all = np.mean(pso_final_rmse)
        pso_final_rmse = np.append(pso_final_rmse,pso_final_rmse_all)
        pso_final_df.loc['strm_rmse'] = pso_final_rmse
        # calculate r2 for pso_final catchment-CN

        # calculate nse for pso_final Catchment-CN

        # put these in their respective dataframes
        to_return = [
            default_df,pso_init_df,pso_final_df,waterwatch_df
        ]
        return to_return
    def plot_water_metrics(self,default_df,waterwatch_df,plots_dir):
        # let's first make a plot of average lai versus streamflow
        # under-prediction to see if increasing ksat via lai might help with
        # catchment-cn underprediction problem
        # get the information we need
        lai_vals = np.array(default_df.loc['lai'])
        lai_vals_no_all = lai_vals[:-1]
        default_strm = default_df.loc['strm']
        default_strm_no_all = default_strm[:-1]
        waterwatch_strm = waterwatch_df.loc['strm']
        waterwatch_strm_no_all = waterwatch_strm[:-1]
        strm_diff_no_all = default_strm_no_all - waterwatch_strm_no_all
        huc_2s = np.arange(len(lai_vals_no_all)) + 1
        savename = os.path.join(
            plots_dir,'lai_verus_default_strm_prediction.png'
        )
        m,b,r,p,stderr = linregress(lai_vals_no_all,strm_diff_no_all)
        r_squared = r**2

        # make the plot
        plt.figure()
        plt.scatter(lai_vals_no_all,strm_diff_no_all)
        plt.plot(lai_vals_no_all,m*lai_vals_no_all+b)
        # plot the R2 value
        plt.text(
            .03, .95,
            'R2={:.2f},p={:.2f}'.format(r_squared,p),
            ha='left', va='bottom'
        )
        plt.ylabel('(Catchment-CN avg. runoff) - (Waterwatch avg. runoff)')
        plt.xlabel('LAI')
        plt.savefig(savename)
        plt.close()
    def plot_maps(self,rmse_dfs,plots_dir,geojson_fname,exp_names,states_shp,
                  make_plots,plot_trim):
        # extract the different error dfs
        default_df = rmse_dfs[0]
        pso_init_df = rmse_dfs[1]
        pso_final_df = rmse_dfs[2]
        waterwatch_df = rmse_dfs[3]
        # what do we want to plot? All for streamflow:
        # overall rmse for default
        default_strm_rmse = np.array(default_df.loc['strm_rmse'])
        default_strm_rmse_avg = default_strm_rmse[-1]
        default_strm_rmse = default_strm_rmse[:-1]
        # r2 for the default
        default_strm_r2 = np.array(default_df.loc['strm_r2'])
        default_strm_r2_avg = default_strm_r2[-1]
        default_strm_r2 = default_strm_r2[:-1]
        # corr for the default
        default_strm_corr = np.array(default_df.loc['strm_corr'])
        default_strm_corr_avg = default_strm_corr[-1]
        default_strm_corr = default_strm_corr[:-1]
        # ubrmse for the default
        default_strm_ubrmse = np.array(default_df.loc['strm_ubrmse'])
        default_strm_ubrmse_avg = default_strm_ubrmse[-1]
        default_strm_ubrmse = default_strm_ubrmse[:-1]
        # nse for the default
        default_strm_nse = np.array(default_df.loc['strm_nse'])
        default_strm_nse_avg = default_strm_nse[-1]
        default_strm_nse = default_strm_nse[:-1]
        # avg prediction difference for default
        default_strm_diff = np.array(default_df.loc['strm_diff'])
        default_strm_diff_avg = default_strm_diff[-1]
        default_strm_diff = default_strm_diff[:-1]
        # overall rmse for firstloc
        init_strm_rmse = np.array(pso_init_df.loc['strm_rmse'])
        init_strm_rmse_avg = init_strm_rmse[-1]
        init_strm_rmse = init_strm_rmse[:-1]
        # overall rmse for final it
        final_strm_rmse = np.array(pso_final_df.loc['strm_rmse'])
        final_strm_rmse_avg = final_strm_rmse[-1]
        final_strm_rmse = final_strm_rmse[:-1]
        # r2 for the final
        final_strm_r2 = np.array(pso_final_df.loc['strm_r2'])
        final_strm_r2_avg = final_strm_r2[-1]
        final_strm_r2 = final_strm_r2[:-1]
        # corr for the final
        final_strm_corr = np.array(pso_final_df.loc['strm_corr'])
        final_strm_corr_avg = final_strm_corr[-1]
        final_strm_corr = final_strm_corr[:-1]
        # ubrmse for the final
        final_strm_ubrmse = np.array(pso_final_df.loc['strm_ubrmse'])
        final_strm_ubrmse_avg = final_strm_ubrmse[-1]
        final_strm_ubrmse = final_strm_ubrmse[:-1]
        # nse for the final
        final_strm_nse = np.array(pso_final_df.loc['strm_nse'])
        final_strm_nse_avg = final_strm_nse[-1]
        final_strm_nse = final_strm_nse[:-1]
        # avg. streamflow for default
        default_strm = np.array(default_df.loc['strm'])
        default_strm_avg = default_strm[-1]
        default_strm = default_strm[:-1]
        # avg. streamflow for first
        init_strm = np.array(pso_init_df.loc['strm'])
        init_strm_avg = init_strm[-1]
        init_strm = init_strm[:-1]
        # avg. streamflow for final
        final_strm = np.array(pso_final_df.loc['strm'])
        final_strm_avg = final_strm[-1]
        final_strm = final_strm[:-1]
        # changes in rmse for default to first
        diff_rmse_strm_init = init_strm_rmse - default_strm_rmse
        diff_rmse_strm_init_avg = init_strm_rmse_avg - default_strm_rmse_avg
        # changes in rmse for default to final
        diff_rmse_strm_final = final_strm_rmse - default_strm_rmse
        diff_rmse_strm_final_avg = final_strm_rmse_avg - default_strm_rmse_avg
        all_diff_rmse_strm_final = np.append(
            diff_rmse_strm_final,diff_rmse_strm_final_avg
        )
        pso_final_df.loc['change_strm_rmse'] = all_diff_rmse_strm_final
        # perc. changes in rmse for default to first
        perc_diff_rmse_strm_init = (
            (init_strm_rmse - default_strm_rmse)/default_strm_rmse
        )
        perc_diff_rmse_strm_init_avg = (
            (init_strm_rmse_avg - default_strm_rmse_avg)/default_strm_rmse_avg
        )
        # perc. changes in rmse for default to final
        perc_diff_rmse_strm_final = (
            (final_strm_rmse - default_strm_rmse)/default_strm_rmse
        )
        perc_diff_rmse_strm_final_avg = (
            (final_strm_rmse_avg - default_strm_rmse_avg)/default_strm_rmse_avg
        )
        all_perc_diff_rmse_strm_final = np.append(
            perc_diff_rmse_strm_final,perc_diff_rmse_strm_final_avg
        )
        pso_final_df.loc['perc_change_strm_rmse'] = all_perc_diff_rmse_strm_final
        # change r2 default to final
        diff_r2_strm_final = final_strm_r2 - default_strm_r2
        diff_r2_strm_final_avg = final_strm_r2_avg - default_strm_r2_avg
        all_diff_r2_strm_final = np.append(
            diff_r2_strm_final,diff_r2_strm_final_avg
        )
        pso_final_df.loc['change_strm_r2'] = all_diff_r2_strm_final
        # change corr default to final
        diff_corr_strm_final = final_strm_corr - default_strm_corr
        diff_corr_strm_final_avg = final_strm_corr_avg - default_strm_corr_avg
        all_diff_corr_strm_final = np.append(
            diff_corr_strm_final,diff_corr_strm_final_avg
        )
        pso_final_df.loc['change_strm_corr'] = all_diff_corr_strm_final
        # perc change ubrmse default to final
        perc_diff_ubrmse_strm_final = (
            (final_strm_ubrmse - default_strm_ubrmse)/default_strm_ubrmse
        )
        perc_diff_ubrmse_strm_final_avg = (
            (final_strm_ubrmse_avg - default_strm_ubrmse_avg)/default_strm_ubrmse_avg
        )
        all_perc_diff_ubrmse_strm_final = np.append(
            perc_diff_ubrmse_strm_final,perc_diff_ubrmse_strm_final_avg
        )
        pso_final_df.loc['perc_change_strm_ubrmse'] = (
            all_perc_diff_ubrmse_strm_final
        )
        # change nse defautl to final
        diff_nse_strm_final = final_strm_nse - default_strm_nse
        diff_nse_strm_final_avg = final_strm_nse_avg - default_strm_nse_avg
        all_diff_nse_strm_final = np.append(
            diff_nse_strm_final,diff_nse_strm_final_avg
        )
        pso_final_df.loc['change_strm_nse'] = all_diff_nse_strm_final
        # changes in avg. strm default to first
        diff_strm_init = init_strm - default_strm
        diff_strm_init_avg = init_strm_avg - default_strm_avg
        # changes in avg. strm deafult to final
        diff_strm_final = final_strm - default_strm
        diff_strm_final_avg = final_strm_avg - default_strm_avg
        all_diff_strm_final = np.append(
            diff_strm_final,diff_strm_final_avg
        )
        pso_final_df.loc['diff_strm'] = all_diff_strm_final
        # perc. changes in avg. strm default to first
        perc_diff_strm_init = (
            (init_strm - default_strm)/default_strm
        )
        perc_diff_strm_init_avg = (
            (init_strm_avg - default_strm_avg)/default_strm_avg
        )
        # perc. chagnes in avg. strm default to final
        perc_diff_strm_final = (
            (final_strm - default_strm)/default_strm
        )
        perc_diff_strm_final_avg = (
            (final_strm_avg - default_strm_avg)/default_strm_avg
        )
        if make_plots:
            # now let's get the shapes that we need for plotting
            huc6s = gpd.read_file(geojson_fname)
            if not plot_trim:
                # now let's get everythin in arrays for proper plotting
                names = [
                    'default_strm_rmse','init_strm_rmse','final_strm_rmse',
                    'default_strm','init_strm','final_strm',
                    'diff_rmse_strm_init','diff_rmse_strm_final',
                    'perc_diff_rmse_strm_init','perc_diff_rmse_strm_final',
                    'diff_strm_init','diff_strm_final',
                    'perc_diff_strm_init','perc_diff_strm_final',
                    'default_strm_diff',
                    'default_strm_r2','default_strm_corr','default_strm_ubrmse',
                    'default_strm_nse','pso_strm_r2','pso_strm_corr',
                    'pso_strm_ubrmse','pso_strm_nse','diff_r2_strm_final',
                    'diff_corr_strm_final','perc_diff_ubrmse_strm_final',
                    'diff_nse_strm_final'
                ]
                vals = [
                    default_strm_rmse,init_strm_rmse,final_strm_rmse,
                    default_strm,init_strm,final_strm,
                    diff_rmse_strm_init,diff_rmse_strm_final,
                    perc_diff_rmse_strm_init,perc_diff_rmse_strm_final,
                    diff_strm_init,diff_strm_final,
                    perc_diff_strm_init,perc_diff_strm_final,
                    default_strm_diff,
                    default_strm_r2,default_strm_corr,default_strm_ubrmse,
                    default_strm_nse,final_strm_r2,final_strm_corr,
                    final_strm_ubrmse,final_strm_nse,diff_r2_strm_final,
                    diff_corr_strm_final,perc_diff_ubrmse_strm_final,
                    diff_nse_strm_final

                ]
                avgs = [
                    default_strm_rmse_avg,init_strm_rmse_avg,final_strm_rmse_avg,
                    default_strm_avg,init_strm_avg,final_strm_avg,
                    diff_rmse_strm_init_avg,diff_rmse_strm_final_avg,
                    perc_diff_rmse_strm_init_avg,perc_diff_rmse_strm_final_avg,
                    diff_strm_init_avg,diff_strm_final_avg,
                    perc_diff_strm_init_avg,perc_diff_strm_final_avg,
                    default_strm_diff_avg,
                    default_strm_r2_avg,default_strm_corr_avg,default_strm_ubrmse_avg,
                    default_strm_nse_avg,final_strm_r2_avg,final_strm_corr_avg,
                    final_strm_ubrmse_avg,final_strm_nse_avg,diff_r2_strm_final_avg,
                    diff_corr_strm_final_avg,perc_diff_ubrmse_strm_final_avg,
                    diff_nse_strm_final_avg
                ]
                types = [
                    'strm_rmse','strm_rmse','strm_rmse',
                    'strm','strm','strm',
                    'diff_strm_rmse','diff_strm_rmse',
                    'perc_diff_strm_rmse','perc_diff_strm_rmse',
                    'diff_strm','diff_strm',
                    'perc_diff_strm','perc_diff_strm',
                    'diff_strm',
                    'strm_r2','strm_r2','strm_rmse',
                    'strm_r2','strm_r2','strm_r2',
                    'strm_rmse','strm_r2','perc_diff_strm_rmse',
                    'perc_diff_strm_rmse','perc_diff_strm_rmse',
                    'perc_diff_strm_rmse'
                ]
                cmaps = {
                    'strm_rmse':'winter',
                    'strm':'winter',
                    'diff_strm_rmse':'bwr',
                    'perc_diff_strm_rmse':'bwr',
                    'diff_strm':'bwr',
                    'perc_diff_strm':'bwr',
                    'strm_r2':'winter'
                }
                vmins = {
                    'strm_rmse':0,
                    'strm':0,
                    'diff_strm_rmse':-35,
                    'perc_diff_strm_rmse':-.25,
                    'diff_strm':-50,
                    'perc_diff_strm':-.25,
                    'strm_r2':0
                }
                vmaxs = {
                    'strm_rmse':40,
                    'strm':125,
                    'diff_strm_rmse':35,
                    'perc_diff_strm_rmse':.25,
                    'diff_strm':50,
                    'perc_diff_strm':.25,
                    'strm_r2':1
                }
            if plot_trim:
                names = [
                    'perc_diff_rmse_strm_final',
                    'diff_nse_strm_final',
                    'diff_corr_strm_final',
                    'perc_diff_ubrmse_strm_final'
                ]
                vals = [
                    perc_diff_rmse_strm_final,
                    diff_nse_strm_final,
                    diff_corr_strm_final,
                    perc_diff_ubrmse_strm_final
                ]
                avgs = [
                    perc_diff_rmse_strm_final_avg,
                    diff_nse_strm_final_avg,
                    diff_corr_strm_final_avg,
                    perc_diff_ubrmse_strm_final_avg
                ]
                types = names
                cmaps = {
                    'perc_diff_rmse_strm_final':'bwr',
                    'diff_nse_strm_final':'bwr',
                    'diff_corr_strm_final':'bwr',
                    'perc_diff_ubrmse_strm_final':'bwr'
                }
                vmins = {
                    'perc_diff_rmse_strm_final':-.4,
                    'diff_nse_strm_final':-.4,
                    'diff_corr_strm_final':-.25,
                    'perc_diff_ubrmse_strm_final':-.4

                }
                vmaxs = {
                    'perc_diff_rmse_strm_final':.4,
                    'diff_nse_strm_final':.4,
                    'diff_corr_strm_final':.25,
                    'perc_diff_ubrmse_strm_final':.4
                }
            print('reading states')
            states = gpd.read_file(states_shp)
            states = states.to_crs('EPSG:4326')
            # get rid of non-conus states since not considering
            non_conus = ['HI','VI','MP','GU','AK','AS','PR']
            states_conus = states
            print('looping non conus')
            for n in non_conus:
                states_conus = states_conus[states_conus.STUSPS != n]
            for n,name in enumerate(names):
                print(name)
                fig,ax = plt.subplots()
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                state_ids = list(states_conus['GEOID'])
                for s,sid in enumerate(state_ids):
                    this_geom = states_conus['geometry'].iloc[s]
                    try:
                        xs,ys = this_geom.exterior.xy
                        ax.fill(xs,ys,fc='none',ec='k',linewidth=.2)
                    except:
                        for geom in this_geom.geoms:
                            xs,ys = geom.exterior.xy
                            ax.fill(xs,ys,fc='none',ec='k',linewidth=.2)
                # get a list of all the hucs
                all_hucs = list(huc6s['huc6'])
                # get our normalize function for getting colors
                norm = mpl.colors.Normalize(
                    vmin=vmins[types[n]],vmax=vmaxs[types[n]]
                )
                this_cmap = mpl.cm.get_cmap(cmaps[types[n]])
                for h,huc in enumerate(all_hucs):
                    this_geom = huc6s['geometry'].iloc[h]
                    xs,ys = this_geom.exterior.xy
                    this_val = vals[n][h]
                    this_val_norm = norm(this_val)
                    this_color = this_cmap(this_val_norm)
                    ax.fill(xs,ys,fc=this_color,ec='k',linewidth=0.5)
                ax.text(
                    -127+2,20+4,'Average {name}: {val:.2f}'.format(
                        name=names[n],val=avgs[n]
                    ),
                    bbox=dict(facecolor='white')
                )
                fig.colorbar(
                    mpl.cm.ScalarMappable(norm=norm, cmap=this_cmap),
                    cax=cax, orientation='vertical'
                )
                this_savename = os.path.join(
                    plots_dir,
                    '{name}_{exp}_huc6.png'.format(
                        name=names[n],exp=exp_names[2]
                    )
                )
                plt.savefig(this_savename,dpi=350,bbox_inches='tight')
                plt.close()
