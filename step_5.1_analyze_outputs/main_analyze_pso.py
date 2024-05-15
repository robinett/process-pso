import sys
sys.path.append('/shared/pso/step_5.1_analyze_outputs/funcs')
sys.path.append('/shared/pso/step_5.1_analyze_outputs/tasks')
import os
import datetime
import numpy as np
from get_timeseries import get_timeseries
from rainfall_runoff_ratio import rainfall_runoff
from plot_timeseries import timeseries
from get_averages_and_error import averages_and_error
from plot_watersheds import plot_wat

def main():
    # what is the base dir?
    base_dir = '/shared/pso/step_5.1_analyze_outputs'
    # where should we save plots?
    plots_dir = os.path.join(
        base_dir,'plots'
    )
    # where should we save computed outputs?
    out_dir = os.path.join(
        base_dir,'outputs'
    )
    save_dir = (
        '/shared/pso/step_5_analyze_outputs/saved_timeseries'
    )
    # what is the start date?
    start = datetime.date(1992,1,1)
    # what is the end date?
    end = datetime.date(2014,12,31)
    # what are the start and end dates for when we should compute error?
    start_err = datetime.date(1995,1,1)
    end_err = datetime.date(2014,12,31)
    # where are the analysis pixels located?
    pixels_fname = (
        '/shared/pso/step_1x_choose_tiles_large/outputs/intersecting_catch_tiles.csv'
    )
    # where is the intersection info located?
    intersection_info_fname = (
        '/shared/pso/step_1x_choose_tiles_large/outputs/intersection_info.pkl'
    )
    # where is the le truth located?
    le_truth_fname = (
        '/shared/pso/step_3x_process_gleam/outputs/' +
        'le_truth_gleam_38a_watts_per_m2_1995-01-01_2014-12-31_' +
        'camels_tiles.csv'
    )
    # where is the streamflow truth located?
    streamflow_truth_fname = (
        '/shared/pso/step_3.1.1x_process_camels/outputs/' +
        'camels_truth_yearly_1995-01-01_2014-12-31_mm_day.csv'
    )
    # what are the names and info of the timeseries that we are going to load?
    timeseries_info = {
        'med-default-pft-g1-1992-2014':{
            'dir':(
                'nan'
            ),
            'load_or_save':'load',
            'default_type':'nan',
            'read_met_forcing':True,
            'timeseries_dir':os.path.join(
                save_dir,
                'med-default-pft-g1-1992-2014'
            )
        },
        'g1-ai-et-strm-spin19921994-test19952014-mae-num1':{
            'dir':(
                '/shared/pso_outputs/g1_ai_et_strm_camels_spin19921994_' +
                'test19952914_mae/num_1/7'
            ),
            'load_or_save':'load',
            'default_type':'pso_output',
            'read_met_forcing':False,
            'timeseries_dir':os.path.join(
                save_dir,
                'g1-ai-et-strm-spin19921994-test19952014-mae-num1'
            )
        },
        'g1-ai-et-strm-spin19921994-test19952014-mae-num4':{
            'dir':(
                '/shared/pso_outputs/g1_ai_et_strm_camels_spin19921994_' +
                'test19952914_mae/num_4/25'
            ),
            'load_or_save':'load',
            'default_type':'pso_output',
            'read_met_forcing':False,
            'timeseries_dir':os.path.join(
                save_dir,
                'g1-ai-et-strm-spin19921994-test19952014-mae-num4'
            )
        }
    }
    # let's get the le truth timeseries
    g = get_timeseries()
    le_obs = g.get_le_obs(le_truth_fname)
    # let's get the streamflow truth timeseries
    strm_obs = g.get_streamflow_obs(streamflow_truth_fname)
    # load the timeseries
    to_load = list(timeseries_info.keys())
    for l,load in enumerate(to_load):
        # get the raw timeseries at the pixel scales
        this_pixel_raw_timeseries = g.get_catchcn(
            timeseries_info[load]['dir'],
            timeseries_info[load]['load_or_save'],
            timeseries_info[load]['default_type'],
            timeseries_info[load]['read_met_forcing'],
            timeseries_info[load]['timeseries_dir'],
            start,
            end,
            pixels_fname
        )
        timeseries_info[load]['pixel_raw_timeseries'] = this_pixel_raw_timeseries
        # get the raw timeseries at the watershed scale
        this_wat_raw_timeseries = g.get_watershed(
            timeseries_info[load]['pixel_raw_timeseries'],
            timeseries_info[load]['read_met_forcing'],
            intersection_info_fname
        )
        timeseries_info[load]['wat_raw_timeseries'] = this_wat_raw_timeseries
        # get the pixel-averaged and error information for pixel-scale outputs
        get_metrics = averages_and_error()
        pixel_avgs = get_metrics.var_avgs(this_pixel_raw_timeseries)
        timeseries_info[load]['pixel_avgs'] = pixel_avgs
        # get the watershed-averaged and error information for watershed-scale
        # ouptts
        wat_avgs = get_metrics.var_avgs(this_wat_raw_timeseries)
        timeseries_info[load]['wat_avgs'] = wat_avgs
        this_le_err_df = get_metrics.var_error(
            this_pixel_raw_timeseries['le'],le_obs,start_err,end_err
        )
        timeseries_info[load]['pixel_le_errors'] = this_le_err_df
        this_wat_err_df = get_metrics.var_error(
            this_wat_raw_timeseries['strm_yr'],strm_obs,start_err,end_err
        )
        timeseries_info[load]['wat_strm_errors'] = this_wat_err_df
    # cool we have the information we need
    # let's start with some timeseries
    p_t = timeseries()
    # le timeseries
    #to_plot = [
    #    timeseries_info[to_load[0]]['pixel_raw_timeseries'],
    #    timeseries_info[to_load[1]]['pixel_raw_timeseries'],
    #    timeseries_info[to_load[2]]['pixel_raw_timeseries'],
    #    {'le':le_obs}
    #]
    #names = [
    #    'default',
    #    'iteration 1',
    #    'iteration 4',
    #    'GLEAM'
    #]
    #sizes = np.repeat(0.5,len(names))
    #obs_idx = 3
    #start_plot = datetime.date(2010,1,1)
    #end_plot = datetime.date(2011,12,31)
    #p_t.plot_one_var(
    #    to_plot,names,'le','W/m2','mae_pso_progression_1_4',
    #    plots_dir,start=start_plot,end=end_plot,
    #    small_preds=sizes,obs_idx=obs_idx
    #)
    # streamflow timeseries
    #to_plot = [
    #    timeseries_info[to_load[0]]['wat_raw_timeseries'],
    #    timeseries_info[to_load[1]]['wat_raw_timeseries'],
    #    timeseries_info[to_load[2]]['wat_raw_timeseries'],
    #    {'strm_yr':strm_obs}
    #]
    #names = [
    #    'default',
    #    'iteration 1',
    #    'iteration 4',
    #    'CAMELS'
    #]
    #sizes = np.repeat(0.5,len(names))
    #obs_idx = 3
    #start_plot = datetime.date(2010,1,1)
    #end_plot = datetime.date(2011,12,31)
    #p_t.plot_one_var(
    #    to_plot,names,'strm_yr','mm/day','mae_pso_progression_1_4',
    #    plots_dir,start=start,end=end,
    #    small_preds=sizes,obs_idx=obs_idx
    #)
    # let's plot the change in J value from iteration 1 to 4 for streamflow
    avg_strm = strm_obs.mean()
    print(avg_strm)
    j_strm_1 = (
        timeseries_info[to_load[1]]['wat_strm_errors'].loc['mae']/
        avg_strm
    )
    j_strm_4 = (
        timeseries_info[to_load[2]]['wat_strm_errors'].loc['mae']/
        avg_strm
    )
    change_j_strm = j_strm_4 - j_strm_1
    change_j_strm = np.array(change_j_strm)
    avg_change_j_strm = np.mean(change_j_strm)
    hucs = list(j_strm_1.index)
    hucs = [
        int(h) for h in hucs
    ]
    p_w = plot_wat()
    p_w.plot_map(
        'change_j_strm_shortenedcolor',hucs,change_j_strm,
        avg_change_j_strm,plots_dir,vmin=-0.2,vmax=0.2,
        cmap='bwr'
    )


if __name__ == '__main__':
    main()
