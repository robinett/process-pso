import sys
sys.path.append('/shared/pso/step_5.1_analyze_outputs/funcs')
import os
import numpy as np
from general_functions import gen_funcs
import datetime
import pandas as pd
import netCDF4 as nc
import pickle as pkl
import copy

class get_timeseries:
    def get_all_timeseries_info(self,timeseries_info,start,end,
                                pixels_fname,intersection_info_fname):
        to_load = list(timeseries_info.keys())
        for l,load in enumerate(to_load):
            # get the raw timeseries at the pixel scales
            this_pixel_raw_timeseries = self.get_catchcn(
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
            this_wat_raw_timeseries = self.get_watershed(
                timeseries_info[load]['pixel_raw_timeseries'],
                timeseries_info[load]['read_met_forcing'],
                intersection_info_fname
            )
            timeseries_info[load]['wat_raw_timeseries'] = this_wat_raw_timeseries
            # get all info for this timeseries
            if timeseries_info[load]['default_type'] == 'pso_output':
                this_all_info_fname = os.path.join(
                    timeseries_info[load]['dir'],
                    '../',
                    '../',
                    'all_info.pkl'
                )
                this_opt_type = timeseries_info[load]['optimization_type']
                this_positions,this_obj = self.get_all_info(
                    this_all_info_fname,this_opt_type
                )
                timeseries_info[load]['obj_vals'] = this_obj
                timeseries_info[load]['positions'] = this_positions
        return timeseries_info
    def get_catchcn(self,fname,load_or_save,default_type,read_met_forcing,
                    timeseries_dir,start,end,
                    pixels_fname,print_vars=False):
        '''
        Function that gets Catchment-CN4.5 timeseries either by loading it from
        where it is stored or by loading a pre-saved .pkl timeseries.
        Inputs:
            fname: the location of the raw Catch-CN4.5 output. Should direct me
            to top directory for the particle number of interest.
            load_or_save: 'load' indicates that this should be loaded from a
            pre-saved directory at timeseries_dir. 'save' indicates that this should be loaded
            from the raw Catchment-CN4.5 directory and then saved to
            timesereis_dir.
            default_type: is this a default output, or an output that was
            directed to elsewhere, such as pso_outputs? options are: 'default'
            which indicates standard CatchCN ouptut, 'pso_output' which means
            copied from a mid-pso run.
            read_met_forcing: should we also read the met forcing for this
            timeseries?
            timeseries_dir: where should the .pkl timeseries be loaded/saved
            to?
            start: what is the start date for loading? (inclusive)
            end: what is the end date for loading? (inclusive)
            print_vars: print the variables availabe to be loaded in the raw
            output?
            pixels_fname: where is the list of the pixels that we want to be
            analyzed?
        Outputs:
            return a dictionary that contains a dataframe for each varaible
            that is loaded over the time period specified.
        '''
        # let's get the name of the experiment
        exp_name = os.path.basename(os.path.normpath(fname))
        # let's get the full directory name
        if default_type == 'default':
            exp_dir = os.path.join(
                fname,
                'output',
                'SMAP_EASEv2_M36',
                'cat',
                'ens0000'
            )
            ldas_domain_fname = os.path.join(
                fname,
                'output',
                'SMAP_EASEv2_M36',
                'rc_out',
                exp_name+'.ldas_domain.txt'
            )
        elif default_type == 'pso_output' or default_type == 'nan':
            exp_dir = fname
            ldas_domain_fname = (
                '/lustre/catchment/exps/' +
                'GEOSldas_CN45_pso_g1_ai_et_strm_camels_' +
                'spin19921994_test19952014_mae/0/output/' +
                'SMAP_EASEv2_M36/rc_out/0.ldas_domain.txt'
            )
        else:
            raise Exception(
                'default_type must be \'default\', \'pso_output\', ' +
                'or \'nan\''
            )
        # now that we have that, let's print the variables if decided by user
        if print_vars:
            # get the file name
            date = start.strftime("%Y%m%d")
            yr = start.strftime("%Y")
            mon = start.strftime("%m")
            ex_file = os.path.join(
                exp_dir,
                'Y'+yr,
                'M'+mn,
                exp_name+'.tavg24_1d_lnd_Nt.'+date+'_1200z.nc4'
            )
            # get the data
            ds = nc.Dataset(ex_file)
            print('variables in the dataset:')
            print(ds.variables)
            if read_met_forcing:
                ex_file_force = os.path.join(
                    exp_dir,
                    'Y'+yr,
                    'M'+mn,
                    exp_name+'.tavg24_1d_lfs_Nt.'+date+'_1200z.nc4'
                )
                ds_force = nc.Dataset(ex_file_force)
                print('met forcing variables in dataset:')
                print(ds_force.variables)
        # get the tiles that exist in this dataset
        run_pixels = np.zeros(0,dtype=int)
        with open(ldas_domain_fname) as fp:
            for line in fp:
                line_txt = line.strip()
                run_pixels = np.append(run_pixels,int(line_txt[0:8]))
        run_pixels = run_pixels[1:]
        # now let's load the dataset from the raw files
        if load_or_save == 'save':
            # set the number of pixels that we will be working with and number
            # of days over which we are running
            num_pixels = len(run_pixels)
            num_days = (end - start).days + 1
            # for trns
            all_times_trns = np.zeros((num_days,num_pixels))
            # for ave_sm
            all_times_ave_sm = np.zeros((num_days,num_pixels))
            # for root_sm
            all_times_root_sm = np.zeros((num_days,num_pixels))
            # for surf_sm
            all_times_surf_sm = np.zeros((num_days,num_pixels))
            # for beta
            all_times_beta = np.zeros((num_days,num_pixels))
            # for infil
            all_times_infil = np.zeros((num_days,num_pixels))
            # for le
            all_times_le = np.zeros((num_days,num_pixels))
            # for runoff
            all_times_runoff = np.zeros((num_days,num_pixels))
            # for baseflow
            all_times_baseflow = np.zeros((num_days,num_pixels))
            # for visible lai
            all_times_laiv = np.zeros((num_days,num_pixels))
            # for total lai
            all_times_lait = np.zeros((num_days,num_pixels))
            # for lat
            all_times_lat = np.zeros((num_days,num_pixels))
            # for lon
            all_times_lon = np.zeros((num_days,num_pixels))
            # if instructed to get met forcing information, let's do this as
            # well
            if read_met_forcing:
                # for tair
                all_times_tair = np.zeros((num_days,num_pixels))
                # for qair
                all_times_qair = np.zeros((num_days,num_pixels))
                # for lwdown
                all_times_lwdown = np.zeros((num_days,num_pixels))
                # for swdown
                all_times_swdown = np.zeros((num_days,num_pixels))
                # for psurf
                all_times_psurf = np.zeros((num_days,num_pixels))
                # for rainf
                all_times_rainf = np.zeros((num_days,num_pixels))
                # for rainfsnowf
                all_times_rainfsnowf = np.zeros((num_days,num_pixels))
            # increment by one day for each loop
            delta = datetime.timedelta(days=1)
            # set current as the start date
            curr = start
            #set the varaibles that we are going to use to keep
            #track of the different time series
            #model generate varaibles
            time = np.arange(
                start,end+datetime.timedelta(days=1),
                datetime.timedelta(days=1)
            )
            time = pd.to_datetime(time)
            d = 0
            # for each day
            while (curr <= end):
                # get this_time
                this_time = time[d]
                if curr.day == 1:
                    # let the user know where we are at
                    print('working on directory {}'.format(
                        exp_dir)
                    )
                    print('working on year {}'.format(
                        curr.year)
                    )
                    print('working on month {}'.format(
                        curr.month)
                    )
                # get the current info
                curr_str = curr.strftime("%Y%m%d")
                curr_yr = curr.strftime("%Y")
                curr_mon = curr.strftime("%m")
                # open the results and forcing files
                # again, make sure formatted correctly 
                this_file = os.path.join(
                    exp_dir,
                    'Y'+curr_yr,
                    'M'+curr_mon,
                    exp_name+'.tavg24_1d_lnd_Nt.'+curr_str+'_1200z.nc4'
                )
                # turn the files into a dataset
                ds = nc.Dataset(this_file,mode='r')
                # if we want to read met forcing, get that file
                if read_met_forcing:
                    this_force = os.path.join(
                        exp_dir,
                        'Y'+curr_yr,
                        'M'+curr_mon,
                        exp_name+'.tavg24_1d_lfs_Nt.'+curr_str+'_1200z.nc4'
                    )
                    ds_force = nc.Dataset(this_force,mode='r')
                # for each file, loop over each pixel for that day and get the
                # information
                all_trns = np.array(ds['EVPTRNS'][:])[0]
                all_times_trns[d,:] = all_trns
                all_ave_sm = np.array(ds['PRMC'][:])[0]
                all_times_ave_sm[d,:] = all_ave_sm
                all_root_sm = np.array(ds['RZMC'][:])[0]
                all_times_root_sm[d,:] = all_root_sm
                all_surf_sm = np.array(ds['SFMC'][:])[0]
                all_times_surf_sm[d,:] = all_surf_sm
                all_le = np.array(ds['LHLAND'][:])[0]
                all_times_le[d,:] = all_le
                all_beta = np.array(ds['BTRAN'][:])[0]
                all_times_beta[d,:] = all_beta
                all_infil = np.array(ds['QINFIL'][:])[0]
                all_times_infil[d,:] = all_infil
                all_runoff = np.array(ds['RUNOFF'][:])[0]
                all_times_runoff[d,:] = all_runoff
                all_baseflow = np.array(ds['BASEFLOW'][:])[0]
                all_times_baseflow[d,:] = all_baseflow
                all_laiv = np.array(ds['CNLAI'][:])[0]
                all_times_laiv[d,:] = all_laiv
                all_lait = np.array(ds['CNTLAI'][:])[0]
                all_times_lait[d,:] = all_lait
                all_lat = np.array(ds['lat'][:])
                all_times_lat[d,:] = all_lat
                all_lon = np.array(ds['lon'][:])
                all_times_lon[d,:] = all_lon
                if read_met_forcing:
                    all_tair = np.array(ds_force['Tair'][:])[0]
                    all_times_tair[d,:] = all_tair
                    all_qair = np.array(ds_force['Qair'][:])[0]
                    all_times_qair[d,:] = all_qair
                    all_lwdown = np.array(ds_force['LWdown'][:])[0]
                    all_times_lwdown[d,:] = all_lwdown
                    all_swdown = np.array(ds_force['SWdown'][:])[0]
                    all_times_swdown[d,:] = all_swdown
                    all_psurf = np.array(ds_force['Psurf'][:])[0]
                    all_times_psurf[d,:] = all_psurf
                    all_rainf = np.array(ds_force['Rainf'][:])[0]
                    all_times_rainf[d,:] = all_rainf
                    all_rainfsnowf = np.array(ds_force['RainfSnowf'][:])[0]
                    all_times_rainfsnowf[d,:] = all_rainfsnowf
                # update the day and return to the top of the loop
                curr += delta
                d += 1
            # if the model thinks there is negative le, just turn it to
            # zero
            all_times_le = np.where(all_times_le<0,0,all_times_le)
            # add each of the variables to out_dict
            # let's make these np arrays into dataframes
            trns_df = pd.DataFrame(
                data=all_times_trns,
                index=time,
                columns=list(run_pixels)
            )
            trns_df.index.name = 'time'
            ave_sm_df = pd.DataFrame(
                data=all_times_ave_sm,
                index=time,
                columns=list(run_pixels)
            )
            ave_sm_df.index.name = 'time'
            root_sm_df = pd.DataFrame(
                data=all_times_root_sm,
                index=time,
                columns=list(run_pixels)
            )
            root_sm_df.index.name = 'time'
            surf_sm_df = pd.DataFrame(
                data=all_times_surf_sm,
                index=time,
                columns=list(run_pixels)
            )
            surf_sm_df.index.name = 'time'
            le_df = pd.DataFrame(
                data=all_times_le,
                index=time,
                columns=list(run_pixels)
            )
            le_df.index.name = 'time'
            beta_df = pd.DataFrame(
                data=all_times_beta,
                index=time,
                columns=list(run_pixels)
            )
            beta_df.index.name = 'time'
            beta_df = pd.DataFrame(
                data=all_times_beta,
                index=time,
                columns=list(run_pixels)
            )
            beta_df.index.name = 'time'
            infil_df = pd.DataFrame(
                data=all_times_infil,
                index=time,
                columns=list(run_pixels)
            )
            infil_df.index.name = 'time'
            runoff_df = pd.DataFrame(
                data=all_times_runoff,
                index=time,
                columns=list(run_pixels)
            )
            runoff_df.index.name = 'time'
            baseflow_df = pd.DataFrame(
                data=all_times_baseflow,
                index=time,
                columns=list(run_pixels)
            )
            baseflow_df.index.name = 'time'
            laiv_df = pd.DataFrame(
                data=all_times_laiv,
                index=time,
                columns=list(run_pixels)
            )
            laiv_df.index.name = 'time'
            lait_df = pd.DataFrame(
                data=all_times_lait,
                index=time,
                columns=list(run_pixels)
            )
            lait_df.index.name = 'time'
            lat_df = pd.DataFrame(
                data=all_times_lat,
                index=time,
                columns=list(run_pixels)
            )
            lat_df.index.name = 'time'
            lon_df = pd.DataFrame(
                data=all_times_lon,
                index=time,
                columns=list(run_pixels)
            )
            lon_df.index.name = 'time'
            if read_met_forcing:
                tair_df = pd.DataFrame(
                    data=all_times_tair,
                    index=time,
                    columns=list(run_pixels)
                )
                tair_df.index.name = 'time'
                qair_df = pd.DataFrame(
                    data=all_times_qair,
                    index=time,
                    columns=list(run_pixels)
                )
                qair_df.index.name = 'time'
                lwdown_df = pd.DataFrame(
                    data=all_times_lwdown,
                    index=time,
                    columns=list(run_pixels)
                )
                lwdown_df.index.name = 'time'
                swdown_df = pd.DataFrame(
                    data=all_times_swdown,
                    index=time,
                    columns=list(run_pixels)
                )
                swdown_df.index.name = 'time'
                psurf_df = pd.DataFrame(
                    data=all_times_psurf,
                    index=time,
                    columns=list(run_pixels)
                )
                psurf_df.index.name = 'time'
                rainf_df = pd.DataFrame(
                    data=all_times_rainf,
                    index=time,
                    columns=list(run_pixels)
                )
                rainf_df.index.name = 'time'
                rainfsnowf_df = pd.DataFrame(
                    data=all_times_rainfsnowf,
                    index=time,
                    columns=list(run_pixels)
                )
                rainfsnowf_df.index.name = 'time'


            out_dict = {
                'trns':trns_df,
                'ave_sm':ave_sm_df,
                'root_sm':root_sm_df,
                'surf_sm':surf_sm_df,
                'le':le_df,
                'beta':beta_df,
                'infil':infil_df,
                'runoff':runoff_df,
                'baseflow':baseflow_df,
                'laiv':laiv_df,
                'lait':lait_df,
                'lat':lat_df,
                'lon':lon_df
            }
            if read_met_forcing:
                out_dict['tair'] = tair_df
                out_dict['qair'] = qair_df
                out_dict['lwdown'] = lwdown_df
                out_dict['swdown'] = swdown_df
                out_dict['psurf'] = psurf_df
                out_dict['rainf'] = rainf_df
                out_dict['rainfsnowf'] = rainfsnowf_df
            # let's create 
            save_dir = timeseries_dir+'.pkl'
            with open(save_dir,'wb') as f:
                pkl.dump(out_dict,f)
            print('saved experiment to {}.'.format(save_dir))
        #otherwise load it
        elif load_or_save == 'load':
            # get where it was saved
            load_dir = timeseries_dir + '.pkl'
            print('Loading experiment from {}'.format(load_dir))
            # open the file
            with open(load_dir,'rb') as f:
                out_dict = pkl.load(f)
            # let's only take the times that we need
            times = list(out_dict['le'].index)
            times = pd.to_datetime(times)
            # let's see where is our start and end idx
            for t,ti in enumerate(times):
                if ti.date() == start:
                    start_idx = t
                if ti.date() == end:
                    end_idx = t
            for ke in out_dict.keys():
                out_dict[ke] = out_dict[ke].iloc[start_idx:end_idx+1]
        # regardless of saving or loading, lets only return the requested
        # pixels
        gen = gen_funcs()
        requested_pixels = gen.get_pixels(pixels_fname)
        for ke in out_dict.keys():
            out_dict[ke] = out_dict[ke][requested_pixels]
        return out_dict
    def get_watershed(self,catchcn_timeseries,read_met_forcing,
                      intersection_info_fname):
        '''
        Function that takes Catchment-CN4.5 outputs and maps them to the
        watershed scale.
        Inputs:
            catchcn_timeseries: a catchment-CN4.5 timeseries, in the format
            delivered from the function above.
            read_met_forcing: Bool. Do we have met forcing data?
            intersection_info: file that contains the intersection info between
            the desired watersheds and the Ctachmentcn tiles. from step 1 of
            the pso.
        Outputs:
            Dictionary of variables at the watershed scale.
        '''
        # get intersection info
        gen = gen_funcs()
        intersection_info = gen.get_intersection_info(
            intersection_info_fname
        )
        # get the variables of interest
        runoff_pixel = catchcn_timeseries['runoff']
        baseflow_pixel = catchcn_timeseries['baseflow']
        le_pixel = catchcn_timeseries['le']
        if read_met_forcing:
            rainfsnowf_pixel = catchcn_timeseries['rainfsnowf']
        # let's see how many days we have
        start = runoff_pixel.index[0]
        end = runoff_pixel.index[-1]
        delta = end - start
        num_days = delta.days + 1
        # get the watersheds that we want
        watersheds = list(intersection_info.keys())
        num_watersheds = len(watersheds)
        # initialize the variables that we want to end up with
        runoff = np.zeros((num_days,num_watersheds))
        baseflow = np.zeros((num_days,num_watersheds))
        le = np.zeros((num_days,num_watersheds))
        if read_met_forcing:
            rainfsnowf = np.zeros((num_days,num_watersheds))
        # loop over all watersheds and convert
        for w,wat in enumerate(watersheds):
            # get the tiles in this watershed
            this_tiles = intersection_info[wat][0]
            # get the percent of each of these tiles in this watershed
            this_perc = intersection_info[wat][1]
            # get the weighted average over the whole watershed and add to
            # array
            # for runoff
            this_runoff_pixel_np = np.array(runoff_pixel[this_tiles])
            this_runoff_pixel_np_avg = np.average(
                this_runoff_pixel_np,axis=1,weights=this_perc
            )
            runoff[:,w] = this_runoff_pixel_np_avg
            # for baseflow
            this_baseflow_pixel_np = np.array(baseflow_pixel[this_tiles])
            this_baseflow_pixel_np_avg = np.average(
                this_baseflow_pixel_np,axis=1,weights=this_perc
            )
            baseflow[:,w] = this_baseflow_pixel_np_avg
            # for le
            this_le_pixel_np = np.array(le_pixel[this_tiles])
            this_le_pixel_np_avg = np.average(
                this_le_pixel_np,axis=1,weights=this_perc
            )
            le[:,w] = this_le_pixel_np_avg
            if read_met_forcing:
                # for rainfsnowf
                this_rainfsnowf_pixel_np = np.array(rainfsnowf_pixel[this_tiles])
                this_rainfsnowf_pixel_np_avg = np.average(
                    this_rainfsnowf_pixel_np,axis=1,weights=this_perc
                )
                rainfsnowf[:,w] = this_rainfsnowf_pixel_np_avg
        # we are going to convert everything here from mm/s (runoff,
        # baseflow) or W/m2 (le) to mm/day
        # because this is going to be a much more relevant metric
        runoff = runoff*86400
        baseflow = baseflow*86400
        le = le/28.94
        if read_met_forcing:
            rainfsnowf = rainfsnowf*86400
        # streamflow will just be the sum of runoff and baseflow
        strm = runoff + baseflow
        # add these all to a dataframe. package in a dictionary and ship out
        times = runoff_pixel.index
        # for runoff
        runoff_df = pd.DataFrame(
            runoff,index=times,columns=watersheds
        )
        runoff_df.index.name = 'time'
        # for baseflow
        baseflow_df = pd.DataFrame(
            baseflow,index=times,columns=watersheds
        )
        baseflow_df.index.name = 'time'
        # for le
        le_df = pd.DataFrame(
            le,index=times,columns=watersheds
        )
        le_df.index.name = 'time'
        # for strm
        strm_df = pd.DataFrame(
            strm,index=times,columns=watersheds
        )
        strm_df.index.name = 'time'
        # unfortunately, we only trust model streamflow outputs at the yearly
        # scale. so let's convert everything here to yearly.
        runoff_yr_df = runoff_df.groupby(runoff_df.index.year).mean()
        runoff_yr_df.index = pd.to_datetime(runoff_yr_df.index,format='%Y')
        baseflow_yr_df = baseflow_df.groupby(baseflow_df.index.year).mean()
        baseflow_yr_df.index = pd.to_datetime(baseflow_yr_df.index,format='%Y')
        strm_yr_df = strm_df.groupby(strm_df.index.year).mean()
        strm_yr_df.index = pd.to_datetime(strm_yr_df.index,format='%Y')
        le_yr_df = le_df.groupby(le_df.index.year).mean()
        le_yr_df.index = pd.to_datetime(le_yr_df.index,format='%Y')
        if read_met_forcing:
            # for rainfsnowf
            rainfsnowf_df = pd.DataFrame(
                rainfsnowf,index=times,columns=watersheds
            )
            rainfsnowf_df.index.name = 'time'
            rainfsnowf_yr_df = rainfsnowf_df.groupby(
                rainfsnowf_df.index.year
            ).mean()
        outs = {
            'runoff':runoff_df,
            'runoff_yr':runoff_yr_df,
            'baseflow':baseflow_df,
            'baseflow_yr':baseflow_yr_df,
            'strm':strm_df,
            'strm_yr':strm_yr_df,
            'le':le_df,
            'le_yr':le_yr_df
        }
        if read_met_forcing:
            outs['rainfsnowf'] = rainfsnowf_df
            outs['rainfsnowf_yr'] = rainfsnowf_yr_df
        return outs
    def get_streamflow_obs(self,fname):
        obs = pd.read_csv(fname)
        obs['time'] = pd.to_datetime(obs['time'],format='%Y%m%d')
        obs = obs.set_index('time')
        obs.columns = obs.columns.map(int)
        return obs
    def get_le_obs(self,fname):
        obs = pd.read_csv(fname)
        obs['time'] = pd.to_datetime(obs['time'],format='%Y-%m-%d')
        obs = obs.set_index('time')
        obs.columns = obs.columns.map(int)
        return obs
    def get_all_info(self,all_info_fname,optimization_type):
        with open(all_info_fname,'rb') as f:
            all_info = pkl.load(f)
        keys = list(all_info.keys())
        num_iterations = len(keys)
        if optimization_type == 'pft':
            param_names = [
                'a0_needleaf_trees',
                'a0_broadleaf_trees',
                'a0_shrub',
                'a0_c3_grass',
                'a0_c4_grass',
                'a0_crop'
            ]
        elif optimization_type == 'ef':
            param_names = [
                'b_needleleaf_trees',
                'b_broadleaf_tress',
                'b_shrub',
                'b_c3_grass',
                'b_c4_grass',
                'b_crop',
                'a0_intercept',
                'a1_precip_coef',
                'a2_canopy_coef'
            ]
        else:
            raise Exception(
                'optimization type must be \'ef\' or \'pft\''
            )
        example_pos = all_info[keys[0]]['positions']
        num_particles,nan = np.shape(example_pos)
        obj_vals = np.zeros((num_iterations,num_particles))
        obj_strm_vals = np.zeros((num_iterations,num_particles))
        obj_et_vals = np.zeros((num_iterations,num_particles))
        positions = {}
        objective = {}
        for p,param in enumerate(param_names):
            this_param = np.zeros((num_iterations,num_particles))
            for i,it in enumerate(keys):
                this_pos = all_info[it]['positions']
                this_param[i,:] = this_pos[:,p]
                if p == 0:
                    this_obj = all_info[it]['obj_out_norm']
                    obj_vals[i,:] = this_obj
                    this_obj_strm = all_info[it]['strm_obj_out_norm']
                    obj_strm_vals[i,:] = this_obj_strm
                    this_obj_et = all_info[it]['et_obj_out_norm']
                    obj_et_vals[i,:] = this_obj_et
            positions[param] = this_param
        objective['all'] = obj_vals
        objective['strm'] = obj_strm_vals
        objective['et'] = obj_et_vals
        return [positions,objective]







