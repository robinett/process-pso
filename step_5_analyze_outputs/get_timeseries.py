import numpy as np
import os
import netCDF4 as nc
import sys
import numpy as np
import copy
import datetime
import pickle as pkl
import pandas as pd

class get_timeseries:
    def __init__(self,start,end):
        self.start = start
        self.end = end
        self.output_middle_dir = (
            'output/SMAP_EASEv2_M36/cat/ens0000'
        )
    def get_pixels(self,step_1_pixels_fname):
        # get the tiles that we are going to run as defined in step 1
        tiles = pd.read_csv(step_1_pixels_fname,header=None)
        # turn this into an np array
        tiles = np.array(tiles).astype(int)
        # make a nice np array
        tiles = tiles.T
        tiles = tiles[0]
        # save to self as pixels for later use
        self.pixels = tiles
        return tiles
    def print_catch_variables(self,exp_dir):
        '''
        Function that will print variables in the Catchment dataset an then
        exit. Prints all varaibles with full description, then just a list of
        all the variables names.
        '''
        exp_results_dir = os.path.join(
            exp_dir,self.output_middle_dir
        )
        exp_tech_name = os.path.basename(os.path.normpath(exp_dir))
        self.exp_tech_name = exp_tech_name
        curr_str = self.start.strftime("%Y%m%d")
        curr_yr = self.start.strftime("%Y")
        curr_mon = self.start.strftime("%m")
        this_file = os.path.join(
            exp_results_dir,
            'Y'+curr_yr,
            'M'+curr_mon,
            exp_tech_name+'.tavg24_1d_lnd_Nt.'+curr_str+'_1200z.nc4'
        )
        ds_pix = nc.Dataset(this_file)
        print('forcing variables in this catchment dataset:')
        print(ds_pix.variables)
        for var in ds_pix.variables:
            print(var)
        this_file = os.path.join(
            exp_results_dir,
            'Y'+curr_yr,
            'M'+curr_mon,
            exp_tech_name+'.tavg24_1d_lfs_Nt.'+curr_str+'_1200z.nc4'
        )
        ds_pix = nc.Dataset(this_file)
        print('variables in this catchment dataset:')
        print(ds_pix.variables)
        for var in ds_pix.variables:
            print(var)
        sys.exit()
    def get_catch_timeseries(self,exp_dir,exp_name,save_bool,
                             load_bool,is_default_experiment,read_met_forcing,
                             timeseries_dir):
        '''
        Function that creates a dictionary of dictionries of compiled model output. The keys to the
        first layer of the dictionary are the model pixel numbers, and the keys to the second
        layer of the dictionaries are the variables of interest.
        Inputs:
            All described in "main.py"
        Outputs:
            out_dir: as described in the function description above.
        '''
        # get the true pixel numbers
        # needed no matter if we load or not
        true_tiles = np.zeros(0)
        # load the catchment tile numbers run in the experiment from the
        # ldas_domain.txt file
        exp_tech_name = os.path.basename(os.path.normpath(exp_dir))
        if is_default_experiment:
            ldas_domain_fname = os.path.join(
                exp_dir,'output/SMAP_EASEv2_M36/rc_out',
                exp_tech_name+'.ldas_domain.txt'
            )
        else:
            ldas_domain_fname = os.path.join(
                (
                    '/shared/exps/GEOSldas_CN45_med_default/0/output/'+
                    'SMAP_EASEv2_M36/rc_out'
                ),'0.ldas_domain.txt'
            )
        with open(ldas_domain_fname) as fp:
            for line in fp:
                line_txt = line.strip()
                true_tiles = np.append(true_tiles,int(line_txt[0:8]))
        true_tiles = true_tiles[1:]
        # dummy holder that will become fake tiles
        catch_pix_idx = np.zeros(len(true_tiles))
        # for all step 1 pixels, get idx of those in context of all 
        # tiles from model
        for p,pi in enumerate(self.pixels):
            try:
                idx = np.where(true_tiles == pi)[0][0]
                catch_pix_idx[p] = idx
            except:
                print('ERROR! User-specified pixel {} does not exist in model output.'.format(pi))
                sys.exit()
        #self.pixels = pixels
        self.catch_pix_idx = catch_pix_idx
        # if datset hasn't been saved then generate it
        if load_bool == False:
            #set the full directory to the results
            exp_results_dir = os.path.join(
                exp_dir,self.output_middle_dir
            )
            # get the starting times
            curr_str = self.start.strftime("%Y%m%d")
            curr_yr = self.start.strftime("%Y")
            curr_mon = self.start.strftime("%m")
            # get the file for the first date
            # files that are saved from during the pso have a different naming
            # structure, so get the appropriate name if so
            if is_default_experiment:
                this_file = os.path.join(
                    exp_results_dir,
                    'Y'+curr_yr,
                    'M'+curr_mon,
                    exp_tech_name+'.tavg24_1d_lnd_Nt.'+curr_str+'_1200z.nc4'
                )
            else:
                this_file = os.path.join(
                    exp_dir,
                    'Y'+curr_yr,
                    'M'+curr_mon,
                    exp_tech_name+'.tavg24_1d_lnd_Nt.'+curr_str+'_1200z.nc4'
                )
            ds_pix = nc.Dataset(this_file)
            # set the number of pixels that we will be working with and number
            # of days over which we are running
            num_pixels = len(self.pixels)
            num_days = (self.end - self.start).days + 1
            # DataFrame for each variables that we will save all output to
            # first we need to get the columns, which are time in first column
            # plus pixel numbers
            pixels_plus = list(self.pixels)
            pixels_plus.insert(0,'time')
            # now create a dataframe for each variables that we are looking to
            # save
            # also create a this_var array to store the daily values at each
            # timestep to then be placed in the df
            # for trns
            trns_df = pd.DataFrame(columns=pixels_plus)
            trns_df = trns_df.set_index('time')
            this_trns = np.zeros(num_pixels)
            # for ave_sm
            ave_sm_df = pd.DataFrame(columns=pixels_plus)
            ave_sm_df = ave_sm_df.set_index('time')
            this_ave_sm = np.zeros(num_pixels)
            # for root_sm
            root_sm_df = pd.DataFrame(columns=pixels_plus)
            root_sm_df = root_sm_df.set_index('time')
            this_root_sm = np.zeros(num_pixels)
            # for surf_sm
            surf_sm_df = pd.DataFrame(columns=pixels_plus)
            surf_sm_df = surf_sm_df.set_index('time')
            this_surf_sm = np.zeros(num_pixels)
            # for beta
            beta_df = pd.DataFrame(columns=pixels_plus)
            beta_df = beta_df.set_index('time')
            this_beta = np.zeros(num_pixels)
            # for infil
            infil_df = pd.DataFrame(columns=pixels_plus)
            infil_df = infil_df.set_index('time')
            this_infil = np.zeros(num_pixels)
            # for le
            le_df = pd.DataFrame(columns=pixels_plus)
            le_df = le_df.set_index('time')
            this_le = np.zeros(num_pixels)
            # for runoff
            runoff_df = pd.DataFrame(columns=pixels_plus)
            runoff_df = runoff_df.set_index('time')
            this_runoff = np.zeros(num_pixels)
            # for baseflow
            baseflow_df = pd.DataFrame(columns=pixels_plus)
            baseflow_df = baseflow_df.set_index('time')
            this_baseflow = np.zeros(num_pixels)
            # for visible lai
            laiv_df = pd.DataFrame(columns=pixels_plus)
            laiv_df = laiv_df.set_index('time')
            this_laiv = np.zeros(num_pixels)
            # for total lai
            lait_df = pd.DataFrame(columns=pixels_plus)
            lait_df = lait_df.set_index('time')
            this_lait = np.zeros(num_pixels)
            # for lat
            lat_df = pd.DataFrame(columns=pixels_plus)
            lat_df = lat_df.set_index('time')
            this_lat = np.zeros(num_pixels)
            # for lon
            lon_df = pd.DataFrame(columns=pixels_plus)
            lon_df = lon_df.set_index('time')
            this_lon = np.zeros(num_pixels)
            # if instructed to get met forcing information, let's do this as
            # well
            if read_met_forcing:
                # for tair
                tair_df = pd.DataFrame(columns=pixels_plus)
                tair_df = tair_df.set_index('time')
                this_tair = np.zeros(num_pixels)
                # for qair
                qair_df = pd.DataFrame(columns=pixels_plus)
                qair_df = qair_df.set_index('time')
                this_qair = np.zeros(num_pixels)
                # for lwdown
                lwdown_df = pd.DataFrame(columns=pixels_plus)
                lwdown_df = lwdown_df.set_index('time')
                this_lwdown = np.zeros(num_pixels)
                # for swdown
                swdown_df = pd.DataFrame(columns=pixels_plus)
                swdown_df = swdown_df.set_index('time')
                this_swdown = np.zeros(num_pixels)
                # for psurf
                psurf_df = pd.DataFrame(columns=pixels_plus)
                psurf_df = psurf_df.set_index('time')
                this_psurf = np.zeros(num_pixels)
                # for rainf
                rainf_df = pd.DataFrame(columns=pixels_plus)
                rainf_df = rainf_df.set_index('time')
                this_rainf = np.zeros(num_pixels)
                # for rainfsnowf
                rainfsnowf_df = pd.DataFrame(columns=pixels_plus)
                rainfsnowf_df = rainfsnowf_df.set_index('time')
                this_rainfsnowf = np.zeros(num_pixels)
            # increment by one day for each loop
            delta = datetime.timedelta(days=1)
            # set current as the start date
            curr = self.start
            #set the varaibles that we are going to use to keep
            #track of the different time series
            #model generate varaibles
            time = np.arange(
                self.start,self.end+datetime.timedelta(days=1),
                datetime.timedelta(days=1)
            )
            time = pd.to_datetime(time)
            d = 0
            # for each day
            while (curr <= self.end):
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
                if is_default_experiment:
                    this_file = os.path.join(
                        exp_results_dir,
                        'Y'+curr_yr,
                        'M'+curr_mon,
                        exp_tech_name+'.tavg24_1d_lnd_Nt.'+curr_str+'_1200z.nc4'
                    )
                    # turn the files into a dataset
                    ds = nc.Dataset(this_file)
                else:
                    this_file = os.path.join(
                        exp_dir,
                        'Y'+curr_yr,
                        'M'+curr_mon,
                        exp_tech_name+'.tavg24_1d_lnd_Nt.'+curr_str+'_1200z.nc4'
                    )
                    # turn the files into a dataset
                    ds = nc.Dataset(this_file)
                # if we want to read met forcing, get that file
                if read_met_forcing and is_default_experiment:
                    this_force = os.path.join(
                        exp_results_dir,
                        'Y'+curr_yr,
                        'M'+curr_mon,
                        exp_tech_name+'.tavg24_1d_lfs_Nt.'+curr_str+'_1200z.nc4'
                    )
                    ds_force = nc.Dataset(this_force)
                elif read_met_forcing and not is_default_experiment:
                    this_force = os.path.join(
                        exp_dir,
                        'Y'+curr_yr,
                        'M'+curr_mon,
                        exp_tech_name+'.tavg24_1d_lfs_Nt.'+curr_str+'_1200z.nc4'
                    )
                    ds_force = nc.Dataset(this_force)
                # for each file, loop over each pixel for that day and get the
                # information
                for p,pix in enumerate(self.pixels):
                    this_trns[p] = ds.variables['EVPTRNS'][0,catch_pix_idx[p]]
                    this_ave_sm[p] = ds.variables['PRMC'][0,catch_pix_idx[p]]
                    this_root_sm[p] = ds.variables['RZMC'][0,catch_pix_idx[p]]
                    this_surf_sm[p] = ds.variables['SFMC'][0,catch_pix_idx[p]]
                    this_le[p] = ds.variables['LHLAND'][0,catch_pix_idx[p]]
                    this_beta[p] = ds.variables['BTRAN'][0,catch_pix_idx[p]]
                    this_infil[p] = ds.variables['QINFIL'][0,catch_pix_idx[p]]
                    this_runoff[p] = ds.variables['RUNOFF'][0,catch_pix_idx[p]]
                    this_baseflow[p] = ds.variables['BASEFLOW'][0,catch_pix_idx[p]]
                    this_laiv[p] = ds.variables['CNLAI'][0,catch_pix_idx[p]]
                    this_lait[p] = ds.variables['CNTLAI'][0,catch_pix_idx[p]]
                    this_lat[p] = np.array(ds.variables['lat'][catch_pix_idx[p]])
                    this_lon[p] = np.array(ds.variables['lon'][catch_pix_idx[p]])
                    if read_met_forcing:
                        this_tair[p] = ds_force.variables['Tair'][0,catch_pix_idx[p]]
                        this_qair[p] = ds_force.variables['Qair'][0,catch_pix_idx[p]]
                        this_lwdown[p] = ds_force.variables['LWdown'][0,catch_pix_idx[p]]
                        this_swdown[p] = ds_force.variables['SWdown'][0,catch_pix_idx[p]]
                        this_psurf[p] = ds_force.variables['Psurf'][0,catch_pix_idx[p]]
                        this_rainf[p] = ds_force.variables['Rainf'][0,catch_pix_idx[p]]
                        this_rainfsnowf[p] = ds_force.variables['RainfSnowf'][0,catch_pix_idx[p]]
                # if the model thinks there is negative le, just turn it to
                # zero
                this_le = np.where(this_le<0,0,this_le)
                # add these arrays to their respective dfs
                trns_df.loc[this_time] = copy.deepcopy(this_trns)
                ave_sm_df.loc[this_time] = copy.deepcopy(this_ave_sm)
                root_sm_df.loc[this_time] = copy.deepcopy(this_root_sm)
                surf_sm_df.loc[this_time] = copy.deepcopy(this_surf_sm)
                le_df.loc[this_time] = copy.deepcopy(this_le)
                beta_df.loc[this_time] = copy.deepcopy(this_beta)
                infil_df.loc[this_time] = copy.deepcopy(this_infil)
                runoff_df.loc[this_time] = copy.deepcopy(this_runoff)
                baseflow_df.loc[this_time] = copy.deepcopy(this_baseflow)
                laiv_df.loc[this_time] = copy.deepcopy(this_laiv)
                lait_df.loc[this_time] = copy.deepcopy(this_lait)
                lat_df.loc[this_time] = copy.deepcopy(this_lat)
                lon_df.loc[this_time] = copy.deepcopy(this_lon)
                if read_met_forcing:
                    tair_df.loc[this_time] = copy.deepcopy(this_tair)
                    qair_df.loc[this_time] = copy.deepcopy(this_qair)
                    lwdown_df.loc[this_time] = copy.deepcopy(this_lwdown)
                    swdown_df.loc[this_time] = copy.deepcopy(this_swdown)
                    psurf_df.loc[this_time] = copy.deepcopy(this_psurf)
                    rainf_df.loc[this_time] = copy.deepcopy(this_rainf)
                    rainfsnowf_df.loc[this_time] =copy.deepcopy(
                        this_rainfsnowf
                    )
                # update the day and return to the top of the loop
                curr += delta
                d += 1
            # add each of the variables to out_dict
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
            if save_bool == True:
                save_dir = os.path.join(
                    timeseries_dir,exp_name+'.pkl'
                )
                with open(save_dir,'wb') as f:
                    pkl.dump(out_dict,f)
                print('saved experiment to {}.'.format(save_dir))
        #otherwise load it
        else:
            # get where it was saved
            load_dir = os.path.join(
                timeseries_dir,exp_name+'.pkl'
            )
            print('Loading experiment from {}'.format(load_dir))
            # open the file
            with open(load_dir,'rb') as f:
                out_dict = pkl.load(f)
            # let's only take the times that we need
            times = list(out_dict['le'].index)
            times = pd.to_datetime(times)
            # let's see where is our start and end idx
            for t,ti in enumerate(times):
                if ti.date() == self.start:
                    start_idx = t
                if ti.date() == self.end:
                    end_idx = t
            for ke in out_dict.keys():
                out_dict[ke] = out_dict[ke].iloc[start_idx:end_idx+1]
        return out_dict
    def get_fluxcom_timeseries(self,fluxcom_dir):
        # load the fluxcom data from previous step
        fluxcom_data = pd.read_csv(fluxcom_dir)
        # set the start and end dates
        start_format = self.start.strftime('%Y-%m-%d')
        end_format = self.end.strftime('%Y-%m-%d')
        # find which rows this is in fluxcom
        fluxcom_time = np.array(fluxcom_data['time'])
        start_idx = np.where(fluxcom_data == start_format)[0][0]
        end_idx = np.where(fluxcom_data == end_format)[0][0]
        # trim to these start and end dates
        fluxcom_trimmed = fluxcom_data.iloc[start_idx:end_idx+1]
        # set the fluxcom columns to be ints instead of strings, aligning with
        # outputs from catchment-CN
        cols = np.array(fluxcom_trimmed.columns)
        cols_no_time = cols[1:]
        new_cols = copy.deepcopy(cols)
        cols_int = cols_no_time.astype(int)
        new_cols[1:] = cols_int
        fluxcom_trimmed.columns = new_cols
        return fluxcom_trimmed
    def get_strm_timeseries(self,strm_dir):
        # load streamflow from previous step
        strm_data = pd.read_csv(strm_dir)
        strm_data = strm_data.set_index('time')
        # trim using start and end dates
        # format the start and end dates
        start_fmt = self.start.strftime('%Y%m')
        end_fmt = self.end.strftime('%Y%m')
        strm_data = strm_data.loc[start_fmt:end_fmt]
        return strm_data
