import pandas as pd
import numpy as np
import sys
import os
import copy
import datetime
import geopandas as gpd

class camels:
    def __init__(self,start,end):
        self.start = start
        self.end  = end
    def get_camels(self,camels_fname):
        # get the tiles that we are going to run as defined in step 1
        camels = pd.read_csv(camels_fname)
        camels = camels.set_index('camel')
        return camels
    def get_streamflow(self,camels_fname,chosen_camels_df):
        # let's get the camels that we are looking for
        chosen_camels = np.array(chosen_camels_df.index)
        cols = chosen_camels
        all_dates = []
        curr = copy.deepcopy(self.start)
        while curr <= self.end:
            all_dates.append(curr.strftime('%Y%m%d'))
            curr += datetime.timedelta(days=1)
        camels_truth = pd.DataFrame(columns=cols,index=all_dates)
        camels_truth.index.name = 'time'
        data_cols = [
            'gauge','year','mon','day','strm','qc'
        ]
        for c,cam in enumerate(chosen_camels):
            camel_str = str(cam).zfill(8)
            file_found = False
            dir_guess = 0
            while not file_found:
                dir_guess_str = str(dir_guess).zfill(2)
                this_fname = os.path.join(
                    camels_fname,
                    dir_guess_str,
                    '{}_streamflow_qc.txt'.format(
                        camel_str
                    )
                )
                file_found = os.path.isfile(this_fname)
                dir_guess += 1
            this_data = pd.read_csv(
                this_fname,
                delim_whitespace=True,
                header=None,
                names=data_cols
            )
            start_idx = np.where(
                (this_data['year'] == self.start.year) &
                (this_data['mon'] == self.start.month) &
                (this_data['day'] == self.start.day)
            )[0][0]
            end_idx = np.where(
                (this_data['year'] == self.end.year) &
                (this_data['mon'] == self.end.month) &
                (this_data['day'] == self.end.day)
            )[0][0]
            this_data_trim = this_data.iloc[
                start_idx:end_idx+1
            ]
            this_strm = np.array(this_data_trim['strm'])
            nan_idx = np.where(this_strm == -999)
            this_strm[nan_idx] = np.nan
            # lets convert
            this_area = chosen_camels_df['area'].loc[cam]
            # strm data oringally in cf/s
            this_strm = this_strm*(0.0003048**3) # km3/s
            this_strm = this_strm/this_area # km/s
            this_strm = this_strm*10e6 # mm/s
            this_strm = this_strm*86400 # mm/d
            strm_mon_avg = np.sum(this_strm)/15/12
            camels_truth[cam] = this_strm
            camels_truth = camels_truth.copy()
        return camels_truth
    def save_truth(self,camels_truth,out_dir):
        fname = os.path.join(
            out_dir,
            'camels_truth_{st}_{end}_mm_day.csv'.format(
                st = self.start,
                end = self.end
            )
        )
        camels_truth.to_csv(fname)












