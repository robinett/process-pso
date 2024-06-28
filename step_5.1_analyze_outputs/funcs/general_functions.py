import pandas as pd
import numpy as np
import pickle as pkl
import sys
import geopandas as gpd
import matplotlib as mpl

class gen_funcs:
    def get_pixels(self,pixels_fname):
        '''
        Function that gets the Catchment-CN4.5 pixels that we are currently
        working with.
        Inputs:
            pixels_fname: fname of the step_1 file where the pixels that we
            need are saved.
        Ouputs:
            pixels: A numpy array containing the pixel numbers as ints
        '''
        # get the pixels that we are going to run as defined in step 1
        pixels = pd.read_csv(pixels_fname,header=None)
        # turn this into an np array
        pixels = np.array(pixels).astype(int)
        # make a nice np array
        pixels = pixels.T
        pixels = pixels[0]
        # return the pixels
        return pixels
    def get_intersection_info(self,fname):
        with open(fname,'rb') as f:
            out = pkl.load(f)
        return out
    def df_to_weekly(self,df):
        df_mon = df.groupby(pd.PeriodIndex(df.index, freq="w")).mean()
        df_mon.index = df_mon.index.to_timestamp()
        return df_mon
    def df_to_monthly(self,df):
        df_mon = df.groupby(pd.PeriodIndex(df.index, freq="M")).mean()
        df_mon.index = df_mon.index.to_timestamp()
        return df_mon
    def df_to_yearly(self,df):
        df_yr = df.groupby(df.index.year).mean()
        df_yr.index = pd.to_datetime(df_yr.index,format='%Y')
        return df_yr
    def add_to_gdf_and_save(self,orig_gdf_fname,out_gdf_fname,
                            vals,vmin,vmax,cmap,nan_replacement=0,
                            subselection=[-1]):
        vals = np.array(vals)
        vals[np.isnan(vals) == True] = nan_replacement
        print('min of vals going into {name}:'.format(name=out_gdf_fname))
        print(np.nanmin(vals))
        print('max of vals going into {name}:'.format(name=out_gdf_fname))
        print(np.nanmax(vals))
        tile_gdf = gpd.read_file(orig_gdf_fname)
        if subselection[0] != -1:
            tile_gdf = tile_gdf.set_index('hru_id')
            tile_gdf = tile_gdf.loc[subselection]
        tile_gdf['vals'] = vals
        norm = mpl.colors.Normalize(vmin=vmin,vmax=vmax)
        this_cmap = mpl.cm.get_cmap(cmap)
        vals_norm = norm(vals)
        colors = this_cmap(vals_norm)
        colors[:,:-1]  = colors[:,:-1]*255
        tile_gdf['color_r'] = colors[:,0]
        tile_gdf['color_g'] = colors[:,1]
        tile_gdf['color_b'] = colors[:,2]
        tile_gdf.to_file(out_gdf_fname)
    def pix_df_to_wat_df(self,df,intersection_info):
        # get watersheds
        watersheds = list(intersection_info.keys())
        # initialize the size of the output
        num_times = len(list(df.index))
        num_watersheds = len(watersheds)
        data_wat = pd.DataFrame()
        data_wat['times'] = list(df.index)
        # loop over all watersheds and convert
        for w,wat in enumerate(watersheds):
            # get the tiles in this watershed
            this_tiles = intersection_info[wat][0]
            # get the percent of each of these tiles in this watershed
            this_perc = intersection_info[wat][1]
            this_data = np.array(df[this_tiles])
            this_data_avg = np.average(
                this_data,axis=1,weights=this_perc
            )
            data_wat[wat] = this_data_avg
        data_wat = data_wat.set_index('times')
        return data_wat

















