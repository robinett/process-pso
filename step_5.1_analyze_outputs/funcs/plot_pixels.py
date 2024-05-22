import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import numpy as np
import sys
import os

class plot_pixels:
    def plot_map(self,name,pixels,vals,avg_val,plots_dir,
                 vmin=np.nan,vmax=np.nan,cmap='rainbow'):
        print('ploting {name}'.format(name=name))
        # load up the tile info
        tile_info = pd.read_csv(
            '/shared/pso/step_1_choose_tiles/data/tile_coord.csv'
        )
        tile_info = tile_info.set_index('tile_id')
        # select lats and lons based off of passed tiles
        this_info = tile_info.loc[pixels]
        lons = np.array(this_info['com_lon'])
        lats = np.array(this_info['com_lat'])
        # add the options available
        if np.isnan(vmin) == True:
            vmin = np.nanmin(vals)
        if np.isnan(vmax) == True:
            vmax = np.nanmax(vals)
        # make hte plot
        fig = plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())
        # add coastline and set the limits
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.STATES,linewidth=0.1)
        ax.set_ylim(bottom=20,top=55)
        ax.set_xlim(left=-127,right=-60)
        # make the scatterplot
        scatter = plt.scatter(
            lons,lats,marker='s',s=1,c=vals,transform=ccrs.PlateCarree(),
            cmap=cmap,vmin=vmin,vmax=vmax
        )
        # add colorbar
        plt.colorbar(scatter)
        # add the average error as text
        ax.text(
            -127+2,20+2,'Average {name}: {val:.4f}'.format(
                name=name,val=avg_val
            ),
            bbox=dict(facecolor='white')
        )
        # save
        savename = '{name}_map.png'.format(
            name=name
        )
        savename = os.path.join(
            plots_dir,savename
        )
        plt.savefig(savename,dpi=300,bbox_inches='tight')
        plt.close()

