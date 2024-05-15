import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

class plot_wat:
    def __init__(self):
        # let's load a bunch of things that are slow to load so that we only
        # have to do this when we first start up the class
        # load states shp and save to self
        states_shp_fname = (
            '/shared/pso/step_1_choose_tiles/data/state_shp'
        )
        print('loading states shp')
        states = gpd.read_file(states_shp_fname)
        states = states.to_crs('EPSG:4326')
        # get rid of non-conus states since not considering
        non_conus = ['HI','VI','MP','GU','AK','AS','PR']
        states_conus = states
        for n in non_conus:
            states_conus = states_conus[states_conus.STUSPS != n]
        self.states_conus = states_conus
        # load the huc6 watershed gpd and save to self
        huc6s_fname = (
            '/shared/pso/step_1_choose_tiles/outputs/chosen_camels.geojson'
        )
        huc6s = gpd.read_file(huc6s_fname)
        self.huc6s = huc6s
    def plot_map(self,name,hucs,vals,avg_val,plots_dir,cmap='rainbow',
                 vmin=np.nan,vmax=np.nan,log=False):
        print('plotting {}'.format(name))
        if np.isnan(vmin) == True:
            vmin = np.min(vals)
        if np.isnan(vmax) == True:
            vmax = np.max(vals)
        hucs_int = [int(h) for h in hucs]
        hucs = hucs_int
        fig,ax = plt.subplots()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        state_ids = list(self.states_conus['GEOID'])
        for s,sid in enumerate(state_ids):
            this_geom = self.states_conus['geometry'].iloc[s]
            try:
                xs,ys = this_geom.exterior.xy
                ax.fill(xs,ys,fc='none',ec='k',linewidth=.1)
            except:
                for geom in this_geom.geoms:
                    xs,ys = geom.exterior.xy
                    ax.fill(xs,ys,fc='none',ec='k',linewidth=.1)
        # get our normalize function for getting colors
        if not log:
            norm = mpl.colors.Normalize(
                vmin=vmin,vmax=vmax
            )
        elif log:
            norm = mpl.colors.LogNorm(
                vmin=vmin,vmax=vmax
            )
        this_cmap = mpl.cm.get_cmap(cmap)
        for h,huc in enumerate(hucs):
            idx = np.where(
                self.huc6s['hru_id'] == huc
            )[0][0]
            this_geom = self.huc6s['geometry'].iloc[idx]
            this_val = vals[h]
            this_val_norm = norm(this_val)
            this_color = this_cmap(this_val_norm)
            if this_geom.geom_type == 'Polygon':
                xs,ys = this_geom.exterior.xy
                ax.fill(xs,ys,fc=this_color,ec='k',linewidth=0.2)
            elif this_geom.geom_type == 'MultiPolygon':
                for this_this_geom in this_geom.geoms:
                    xs,ys = this_this_geom.exterior.xy
                    ax.fill(xs,ys,fc=this_color,ec='k',linewidth=0.2)
            else:
                raise IOError('Shape is not a polygon')
        ax.text(
            -127+2,20+4,'Average {name}: {val:.4f}'.format(
                name=name,val=avg_val
            ),
            bbox=dict(facecolor='white')
        )
        fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=this_cmap),
            cax=cax, orientation='vertical'
        )
        this_savename = os.path.join(
            plots_dir,
            '{name}_map.png'.format(
                name=name
            )
        )
        plt.savefig(this_savename,dpi=350,bbox_inches='tight')
        plt.close()
