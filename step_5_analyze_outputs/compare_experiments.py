import sys
import matplotlib.pyplot as plt
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import matplotlib as mpl
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

class compare_experiments:
    def __init__(self):
        pass
    def plot_diff(self,exp_1_names,exp_2_names,exp_1_pix_pso_df,exp_2_pix_pso_df,
                  exp_1_wat_pso_df,exp_2_wat_pso_df,geojson_fname,states_shp,
                  plots_dir):
        # lets make some watershed scale difference plots
        # perc change in le rmse between experiments
        exp_perc_change_le_rmse = (
            exp_1_pix_pso_df.loc['perc_change_le_rmse'] -
            exp_2_pix_pso_df.loc['perc_change_le_rmse']
        )
        avg_exp_perc_change_le_rmse = exp_perc_change_le_rmse['all']
        # change in le r2 between experiments
        exp_change_le_r2 = (
            exp_1_pix_pso_df.loc['change_le_r2'] -
            exp_2_pix_pso_df.loc['change_le_r2']
        )
        avg_exp_change_le_r2 = exp_change_le_r2['all']
        # change in corr between experiments
        exp_change_le_corr = (
            exp_1_pix_pso_df.loc['change_le_corr'] -
            exp_2_pix_pso_df.loc['change_le_corr']
        )
        avg_exp_change_le_corr = exp_change_le_corr['all']
        # perc change in le ubrmse between experiments
        exp_perc_change_le_ubrmse = (
            exp_1_pix_pso_df.loc['perc_change_le_ubrmse'] -
            exp_2_pix_pso_df.loc['perc_change_le_ubrmse']
        )
        avg_exp_perc_change_le_ubrmse = exp_perc_change_le_ubrmse['all']
        # difference of average le
        exp_diff_ave_le = (
            exp_1_pix_pso_df.loc['ave_le'] -
            exp_2_pix_pso_df.loc['ave_le']
        )
        avg_exp_diff_ave_le = exp_diff_ave_le['all']
        # put values to be plotted into list for plotting
        vals = [
            exp_perc_change_le_rmse,exp_change_le_r2,exp_change_le_corr,
            exp_perc_change_le_ubrmse,exp_diff_ave_le
        ]
        # put the averages that correspond to these values
        avgs = [
            avg_exp_perc_change_le_rmse,avg_exp_change_le_r2,avg_exp_change_le_corr,
            avg_exp_perc_change_le_ubrmse,avg_exp_diff_ave_le
        ]
        # put the name the corresponds to each of these values
        names = [
            'exp_perc_change_le_rmse','exp_change_le_r2','exp_change_le_corr',
            'exp_perc_change_le_ubrmse','exp_diff_ave_le'
        ]
        # vals, avgs, and names need to all be the same length. if this isn't
        # true stop here and inform the user
        len_vals = len(vals)
        len_avgs = len(avgs)
        len_names = len(names)
        if ((len_vals != len_avgs) or
            (len_vals != len_names) or
            (len_avgs != len_names)):
            print('vals, avgs, and names must all be the same length!')
            print('go back and correct this!')
            sys.exit()
        types = names
        cmaps = {
            'exp_perc_change_le_rmse':'bwr',
            'exp_change_le_r2':'bwr',
            'exp_change_le_corr':'bwr',
            'exp_perc_change_le_ubrmse':'bwr',
            'exp_diff_ave_le':'bwr'
        }
        vmins = {
            'exp_perc_change_le_rmse':-.5,
            'exp_change_le_r2':-.4,
            'exp_change_le_corr':-.2,
            'exp_perc_change_le_ubrmse':-.3,
            'exp_diff_ave_le':-30
        }
        vmaxs = {
            'exp_perc_change_le_rmse':.5,
            'exp_change_le_r2':.4,
            'exp_change_le_corr':.2,
            'exp_perc_change_le_ubrmse':.3,
            'exp_diff_ave_le':30
        }
        for p in range(len(vals)):
            # let's first plot the rmse of default experiment versus fluxcom
            # create the figure and set the projection
            fig = plt.figure()
            ax = plt.axes(projection=ccrs.PlateCarree())
            # add coastline and set the limits
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.STATES,linewidth=0.1)
            ax.set_ylim(bottom=20,top=55)
            ax.set_xlim(left=-127,right=-60)
            # set extent isn't working with this weird bootleg version of cartopy
            # i've had to install on AWS--if it is fixed then could go back to this
            #if extent == 'conus':
            #    ax.set_extent([-127,-60,20,55],crs=ccrs.PlateCarree)
            #elif extent == 'global':
            #    ax.set_global()
            # define lats, lons, values
            lons = exp_1_pix_pso_df.loc['lon']
            lats = exp_1_pix_pso_df.loc['lat']
            val = vals[p]
            # make the scatterplot
            scatter = plt.scatter(
                lons,lats,marker='s',s=1,c=val,transform=ccrs.PlateCarree(),
                cmap=cmaps[types[p]],vmin=vmins[types[p]],vmax=vmaxs[types[p]]
            )
            # add colorbar
            plt.colorbar(scatter)
            # add the average error as text
            ax.text(
                -127+2,20+2,'Average {name}: {val:.2f}'.format(
                    name=names[p],val=avgs[p]
                ),
                bbox=dict(facecolor='white')
            )
            # save
            savename = '{name}_{exp_1}_vs_{exp_2}_optimization_pixels.png'.format(
                name=names[p],exp_1=exp_1_names[2],exp_2=exp_2_names[2]
            )
            savename = os.path.join(
                plots_dir,savename
            )
            plt.savefig(savename,dpi=300,bbox_inches='tight')
            plt.close()
        # now lets do this for our streamflow friends
        # percent change in strm rmse
        # change in strm perc rmse between experiments
        print(exp_1_wat_pso_df)
        print(exp_2_wat_pso_df)
        exp_perc_change_strm_rmse = (
            exp_1_wat_pso_df.loc['perc_change_strm_rmse'] -
            exp_2_wat_pso_df.loc['perc_change_strm_rmse']
        )
        avg_exp_perc_change_strm_rmse = exp_perc_change_strm_rmse['all']
        exp_perc_change_strm_rmse = np.array(exp_perc_change_strm_rmse)
        exp_perc_change_strm_rmse = exp_perc_change_strm_rmse[:-1]
        # change in strm r2 between experiments
        exp_change_strm_r2 = (
            exp_1_wat_pso_df.loc['change_strm_r2'] -
            exp_2_wat_pso_df.loc['change_strm_r2']
        )
        avg_exp_change_strm_r2 = exp_change_strm_r2['all']
        exp_change_strm_r2 = np.array(exp_change_strm_r2)
        exp_change_strm_r2 = exp_change_strm_r2[:-1]
        # change in strm corr between experiments
        exp_change_strm_corr = (
            exp_1_wat_pso_df.loc['change_strm_corr'] -
            exp_2_wat_pso_df.loc['change_strm_corr']
        )
        avg_exp_change_strm_corr = exp_change_strm_corr['all']
        exp_change_strm_corr = np.array(exp_change_strm_corr)
        exp_change_strm_corr = exp_change_strm_corr[:-1]
        # change in strm perc ubrmse between experiments
        exp_perc_change_strm_ubrmse = (
            exp_1_wat_pso_df.loc['perc_change_strm_ubrmse'] -
            exp_2_wat_pso_df.loc['perc_change_strm_ubrmse']
        )
        avg_exp_perc_change_strm_ubrmse = exp_perc_change_strm_ubrmse['all']
        exp_perc_change_strm_ubrmse = np.array(exp_perc_change_strm_ubrmse)
        exp_perc_change_strm_ubrmse = exp_perc_change_strm_ubrmse[:-1]
        # change in strm nse between experiments
        exp_change_strm_nse = (
            exp_1_wat_pso_df.loc['change_strm_nse'] -
            exp_2_wat_pso_df.loc['change_strm_nse']
        )
        avg_exp_change_strm_nse = exp_change_strm_nse['all']
        exp_change_strm_nse = np.array(exp_change_strm_nse)
        exp_change_strm_nse = exp_change_strm_nse[:-1]
        # change in strm between experiments
        exp_diff_strm = (
            exp_1_wat_pso_df.loc['diff_strm'] -
            exp_2_wat_pso_df.loc['diff_strm']
        )
        avg_exp_diff_strm = exp_diff_strm['all']
        exp_diff_strm = np.array(exp_diff_strm)
        exp_diff_strm = exp_diff_strm[:-1]
        # and plot this data
        # now let's get the shapes that we need for plotting
        huc6s = gpd.read_file(geojson_fname)
        # now let's get everythin in arrays for proper plotting
        names = [
            'exp_perc_change_strm_rmse','exp_change_strm_r2','exp_change_strm_corr',
            'exp_perc_change_strm_ubrmse','exp_change_strm_nse','exp_diff_strm'
        ]
        vals = [
            exp_perc_change_strm_rmse,exp_change_strm_r2,exp_change_strm_corr,
            exp_perc_change_strm_ubrmse,exp_change_strm_nse,exp_diff_strm
        ]
        avgs = [
            avg_exp_perc_change_strm_rmse,avg_exp_change_strm_r2,avg_exp_change_strm_corr,
            avg_exp_perc_change_strm_ubrmse,avg_exp_change_strm_nse,avg_exp_diff_strm
        ]
        types = names
        cmaps = {
            'exp_perc_change_strm_rmse':'bwr',
            'exp_change_strm_r2':'bwr',
            'exp_change_strm_corr':'bwr',
            'exp_perc_change_strm_ubrmse':'bwr',
            'exp_change_strm_nse':'bwr',
            'exp_diff_strm':'bwr'
        }
        vmins = {
            'exp_perc_change_strm_rmse':-.2,
            'exp_change_strm_r2':-1,
            'exp_change_strm_corr':-1,
            'exp_perc_change_strm_ubrmse':-.1,
            'exp_change_strm_nse':-1,
            'exp_diff_strm':-20
        }
        vmaxs = {
            'exp_perc_change_strm_rmse':.2,
            'exp_change_strm_r2':1,
            'exp_change_strm_corr':1,
            'exp_perc_change_strm_ubrmse':.1,
            'exp_change_strm_nse':1,
            'exp_diff_strm':20
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
                '{name}_{exp_1}_vs_{exp_2}_huc6.png'.format(
                    name=names[n],exp_1=exp_1_names[2],exp_2=exp_2_names[2]
                )
            )
            plt.savefig(this_savename,dpi=350,bbox_inches='tight')
            plt.close()
