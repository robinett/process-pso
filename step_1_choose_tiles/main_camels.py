import os
import sys
from choose_tiles_camels import choose

def main():
    # set main directory
    main_dir = (
        '/shared/pso/step_1_choose_tiles'
    )
    # set directory where data is stored
    data_dir = os.path.join(
        main_dir,'data'
    )
    # set directory to save outputs
    out_dir = os.path.join(
        main_dir,'outputs'
    )
    # set directory to save plots
    plots_dir = os.path.join(
        main_dir,'plots'
    )
    # file names for the camels watershed boundaries shapefiles
    camels_boundaries_fname = os.path.join(
        data_dir,
        'camels',
        'shapefile'
    )
    # what is approximately the maximum number of tiles that we want to run in
    # the optimization?
    max_num_tiles = 1000
    # file for catchment tile information
    tile_fname = os.path.join(data_dir,'tile_coord.csv')
    # file name for the final output dictionary
    dictionary_name = os.path.join(out_dir,'intersection_info.pkl')
    # file name for output of all Catchment tiles that need to be run
    all_tile_fname = os.path.join(out_dir,'intersecting_catch_tiles.csv')
    # name for the shapefile that contains state information for backgroun of plot
    states_shp_fname = os.path.join(data_dir,'state_shp/tl_2022_us_state.shp')
    # name of where to save the formatted include file for pso run
    include_fname = os.path.join(out_dir,'include')
    # name of where to save geodataframe with all the chosen camels information
    chosen_camels_gdf_fname = os.path.join(out_dir,'chosen_camels.geojson')
    # name of where to save the gdf with all tiles
    chosen_tiles_gdf_fname = os.path.join(out_dir,'chosen_tiles.geojson')
    # fname of where to save the selected camels ids
    chosen_camels_fname = os.path.join(out_dir,'chosen_camels.csv')
    # do we want to reprocess the data? If true, does so and saves. If false,
    # just loads saved processed data
    reprocess = True
    # do we want to plot maps?
    plot_maps = False
    # initiate an instance of class choose
    ch = choose()
    if reprocess:
        # get this catchment tile intersection infor from these
        intersection_info,all_intersecting,tile_shapes,camels_sel,camel_shapes = (
            ch.get_catch_tiles(tile_fname,camels_boundaries_fname,max_num_tiles)
        )
        print('number of total tiles needed to run:')
        print(len(all_intersecting))
        ch.save_dictionary(intersection_info,dictionary_name)
        # save the list of tiles to be run by Catchment in a later step
        ch.save_array_as_csv(all_intersecting,all_tile_fname)
        ch.save_df_as_csv(camels_sel,chosen_camels_fname)
        ch.create_include_file(all_intersecting,include_fname)
        ch.save_gdf(tile_shapes,chosen_tiles_gdf_fname)
        ch.save_gdf(camel_shapes,chosen_camels_gdf_fname)
    if plot_maps:
        # load the stuff here
        print('loading intersection info')
        intersection_info = ch.load_dictionary(dictionary_name)
        print('loading tiles_shp')
        tiles_shp = ch.load_gdf(chosen_tiles_gdf_fname,index_name='tile')
        print(tiles_shp)
        print('loading camels_shp')
        camels_shp = ch.load_gdf(chosen_camels_gdf_fname,index_name='hru_id')
        print(camels_shp)
        print('loading states shp')
        states_shp = ch.load_gdf(states_shp_fname)
        print(states_shp)
        # plot the stuff here
        ch.plot_hucs_tiles(
            intersection_info,tiles_shp,camels_shp,states_shp,plots_dir
        )
    # save the catchment tile info to be used in future steps
if __name__ == '__main__':
    main()
