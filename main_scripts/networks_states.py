#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 08:39:43 2022
@author: evelynm
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path
import shapely
import os
import pickle
import numpy as np

# on climada_petals branch feature/networks until merged into develop!!
from climada_petals.engine.networks.nw_preps import NetworkPreprocess, PowerlinePreprocess, RoadPreprocess
from climada_petals.engine.networks.nw_base import Network
from climada_petals.engine.networks.nw_calcs import Graph
from climada_petals.engine.networks import nw_utils as nwu
from climada_petals.entity.exposures.openstreetmap import osm_dataloader as osm
from climada_petals.util.constants import DICT_GEOFABRIK

from climada.util import coordinates as u_coords
from climada.hazard.base import Hazard

# general paths & constants
PATH_DATA = '/cluster/work/climate/evelynm/nw_inputs/'
PATH_DATA_OSM = PATH_DATA +'osm_countries/'
PATH_DATA_HVMV = PATH_DATA +'power/hvmv_global.shp'
PATH_DATA_PP = PATH_DATA +'power/global_power_plant_database.csv'
PATH_DATA_CT = PATH_DATA +'opencellid_global_1km_int.tif'
PATH_DATA_POP = PATH_DATA + 'worldpop/'
PATH_DEPS = PATH_DATA + 'dependencies/dependencies_default.csv'
PATH_SAVE = '/cluster/work/climate/evelynm/nw_outputs/'
PATH_EL_CONS_GLOBAL = PATH_DATA +'power/final_consumption_iea_global.csv'
PATH_ET = PATH_DATA +'power/electrification_targets.tif'
PATH_FRICTION = PATH_DATA + 'friction/202001_Global_Walking_Only_Friction_Surface_2019.tif'

import sys
cntry = sys.argv[1]
state = sys.argv[2]

# =============================================================================
# Load Infra Data
# =============================================================================

# country info
iso3 = u_coords.country_to_iso(cntry)
path_worldpop_cntry = PATH_DATA_POP + f'{iso3.lower()}_ppp_2020_1km_Aggregated_UNadj.tif'
path_el_consump_cntry = PATH_DATA + f'power/Electricity consumption per capita - {cntry}.csv'
path_elgen_cntry = PATH_DATA + f'power/Electricity generation by source - {cntry}.csv'
path_el_imp_exp_cntry = PATH_DATA + f'power/Electricity imports vs. exports - {cntry}.csv'
path_save_cntry = PATH_SAVE + f'{iso3}/'
if not os.path.isdir(path_save_cntry):
    os.mkdir(path_save_cntry)

# state info
path_osm_state = PATH_DATA_OSM+f'{state}'+'-latest.osm.pbf'
path_save_state = path_save_cntry + f'{state}/'
if not os.path.isdir(path_save_state):
    os.mkdir(path_save_state)

# STATE SHAPE    
states, cntry_shape = u_coords.get_admin1_info([cntry])
pos_state = np.where([state_dict['name_en']==state for state_dict in states[iso3]])[0][0]
state_shape = cntry_shape[iso3][pos_state]

# OSM Query Object
import urllib
if iso3=='USA':
    # get state osm.pbf from geofabrik
    url_osm_state = f'https://download.geofabrik.de/north-america/us/{state.lower()}-latest.osm.pbf'
    if not Path(path_osm_state).is_file():
        print(f"downloading osm pbf file for {state}")
        urllib.request.urlretrieve(url_osm_state, path_osm_state)
elif iso3=='CHN':
    osm.OSMRaw().get_data_geofabrik(iso3, file_format='pbf', save_path=PATH_DATA_OSM)
    osm.OSMRaw().extract_from_shapes([state_shape], path_poly=PATH_DATA_OSM+f'{state}.poly', 
                                     path_extract=path_osm_state, path_parentfile=PATH_DATA_OSM+'china-latest.osm.pbf',
                                     overwrite=False)
else:
    osm.OSMRaw().extract_from_shapes([state_shape], path_poly=PATH_DATA_OSM+f'{state}.poly', 
                                     path_extract=path_osm_state, path_parentfile=PATH_DATA_OSM+'planet-latest.osm.pbf',
                                     overwrite=False)
StateFileQuery = osm.OSMFileQuery(path_osm_state)

# POWER LINES
gdf_powerlines = gpd.read_file(PATH_DATA_HVMV, mask=state_shape)
gdf_powerlines['osm_id'] = 'n/a'
gdf_powerlines['ci_type'] = 'n/a' #random, preprocessing needs another column
gdf_powerlines = gdf_powerlines[['osm_id', 'geometry', 'ci_type']]

# POWER PLANTS
# try WRI power plant database, then OSM
gdf_pp = nwu.PowerFunctionalData().load_pplants_wri(PATH_DATA_PP, iso3)
if gdf_pp.empty:
    gdf_pp = StateFileQuery.retrieve_cis('power')
    if len(gdf_pp[gdf_pp.power=='plant'])>1:
        gdf_pp = gdf_pp[gdf_pp.power=='plant']
    else:
        # last 'resort': take generators from OSM
        gdf_pp = gdf_pp[gdf_pp.power=='generator']       
    gdf_pp['geometry'] = gdf_pp.geometry.apply(lambda geom: geom.centroid)
    gdf_pp = gdf_pp[['name', 'power', 'geometry']]
gdf_pp = gdf_pp[gdf_pp.geometry.within(state_shape)]

# PEOPLE
nwu.get_worldpop_data(iso3, PATH_DATA_POP)
meta, data = u_coords.read_raster(path_worldpop_cntry, geometry=[state_shape])
grid = u_coords.raster_to_meshgrid(meta['transform'], meta['width'], 
                                       meta['height'])                                               
gdf_people = gpd.GeoDataFrame({'counts': data.squeeze().flatten(), 
                        'geometry': gpd.points_from_xy(
                                grid[0].flatten(), grid[1].flatten())})
gdf_people = gdf_people[gdf_people.counts!=0].reset_index(drop=True)
pop_thresh = nwu.get_pop_cutoff(gdf_people, 0.1)
gdf_people = gdf_people[gdf_people.counts>=pop_thresh].reset_index(drop=True)
# assign electricity consumption & production
gdf_people['electrified'] = nwu.PowerFunctionalData().assign_el_targets(
    gdf_people, iso3, state_shape, PATH_ET)
gdf_people, gdf_pp = nwu.PowerFunctionalData().assign_el_prod_consump(
    gdf_people, gdf_pp, iso3, PATH_EL_CONS_GLOBAL)

# HEALTH FACILITIES
# from osm
gdf_health = StateFileQuery.retrieve_cis('healthcare') 
gdf_health['geometry'] = gdf_health.geometry.apply(lambda geom: geom.centroid)
gdf_health = gdf_health[['name', 'geometry']]

# EDUC. FACILITIES
# from osm
gdf_educ = StateFileQuery.retrieve_cis('education')
gdf_educ['geometry'] = gdf_educ.geometry.apply(lambda geom: geom.centroid)
gdf_educ = gdf_educ[['name', 'geometry']]

# TELECOM
# cells from rasterized opencellID (via WB)
path_ct_state = path_save_state+'celltowers.tif'
if not Path(path_ct_state).is_file():
    if state_shape.type=='Polygon':
        geo_mask = [state_shape]
    else:
        geo_mask = [mp for mp in state_shape]
    meta_ct, arr_ct = u_coords.read_raster(PATH_DATA_CT, src_crs={'epsg':'4326'},
                                           geometry=geo_mask)
    u_coords.write_raster(path_ct_state, arr_ct, meta_ct)
gdf_cells = nwu.load_resampled_raster(path_ct_state, 1/5)

# ROADS
# from osm; by default, take all types of roads
gdf_roads = StateFileQuery.retrieve_cis('road')
frac_unclass = gdf_roads.groupby('highway').size()['unclassified']/len(gdf_roads)
if frac_unclass < 0.2:
    # if unclassified road fraction is quite small, take only main roads
    gdf_roads = gdf_roads[(gdf_roads.highway != 'residential') & 
                          (gdf_roads.highway != 'unclassified')]
gdf_roads = gdf_roads[gdf_roads.geometry.type=='LineString']
gdf_roads = gdf_roads[['osm_id','highway', 'geometry']]

# =============================================================================
# # Graphs
# =============================================================================

# POWER LINES
gdf_power_edges, gdf_power_nodes = PowerlinePreprocess().preprocess(
    gdf_edges=gdf_powerlines)
power_network = Network(gdf_power_edges, gdf_power_nodes)
power_graph = Graph(power_network, directed=False)

iter_count = 0
while (len(power_graph.graph.clusters())>1) and (iter_count<8):
    iter_count+=1
    power_graph.link_clusters(dist_thresh=200000)
    
power_network = Network().from_graphs([power_graph.graph.as_directed()])
# power_graph.graph.clusters().summary(): 

# PEOPLE
__, gdf_people_nodes = NetworkPreprocess('people').preprocess(
    gdf_nodes=gdf_people)
people_network = Network(nodes=gdf_people_nodes)

# POWER PLANTS
__, gdf_pp_nodes = NetworkPreprocess('power_plant').preprocess(
    gdf_nodes=gdf_pp)
pplant_network = Network(nodes=gpd.GeoDataFrame(gdf_pp_nodes))

# HEALTHCARE
__, gdf_health_nodes = NetworkPreprocess('health').preprocess(
    gdf_nodes=gdf_health)
health_network = Network(nodes=gdf_health_nodes)

# EDUC
__, gdf_educ_nodes = NetworkPreprocess('education').preprocess(
    gdf_nodes=gdf_educ)
educ_network = Network(nodes=gdf_educ_nodes)

# ROAD
gdf_road_edges, gdf_road_nodes = RoadPreprocess().preprocess(
    gdf_edges=gdf_roads)
road_network = Network(gdf_road_edges, gdf_road_nodes)
# easy workaround for doubling edges
road_graph = Graph(road_network, directed=False)
iter_count = 0
while (len(road_graph.graph.clusters())>1) and (iter_count<5):
    iter_count+=1
    road_graph.link_clusters(dist_thresh=30000)

road_network = Network().from_graphs([road_graph.graph.as_directed()])
#road_graph.graph.clusters().summary()

# TELECOM
__, gdf_tele_nodes = NetworkPreprocess('celltower').preprocess(gdf_nodes=gdf_cells)
tele_network = Network(nodes=gdf_tele_nodes)

# MULTINET
cis_network = Network.from_nws([pplant_network, power_network,
                                people_network, health_network, educ_network,
                                road_network, tele_network])
cis_network.initialize_funcstates()

# =============================================================================
# Interdependencies
# =============================================================================

# load default dependency table
df_dependencies = pd.read_csv(PATH_DEPS, sep=',', header=0)
df_dependencies.loc[np.isnan(df_dependencies.thresh_dist), 'thresh_dist'] = np.inf

# refine dependency table
cond = ('electrified', True)
df_dependencies['conditions'] = None
df_dependencies['conditions'] = df_dependencies['conditions'].astype(object)
idx = np.where((df_dependencies.source=='power_line') & (
    df_dependencies.target=='people'))[0][0]
df_dependencies.at[idx, 'conditions'] = cond
df_dependencies = nwu.set_travel_distance_threshs(df_dependencies, iso3, hrs_max=1)
k=3
if frac_unclass >= 0.2:
    # if all roads are taken, set a shorter dist thresh & allow only one access point, else leave original main-road thresh
    df_dependencies.loc[(df_dependencies.source=='road') &
                        (df_dependencies.target=='people'), 'thresh_dist'] = 2000
    df_dependencies.loc[(df_dependencies.source=='road') &
                        (df_dependencies.target=='people'), 'single_link'] = True
    k=1

cis_graph = Graph(cis_network, directed=True)

# create "missing physical structures" - needed for real world flows
cis_graph.link_vertices_closest_k(
    'power_line', 'power_plant',dist_thresh=np.inf, link_name='power_line', bidir=True, k=1)
cis_graph.link_vertices_closest_k(
    'road', 'people',  link_name='road', 
    dist_thresh=df_dependencies.loc[(df_dependencies.source=='road') & 
                                    (df_dependencies.target=='people'),
                                    'thresh_dist'].values[0], bidir=True, k=k)
cis_graph.link_vertices_closest_k(
    'road', 'health',  link_name='road', dist_thresh=np.inf, bidir=True, k=1)
cis_graph.link_vertices_closest_k(
    'road', 'education', link_name='road', dist_thresh=np.inf, bidir=True, k=1)

# place dependencies
friction_surf = Hazard('FRIC').from_raster(
    PATH_FRICTION, geometry=[state_shape.convex_hull.buffer(0.1)])

for __, row in df_dependencies.iterrows():
    cis_graph.place_dependency(row.source, row.target, 
                               single_link=row.single_link,
                               access_cnstr=row.access_cnstr, 
                               dist_thresh=row.thresh_dist,
                               preselect=False,
                               friction_surf=friction_surf,
                               dur_thresh=60,
                               cond=row.conditions)
cis_network = cis_graph.return_network()

# =============================================================================
# Base State
# =============================================================================

cis_network.initialize_funcstates()
for __, row in df_dependencies.iterrows():
    cis_network.initialize_capacity(row.source, row.target)
for __, row in df_dependencies[
        df_dependencies['type_I']=='enduser'].iterrows():
    cis_network.initialize_supply(row.source)
    
cis_graph = Graph(cis_network, directed=True)
cis_graph.cascade(df_dependencies, p_source='power_plant', p_sink='power_line', 
                  source_var='el_generation', demand_var='el_consumption',
                  preselect=False, initial=True, friction_surf=friction_surf, 
                  dur_thresh=60)
cis_network = cis_graph.return_network()

cis_network.nodes.to_feather(path_save_state+'cis_nw_nodes')
cis_network.edges.to_feather(path_save_state+'cis_nw_edges')

base_stats = nwu.number_noservices(cis_graph,
                         services=['power', 'healthcare', 'education', 
                                   'telecom', 'mobility'])

with open(path_save_state +f'base_stats_{iso3}_{state}.pkl', 'wb') as f:
    pickle.dump(base_stats, f) 

# save country-specific dependency-table.
df_dependencies.to_csv(path_save_state +f'dependency_table_{iso3}_{state}.csv')
