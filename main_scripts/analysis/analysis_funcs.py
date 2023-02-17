#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 14:03:09 2023

@author: evelynm
"""

import geopandas as gpd
import pickle
import numpy as np
import glob
import os
import copy
import scipy

import logging
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel('INFO')

# =============================================================================
# Constants
# =============================================================================
def service_dict():
    """
    ci types as coded to ci types properly worded
    """
    return {'power_line' : 'electricity',
           'celltower' : 'mobile communications',
           'health': 'healthcare facilities',
           'education' : 'educational facility',
           'road' : 'roads'}

def service_dict2():
    """service supply variables from raw output 
    to basic service names"""
    return {'actual_supply_power_line_people' : 'electricity',
           'actual_supply_celltower_people' : 'mobile /n communications',
           'actual_supply_health_people': 'healthcare',
           'actual_supply_education_people' : 'education',
           'actual_supply_road_people' : 'mobility'}

def inc_class_dict():
    return {'ATG': 1,
 'BGD': 4,
 'CUB': 2,
 'GTM': 2,
 'HTI': 4,
 'KHM': 3,
 'LKA': 3,
 'MDG': 4,
 'MEX': 2,
 'MOZ': 4,
 'PHL': 3,
 'VNM': 3,
 'PRI': 1,
 'USA Florida': 1,
 'USA Louisiana': 1,
 'USA Texas': 1,
 'CHN Fujian': 2,
 'CHN Hainan': 2}


def pop_density_dict():
    return {'ATG': 222,
 'BGD': 1265,
 'CUB': 110,
 'GTM': 167,
 'HTI': 413,
 'KHM': 95,
 'LKA': 354,
 'MDG': 47,
 'MEX': 66,
 'MOZ': 39,
 'PHL': 367,
 'VNM': 313,
 'PRI': 360,
 'USA Florida': 136,
 'USA Louisiana': 41,
 'USA Texas': 43,
 'CHN Fujian': 350,
 'CHN Hainan': 276}

# =============================================================================
# Util functions
# =============================================================================


def load_dict(path_dict):
    # load any dict saved as pkl
    with open(path_dict, 'rb') as stats_dict:
         stats_dict = pickle.load(stats_dict)
    return stats_dict

def save_dict(dict_var, save_path):
    # save any dict as pkl
    with open(save_path, 'wb') as f:
        pickle.dump(dict_var, f)


def move_gdfs(path_cntry_folder, haz_type, valid_events=[], small_events=[]):
    """
    move old or small event gdfs to respective sub-folders.
    """
    if haz_type == 'TC':
        paths_result_files = [file for file in glob.glob(path_cntry_folder +'cascade_results*') 
                              if not 'DFO' in file]
        # select only result files that are still in valid hazard event selection
        paths_valid_resfiles = [path for path in paths_result_files 
                                if path.split('_')[-1] in valid_events]
        if len(small_events) > 0:
            paths_small_resfiles = [path for path in paths_result_files 
                                    if path.split('_')[-1] in small_events]
            for path in paths_small_resfiles:
                new_path = path_cntry_folder+f"small_events/cascade_results_{path.split('_')[-1]}"
                os.rename(path, new_path)
            paths_valid_resfiles.extend(paths_small_resfiles)
            
        paths_old_resfiles = set(paths_result_files).difference(
            paths_valid_resfiles)
        
        for path in paths_old_resfiles:
            new_path = path_cntry_folder+f"old/cascade_results_{path.split('_')[-1]}"
            os.rename(path, new_path)
        
    elif haz_type == 'FL':
        paths_result_files = glob.glob(path_cntry_folder + 'cascade_results_DFO*')
        # select only result files that are still in valid hazard event selection
        paths_valid_resfiles = [path for path in paths_result_files 
                                if 'DFO_'+ path.split('_')[-1] in valid_events]
        if len(small_events) > 0:
            paths_small_resfiles = [path for path in paths_result_files 
                                    if path.split('_')[-1] in small_events]
            for path in paths_small_resfiles:
                new_path = path_cntry_folder+f"small_events/cascade_results_DFO{path.split('_')[-1]}"
                os.rename(path, new_path)
                
        paths_valid_resfiles.extend(paths_small_resfiles)        
        paths_old_resfiles = set(paths_result_files).difference(
            paths_valid_resfiles)
        
        for path in paths_old_resfiles:
            new_path = path_cntry_folder+f"old/cascade_results_DFO{path.split('_')[-1]}"
            os.rename(path, new_path)
            
    else:
        LOGGER.error('not implemented')

    
def load_gdf_dict(path_cntry_folder, haz_type, valid_events):
    # get all result dataframes filepaths
    if haz_type == 'TC':
        paths_result_files = [file for file in glob.glob(path_cntry_folder +'cascade_results*') 
                              if not 'DFO' in file]
        # select only result files that are still in valid hazard event selection
        paths_valid_resfiles = [path for path in paths_result_files 
                                if path.split('_')[-1] in valid_events]
        
    elif haz_type == 'FL':
        paths_result_files = glob.glob(path_cntry_folder + 'cascade_results_DFO*')
        # select only result files that are still in valid hazard event selection
        paths_valid_resfiles = [path for path in paths_result_files 
                                if 'DFO_'+ path.split('_')[-1] in valid_events]
    elif haz_type == 'RF':
        paths_result_files = [file for file in glob.glob(path_cntry_folder +'cascade_results*') 
                              if not 'DFO' in file]
        # select only result files that are still in valid hazard event selection
        paths_valid_resfiles = [path for path in paths_result_files 
                                if path.split('_')[-1] in valid_events]    
    # read in all result gdfs
    gdf_list= []
    name_list = []
    for file_path in paths_valid_resfiles:
        event_name = file_path.split('_')[-1] if haz_type in ['TC', 'RF'] else 'DFO_'+ file_path.split('_')[-1]
        name_list.append(event_name)
        gdf_list.append(gpd.read_feather(file_path)) 
    
    # make dict
    return dict(zip(name_list, gdf_list))

# =============================================================================
# Cascade State and Access State
# =============================================================================

def get_cascstate(gdf):
    """
    for infrastructure items. 0: functional state, 1: dysfunctional state, 
    2: cascaded dysfunctional state
    """
    casc_state = [0]* len(gdf)
    for i in range(len(gdf)):
        if ((gdf.func_tot.iloc[i]==0) & (gdf.func_internal.iloc[i]==0)):
            casc_state[i] = 1
        elif ((gdf.func_tot.iloc[i] ==0) & (gdf.func_internal.iloc[i] >0)):
            casc_state[i] = 2
    return casc_state                                   

def get_accessstates(gdf, node_gdf_orig):
    """
    1 - accessible, 0 - inaccessible from beginning on, -1 - disrupted due to 
    disaster. Careful - Changes gdf entries!
    """
    
    services = [colname for colname in gdf.columns if 'actual_supply_' in colname] 
    for service in services:
        serv_level = gdf[gdf.ci_type=='people'][service].values
        serv_level_orig = node_gdf_orig[node_gdf_orig.ci_type=='people'][service].values
        serv_level[(serv_level==0.) & (serv_level_orig==1.)]= -1.
        gdf.loc[gdf.ci_type=='people', service] = serv_level
    return gdf
 
def get_casc_and_access_states(gdf, gdf_nodes_orig, save_path=None):
    """
    re-calculate cascade-states for now, irrespective of whether it's been calculated.
    """
    if 'casc_state' in gdf.columns:
        print('Cascade and access analysis was already performed. Aborting')
        return gdf
    gdf['casc_state'] = get_cascstate(gdf)
    gdf = get_accessstates(gdf, gdf_nodes_orig)
    if save_path is not None:
        gdf.to_feather(save_path)
    return gdf

# =============================================================================
# Event Pre-Selections
# =============================================================================
 
def select_insignificant_events(disr_rate_dict, cutoff_popimp):
    """
    check which events do not have a "significant enough" impact magnitude, based on 
    fraction of total population impacted.
    """
    drop_events = []
    for event_id, imp_dicts in disr_rate_dict.items():
        if imp_dicts['people'] < cutoff_popimp:
            drop_events.append(event_id)
    return drop_events


# =============================================================================
# Adding up results
# =============================================================================

def sum_impacts(gdf_list, save_path=None):
    # sum up amounts of service failure events across all result gdfs
    services = [colname for colname in gdf_list[0].columns if 'actual_supply_' in colname]
    gdf_services = gdf_list[0][gdf_list[0].ci_type=='people'][['counts', 'geometry']]
    
    for service in services:
        service_counts = np.array(gdf_list[0][gdf_list[0].ci_type=='people'][service].values)
        for gdf in gdf_list[1:]:
            service_counts = np.vstack([service_counts, np.array(gdf[gdf.ci_type=='people'][service].values)])
        failures = np.ma.masked_greater_equal(service_counts, 0).sum(axis=0).filled(np.nan)
        inavails = np.ma.masked_not_equal(service_counts, 0).sum(axis=0).filled(np.nan)
        failures[~np.isnan(inavails)] = 0.
        failures[np.isnan(failures)] = 1.
        gdf_services[service] = failures
    
    impact_counts = np.round(np.array(gdf_list[0][gdf_list[0].ci_type=='people']['imp_dir'].values/
                             gdf_list[0][gdf_list[0].ci_type=='people']['counts'].values), 0)
    for gdf in gdf_list[1:]:
        impact_counts += np.round(np.array(gdf[gdf.ci_type=='people']['imp_dir'].values/gdf[gdf.ci_type=='people']['counts'].values), 0)
    gdf_services['imp_dir'] = impact_counts*-1 
    
    if save_path is not None:
        gdf_services.to_feather(save_path)
        
    return gdf_services


def sum_impacts_popweighted(gdf_summed, inv=False, save_path=None):
    """
    inv : bool
        if true, invert multiplication: disruption numbers are positive, with service are negative
        default: disruption numbers are negative, with service are positive
    """
    factor = -1 if inv else 1
    gdf_summed_popweighted = gdf_summed.apply(lambda row: row[2:]*factor*row.counts, axis=1)
    gdf_summed_popweighted[['counts', 'geometry']] = gdf_summed[['counts', 'geometry']]
    gdf_summed_popweighted['lat'] = gdf_summed_popweighted.geometry.y
    gdf_summed_popweighted['lon'] = gdf_summed_popweighted.geometry.x
    # path_cntry_folder+'summed_pop_impacts_inv.csv'
    if save_path is not None:
        gdf_summed_popweighted.to_feather(save_path)
    return gdf_summed_popweighted



def get_total_imps(event_stat_dict):
    event_keys = list(event_stat_dict.keys())
    total_imps = copy.deepcopy(event_stat_dict[event_keys[0]])
    for service, imp_val in total_imps.items():
        for event in event_keys[1:]:
            total_imps[service]+=event_stat_dict[event][service]
    return total_imps


# =============================================================================
# Meta-Analysis
# =============================================================================

def get_worst_events(event_stats_dict, frac=0.1, n_min=3, mode='service'):
    """
    mode = service or functionality
    """
    if n_min < np.ceil(len(event_stats_dict)*frac):
        n_min = int(np.ceil(len(event_stats_dict)*frac))
    event_sums = {}
    for event_name, serv_dict in event_stats_dict.items():
        if mode=='service':
            event_sums[event_name]=serv_dict['power']+serv_dict['healthcare']+serv_dict['education']+serv_dict['telecom']+serv_dict['mobility']
        elif mode=='functionality':
            event_sums[event_name]=serv_dict['power_line']+serv_dict['health']+serv_dict['education']+serv_dict['celltower']+serv_dict['road']+serv_dict['people']
    return sorted(zip(event_sums.values(),event_sums.keys()), reverse=True)[:n_min]

def compare_impact_rankings(disr_rate_dict, destruction_rate_dict):
    dir_imps = []
    serv_imps = []
    for event_name, serv_dict in disr_rate_dict.items():
        dir_imps.append(serv_dict['power']+serv_dict['healthcare']+serv_dict['education']+serv_dict['telecom']+serv_dict['mobility'])
        serv_imps.append(np.array(list(destruction_rate_dict[event_name].values())).sum())
    return scipy.stats.spearmanr(np.array(dir_imps), np.array(serv_imps)), [scipy.stats.rankdata(dir_imps),
                                                                            scipy.stats.rankdata(serv_imps)]


def get_diverging_events(event_stat_dict, ranklist, threshold=0.2):
    event_names = np.array(list(event_stat_dict.keys()))
    return event_names[np.abs(ranklist[0]-ranklist[1])>(threshold*len(event_names))]

# =============================================================================
# Conversions
# =============================================================================


def access_rate_conversion(base_stat_dict, gdf_nodes_orig, abs_num=False):
    """
    from number of ppl without access to number of poeple with access
    and access rates
    """
    total_pop = gdf_nodes_orig[gdf_nodes_orig.ci_type=='people'].counts.sum()
    access_rate_dict = {}
    factor = 1/total_pop if not abs_num else 1
    for key, value in base_stat_dict.items():
        access_rate_dict[key] = (total_pop-value)* factor
    return access_rate_dict


def disruption_rate_conversion(event_stats_dict, access_num_dict):
    """
    disruption rate: number of people who lost service access relative
    to those with initial service access
    """
    disruption_rate_dict = copy.deepcopy(event_stats_dict)
    for event_name, event_dict in disruption_rate_dict.items():
        for bs, event_imp in event_dict.items():
            disruption_rate_dict[event_name][bs] = event_imp/access_num_dict[bs]
    return disruption_rate_dict

def destruction_rate_conversion(dict_gdfs): 
    """
    destruction rate: fraction of ci compnents that are internally destroyed
    (only numbers, does not account for length of line structures)
    """
    dicts_structimps = {}
    for event_name, gdf in dict_gdfs.items():
        dict_structimps = {}
        for ci_type, ci_name in zip(['power_line','health', 'education', 'celltower', 'road', 'people'],
                                    ['power lines', 'hospitals', 'schools', 'cell towers', 'roads', 'people']):
            dict_structimps[ci_type] = np.round(
                len(gdf[(gdf.ci_type==ci_type)&(gdf.func_internal==0)])/len(gdf[(gdf.ci_type==ci_type)])*100,1)
        dict_structimps['people'] = np.round(gdf[gdf.ci_type=='people'].imp_dir.sum()/gdf[gdf.ci_type=='people'].counts.sum()*100,
                                                 1)
        dicts_structimps[event_name] = dict_structimps
    return dicts_structimps

