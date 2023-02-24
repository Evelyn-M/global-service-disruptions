#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 14:35:00 2023

@author: evelynm
"""
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable


import numpy as np
import analysis.analysis_funcs as af
# =============================================================================
# Visualizing results
# =============================================================================

# Color definitions

class InfraColorMaps:
    def __init__(self):
        self.service_col_dict = {-1. : '#FF5733', 0. : 'grey', 1. : 'green'}
        self.service_col_map = ListedColormap(['#FF5733', 'grey', 'green'])
        self.servicecum_col_dict = {-9. : '#581845', -8. : '#581845',
                                -7. : '#581845', -6. : '#581845',  
                                -5. : '#581845', -4. : '#581845',
                                -3. : '#900C3F', -2. : '#C70039', 
                                -1. : '#FF5733', 0. : 'grey', 
                                 1. : 'green'}
        self.servicecum_col_map = ListedColormap(['#581845',  '#581845',
                                '#581845',  '#581845',  '#581845', '#581845',
                               '#900C3F', '#C70039',  '#FF5733', 'grey', 'green'])
        self.casc_col_dict = {0. : 'blue', 1. : 'magenta', 2. : 'yellow'}
        self.casc_col_map = ListedColormap(['blue','magenta','yellow'])

def _two_slope_norm(vmin=-10, vcenter=0, vmax=1):
    """
    Two Slope Norm example from
    https://matplotlib.org/stable/tutorials/colors/colormapnorms.html
    """
    #cols_access = plt.cm.Greens(np.linspace(0.75, 1, 25))
    cols_access = plt.cm.Greens(np.linspace(0.2, 0.3, 10))
    cols_inavail = plt.cm.Greys(np.linspace(0.2, 0.3, 11))
    cols_disrupt = plt.cm.magma(np.linspace(0, 0.75, 20))
    all_colors = np.vstack((cols_disrupt, cols_inavail, cols_access))
    segment_colmap = colors.LinearSegmentedColormap.from_list('service_states', all_colors)
    divnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    return segment_colmap, divnorm      


def _get_extent(gdf):
    buffer_deg = 0.1
    sub_gdf = gdf[gdf.geometry.type == 'Point']
    return (min(sub_gdf.geometry.x)-buffer_deg, max(sub_gdf.geometry.x)+buffer_deg,
                     min(sub_gdf.geometry.y)-buffer_deg, max(sub_gdf.geometry.y)+buffer_deg)


def service_cumimpact_plot(gdf_cumimpacts, haz_type, save_path=None):
    """
    per basic service, people cluster with and without access to that service
    """
    border = cfeature.NaturalEarthFeature('cultural', 'admin_0_countries', '50m')
    services = [colname for colname in gdf_cumimpacts.columns if 'actual_supply_' in colname]
    services.append('imp_dir')
    #ci_types = set(gdf_cumimpacts.ci_type).difference({'people'})
    f, axes = plt.subplots(3, int(np.ceil(len(services)/3)), 
                           subplot_kw=dict(projection=ccrs.PlateCarree()),
                           figsize=(16,16))

    for service, ax in zip(services, axes.flatten()[:len(services)]):
        #ax.set_extent(_get_extent(gdf_cumimpacts), ccrs.PlateCarree()) # causes segmentation fault with current version of matplotlib for some reason.
        ax.add_feature(border, facecolor='none', edgecolor='0.5')
        vmin = np.min(gdf_cumimpacts.iloc[:,2:].values)
        segment_map, divnorm = _two_slope_norm(vmin=vmin, vcenter=0, vmax=1)
        pcm = ax.scatter(gdf_cumimpacts.geometry.x, 
                         gdf_cumimpacts.geometry.y, 
                         c=gdf_cumimpacts[service], norm=divnorm,
                         cmap=segment_map, transform=ccrs.PlateCarree(),
                         s=0.05)
        cb = f.colorbar(pcm, shrink=0.6, ax=ax)
        tick_list = list(np.arange(vmin,0,5))
        tick_list.extend([0, 1])
        cb.set_ticks(tick_list)
        #plt.show()
        if service != 'imp_dir':
            ax.set_title(f'Disruptions in access to {af.cinames_dict()[service[14:-7]]}', 
                         weight='bold', fontsize=12) 
        else:
            ax.set_title('Direct impact pattern', weight='bold', fontsize=12) 
        
    if len(services)%2>0:
        f.delaxes(axes[2,-1])
    f.subplots_adjust(bottom=0.05, top=0.95)  
                            
    if save_path:
        plt.savefig(f'{save_path}'+f'service_disruptions_cum_{haz_type}.png', 
                    format='png', dpi=300,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)


def plot_rank_correlation(ranklist, spearmanr_res, event_names, haz_type, save_path=None):
    f, ax = plt.subplots(1, 1, figsize=(8,8))
    ax.scatter( ranklist[1], ranklist[0])
    plt.title("rank correlation ")
    plt.xlabel("structural impact rank")
    plt.ylabel("service disruption impact rank")
    plt.axline((min(ranklist[0])-1, min(ranklist[0])-1), slope=1)
    plt.text(min(ranklist[0]), max(ranklist[0]), f'Spearman rank corr. coeff.: {np.round(spearmanr_res[0], 3)}')
    for i, txt in enumerate(event_names):
        ax.annotate(txt, (ranklist[1][i]-1, ranklist[0][i]+0.3))
    if save_path is not None:
        plt.savefig(f'{save_path}'+f'event_rankings_{haz_type}.png', 
                    format='png', dpi=72,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)
    
    plt.show()



def casc_factor_boxplots(df_factor_c, df_factor_b, haz_type, save_path=None):
    """
    make boxplots for failure cascade factors of basic services,
    for two cascade metrics ("factor b and factor c")
    """
    f, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    my_cmap = plt.get_cmap("Set3")
    
    ax1 = axes.flatten()[0]
    ax2 = axes.flatten()[1]
    
    divider = make_axes_locatable(ax1)
    ax1b = divider.new_vertical(size="15%", pad=0.1)
    f.add_axes(ax1b)
    
    divider = make_axes_locatable(ax2)
    ax2b = divider.new_vertical(size="15%", pad=0.1)
    f.add_axes(ax2b)
        
    facc_med = df_factor_c['median'][:-1]
    facb_med = df_factor_b['median'][:-1]
    facc_max = np.nanmax(df_factor_c.values.flatten()[df_factor_c.values.flatten()!=np.inf])
    facb_max = np.nanmax(df_factor_b.values.flatten()[df_factor_b.values.flatten()!=np.inf])
    
    labels = df_factor_c.index.values[:-1]
    
    bplot_c = df_factor_c.iloc[:-1,:].T.boxplot(ax=ax1, grid=False,
                                                patch_artist=True, return_type='both',
                                                medianprops = dict(linestyle='-', linewidth=2, color='k'),
                                                whiskerprops=dict(linestyle='-', linewidth=1, color='k'))
    df_factor_c.iloc[:-1,:].T.boxplot(ax=ax1b, grid=False,
                                                patch_artist=True, return_type='both',
                                                medianprops = dict(linestyle='-', linewidth=2, color='k'),
                                                whiskerprops=dict(linestyle='-', linewidth=1, color='k'))
    
    ax1.set_ylabel('Resilience Cascade Factor', fontsize=16)
    for i in range(len(facc_med)):
        ax1.text(i+0.8, facc_med[i]+0.2, '%.2f' % facc_med[i], 
                 verticalalignment='center', fontsize=16)
    
    ax1_ylim = np.nanmax([9, np.nanmax(np.percentile(df_factor_c.values, 90, axis=1))])
    ax1.set_ylim([0, ax1_ylim])
    ax1b.set_ylim([np.max([9, facc_max-5]), np.max([10, facc_max+1])])
    
    bplot_b = df_factor_b.iloc[:-1,:].T.boxplot(ax=ax2, grid=False,
                                                patch_artist=True, return_type='both',
                                                medianprops = dict(linestyle='-', linewidth=2, color='k'),
                                                whiskerprops=dict(linestyle='-', linewidth=1, color='k'))
    df_factor_b.iloc[:-1,:].T.boxplot(ax=ax2b, grid=False,
                                                patch_artist=True, return_type='both',
                                                medianprops = dict(linestyle='-', linewidth=2, color='k'),
                                                whiskerprops=dict(linestyle='-', linewidth=1, color='k'))
    for i in range(len(facb_med)):
        ax2.text(i+0.8, facb_med[i]+0.2, '%.2f' % facb_med[i], 
                 verticalalignment='center', fontsize=16)
        
    ax2.set_ylabel('Spatial Cascade Factor', fontsize=16)
    
    ax2_ylim = np.nanmax([10, np.nanmax(np.percentile(df_factor_b.values, 90, axis=1))])
    ax2.set_ylim([1, ax2_ylim])
    ax2b.set_ylim([np.max([10, facb_max-5]), np.max([11, facb_max+1])])
    
    ax1.set_xticks([])
    ax1.spines['top'].set_visible(False)
    ax2.set_xticks([])
    ax2.spines['top'].set_visible(False)
    ax1b.set_xticks([])
    ax1b.spines['bottom'].set_visible(False)
    ax2b.set_xticks([])
    ax2b.spines['bottom'].set_visible(False)
    
    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax1b.transAxes, color='k', clip_on=False)
    ax1b.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax1b.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
    kwargs.update(transform=ax1.transAxes)  # switch to the bottom axes
    ax1.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
    
    kwargs = dict(transform=ax2b.transAxes, color='k', clip_on=False)
    ax2b.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax2b.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
        
    # fill with colors
    colors = my_cmap.colors
    for bplot in (bplot_c, bplot_b):
        for patch, whisk, med, color in zip(bplot[1]['boxes'],bplot[1]['whiskers'],bplot[1]['whiskers'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
            patch.set_edgecolor('k') # or try 'black'
            patch.set_linewidth(1)
    legend_elements = [Patch(facecolor=my_cmap.colors[i], alpha=0.6, edgecolor='k',
                         label=list(labels)[i]) for i in range(len(labels))]
    f.legend(handles=legend_elements, frameon=False, fontsize=16,
            bbox_to_anchor=(0.9, 0.1),
            ncol=len(labels))        
    #f.suptitle(f'Cascade factors {cntry}, {haz_type}', fontsize=20)
    
    if save_path is not None:
        plt.savefig(f'{save_path}'+f'cascfactor_boxplots_{haz_type}.png', 
                    format='png', dpi=150,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)
    plt.show()

def plot_relative_impacts_bars(imp_dict_relb, imp_dict_rela, haz_type, cntry,
                               save_path=None):
    """both in one
    relb - rel to base state availability
    rela - rel to direclty affected
    """ 
    f, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    width = 0.8
    x = np.arange(5)
    
    ax1 = axes.flatten()[0]
    ax2 = axes.flatten()[1]
    
    relb_base = imp_dict_relb.pop('people')
    rela_base = imp_dict_rela.pop('people')
    relb_vals = np.round(np.array(list(imp_dict_relb.values())),4)
    rela_vals = np.round(np.array(list(imp_dict_rela.values())),4)
    
    labels = imp_dict_relb.keys()
    
    rects1 = ax1.bar(x, relb_vals, width, color='red', label='rel. to pop. with service access in base state')
    ax1.set_ylabel('Impact Factor')
    ax1.set_xticks(x, labels)
    ax1.plot([0.-width/2, 4+width/2], [relb_base, relb_base], "k--", linewidth=2)
    ax1.legend()
    ax1.bar_label(rects1, padding=3)
    
    rects2 = ax2.bar(x, rela_vals, width, color='blue', label='rel. to directly affected pop.')
    ax2.set_ylabel('Impact Factor')
    ax2.set_xticks(x, labels)
    ax2.plot([0.-width/2, 4+width/2], [rela_base, rela_base], "k--", linewidth=2)
    ax2.legend()
    ax2.bar_label(rects2, padding=3)
    year_range = '1980-2020' if haz_type=='TC' else '2002-2018'
    f.suptitle( f"Average relative service disruptions from {haz_type}s, {cntry}, {year_range}",  fontsize=16)
    f.tight_layout()
    if save_path is not None:
        plt.savefig(f'{save_path}'+f'rel_serv_impacts_bars_{haz_type}.png', 
                    format='png', dpi=72,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)
    plt.show()

def plot_total_impacts_bars(imp_tot_dict, haz_type, cntry, save_path=None):
    f, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    
    width = 0.8
    y = np.arange(len(imp_tot_dict.keys())) 
    imp_vals = np.round(np.array(list(imp_tot_dict.values())),-3)
    
    labels = imp_tot_dict.keys()
    
    rects1 = ax1.barh(y, imp_vals, width, color='gray', label='total amount of people impacted')
    ax1.set_ylabel('Basic Service Type')
    ax1.set_yticks(y, labels)
    ax1.legend()
    ax1.bar_label(rects1, labels=["{:,}".format(int(num)) for num in imp_vals], padding=-5)
    ax1.ticklabel_format(useOffset=False, style='plain', axis='x')
    year_range = '1980-2020' if haz_type=='TC' else '2002-2018'
    f.suptitle( f"Cumulative service disruptions from {haz_type}s, {cntry}, {year_range}",  fontsize=16)
    f.tight_layout()
    if save_path is not None:
        plt.savefig(f'{save_path}'+f'tot_serv_impacts_bars_{haz_type}.png', 
                    format='png', dpi=72,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)
    plt.show()

def plot_worst_tc_tracks(worst_events, save_path=None):
    
    from climada.hazard import TCTracks
   
    
    tr_1 = TCTracks.from_ibtracs_netcdf(storm_id=worst_events[0])
    
    for event_id in worst_events[1:]:
        tr_1.append(TCTracks.from_ibtracs_netcdf(storm_id=event_id).data)

    ax = tr_1.plot();
    ax.get_legend()._loc = 2 # correct legend location
    ax.set_title('worst event tracks');
    ax.figure.tight_layout()
    if save_path is not None:
        ax.figure.savefig(f'{save_path}'+'worst_events_TC.png', 
                    format='png', dpi=150,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)

def plot_worst_floods(worst_events, path_cntry_folder, iso3, save_path=None):
    
    from climada.hazard import Hazard
    
    path_haz = path_cntry_folder+f'flood_{iso3}.hdf5'
    hazards = Hazard('FL').from_hdf5(path_haz)
    hazards = hazards.select(event_names=worst_events)

    ax = hazards.plot_intensity(0)
    ax.set_title(f'Worst flood events {iso3}');
    ax.figure.tight_layout()
    if save_path is not None:
        ax.figure.savefig(f'{save_path}'+'worst_events_FL.png', 
                    format='png', dpi=150,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)
