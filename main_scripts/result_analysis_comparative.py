#!/usr/bin/env python
# coding: utf-8



import geopandas as gpd
import numpy as np
import os
from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable


import analysis.analysis_funcs as af



# =============================================================================
# # Creating Average Impact Cascade Factors
# =============================================================================


# ### Cascade Factor A (all service losses : all directly affected)


haz_type='TC'
# countries
for iso3 in ['ATG', 'BGD', 'CUB', 'GTM', 'HTI', 'KHM', 'LKA', 'MDG', 'MEX', 'MOZ', 'PHL', 'VNM', 'PRI']:
    path_cntry_folder = '/cluster/work/climate/evelynm/nw_outputs/'+f'{iso3}/'
    total_imps = af.load_dict(path_cntry_folder+f'total_service_disruptions_{iso3}_{haz_type}.pkl')
    total_imps_rela = {key : total_imps[key]/total_imps['people'] for key in total_imps.keys()}
    af.save_dict(total_imps_rela, path_cntry_folder+f'total_service_disruptions_rela_{iso3}_{haz_type}.pkl')
# states
for iso3, state in zip(['USA', 'USA', 'USA', 'CHN', 'CHN'], ['Florida', 'Louisiana', 'Texas', 'Fujian', 'Hainan']) :
    path_cntry_folder = '/cluster/work/climate/evelynm/nw_outputs/'+f'{iso3}/'
    total_imps = af.af.load_dict(path_cntry_folder+f'{state}/total_service_disruptions_{iso3}_{haz_type}_{state}.pkl')
    total_imps_rela = {key : total_imps[key]/total_imps['people'] for key in total_imps.keys()}
    af.save_dict(total_imps_rela, path_cntry_folder+f'{state}/total_service_disruptions_rela_{iso3}_{state}_{haz_type}.pkl')



# ### Cascade Factor B (spatial cascade - all service losses : service losses of dir. affected)


haz_type='TC'
# countries
for iso3 in ['ATG', 'BGD', 'CUB', 'GTM', 'HTI', 'KHM', 'LKA', 'MDG', 'MEX', 'MOZ', 'PHL', 'VNM', 'PRI']:
    path_cntry_folder = '/cluster/work/climate/evelynm/nw_outputs/'+f'{iso3}/'
    total_imps_relb={}
    gdf_summed_popweighted = gpd.GeoDataFrame(pd.read_csv(path_cntry_folder+f'summed_pop_impacts_{haz_type}_{iso3}.csv'))
    for variable, service in af.service_dict2().items():
        lost_service = gdf_summed_popweighted[gdf_summed_popweighted[variable]<0][variable].sum()
        sub_gdf = gdf_summed_popweighted[(gdf_summed_popweighted['imp_dir']<0) & (gdf_summed_popweighted[variable]<0)]
        lost_service_while_aff = np.maximum(sub_gdf[variable], sub_gdf['imp_dir']).sum()
        total_imps_relb[service] = np.round(lost_service/lost_service_while_aff,2)
    total_imps_relb['people']=1
    af.save_dict(total_imps_relb, path_cntry_folder+f'total_service_disruptions_relb_{iso3}_{haz_type}.pkl')
# states
for iso3, state in zip(['USA', 'USA', 'USA', 'CHN', 'CHN'], ['Florida', 'Louisiana', 'Texas', 'Fujian', 'Hainan']) :
    path_cntry_folder = '/cluster/work/climate/evelynm/nw_outputs/'+f'{iso3}/'
    total_imps_relb={}
    gdf_summed_popweighted = gpd.GeoDataFrame(pd.read_csv(path_cntry_folder+f'{state}/summed_pop_impacts_{haz_type}_{iso3}_{state}.csv'))
    for variable, service in af.service_dict2().items():
        lost_service = gdf_summed_popweighted[gdf_summed_popweighted[variable]<0][variable].sum()
        sub_gdf = gdf_summed_popweighted[(gdf_summed_popweighted['imp_dir']<0) & (gdf_summed_popweighted[variable]<0)]
        lost_service_while_aff = np.maximum(sub_gdf[variable], sub_gdf['imp_dir']).sum()
        total_imps_relb[service] = np.round(lost_service/lost_service_while_aff,2)
    total_imps_relb['people']=1
    af.save_dict(total_imps_relb, path_cntry_folder+f'{state}/total_service_disruptions_relb_{iso3}_{state}_{haz_type}.pkl')    
                                              


# ### Cascade Factor C (number of service losses : number of service access in dir. affected area)


haz_type='TC'
# countries
for iso3 in ['ATG', 'BGD', 'CUB', 'GTM', 'HTI', 'KHM', 'LKA', 'MDG', 'MEX', 'MOZ', 'PHL', 'VNM', 'PRI']:
    path_cntry_folder = '/cluster/work/climate/evelynm/nw_outputs/'+f'{iso3}/'    
    total_imps_relc={}
    gdf_summed_popweighted = gpd.GeoDataFrame(pd.read_csv(path_cntry_folder+f'summed_pop_impacts_{haz_type}_{iso3}.csv'))
    for variable, service in af.service_dict2().items():
        lost_service = gdf_summed_popweighted[gdf_summed_popweighted[variable]<0][variable].sum()
        # has or had service, in directly affected area
        had_service_affected = gdf_summed_popweighted[(gdf_summed_popweighted['imp_dir']<0) & 
                                                      (gdf_summed_popweighted[variable]!=0)]['imp_dir'].sum()
        total_imps_relc[service] = np.round(lost_service/had_service_affected,2)
    total_imps_relc['people']=1
    af.save_dict(total_imps_relc, path_cntry_folder+f'total_service_disruptions_relc_{iso3}_{haz_type}.pkl')
# states
for iso3, state in zip(['USA', 'USA', 'USA', 'CHN', 'CHN'], ['Florida', 'Louisiana', 'Texas', 'Fujian', 'Hainan']) :
    path_cntry_folder = '/cluster/work/climate/evelynm/nw_outputs/'+f'{iso3}/'
    total_imps_relc={}
    gdf_summed_popweighted = gpd.GeoDataFrame(pd.read_csv(path_cntry_folder+f'{state}/summed_pop_impacts_{haz_type}_{iso3}_{state}.csv'))
    for variable, service in af.service_dict2().items():
        lost_service = gdf_summed_popweighted[gdf_summed_popweighted[variable]<0][variable].sum()
        # has or had service, in directly affected area
        had_service_affected = gdf_summed_popweighted[(gdf_summed_popweighted['imp_dir']<0) & 
                                                      (gdf_summed_popweighted[variable]!=0)]['imp_dir'].sum()
        total_imps_relc[service] = np.round(lost_service/had_service_affected,2)
    total_imps_relc['people']=1
    af.save_dict(total_imps_relc, path_cntry_folder+f'{state}/total_service_disruptions_relc_{iso3}_{state}_{haz_type}.pkl')
                                              


gdf_summed_popweighted = gpd.GeoDataFrame(pd.read_csv(path_cntry_folder+f'summed_pop_impacts_{haz_type}_{iso3}.csv'))


# ## Creating event-wise impact cascade factors

# only for "Factor C"
perevent_factor_c={}

haz_type = 'TC'
for iso3 in ['ATG', 'BGD', 'CUB', 'GTM', 'HTI', 'KHM', 'LKA', 'MDG', 'MEX', 'MOZ',  'PHL', 'VNM', 'PRI']:
    path_cntry_folder = '/cluster/work/climate/evelynm/nw_outputs/'+f'{iso3}/'
    if not os.path.isfile(path_cntry_folder+f'perevent_factor_c_{iso3}_{haz_type}.pkl'):
    # define necessary paths
        path_event_stats = path_cntry_folder+f'service_stats_{haz_type}_{iso3}.pkl'
        dict_event_stats = af.load_dict(path_event_stats)
        dict_gdfs = af.load_gdf_dict(path_cntry_folder, haz_type, valid_events=list(dict_event_stats.keys()))

        perevent_factor_c[iso3] = {}

        for event_id, imp_gdf in dict_gdfs.items():
            perevent_factor_c[iso3][event_id] = {}
            imp_gdf = imp_gdf[imp_gdf.ci_type=='people']
            for variable, service in af.service_dict2().items():
                lost_service = imp_gdf[imp_gdf[variable]<0]['counts'].sum()
                # has or had service, in directly affected area
                had_service_affected = imp_gdf[(imp_gdf['imp_dir']>0) & (imp_gdf[variable]!=0)]['counts'].sum()
                perevent_factor_c[iso3][event_id][service] = np.round(np.abs(lost_service/had_service_affected),2)
                if lost_service==0:
                    perevent_factor_c[iso3][event_id][service] = 0
            perevent_factor_c[iso3][event_id]['population'] = imp_gdf[(imp_gdf['imp_dir']>0)]['counts'].sum()/imp_gdf['counts'].sum()

        # convert to dataframe    
        df_factor_c = pd.DataFrame.from_dict(perevent_factor_c[iso3])
        # get rid of all-0 events
        drop_cols = df_factor_c.columns[[np.where(df_factor_c.apply(lambda col: np.all(col==0)))[0]]].values
        for drop_col in drop_cols:
            df_factor_c.pop(drop_col)
        df_factor_c['median'] = df_factor_c.apply(lambda row: np.median(row.values), axis=1)    
        perevent_factor_c[iso3] = df_factor_c
        df_factor_c.to_csv(path_cntry_folder+f'perevent_factor_c_{iso3}_{haz_type}.pkl')
    else:
        perevent_factor_c[iso3] = pd.read_csv(path_cntry_folder+f'perevent_factor_c_{iso3}_{haz_type}.pkl',
                                             index_col=0)
    print(iso3)
    print(perevent_factor_c[iso3])

for iso3, state in zip(['USA', 'USA', 'USA', 'CHN', 'CHN'], ['Florida', 'Louisiana', 'Texas', 'Fujian', 'Hainan']) :
    # define necessary paths
    path_cntry_folder = '/cluster/work/climate/evelynm/nw_outputs/'+f'{iso3}/'
    path_event_stats = path_cntry_folder+f'{state}/service_stats_{haz_type}_{iso3}_{state}.pkl'
    dict_event_stats = af.load_dict(path_event_stats)
    dict_gdfs = af.load_gdf_dict(path_cntry_folder+f'{state}/', haz_type, valid_events=list(dict_event_stats.keys()))
    
    perevent_factor_c[f'{iso3} {state}'] = {}
    
    for event_id, imp_gdf in dict_gdfs.items():
        perevent_factor_c[f'{iso3} {state}'][event_id] = {}
        imp_gdf = imp_gdf[imp_gdf.ci_type=='people']
        for variable, service in af.service_dict2().items():
            lost_service = imp_gdf[imp_gdf[variable]<0]['counts'].sum()
            # has or had service, in directly affected area
            had_service_affected = imp_gdf[(imp_gdf['imp_dir']>0) & (imp_gdf[variable]!=0)]['counts'].sum()
            perevent_factor_c[f'{iso3} {state}'][event_id][service] = np.round(np.abs(lost_service/had_service_affected),2)
            if lost_service==0:
                perevent_factor_c[f'{iso3} {state}'][event_id][service] = 0
        perevent_factor_c[f'{iso3} {state}'][event_id]['population'] = imp_gdf[(imp_gdf['imp_dir']>0)]['counts'].sum()/imp_gdf['counts'].sum()
    
    # convert to dataframe    
    df_factor_c = pd.DataFrame.from_dict(perevent_factor_c[f'{iso3} {state}'])
    # get rid of all-0 events
    drop_cols = df_factor_c.columns[[np.where(df_factor_c.apply(lambda col: np.all(col==0)))[0]]].values
    for drop_col in drop_cols:
        df_factor_c.pop(drop_col)
    df_factor_c['median'] = df_factor_c.apply(lambda row: np.median(row.values), axis=1)    
    perevent_factor_c[f'{iso3} {state}'] = df_factor_c
    

# save entire dict
af.save_dict(perevent_factor_c, f'/cluster/work/climate/evelynm/nw_outputs/factor_c_eventwise_{haztype}_allcntrs.pkl')


# ### Loading Impact Cascade Factors


# Comparison 1: Cascading factors
haz_type = 'TC'
total_imps_rela={}
total_imps_relb={}
total_imps_relc={}
perevent_factor_c={}

for iso3 in ['ATG', 'BGD', 'CUB', 'GTM', 'HTI', 'KHM', 'LKA', 'MDG', 'MEX', 'MOZ',  'PHL', 'VNM', 'PRI']:
    path_cntry_folder = '/cluster/work/climate/evelynm/nw_outputs/'+f'{iso3}/'
    total_imps_rela[iso3] = af.load_dict( path_cntry_folder+f'total_service_disruptions_rela_{iso3}_{haz_type}.pkl')
    total_imps_relb[iso3] = af.load_dict( path_cntry_folder+f'total_service_disruptions_relb_{iso3}_{haz_type}.pkl')
    total_imps_relc[iso3] = af.load_dict( path_cntry_folder+f'total_service_disruptions_relc_{iso3}_{haz_type}.pkl')
for iso3, state in zip(['USA', 'USA', 'USA', 'CHN', 'CHN'], ['Florida', 'Louisiana', 'Texas', 'Fujian', 'Hainan']) :
    path_cntry_folder = '/cluster/work/climate/evelynm/nw_outputs/'+f'{iso3}/'
    total_imps_rela[f'{iso3} {state}'] = af.load_dict(path_cntry_folder+f'{state}/total_service_disruptions_rela_{iso3}_{state}_{haz_type}.pkl')
    total_imps_relb[f'{iso3} {state}'] = af.load_dict(path_cntry_folder+f'{state}/total_service_disruptions_relb_{iso3}_{state}_{haz_type}.pkl')
    total_imps_relc[f'{iso3} {state}'] = af.load_dict(path_cntry_folder+f'{state}/total_service_disruptions_relc_{iso3}_{state}_{haz_type}.pkl')

perevent_factor_c = af.load_dict( f'/cluster/work/climate/evelynm/nw_outputs/factor_c_eventwise_{haztype}_allcntrs.pkl')


#print(total_imps_relb)
#print(total_imps_rela)
#print(total_imps_relc)


iso3 = 'HTI'
df_factor_c = perevent_factor_c[iso3]
df_factor_c.T.plot(kind='box')


# In[53]:


df_factor_c.T.plot(x='population', y='mobility', kind='scatter')


# In[54]:


df_factor_c.T.plot(x='population', y='electricity', kind='scatter')


# In[55]:


df_factor_c.T.plot(x='population', y='healthcare', kind='scatter')


# In[56]:


df_factor_c.T.plot(x='population', y='education', kind='scatter')


# ## Overview Plots

# In[5]:


def plot_relative_impacts_bars(imp_dict_relb_all, imp_dict_rela_all, imp_dict_relc_all, save_path=None):
    """all in one
    relb - rel to base state availability
    rela - rel to direclty affected
    """ 
    f, axes = plt.subplots(int(len(imp_dict_relb_all.keys())/3), 9, figsize=(15, 0.8*len(imp_dict_relb_all.keys())),
                          sharex=True, sharey=True)
    
    width = 0.8
    x = np.arange(5)
    axes = axes.flatten()
            
    i=0
    for out_a, out_b, out_c in zip( imp_dict_rela_all.items(), imp_dict_relb_all.items(), imp_dict_relc_all.items()):
        
        ax1=axes[i]
        ax2=axes[i+1]
        ax3=axes[i+2]
        
        iso3 = out_a[0]
        imp_dict_rela = out_a[1]
        imp_dict_relb = out_b[1]
        imp_dict_relc = out_c[1]
        
        relb_base = imp_dict_relb.pop('people')
        rela_base = imp_dict_rela.pop('people')
        relc_base = imp_dict_relc.pop('people')
        relb_vals = np.round(np.array(list(imp_dict_relb.values())),2)
        rela_vals = np.round(np.array(list(imp_dict_rela.values())),2)
        relc_vals = np.round(np.array(list(imp_dict_relc.values())),2)
        
        labels = imp_dict_relb.keys()

        rects1 = ax1.bar(x, rela_vals, width, color='red', label='rel. to pop. with service loss in dir. affected area')
        ax1.title.set_text(f'{iso3}')
        ax1.set_xticks(x, labels, rotation=90)
        ax1.plot([0.-width/2, 4+width/2], [relb_base, relb_base], "k--", linewidth=2)
        #ax1.legend()
        ax1.bar_label(rects1, padding=3)

        rects2 = ax2.bar(x, relb_vals, width, color='blue', label='rel. to directly affected pop.')
        ax2.set_xticks(x, labels, rotation=90)
        ax2.title.set_text(f'{iso3}')
        ax2.plot([0.-width/2, 4+width/2], [rela_base, rela_base], "k--", linewidth=2)
        #ax2.legend()
        ax2.bar_label(rects2, padding=3)
        
        rects3 = ax3.bar(x, relc_vals, width, color='green', label='rel. to pop. with initial services in dir. affected ar')
        ax3.set_xticks(x, labels, rotation=90)
        ax3.title.set_text(f'{iso3}')
        ax3.plot([0.-width/2, 4+width/2], [relc_base, relc_base], "k--", linewidth=2)
        #ax2.legend()
        ax3.bar_label(rects3, padding=3)
        i+=3
        
    legend_elements = [Patch(facecolor='r', edgecolor='r',
                         label='Factor A'),
                      Patch(facecolor='b', edgecolor='b',
                         label='Factor B'),
                      Patch(facecolor='g', edgecolor='g',
                         label='Factor C')]
    f.legend(handles=legend_elements, frameon=False, fontsize=16,
            bbox_to_anchor=(0.7, 0),
            ncol=len(labels))    
    if save_path is not None:
        plt.savefig(f'{save_path}.png', 
                    format='png', dpi=72,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)
    plt.tight_layout()
    plt.show()


# In[147]:


plot_relative_impacts_bars(deepcopy(total_imps_relb), deepcopy(total_imps_rela), deepcopy(total_imps_relc), save_path='/cluster/work/climate/evelynm/nw_outputs/impact_factors_TC_allcntries_both')


# In[6]:



def plot_relative_impacts_bars_2(imp_dict_relb_all, save_path=None):
    """both in one
    relb - rel to base state availability
    rela - rel to direclty affected
    """ 
    f, axes = plt.subplots(3,6, figsize=(15, 10),
                          sharex=True, sharey=True)
    
    width = 0.8
    x = np.arange(5)
    axes = axes.flatten()
    my_cmap = plt.get_cmap("Set3") #
    i=0
    for iso3, imp_dict_relb in imp_dict_relb_all.items():
        
        ax1=axes[i] 
        divider = make_axes_locatable(ax1)
        ax2 = divider.new_vertical(size="40%", pad=0.1)
        f.add_axes(ax2)
        
        relb_base = imp_dict_relb.pop('people')
        relb_vals = np.round(np.array(list(imp_dict_relb.values())),2)
        labels = imp_dict_relb.keys()
        
        rects1 = ax1.bar(x, relb_vals, width, color=my_cmap.colors, label='rel. to pop. with service access in base state')
        if (i%6)==0:
            ax1.set_ylabel('Resilience Cascade Factor')
        ax1.set_xticks([])
        ax1.plot([0.-width/2, 4+width/2], [relb_base, relb_base], "k--", linewidth=2)
        ax1.bar_label(rects1, padding=3)
        ax1.set_ylim(0, 5)
        ax1.spines['top'].set_visible(False)
        rects1 = ax2.bar(x, relb_vals, width, color=my_cmap.colors, label='')
        ax2.set_title(f'{iso3}',y=1.0, pad=-14) #title.set_text(
        #plt.rcParams['axes.titley'] = 1.0    # y is in axes-relative coordinates.
        #plt.rcParams['axes.titlepad'] = -14  # pad is in points...
        ax2.plot([0.-width/2, 4+width/2], [relb_base, relb_base], "k--", linewidth=2)
        ax2.set_ylim(9, 11.5)
        if (i%6)!=0:
            ax2.yaxis.set_ticklabels([])
        ax2.bar_label(rects1, padding=3)
        ax2.tick_params(bottom=False, labelbottom=False)
        ax2.spines['bottom'].set_visible(False)
        
        d = .015  # how big to make the diagonal lines in axes coordinates
        # arguments to pass to plot, just so we don't keep repeating them
        kwargs = dict(transform=ax2.transAxes, color='k', clip_on=False)
        ax2.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
        ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
        kwargs.update(transform=ax1.transAxes)  # switch to the bottom axes
        ax1.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
        ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
        i+=1
        
    #f.suptitle( f"Average relative service disruptions from {haz_type}, all countries , {year_range}",  fontsize=16)
    #f.tight_layout()
    remainder=(len(imp_dict_relb_all.keys())%3)*-1
    
    legend_elements = [Patch(facecolor=my_cmap.colors[i], edgecolor=my_cmap.colors[i],
                         label=list(labels)[i]) for i in range(len(labels))]
    f.legend(handles=legend_elements, frameon=False, fontsize=16,
            bbox_to_anchor=(0.9, 0),
            ncol=len(labels))

    if remainder < 0:
        for ax in axes[remainder:]:
            ax.remove()
    if save_path is not None:
        plt.savefig(f'{save_path}.png', 
                    format='png', dpi=150,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)
    plt.tight_layout()
    plt.show()


# In[ ]:


def plot_cascfactor_boxplots(perevent_factor_c, save_path=None):
    """both in one
    relb - rel to base state availability
    rela - rel to direclty affected
    """ 
    f, axes = plt.subplots(3,6, figsize=(15, 10),
                          sharex=True, sharey=True)
    
    width = 0.8
    x = np.arange(5)
    axes = axes.flatten()
    my_cmap = plt.get_cmap("Set3") #
    i=0
    for iso3, imp_dict_relb in imp_dict_relb_all.items():
        
        ax1=axes[i] 
        divider = make_axes_locatable(ax1)
        ax2 = divider.new_vertical(size="40%", pad=0.1)
        f.add_axes(ax2)
        
        relb_base = perevent_factor_c.drop('people')
        relb_vals = np.round(np.array(list(imp_dict_relb.values())),2)
        labels = imp_dict_relb.keys()
        
        rects1 = ax1.bar(x, relb_vals, width, color=my_cmap.colors)
        if (i%6)==0:
            ax1.set_ylabel('Resilience Cascade Factor')
        ax1.set_xticks([])
        ax1.plot([0.-width/2, 4+width/2], [relb_base, relb_base], "k--", linewidth=2)
        ax1.bar_label(rects1, padding=3)
        ax1.set_ylim(0, 5)
        ax1.spines['top'].set_visible(False)
        rects1 = ax2.bar(x, relb_vals, width, color=my_cmap.colors, label='')
        ax2.set_title(f'{iso3}',y=1.0, pad=-14) #title.set_text(
        #plt.rcParams['axes.titley'] = 1.0    # y is in axes-relative coordinates.
        #plt.rcParams['axes.titlepad'] = -14  # pad is in points...
        ax2.plot([0.-width/2, 4+width/2], [relb_base, relb_base], "k--", linewidth=2)
        ax2.set_ylim(9, 11.5)
        if (i%6)!=0:
            ax2.yaxis.set_ticklabels([])
        ax2.bar_label(rects1, padding=3)
        ax2.tick_params(bottom=False, labelbottom=False)
        ax2.spines['bottom'].set_visible(False)
        
        d = .015  # how big to make the diagonal lines in axes coordinates
        # arguments to pass to plot, just so we don't keep repeating them
        kwargs = dict(transform=ax2.transAxes, color='k', clip_on=False)
        ax2.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
        ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
        kwargs.update(transform=ax1.transAxes)  # switch to the bottom axes
        ax1.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
        ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
        i+=1
        
    #f.suptitle( f"Average relative service disruptions from {haz_type}, all countries , {year_range}",  fontsize=16)
    #f.tight_layout()
    remainder=(len(imp_dict_relb_all.keys())%3)*-1
    
    legend_elements = [Patch(facecolor=my_cmap.colors[i], edgecolor=my_cmap.colors[i],
                         label=list(labels)[i]) for i in range(len(labels))]
    f.legend(handles=legend_elements, frameon=False, fontsize=16,
            bbox_to_anchor=(0.9, 0),
            ncol=len(labels))

    if remainder < 0:
        for ax in axes[remainder:]:
            ax.remove()
    if save_path is not None:
        plt.savefig(f'{save_path}.png', 
                    format='png', dpi=150,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)
    plt.tight_layout()
    plt.show()


# In[126]:


plot_relative_impacts_bars_2(deepcopy(total_imps_relc), save_path='/cluster/work/climate/evelynm/nw_outputs/impact_factors_TC_allcntries_c')


# In[127]:


plot_relative_impacts_bars_2(deepcopy(total_imps_relb), save_path='/cluster/work/climate/evelynm/nw_outputs/impact_factors_TC_allcntries_b')


# In[128]:


plot_relative_impacts_bars_2(deepcopy(total_imps_rela), save_path='/cluster/work/climate/evelynm/nw_outputs/impact_factors_TC_allcntries_a')


# In[29]:


dict(zip(total_imps_relb[list(total_imps_relb.keys())[0]].keys(), [[],[],[],[],[]]))





# In[31]:


def scatter_cascade_ranking(imp_dict_relb_all, ref_dict, label, annotate=False, save_path=None):
    summary_dict = {}   
    for iso3, imp_dict in imp_dict_relb_all.items():
        imp_dict.pop('people')
        summary_dict[iso3] = np.array(list(imp_dict.values())).sum()/5
        
    fig, ax = plt.subplots()
    x = [ref_dict[iso3] for iso3 in summary_dict.keys()]
    y = [value for value in summary_dict.values()]
    z = [iso3 for iso3 in imp_dict_relb_all.keys()]
    
    ax.scatter(x,y)
    if annotate:    
        for i, txt in enumerate(z):
            ax.annotate(txt, (x[i]+0.2, y[i]+0.2))
    
    #plt.xticks(np.arange(5.5))
    plt.xlabel(label)
    plt.ylabel('average cascade factor')
    
    if save_path is not None:
        plt.savefig(f'{save_path}.png', 
                    format='png', dpi=72,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)
    plt.show()
                


# In[30]:


scatter_cascade_ranking(deepcopy(total_imps_relc), af.inc_class_dict(), annotate=True, label='World Bank Income Class')


# In[ ]:


scatter_cascade_ranking_perservice(imp_dict_relb_all, ref_dict, label, save_path=None)


# In[145]:


scatter_cascade_ranking(deepcopy(total_imps_relc), af.inc_class_dict(), label='World Bank Income Class')


# In[15]:


scatter_cascade_ranking(deepcopy(total_imps_relb), af.pop_density_dict(), label='Population density')


# In[32]:


scatter_cascade_ranking(deepcopy(total_imps_relc), af.pop_density_dict(), annotate=True, label='Population density')


# In[10]:


import seaborn as sns
def scatter_cascade_ranking_perservice(imp_dict_relb_all, save_path=None):
    
    f, ax = plt.subplots(1,1, figsize=(10, 10),
                          sharex=True, sharey=True)
    
    summary_dict = dict(zip(total_imps_relb[list(total_imps_relb.keys())[0]].keys(), [[],[],[],[],[]]))  
    for service in summary_dict.keys():
        summary_dict[service] = [cntry_dict[service] for cntry_dict in imp_dict_relb_all.values()]
    summary_df = pd.DataFrame.from_dict(summary_dict)
    sns.catplot(summary_df)
    #plt.xticks(np.arange(5))
    #plt.xlabel(label)
    plt.ylabel('Average Cascade factor')
    if save_path is not None:
        plt.savefig(f'{save_path}.png', 
                    format='png', dpi=72,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)
    plt.show()


# In[12]:


import seaborn as sns
def scatter_cascade_ranking_perservice(imp_dict_relb_all, save_path=None):
    
    summary_dict = dict(zip(total_imps_relb[list(total_imps_relb.keys())[0]].keys(), [[],[],[],[],[]]))  
    for service in summary_dict.keys():
        summary_dict[service] = [cntry_dict[service] for cntry_dict in imp_dict_relb_all.values()]
    summary_df = pd.DataFrame.from_dict(summary_dict)
    print(summary_df)
    sns.catplot(summary_df)


# In[147]:


col_list = ['casc_factor_b','casc_factor_c', 'service', 'iso3']
cascade_factors_df = pd.DataFrame(columns=col_list)
for items_b, items_c in zip(total_imps_relb.items(), total_imps_relc.items()):
    cntry_df = pd.DataFrame(columns=col_list)
    cntry_df['casc_factor_b'] = list(items_b[1].values())[:-1]
    cntry_df['casc_factor_c'] = list(items_c[1].values())[:-1]
    cntry_df['service'] = list(cntry_dict.keys())[:-1]
    cntry_df['iso3'] = items_b[0]
    cascade_factors_df = cascade_factors_df.append(cntry_df, ignore_index=True)


# In[148]:


cascade_factors_df['WB_inc_class'] = cascade_factors_df.apply(lambda row: inc_class_dict()[row.iso3], axis=1)


# In[149]:


cascade_factors_df['pop_density'] = cascade_factors_df.apply(lambda row: pop_density_dict()[row.iso3], axis=1)


# In[150]:


sns.catplot(data=cascade_factors_df, x="service", y="casc_factor_b", kind='box', hue="WB_inc_class")


# In[151]:


sns.catplot(data=cascade_factors_df, x="service", y="casc_factor_c", kind='box', hue="WB_inc_class")


# In[152]:


sns.catplot(data=cascade_factors_df, x="WB_inc_class", y="casc_factor_b", kind='box', hue="service")


# In[153]:


sns.catplot(data=cascade_factors_df, x="WB_inc_class", y="casc_factor_c", kind='box', hue="service")


# In[154]:


sns.catplot(data=cascade_factors_df, x="service", y="casc_factor_b", hue="WB_inc_class")


# In[155]:


sns.catplot(data=cascade_factors_df, x="service", y="casc_factor_c", hue="WB_inc_class")


# In[159]:


sns.catplot(data=cascade_factors_df, x="service", y="casc_factor_b", col="WB_inc_class")


# In[158]:


sns.catplot(data=cascade_factors_df, x="service", y="casc_factor_c", col="WB_inc_class")


# In[156]:


sns.scatterplot(data=cascade_factors_df, x="pop_density", y="casc_factor_b", hue="service")


# In[157]:


sns.scatterplot(data=cascade_factors_df, x="pop_density", y="casc_factor_c", hue="service")


# In[34]:


def scatter_cascade_ranking_perservice(imp_dict_relb_all, ref_dict, label, save_path=None):
    
    f, axes = plt.subplots(3,2, figsize=(15, 10),
                          sharex=True, sharey=True)
    axes = axes.flatten()
    
    summary_dict = dict(zip(total_imps_relb[list(total_imps_relb.keys())[0]].keys(), [[],[],[],[],[]]))  
    for service in summary_dict.keys():
        summary_dict[service] = [cntry_dict[service] for cntry_dict in imp_dict_relb_all.values()]
    for ax in axes:
        ax.scatter([ref_dict[iso3] for iso3 in summary_dict.keys()], summary_dict.values())
    #plt.xticks(np.arange(5))
    plt.xlabel(label)
    plt.ylabel('Average Cascade factor')
    if save_path is not None:
        plt.savefig(f'{save_path}.png', 
                    format='png', dpi=72,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)
    plt.show()


# In[35]:


def calc_infra_density(iso3, state=None):
    path_nw_folder = '/cluster/work/climate/evelynm/nw_outputs/'+f'{iso3}/'
    infra_density_dict = {}
    if state:
        path_nw_folder+=f'{state}/'
    df_edges = gpd.read_feather(path_nw_folder+'cis_nw_edges')
    df_nodes = gpd.read_feather(path_nw_folder+'cis_nw_nodes')
    pop_count = df_nodes[df_nodes.ci_type=='people'].counts.sum()
    
    for ci_type in ['health', 'education', 'celltower', 'power_plant']:
        infra_density_dict[ci_type] = len(df_nodes[df_nodes.ci_type==ci_type])/pop_count
    for ci_type in ['power_line', 'road']:
        infra_density_dict[ci_type] = df_edges[df_edges.ci_type=='power_line']['distance'].sum()/pop_count
    return infra_density_dict


# In[37]:


iso3='HTI'
calc_infra_density(iso3, state=None)


# In[38]:


infra_density_dict = {}
for key in total_imps_relc.keys():
    try:
        iso3, state = key.split(' ')
    except ValueError:
        iso3 = key
        state = None
    infra_density_dict[key] = calc_infra_density(iso3, state)

pd.DataFrame.from_dict(infra_density_dict)


# In[39]:


df_density_casc = pd.concat([pd.DataFrame.from_dict(total_imps_relc).T, pd.DataFrame.from_dict(infra_density_dict).T],
                           axis=1)


# In[40]:


df_density_casc


# In[59]:


def scatter_cascade_density(df_density_casc, ci_type, service_type, annotate=False, save_path=None):
        
    fig, ax = plt.subplots()
    if ci_type=='education':
        x = df_density_casc[ci_type].iloc[:,1].values
        y = df_density_casc[service_type].iloc[:,0].values
    else:
        x = df_density_casc[ci_type].values 
        y = df_density_casc[service_type].values
    z = df_density_casc.index.tolist()

    ax.scatter(x,y)
    if annotate:    
        for i, txt in enumerate(z):
            ax.annotate(txt, (x[i]*1.05, y[i]*1.025))
    
    #plt.xticks(np.arange(5.5))
    plt.xlabel(f'infrastructure density {ci_type} (metres or units per cap.)')
    plt.ylabel(f'cascade factor {service_type}')
    
    if save_path is not None:
        plt.savefig(f'{save_path}.png', 
                    format='png', dpi=72,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)
    plt.show()


# In[50]:


scatter_cascade_density(df_density_casc, 'power_line', 'electricity', annotate=True, save_path=None)


# In[51]:


scatter_cascade_density(df_density_casc, 'power_plant', 'electricity', annotate=True, save_path=None)


# In[53]:


scatter_cascade_density(df_density_casc, 'health', 'healthcare', annotate=False, save_path=None)


# In[61]:


scatter_cascade_density(df_density_casc, 'education','education', annotate=True, save_path=None)


# In[63]:


scatter_cascade_density(df_density_casc, 'road','mobility', annotate=True, save_path=None)


# In[ ]:




