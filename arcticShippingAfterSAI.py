#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 13:59:02 2024

@author: arielmor
"""

def discrete_cmap(N, base_cmap=None):
    import matplotlib.pyplot as plt
    import numpy as np

    """Create an N-bin discrete colormap from the specified input map"""
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    #    The following works for string, None, or a colormap instance:
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def get_colormap(colormap,vmins,vmaxs,levs,extend):
    import matplotlib as mpl
    from matplotlib.colors import ListedColormap
    import numpy as np
    import cmocean
    import cmasher as cmr
    from matplotlib import cm
    
    #########################################################
    # create discrete colormaps from existing continuous maps
    # first make default discrete blue-red colormap 
    # replace center colors with white at 0
    #########################################################
    
    if colormap == 'PiYG':
        ## purple-green
        cmap = cm.get_cmap('PiYG', (levs)) 
        newcolors = cmap(np.linspace(0, 1, 256))
        newcolors[120:136, :] = np.array([1, 1, 1, 1])
        cmap = ListedColormap(newcolors)
        
    elif colormap == 'BrBG':
        ## blue-orange
        cmap = mpl.cm.BrBG
        newcolors = cmap(np.linspace(0, 1, 256))
        newcolors[120:136, :] = np.array([1, 1, 1, 1])
        cmap = ListedColormap(newcolors)
        
    elif colormap == 'ice':        
        cmap = cm.get_cmap('cmr.arctic', (levs)) 
        
    elif colormap == 'freeze':
        cmap = cm.get_cmap('cmr.freeze', (levs))
        
    elif colormap == 'freeze_r':
        cmap = cmr.get_sub_cmap('cmr.rainforest', 0, 0.96, N = levs)
        
    elif colormap == 'navigableDays':
        cmap = cmr.get_sub_cmap('cmr.freeze', 0.2, 1, N=levs)
        cmap = cmap.with_extremes(under='black')
        
    elif colormap == 'iceburn':
        cmap = cm.get_cmap('cmr.prinsenvlag', (levs))
        newcolors = cmap(np.linspace(0, 1, 256))
        newcolors[120:136, :] = np.array([1, 1, 1, 1])
        cmap = ListedColormap(newcolors)
        
    bounds = np.linspace(vmins,vmaxs,levs)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend=extend)
    return cmap, bounds, norm


def make_timeseries(numEns,var,lat,lon,latmax,latmin,lonmax,lonmin,dataDict):
    import numpy as np
    import warnings
    # import xarray as xr
        
    if len(lat) == 85: lat = lat[41:]
    
    lengthDictionary = len(dataDict)
    
    if lengthDictionary > 1: ens = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010']
    else: ens = ['001']
    numEns = len(ens)
    
    latmin_ind = int(np.abs(latmin-lat).argmin())
    latmax_ind = int(np.abs(latmax-lat).argmin())+1
    lonmin_ind = int(np.abs(lonmin-lon).argmin())
    lonmax_ind = int(np.abs(lonmax-lon).argmin())+1

    
    # Latitude weighting
    lonmesh,latmesh = np.meshgrid(lon,lat)
    
    # Mask out ocean and non-permafrost land:
    weights2D = {}
    for i in range(numEns):
        weights2D[ens[i]] = np.full((dataDict[ens[i]].shape[0],len(lat),len(lon)),np.nan) 
        for iyear in range(dataDict[ens[0]].shape[0]):
            weights2D[ens[i]][iyear,:,:] = np.cos(np.deg2rad(latmesh))
            weights2D[ens[i]][iyear,:,:][np.isnan(dataDict[ens[i]][iyear,:,:])] = np.nan
    
    # Annual time series for each ensemble member
    ensMemberTS = {}
    for ensNum in range(numEns):
        warnings.simplefilter("ignore")
                                            
        ensMasked         = dataDict[ens[ensNum]]
        ensMasked_grouped = ensMasked[:,latmin_ind:latmax_ind,lonmin_ind:lonmax_ind]
        ensMasked_grouped = np.ma.MaskedArray(ensMasked_grouped, mask=np.isnan(ensMasked_grouped))
        weights           = np.ma.asanyarray(weights2D[ens[ensNum]][
                                :,latmin_ind:latmax_ind,lonmin_ind:lonmax_ind])
        weights.mask      = ensMasked_grouped.mask
        ensMemberTS[ens[ensNum]] = np.array([np.ma.average(
                                            ensMasked_grouped[i],
                                            weights=weights[i]
                                            ) for i in range((ensMasked_grouped.shape)[0])])
    return ensMemberTS



def make_ensemble_mean_timeseries(ensMemberTS,numEns):
    ensMeanTS = 0
    if type(ensMemberTS).__module__ == 'numpy':
        for val in ensMemberTS:
            ensMeanTS += val
        ensMeanTS = ensMeanTS/numEns 
    else:
        for val in ensMemberTS.values():
            ensMeanTS += val
        ensMeanTS = ensMeanTS/numEns 
    return ensMeanTS



def membersAndMeanTimeSeries(controlMems,feedbackMems,feedbacklowerMems,controlMean,feedbackMean,feedbacklowerMean,ymin,ymax,ylabels,title,figDir,figName):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    
    ens = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010']
    
    fig, ax = plt.subplots(1,1, figsize=(9,5))
    for i in range(len(controlMems)):
        ax.plot(np.linspace(2015,2069,55),controlMems[ens[i]],color='xkcd:light red',label='SSP2-4.5',linewidth=0.8)
        ax.plot(np.linspace(2035,2069,35),feedbackMems[ens[i]],color='xkcd:sky blue',label='ARISE-SAI-1.5',linestyle='--',linewidth=0.85)
        ax.plot(np.linspace(2035,2069,35),feedbacklowerMems[ens[i]],color='xkcd:cobalt',label='ARISE-SAI-1.0',linestyle='dotted',linewidth=0.85)
    if type(controlMean) == dict:
        ax.plot(np.linspace(2015,2069,55),controlMean[ens[0]],color='xkcd:dark red',label='SSP2-4.5',linewidth=2)
        ax.plot(np.linspace(2035,2069,35),feedbackMean[ens[0]],color='xkcd:ocean blue',label='ARISE-SAI-1.5',linestyle='--',linewidth=2.5)
        ax.plot(np.linspace(2035,2069,35),feedbacklowerMean[ens[0]],color='xkcd:deep blue',label='ARISE-SAI-1.0',linestyle='dotted',linewidth=2.5)
    else:
        ax.plot(np.linspace(2015,2069,55),controlMean,color='xkcd:dark red',label='SSP2-4.5',linewidth=2)
        ax.plot(np.linspace(2035,2069,35),feedbackMean,color='xkcd:ocean blue',label='ARISE-SAI-1.5',linestyle='--',linewidth=2.5)
        ax.plot(np.linspace(2035,2069,35),feedbacklowerMean,color='xkcd:deep blue',label='ARISE-SAI-1.0',linestyle='dotted',linewidth=2.5)

    
    custom_lines = [Line2D([0], [0], color='xkcd:dark red', lw=2),
                    Line2D([0], [0], color='xkcd:ocean blue', lw=2, linestyle='--'),
                    Line2D([0], [0], color='xkcd:deep blue', lw=2, linestyle='dotted')]
    plt.legend(custom_lines, ['SSP2-4.5','ARISE-1.5','ARISE-1.0'], fancybox=True, fontsize=12, loc='best')
    plt.xlim([2015,2069])
    plt.ylim([ymin,ymax])
    plt.ylabel(str(ylabels), fontsize=13)
    plt.title(str(title), fontweight='bold', fontsize=15)
    plt.yticks(fontsize=13)
    plt.xticks(fontsize=13)
    plt.savefig(figDir + figName + '.pdf',
                dpi=2000, bbox_inches='tight')
    
    return fig, ax



def make_maps(var1,latitude,longitude,vmins,vmaxs,levs,mycmap,label,title,savetitle,extend,seaIce,hatching,varForHatching):
    from matplotlib import colors as c
    cmapLand = c.ListedColormap(['xkcd:gray','none'])
    cmapToPlot, bounds, norm1 = get_colormap(mycmap,vmins,vmaxs,levs,extend)
    
    
    import cartopy.crs as ccrs
    from cartopy.util import add_cyclic_point, add_cyclic
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    import cartopy.feature as cfeature
    import matplotlib.ticker as mticker
    import matplotlib.pyplot as plt
    import matplotlib.path as mpath
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    import matplotlib as mpl
    import numpy as np
    import xarray as xr
    hfont = {'fontname':'Verdana'}
    landmask = np.load('/Users/arielmor/Desktop/SAI/crops/data/landMask_from_global.npy')
    
    #########################################################
    # land mask
    #########################################################
    datadir = '/Users/arielmor/Desktop/SAI/data/ARISE/data'
    figureDir = '/Users/arielmor/Desktop/SAI/ArcticShipping/figures/'
    ds = xr.open_dataset(datadir + '/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001.clm2.h0.ALT.203501-206912_NH.nc')
    lat = ds.lat; lon2 = ds.lon
    ds.close()
    
    
    #########################################################
    # make single North Pole stereographic filled contour map
    #########################################################
    # Add cyclic point
    if seaIce:
        lons2 = np.linspace(0,360,288)
        var, lon = add_cyclic(var1, x=lons2)
    else:
        var,lon = add_cyclic_point(var1,coord=longitude)
    
    if vmins < 0. and vmaxs > 0.:
        norm1 = mcolors.TwoSlopeNorm(vmin=vmins, vcenter=0, vmax=vmaxs)
    else:
        norm1 = mcolors.Normalize(vmin=vmins, vmax=vmaxs)
    
    ## Create figure
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
    ax.set_extent([180, -180, 50, 90], crs=ccrs.PlateCarree())
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.set_facecolor('0.8')
    
    ## field to be plotted
    cf1 = ax.pcolormesh(lon,latitude,var,transform=ccrs.PlateCarree(), 
                  norm=norm1, cmap=cmapToPlot)
    
    ## add hatching if necessary
    if hatching:
        none_map = c.ListedColormap(['none'])
        hatch1 = ax.pcolor(lon, latitude, varForHatching, transform=ccrs.PlateCarree(), cmap=none_map,
                        hatch='X X', edgecolor='k', lw=0, zorder=2)
   
    land_50m = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                    edgecolor='k',
                                    facecolor='0.8',
                                    linewidth=0.4)
    ax.add_feature(land_50m)
    ### for plotting sea ice concentration:
    # cs = plt.contour(lon, latitude, var, [50], colors='r', transform=ccrs.PlateCarree())
    ax.set_extent([180, -180, 60, 90], crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.5)        
    
    ## add lat/lon grid lines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), 
                      draw_labels=True,
                      linewidth=1, color='C7', 
                  alpha=0.8, linestyle=':',
                  x_inline=False,
                  y_inline=True,
                  rotate_labels=False)
    gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120])
    gl.ylocator = mticker.FixedLocator([60, 70, 80])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.ylabel_style = {'size': 11, 'rotation':20}
    plt.draw()  # Enable the use of `gl._labels`

    for ea in gl.label_artists:
        # return ea of mpl.text type, e.g. Text(135, 30, '30°N') 
        pos = ea.get_position()
        ## put labels over ocean and not over land
        if pos[0] == 150:
            ea.set_position([0, pos[1]])
    

    if label == ' ':
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm1, cmap=cmapToPlot),
             ax=ax, orientation='horizontal',
             extend=extend, fraction=0.051,
             ticks=[0,31,61,92,122,153,184,214,245,275,306,337])
        cbar.ax.tick_params(labelsize=12)
        cbar.ax.set_xticklabels(['Mar 1','Apr 1','May 1',
                                        'Jun 1','Jul 1','Aug 1','Sep 1','Oct 1',
                                        'Nov 1','Dec 1','Jan 1','Feb 1'])
        cbar.ax.tick_params(rotation=30)
    else:
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm1, cmap=cmapToPlot),
             ax=ax, orientation='horizontal',
             extend=extend, fraction=0.051) # change orientation if needed
        cbar.ax.tick_params(labelsize=14)
        cbar.ax.tick_params(labelsize=14)
    cbar.set_label(str(label), fontsize=14, fontweight='bold')
    plt.title(str(title), fontsize=16, fontweight='bold', **hfont, y=1.07)
    ## Save figure
    plt.savefig(figureDir + str(savetitle) + '.png', dpi=2000, bbox_inches='tight')
    return fig, ax



def globalLandMask():
    from global_land_mask import globe
    import numpy as np
    
    lat = np.linspace(50.418848, 90., 43)
    lon = np.linspace(-178.75, 180., 288)
    land = np.full((len(lat),len(lon)),np.nan)
    
    for ilat in range(len(lat)):
        for ilon in range(len(lon)):
            land[ilat,ilon] = globe.is_land(lat[ilat],lon[ilon])
    
    land2 = np.concatenate((land[:,143:], land[:,:143]), axis = 1)
                                               
    np.save('/Users/arielmor/Desktop/SAI/ArcticShipping/data/landMask_from_global.npy', land2)
            
    return



def seaIceRegions(figureDir):
    import cartopy.crs as ccrs
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    import cartopy.feature as cfeature
    import matplotlib.ticker as mticker
    import matplotlib.pyplot as plt
    import matplotlib.path as mpath
    import matplotlib.patches as mpatches
    import numpy as np
    
    figureDir = '/Users/arielmor/Desktop/SAI/ArcticShipping/figures/'
    
    ###########################################
    # make single North Pole stereographic map
    ###########################################
    #### above Arctic circle
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
    ax.set_extent([180, -180, 66.5, 90], crs=ccrs.PlateCarree())
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    SOURCE = 'Natural Earth'
    LICENSE = 'public domain'

    ax.add_feature(cfeature.LAND, facecolor='xkcd:light gray')
    ax.add_feature(cfeature.COASTLINE)
    
    ## add lat/lon grid lines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), 
                      draw_labels=True,
                      linewidth=1, color='C7', 
                  alpha=0.8, linestyle=':',
                  x_inline=False,
                  y_inline=True,
                  rotate_labels=False)
    gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120])
    gl.ylocator = mticker.FixedLocator([60, 70, 80])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.ylabel_style = {'size': 9, 'rotation':20}
    plt.draw()  # Enable the use of `gl._labels`

    for ea in gl.label_artists:
        # return ea of mpl.text type, e.g. Text(135, 30, '30°N') 
        pos = ea.get_position()
        ## put labels over ocean and not over land
        if pos[0] == 150:
            ea.set_position([0, pos[1]])
    
    plt.savefig(figureDir + '/above_arctic_circle.pdf', dpi=1200, bbox_inches='tight')
    
    
    #### canada
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal(central_longitude=262.7, cutoff=54,
                                                                   standard_parallels=(33,45)))
    ax.set_extent([180, 350, 60, 80], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='xkcd:light gray')
    ax.add_feature(cfeature.COASTLINE)
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), 
                      draw_labels=True,
                      linewidth=1, color='C7', 
                  alpha=0.8, linestyle=':',
                  x_inline=False,
                  y_inline=True,
                  rotate_labels=False)
    gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120])
    gl.ylocator = mticker.FixedLocator([60, 70, 80])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.ylabel_style = {'size': 9, 'rotation':20}
    plt.draw()  # Enable the use of `gl._labels`

    for ea in gl.label_artists:
        # return ea of mpl.text type, e.g. Text(135, 30, '30°N') 
        pos = ea.get_position()
        ## put labels over ocean and not over land
        if pos[0] == 150:
            ea.set_position([0, pos[1]])
            
    ax.add_patch(mpatches.Rectangle(xy=[190,61],width=140,height=18,
                                    facecolor='none',edgecolor='k',
                                    linewidth=3.5,
                                    transform=ccrs.PlateCarree()))

    plt.savefig(figureDir + '/canada_coast.pdf', dpi=1200, bbox_inches='tight')
    
    
    #### russia
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal(central_longitude=95, cutoff=59,
                                                                   standard_parallels=(33,45)))
    ax.set_extent([0, 195, 60, 80], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='xkcd:light gray')
    ax.add_feature(cfeature.COASTLINE)
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), 
                      draw_labels=True,
                      linewidth=1, color='C7', 
                  alpha=0.8, linestyle=':',
                  x_inline=False,
                  y_inline=True,
                  rotate_labels=False)
    gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120])
    gl.ylocator = mticker.FixedLocator([60, 70, 80])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.ylabel_style = {'size': 9, 'rotation':20}
    plt.draw()  # Enable the use of `gl._labels`

    for ea in gl.label_artists:
        # return ea of mpl.text type, e.g. Text(135, 30, '30°N') 
        pos = ea.get_position()
        ## put labels over ocean and not over land
        if pos[0] == 150:
            ea.set_position([0, pos[1]])
    
    ax.add_patch(mpatches.Rectangle(xy=[0,64],width=190,height=16,
                                    facecolor='none',edgecolor='k',
                                    linewidth=3.5,
                                    transform=ccrs.PlateCarree()))

    plt.savefig(figureDir + '/russia_coast.pdf', dpi=1200, bbox_inches='tight')
    
    return


def seaIceAreaExpt():
    # I/O
    import gc
    
    # analysis
    import numpy as np
    import xarray as xr
    import pandas as pd


    dataDir = '/Users/arielmor/Desktop/SAI/ArcticShipping/data/'
    ens = ['001','002','003','004','005','006','007','008','009','010']
    
    
    landMask = np.load(dataDir + 'landMask_from_global_edited.npy')
    landMask2 = 1-landMask
    landMask = np.where(landMask2 == 0., np.nan, 1.)
    
    landMaskARISE = np.tile(landMask2,(35, 1, 1))
    landMaskSSP   = np.tile(landMask2, (55, 1, 1))
    
    #### read data
    # make empty dictionaries to hold all ensemble members
    iceAreaCONTROL  = {}
    iceAreaFEEDBACK = {}
    iceAreaFEEDBACK_LOWER = {}
    
    
    # loop through each file and store in dictionary
    for ensNum in range(len(ens)):
        ## ARISE-1.5
        ds = xr.open_dataset(dataDir + 
                             'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.' + str(ens[ensNum]) +
                             '.cice.h.aice_only.203501-206912_RG_NH.nc', decode_times=False)
        # get the correct time stamp since xarray has trouble reading it
        ds['time'] = pd.date_range(start=pd.to_datetime("2035-01-01"), 
                                               periods=ds.sizes['time'], freq='MS')
        latIce = ds.lat; lonIce = ds.lon
        iceAreaFEEDBACK[ens[ensNum]] = (ds.aice*100.).groupby('time.year').mean(dim='time', skipna=True) * landMaskARISE
        # iceAreaFEEDBACK[ens[ensNum]] = xr.where(landMask == 1, iceAreaFEEDBACK[ens[ensNum]], np.nan)
        ds.close()
        
        ## ARISE-1.0
        ds = xr.open_dataset(dataDir + 
                             'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-LOWER-0.5.' + str(ens[ensNum]) +
                             '.cice.h.aice_only.203501-206912_RG_NH.nc', decode_times=False)
        # get the correct time stamp since xarray has trouble reading it
        ds['time'] = pd.date_range(start=pd.to_datetime("2035-01-01"), 
                                               periods=ds.sizes['time'], freq='MS')
        latIce = ds.lat; lonIce = ds.lon
        iceAreaFEEDBACK_LOWER[ens[ensNum]] = (ds.aice*100.).groupby('time.year').mean(dim='time', skipna=True) * landMaskARISE
        # iceAreaFEEDBACK_LOWER[ens[ensNum]] = xr.where(landMask == 1, iceAreaFEEDBACK_LOWER[ens[ensNum]], np.nan)
        ds.close()
        
        ## SSP
        ds = xr.open_dataset(dataDir + 
                             'b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.' + str(ens[ensNum]) +
                             '.cice.h.aice_only.201501-206912_RG_NH.nc')
        ds['time'] = pd.date_range(start=pd.to_datetime("2015-01-01"), 
                                               periods=ds.sizes['time'], freq='MS')
        iceAreaCONTROL[ens[ensNum]] = (ds.aice*100.).groupby('time.year').mean(dim='time', skipna=True) * landMaskSSP
        # iceAreaCONTROL[ens[ensNum]] = xr.where(landMask == 1, iceAreaCONTROL[ens[ensNum]], np.nan)
        ds.close()
        gc.collect()
        
    
        
    #### climatologies
    iceAreaCONTROL_diff_pos_ens = {}
    iceAreaCONTROL_diff_neg_ens = {}
    iceAreaCONTROL_diff_zero_ens = {}
    iceAreaFEEDBACK_diff_pos_ens = {}
    iceAreaFEEDBACK_diff_neg_ens = {}
    iceAreaFEEDBACK_diff_zero_ens = {}
    iceAreaFEEDBACK_LOWER_diff_pos_ens = {}
    iceAreaFEEDBACK_LOWER_diff_neg_ens = {}
    iceAreaFEEDBACK_LOWER_diff_zero_ens = {}
    for i in range(len(ens)):
        iceAreaCONTROL_diff_pos_ens[ens[i]] = np.where((np.nanmean(iceAreaCONTROL[ens[i]][-5:,:,:],axis=0) - np.nanmean(
                                                    iceAreaCONTROL[ens[i]][:5,:,:],axis=0)) > 0., 1, np.nan)
        iceAreaCONTROL_diff_neg_ens[ens[i]] = np.where((np.nanmean(iceAreaCONTROL[ens[i]][-5:,:,:],axis=0) - np.nanmean(
                                                    iceAreaCONTROL[ens[i]][:5,:,:],axis=0)) < 0., 1, np.nan)
        iceAreaCONTROL_diff_zero_ens[ens[i]] = np.where((np.nanmean(iceAreaCONTROL[ens[i]][-5:,:,:],axis=0) - np.nanmean(
                                                    iceAreaCONTROL[ens[i]][:5,:,:],axis=0)) == 0., 1, np.nan)
        iceAreaFEEDBACK_diff_pos_ens[ens[i]] = np.where((np.nanmean(iceAreaFEEDBACK[ens[i]][-5:,:,:],axis=0) - np.nanmean(
                                                    iceAreaFEEDBACK[ens[i]][:5,:,:],axis=0)) > 0., 1, np.nan)
        iceAreaFEEDBACK_diff_neg_ens[ens[i]] = np.where((np.nanmean(iceAreaFEEDBACK[ens[i]][-5:,:,:],axis=0) - np.nanmean(
                                                    iceAreaFEEDBACK[ens[i]][:5,:,:],axis=0)) < 0., 1, np.nan)
        iceAreaFEEDBACK_diff_zero_ens[ens[i]] = np.where((np.nanmean(iceAreaFEEDBACK[ens[i]][-5:,:,:],axis=0) - np.nanmean(
                                                    iceAreaFEEDBACK[ens[i]][:5,:,:],axis=0)) == 0., 1, np.nan)
        iceAreaFEEDBACK_LOWER_diff_pos_ens[ens[i]] = np.where((np.nanmean(iceAreaFEEDBACK_LOWER[ens[i]][-5:,:,:],axis=0) - np.nanmean(
                                                    iceAreaFEEDBACK_LOWER[ens[i]][:5,:,:],axis=0)) > 0., 1, np.nan)
        iceAreaFEEDBACK_LOWER_diff_neg_ens[ens[i]] = np.where((np.nanmean(iceAreaFEEDBACK_LOWER[ens[i]][-5:,:,:],axis=0) - np.nanmean(
                                                    iceAreaFEEDBACK_LOWER[ens[i]][:5,:,:],axis=0)) < 0., 1, np.nan)
        iceAreaFEEDBACK_LOWER_diff_zero_ens[ens[i]] = np.where((np.nanmean(iceAreaFEEDBACK_LOWER[ens[i]][-5:,:,:],axis=0) - np.nanmean(
                                                    iceAreaFEEDBACK_LOWER[ens[i]][:5,:,:],axis=0)) == 0., 1, np.nan)
    
        
    iceAreaCONTROL_diff_pos = np.where((np.nansum(np.stack(iceAreaCONTROL_diff_pos_ens.values()),axis=0)) >= 8., 1., np.nan)
    iceAreaCONTROL_diff_neg = np.where((np.nansum(np.stack(iceAreaCONTROL_diff_neg_ens.values()),axis=0)) >= 8., 1., np.nan)
    iceAreaCONTROL_diff_zero = np.where((np.nansum(np.stack(iceAreaCONTROL_diff_zero_ens.values()),axis=0)) >= 8., 1., np.nan)
    iceAreaFEEDBACK_diff_pos = np.where((np.nansum(np.stack(iceAreaFEEDBACK_diff_pos_ens.values()),axis=0)) >= 8., 1., np.nan)
    iceAreaFEEDBACK_diff_neg = np.where((np.nansum(np.stack(iceAreaFEEDBACK_diff_neg_ens.values()),axis=0)) >= 8., 1., np.nan)
    iceAreaFEEDBACK_diff_zero = np.where((np.nansum(np.stack(iceAreaFEEDBACK_diff_zero_ens.values()),axis=0)) >= 8., 1., np.nan)
    iceAreaFEEDBACK_LOWER_diff_pos = np.where((np.nansum(np.stack(iceAreaFEEDBACK_LOWER_diff_pos_ens.values()),axis=0)) >= 8., 1., np.nan)
    iceAreaFEEDBACK_LOWER_diff_neg = np.where((np.nansum(np.stack(iceAreaFEEDBACK_LOWER_diff_neg_ens.values()),axis=0)) >= 8., 1., np.nan)
    iceAreaFEEDBACK_LOWER_diff_zero = np.where((np.nansum(np.stack(iceAreaFEEDBACK_LOWER_diff_zero_ens.values()),axis=0)) >= 8., 1., np.nan)
    
    
    iceAreaCONTROL_agreement = np.nansum(np.stack((iceAreaCONTROL_diff_pos,
                                                  iceAreaCONTROL_diff_neg),axis=0), axis=0)
    iceAreaCONTROL_agreement[iceAreaCONTROL_agreement == 0.] = np.nan
    iceAreaFEEDBACK_agreement = np.nansum(np.stack((iceAreaFEEDBACK_diff_pos,
                                                  iceAreaFEEDBACK_diff_neg),axis=0), axis=0)
    iceAreaFEEDBACK_agreement[iceAreaFEEDBACK_agreement == 0.] = np.nan
    iceAreaFEEDBACK_LOWER_agreement = np.nansum(np.stack((iceAreaFEEDBACK_LOWER_diff_pos,
                                                  iceAreaFEEDBACK_LOWER_diff_neg),axis=0), axis=0)
    iceAreaFEEDBACK_LOWER_agreement[iceAreaFEEDBACK_LOWER_agreement == 0.] = np.nan
    
    
    iceAreaFEEDBACK_climo = np.nanmean(np.stack(iceAreaFEEDBACK.values()),axis=0)#make_ensemble_mean_timeseries(iceAreaFEEDBACK, 10)
    iceAreaFEEDBACK_LOWER_climo = np.nanmean(np.stack(iceAreaFEEDBACK_LOWER.values()),axis=0)
    iceAreaCONTROL_climo  = np.nanmean(np.stack(iceAreaCONTROL.values()),axis=0) #make_ensemble_mean_timeseries(iceAreaCONTROL, 10)
    
    iceAreaFEEDBACK_20352044 = np.nanmean(iceAreaFEEDBACK_climo[:5,:,:],axis=0)#.sel(year=slice('2035', '2044')).mean(dim='year')
    iceAreaFEEDBACK_LOWER_20352044 = np.nanmean(iceAreaFEEDBACK_LOWER_climo[:5,:,:],axis=0)
    iceAreaCONTROL_20352044  = np.nanmean(iceAreaCONTROL_climo[20:25,:,:],axis=0)#.sel(year=slice('2035', '2044')).mean(dim='year')

    iceAreaFEEDBACK_20602069 = np.nanmean(iceAreaFEEDBACK_climo[-5:,:,:],axis=0)#.sel(year=slice('2060', '2069')).mean(dim='year')
    iceAreaFEEDBACK_LOWER_20602069 = np.nanmean(iceAreaFEEDBACK_LOWER_climo[-5:,:,:],axis=0)
    iceAreaCONTROL_20602069  = np.nanmean(iceAreaCONTROL_climo[-5:,:,:],axis=0)#.sel(year=slice('2060', '2069')).mean(dim='year')
    
    
    diffFEEDBACK = iceAreaFEEDBACK_20602069 - iceAreaFEEDBACK_20352044
    diffFEEDBACK_LOWER = iceAreaFEEDBACK_LOWER_20602069 - iceAreaFEEDBACK_LOWER_20352044
    diffCONTROL  = iceAreaCONTROL_20602069 - iceAreaCONTROL_20352044
    
    diffFEEDBACK_ens = {}
    diffFEEDBACK_LOWER_ens = {}
    for ensNum in range(len(ens)):
        diffFEEDBACK_ens[ens[ensNum]] = np.nanmean(iceAreaFEEDBACK[ens[ensNum]][-10:,:,:],axis=0) - np.nanmean(iceAreaFEEDBACK[ens[ensNum]][:10,:,:],axis=0)
        diffFEEDBACK_LOWER_ens[ens[ensNum]] = np.nanmean(iceAreaFEEDBACK_LOWER[ens[ensNum]][-10:,:,:],axis=0) - np.nanmean(iceAreaFEEDBACK_LOWER[ens[ensNum]][:10,:,:],axis=0)
        
    #### climatological maps    
    fig, ax = make_maps(iceAreaCONTROL_20352044, latIce, lonIce, 0, 100, 21, 'ice', 'sea ice concentration (%)', 
                        'a) SSP2-4.5, 2035-2039',
                        'Fig2a_control_ice_fraction_20352044', 'neither', True, False, None)
    
    fig, ax = make_maps(iceAreaCONTROL_20602069, latIce, lonIce, 0, 100, 21, 'ice', 'sea ice concentration (%)', 
                        'b) SSP2-4.5, 2065-2069',
                        'Fig2b_control_ice_fraction_20602069', 'neither', True, False, None)
    
    fig, ax = make_maps(diffCONTROL, latIce, lonIce, -15, 15, 21, 'iceburn', 'sea ice concentration difference (%)', 
                        'c) SSP2-4.5, 2065-2069 minus 2035-2039',
                        'Fig2c_control_ice_fraction_diff', 'both', True, True, iceAreaCONTROL_agreement)
    
    fig, ax = make_maps(iceAreaFEEDBACK_20352044, latIce, lonIce, 0, 100, 21, 'ice', 'sea ice concentration (%)', 
                        'd) ARISE-1.5, 2035-2039',
                        'Fig2d_feedback_ice_fraction_20352044', 'neither', True, False, None)
    
    fig, ax = make_maps(iceAreaFEEDBACK_20602069, latIce, lonIce, 0, 100, 21, 'ice', 'sea ice concentration (%)', 
                        'e) ARISE-1.5, 2065-2069',
                        'Fig2e_feedback_ice_fraction_20602069', 'neither', True, False, None)
    
    fig, ax = make_maps(diffFEEDBACK, latIce, lonIce, -15, 15, 21, 'iceburn', 'sea ice concentration difference (%)', 
                        'f) ARISE-1.5, 2065-2069 minus 2035-2039',
                        'Fig2f_feedback_ice_fraction_diff', 'both', True, True, iceAreaFEEDBACK_agreement)
    
    fig, ax = make_maps(iceAreaFEEDBACK_LOWER_20352044, latIce, lonIce, 0, 100, 21, 'ice', 'sea ice concentration (%)', 
                        'g) ARISE-1.0, 2035-2039',
                        'Fig2g_feedback_lower_ice_fraction_20352044', 'neither', True, False, None)
    
    fig, ax = make_maps(iceAreaFEEDBACK_LOWER_20602069, latIce, lonIce, 0, 100, 21, 'iceburn', 'sea ice concentration (%)', 
                        'h) ARISE-1.0, 2065-2069',
                        'Fig2h_feedback_lower_ice_fraction_20602069', 'ice', True, False, None)
    
    fig, ax = make_maps(diffFEEDBACK_LOWER, latIce, lonIce, -15, 15, 21, 'iceburn', 'sea ice concentration difference (%)', 
                        'i) ARISE-1.0, 2065-2069 minus 2035-2039',
                        'Fig2i_feedback_lower_ice_fraction_diff', 'both', True, True, iceAreaFEEDBACK_LOWER_agreement)
    
    for ensNum in range(len(ens)):
        fig, ax = make_maps(diffFEEDBACK_ens[ens[ensNum]], latIce, lonIce, -15, 15, 21, 'iceburn', 'sea ice concentration difference (%)', 
                            'ARISE-1.5, 2065-2069 minus 2035-2039',
                            'feedback_ice_fraction_diff_' + str(ens[ensNum]), 'both', True, False, None)
        fig, ax = make_maps(diffFEEDBACK_LOWER_ens[ens[ensNum]], latIce, lonIce, -15, 15, 21, 'iceburn', 'sea ice concentration difference (%)', 
                            'ARISE-1.0, 2065-2069 minus 2035-2039',
                            'feedback_lower_ice_fraction_diff_' + str(ens[ensNum]), 'both', True, False, None)
        
        
    #### time series      
    from make_timeseries import make_timeseries
    from plottingFunctions import membersAndMeanTimeSeries
    
    ## Arctic Circle  
    sicTimseriesCONTROL_ens = make_timeseries(10, 'sithick', latIce, lonIce, 91, 66.5, 360, 0, iceAreaCONTROL)
    sicTimseriesFEEDBACK_ens = make_timeseries(10, 'sithick', latIce, lonIce, 91, 66.5, 360, 0, iceAreaFEEDBACK)
    sicTimseriesFEEDBACK_LOWER_ens = make_timeseries(10, 'sithick', latIce, lonIce, 91, 66.5, 360, 0, iceAreaFEEDBACK_LOWER)
    sicTimseriesCONTROL_mean = make_ensemble_mean_timeseries(sicTimseriesCONTROL_ens, 10)
    sicTimseriesFEEDBACK_mean = make_ensemble_mean_timeseries(sicTimseriesFEEDBACK_ens, 10)
    sicTimseriesFEEDBACK_LOWER_mean = make_ensemble_mean_timeseries(sicTimseriesFEEDBACK_LOWER_ens, 10)
    print("% change SSP: ", (sicTimseriesCONTROL_mean[-1] - sicTimseriesCONTROL_mean[20])/sicTimseriesCONTROL_mean[20] * 100.)
    print("% change ARISE: ", (sicTimseriesFEEDBACK_mean[-1] - sicTimseriesFEEDBACK_mean[0])/sicTimseriesFEEDBACK_mean[0] * 100.)
    print("% change ARISE: ", (sicTimseriesFEEDBACK_LOWER_mean[-1] - sicTimseriesFEEDBACK_LOWER_mean[0])/sicTimseriesFEEDBACK_LOWER_mean[0] * 100.)
    
    fig, ax = membersAndMeanTimeSeries(sicTimseriesCONTROL_ens, sicTimseriesFEEDBACK_ens, sicTimseriesFEEDBACK_LOWER_ens,
                                       sicTimseriesCONTROL_mean, sicTimseriesFEEDBACK_mean, sicTimseriesFEEDBACK_LOWER_mean,
                                       25, 60,'concentration (%)', 'a) Sea ice concentration, above Arctic Circle', 
                                       '/Users/arielmor/Desktop/SAI/ArcticShipping/figures/', 
                                       'FigS1a_sic_timeseries_annual_entire_AO')
    
    ## Canada coast
    sicTimseriesCONTROL_canada_ens = make_timeseries(10, 'sithick', latIce, lonIce, 79, 61, 330, 190, iceAreaCONTROL)
    sicTimseriesFEEDBACK_canada_ens = make_timeseries(10, 'sithick', latIce, lonIce, 79, 61, 330, 190, iceAreaFEEDBACK)
    sicTimseriesFEEDBACK_LOWER_canada_ens = make_timeseries(10, 'sithick', latIce, lonIce, 79, 61, 330, 190, iceAreaFEEDBACK_LOWER)
    sicTimseriesCONTROL_canada_mean = make_ensemble_mean_timeseries(sicTimseriesCONTROL_canada_ens, 10)
    sicTimseriesFEEDBACK_canada_mean = make_ensemble_mean_timeseries(sicTimseriesFEEDBACK_canada_ens, 10)
    sicTimseriesFEEDBACK_LOWER_canada_mean = make_ensemble_mean_timeseries(sicTimseriesFEEDBACK_LOWER_canada_ens, 10)
    print("% change SSP: ", (sicTimseriesCONTROL_canada_mean[-1] - sicTimseriesCONTROL_canada_mean[20])/sicTimseriesCONTROL_canada_mean[20] * 100.)
    print("% change ARISE: ", (sicTimseriesFEEDBACK_canada_mean[-1] - sicTimseriesFEEDBACK_canada_mean[0])/sicTimseriesFEEDBACK_canada_mean[0] * 100.)
    print("% change ARISE: ", (sicTimseriesFEEDBACK_LOWER_canada_mean[-1] - sicTimseriesFEEDBACK_LOWER_canada_mean[0])/sicTimseriesFEEDBACK_LOWER_canada_mean[0] * 100.)
    
    
    fig, ax = membersAndMeanTimeSeries(sicTimseriesCONTROL_canada_ens, sicTimseriesFEEDBACK_canada_ens, sicTimseriesFEEDBACK_LOWER_canada_ens,
                                       sicTimseriesCONTROL_canada_mean, sicTimseriesFEEDBACK_canada_mean, sicTimseriesFEEDBACK_LOWER_canada_mean, 
                                       25, 80,'concentration (%)', 'b) Northwest Passage', 
                                       '/Users/arielmor/Desktop/SAI/ArcticShipping/figures/', 
                                       'FigS1b_sic_timeseries_annual_canada_coast')
    
    ## Russia coast
    sicTimseriesCONTROL_russia_ens = make_timeseries(10, 'sithick', latIce, lonIce, 80, 64, 190, 0, iceAreaCONTROL)
    sicTimseriesFEEDBACK_russia_ens = make_timeseries(10, 'sithick', latIce, lonIce, 80, 64, 190, 0, iceAreaFEEDBACK)
    sicTimseriesFEEDBACK_LOWER_russia_ens = make_timeseries(10, 'sithick', latIce, lonIce, 80, 64, 190, 0, iceAreaFEEDBACK_LOWER)
    sicTimseriesCONTROL_russia_mean = make_ensemble_mean_timeseries(sicTimseriesCONTROL_russia_ens, 10)
    sicTimseriesFEEDBACK_russia_mean = make_ensemble_mean_timeseries(sicTimseriesFEEDBACK_russia_ens, 10)
    sicTimseriesFEEDBACK_LOWER_russia_mean = make_ensemble_mean_timeseries(sicTimseriesFEEDBACK_LOWER_russia_ens, 10)
    print("% change SSP: ", (sicTimseriesCONTROL_russia_mean[-1] - sicTimseriesCONTROL_russia_mean[20])/sicTimseriesCONTROL_russia_mean[20] * 100.)
    print("% change ARISE: ", (sicTimseriesFEEDBACK_russia_mean[-1] - sicTimseriesFEEDBACK_russia_mean[0])/sicTimseriesFEEDBACK_russia_mean[0] * 100.)
    print("% change ARISE: ", (sicTimseriesFEEDBACK_LOWER_russia_mean[-1] - sicTimseriesFEEDBACK_LOWER_russia_mean[0])/sicTimseriesFEEDBACK_LOWER_russia_mean[0] * 100.)
    
    
    fig, ax = membersAndMeanTimeSeries(sicTimseriesCONTROL_russia_ens, sicTimseriesFEEDBACK_russia_ens, sicTimseriesFEEDBACK_LOWER_russia_ens, 
                                       sicTimseriesCONTROL_russia_mean, sicTimseriesFEEDBACK_russia_mean, sicTimseriesFEEDBACK_LOWER_russia_mean,
                                       25, 80,'concentration (%)', 'c) Northern Sea Route', 
                                       '/Users/arielmor/Desktop/SAI/ArcticShipping/figures/', 
                                       'FigS1c_sic_timeseries_annual_russia_coast')
    
    return


def seaIceThickExpt():
    # analysis
    import numpy as np
    import xarray as xr
    import pandas as pd
    
    # viz
    from matplotlib import cm

    
    # dataDir = '/Users/arielmor/Desktop/REU/notebooks/data/'
    dataDir = '/Users/arielmor/Desktop/SAI/ArcticShipping/data/'
    ens = ['001','002','003','004','005','006','007','008','009','010']
    
    #### landmask
    landMask = np.load(dataDir + 'landMask_from_global_edited.npy')
    
    
    #### read hi (grid cell mean ice depth)
    hiCONTROL  = {}
    hiFEEDBACK = {}
    hiFEEDBACK_LOWER = {}
    
    
    # loop through each file and store in dictionary
    dataDir = '/Users/arielmor/Desktop/SAI/ArcticShipping/data/'
    ens = ['001','002','003','004','005','006','007','008', '009', '010']
    for ensNum in range(len(ens)):
        ## ARISE
        print(ensNum)
        ds = xr.open_dataset(dataDir + 
                             'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.' + str(ens[ensNum]) +
                             '.cice.h.hi_only.203501-206912_RG_NH.nc')
        # get the correct time stamp since xarray has trouble reading it
        ds['time'] = xr.cftime_range(start = pd.to_datetime("2035-01-01"), 
                                     periods = ds.sizes['time'], freq = "MS", calendar = "noleap")
        latIce = ds.lat; lonIce = ds.lon
        
        # annual mean ice thickness
        hiFEEDBACK[ens[ensNum]] = (ds.hi).groupby('time.year').mean(dim='time', skipna=True)
        hiFEEDBACK[ens[ensNum]] = np.where(landMask == 1., np.nan, hiFEEDBACK[ens[ensNum]])
        ds.close()
        
        ## ARISE-1.0
        dsLOWER = xr.open_dataset(dataDir + 
                             'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-LOWER-0.5.' + str(ens[ensNum]) +
                             '.cice.h.hi_only.203501-206912_RG_NH.nc')
        # get the correct time stamp since xarray has trouble reading it
        dsLOWER['time'] = xr.cftime_range(start = pd.to_datetime("2035-01-01"), 
                                     periods = dsLOWER.sizes['time'], freq = "MS", calendar = "noleap")
       
        # annual mean ice thickness
        hiFEEDBACK_LOWER[ens[ensNum]] = (dsLOWER.hi).groupby('time.year').mean(dim='time', skipna=True)
        hiFEEDBACK_LOWER[ens[ensNum]] = np.where(landMask == 1., np.nan, hiFEEDBACK_LOWER[ens[ensNum]])
        dsLOWER.close()
        
        ## SSP
        ds = xr.open_dataset(dataDir + 
                             'b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.' + str(ens[ensNum]) +
                             '.cice.h.hi_only.201501-206912_RG_NH.nc')
        ds['time'] = xr.cftime_range(start = pd.to_datetime("2015-01-01"), 
                                     periods = ds.sizes['time'], freq = "MS", calendar = "noleap")
        
        # annual mean
        hiCONTROL[ens[ensNum]] = (ds.hi).groupby('time.year').mean(dim='time', skipna=True)
        hiCONTROL[ens[ensNum]] = np.where(landMask == 1., np.nan, hiCONTROL[ens[ensNum]])
        ds.close()
    
    
    
    #### agreement with ens
    hiCONTROL_diff_pos_ens = {}
    hiCONTROL_diff_neg_ens = {}
    hiCONTROL_diff_zero_ens = {}
    hiFEEDBACK_diff_pos_ens = {}
    hiFEEDBACK_diff_neg_ens = {}
    hiFEEDBACK_diff_zero_ens = {}
    hiFEEDBACK_LOWER_diff_pos_ens = {}
    hiFEEDBACK_LOWER_diff_neg_ens = {}
    hiFEEDBACK_LOWER_diff_zero_ens = {}
    for i in range(len(ens)):
        hiCONTROL_diff_pos_ens[ens[i]] = np.where((np.nanmean(hiCONTROL[ens[i]][-5:,:,:],axis=0) - np.nanmean(
                                                    hiCONTROL[ens[i]][:5,:,:],axis=0)) > 0., 1, np.nan)
        hiCONTROL_diff_neg_ens[ens[i]] = np.where((np.nanmean(hiCONTROL[ens[i]][-5:,:,:],axis=0) - np.nanmean(
                                                    hiCONTROL[ens[i]][:5,:,:],axis=0)) < 0., 1, np.nan)
        hiCONTROL_diff_zero_ens[ens[i]] = np.where((np.nanmean(hiCONTROL[ens[i]][-5:,:,:],axis=0) - np.nanmean(
                                                    hiCONTROL[ens[i]][:5,:,:],axis=0)) == 0., 1, np.nan)
        hiFEEDBACK_diff_pos_ens[ens[i]] = np.where((np.nanmean(hiFEEDBACK[ens[i]][-5:,:,:],axis=0) - np.nanmean(
                                                    hiFEEDBACK[ens[i]][:5,:,:],axis=0)) > 0., 1, np.nan)
        hiFEEDBACK_diff_neg_ens[ens[i]] = np.where((np.nanmean(hiFEEDBACK[ens[i]][-5:,:,:],axis=0) - np.nanmean(
                                                    hiFEEDBACK[ens[i]][:5,:,:],axis=0)) < 0., 1, np.nan)
        hiFEEDBACK_diff_zero_ens[ens[i]] = np.where((np.nanmean(hiFEEDBACK[ens[i]][-5:,:,:],axis=0) - np.nanmean(
                                                    hiFEEDBACK[ens[i]][:5,:,:],axis=0)) == 0., 1, np.nan)
        hiFEEDBACK_LOWER_diff_pos_ens[ens[i]] = np.where((np.nanmean(hiFEEDBACK_LOWER[ens[i]][-5:,:,:],axis=0) - np.nanmean(
                                                    hiFEEDBACK_LOWER[ens[i]][:5,:,:],axis=0)) > 0., 1, np.nan)
        hiFEEDBACK_LOWER_diff_neg_ens[ens[i]] = np.where((np.nanmean(hiFEEDBACK_LOWER[ens[i]][-5:,:,:],axis=0) - np.nanmean(
                                                    hiFEEDBACK_LOWER[ens[i]][:5,:,:],axis=0)) < 0., 1, np.nan)
        hiFEEDBACK_LOWER_diff_zero_ens[ens[i]] = np.where((np.nanmean(hiFEEDBACK_LOWER[ens[i]][-5:,:,:],axis=0) - np.nanmean(
                                                    hiFEEDBACK_LOWER[ens[i]][:5,:,:],axis=0)) == 0., 1, np.nan)
    
        
    hiCONTROL_diff_pos = np.where((np.nansum(np.stack(hiCONTROL_diff_pos_ens.values()),axis=0)) >= 8., 1., np.nan)
    hiCONTROL_diff_neg = np.where((np.nansum(np.stack(hiCONTROL_diff_neg_ens.values()),axis=0)) >= 8., 1., np.nan)
    hiCONTROL_diff_zero = np.where((np.nansum(np.stack(hiCONTROL_diff_zero_ens.values()),axis=0)) >= 8., 1., np.nan)
    hiFEEDBACK_diff_pos = np.where((np.nansum(np.stack(hiFEEDBACK_diff_pos_ens.values()),axis=0)) >= 8., 1., np.nan)
    hiFEEDBACK_diff_neg = np.where((np.nansum(np.stack(hiFEEDBACK_diff_neg_ens.values()),axis=0)) >= 8., 1., np.nan)
    hiFEEDBACK_diff_zero = np.where((np.nansum(np.stack(hiFEEDBACK_diff_zero_ens.values()),axis=0)) >= 8., 1., np.nan)
    hiFEEDBACK_LOWER_diff_pos = np.where((np.nansum(np.stack(hiFEEDBACK_LOWER_diff_pos_ens.values()),axis=0)) >= 8., 1., np.nan)
    hiFEEDBACK_LOWER_diff_neg = np.where((np.nansum(np.stack(hiFEEDBACK_LOWER_diff_neg_ens.values()),axis=0)) >= 8., 1., np.nan)
    hiFEEDBACK_LOWER_diff_zero = np.where((np.nansum(np.stack(hiFEEDBACK_LOWER_diff_zero_ens.values()),axis=0)) >= 8., 1., np.nan)
    
    
    hiCONTROL_agreement = np.nansum(np.stack((hiCONTROL_diff_pos,
                                                  hiCONTROL_diff_neg),axis=0), axis=0)
    hiCONTROL_agreement[hiCONTROL_agreement == 0.] = np.nan
    hiFEEDBACK_agreement = np.nansum(np.stack((hiFEEDBACK_diff_pos,
                                                  hiFEEDBACK_diff_neg),axis=0), axis=0)
    hiFEEDBACK_agreement[hiFEEDBACK_agreement == 0.] = np.nan
    hiFEEDBACK_LOWER_agreement = np.nansum(np.stack((hiFEEDBACK_LOWER_diff_pos,
                                                  hiFEEDBACK_LOWER_diff_neg),axis=0), axis=0)
    hiFEEDBACK_LOWER_agreement[hiFEEDBACK_LOWER_agreement == 0.] = np.nan
    
    
    #### ensemble means -  grid-cell mean ice thickness
    hiFEEDBACK_climo = {}
    hiFEEDBACK_climo[ens[0]] = np.nanmean(np.stack((hiFEEDBACK.values())), axis=0)
    hiFEEDBACK_LOWER_climo = {}
    hiFEEDBACK_LOWER_climo[ens[0]] =  np.nanmean(np.stack((hiFEEDBACK_LOWER.values())),axis=0)
    hiCONTROL_climo = {}
    hiCONTROL_climo[ens[0]]  = np.nanmean(np.stack((hiCONTROL.values())),axis=0)
    
    hiFEEDBACK_20352044 = np.nanmean(hiFEEDBACK_climo[ens[0]][:5,:,:],axis=0)#.sel(year=slice('2035', '2044')).mean(dim='year')
    hiFEEDBACK_LOWER_20352044 = np.nanmean(hiFEEDBACK_LOWER_climo[ens[0]][:5,:,:],axis=0)
    hiCONTROL_20352044  = np.nanmean(hiCONTROL_climo[ens[0]][20:25,:,:],axis=0)#.sel(year=slice('2035', '2044')).mean(dim='year')

    hiFEEDBACK_20602069 = np.nanmean(hiFEEDBACK_climo[ens[0]][-5:,:,:],axis=0)#.sel(year=slice('2060', '2069')).mean(dim='year')
    hiFEEDBACK_LOWER_20602069 = np.nanmean(hiFEEDBACK_LOWER_climo[ens[0]][-5:,:,:],axis=0)
    hiCONTROL_20602069  = np.nanmean(hiCONTROL_climo[ens[0]][-5:,:,:],axis=0)#(year=slice('2060', '2069')).mean(dim='year')
    
    hiDiffFEEDBACK = hiFEEDBACK_20602069 - hiFEEDBACK_20352044
    hiDiffFEEDBACK_LOWER = hiFEEDBACK_LOWER_20602069 - hiFEEDBACK_LOWER_20352044
    hiDiffCONTROL  = hiCONTROL_20602069 - hiCONTROL_20352044
    

    
    #### annual mean maps
    fig, ax = make_maps(hiCONTROL_20352044, latIce, lonIce, 0, 2, 21, 'ice', 'sea ice thickness (m)', 
                        'a) SSP2-4.5, 2035-2039',
                        'Fig3a_control_ice_thickness_20352044', 'max', True, False, None)
    
    fig, ax = make_maps(hiCONTROL_20602069, latIce, lonIce, 0, 2, 21, 'ice', 'sea ice thickness (m)', 
                        'b) SSP2-4.5, 2065-2069',
                        'Fig3b_control_ice_thickness_20602069', 'max', True, False, None)
    
    fig, ax = make_maps(hiDiffCONTROL, latIce, lonIce, -0.5, 0.5, 21, 'iceburn', 'sea ice thickness difference (m)', 
                        'c) SSP2-4.5, 2065-2069 minus 2035-2039',
                        'Fig3c_control_ice_thickness_diff', 'both', True, True, hiCONTROL_agreement)
    
    fig, ax = make_maps(hiFEEDBACK_20352044, latIce, lonIce, 0, 2, 21, 'ice', 'sea ice thickness (m)', 
                        'd) ARISE-1.5, 2035-2039',
                        'Fig3d_feedback_ice_thickness_20352044', 'max', True, False, None)
    
    fig, ax = make_maps(hiFEEDBACK_20602069, latIce, lonIce, 0, 2, 21, 'ice', 'ice %', 
                        'e) ARISE-1.5, 2065-2069',
                        'Fig3e_feedback_ice_thickness_20602069', 'max', True, False, None)
    
    fig, ax = make_maps(hiDiffFEEDBACK, latIce, lonIce, -0.5, 0.5, 21, 'iceburn', 'sea ice thickness difference (m)', 
                        'f) ARISE-1.5, 2065-2069 minus 2035-2039',
                        'Fig3f_feedback_ice_thickness_diff', 'both', True, True, hiFEEDBACK_agreement)
    
    fig, ax = make_maps(hiFEEDBACK_LOWER_20352044, latIce, lonIce, 0, 2, 21, 'ice', 'ice %', 
                        'g) ARISE-1.0, 2035-2039',
                        'Fig3g_feedback_lower_ice_thickness_20352044', 'max', True, False, None)
    
    fig, ax = make_maps(hiFEEDBACK_LOWER_20602069, latIce, lonIce, 0, 2, 21, 'ice', 'ice %', 
                        'h) ARISE-1.0, 2065-2069',
                        'Fig3h_feedback_lower_ice_thickness_20602069', 'max', True, False, None)
    
    fig, ax = make_maps(hiDiffFEEDBACK_LOWER, latIce, lonIce, -0.5, 0.5, 21, 'iceburn', 'sea ice thickness difference (m)', 
                        'i) ARISE-1.0, 2065-2069 minus 2035-2039',
                        'Fig3i_feedback_lower_ice_thickness_diff', 'both', True, True, hiFEEDBACK_LOWER_agreement)
    
    
    #### time series
    hiTimeseriesCONTROL_ens = make_timeseries(10, 'hi', latIce, lonIce, 
                                              92, 66.5, 360, 0, hiCONTROL)
    hiTimeseriesCONTROL_mean = make_ensemble_mean_timeseries(hiTimeseriesCONTROL_ens,10)
    
    hiTimeseriesFEEDBACK_ens = make_timeseries(10, 'hi', latIce, lonIce,
                                               92, 66.5, 360, 0, hiFEEDBACK)
    hiTimeseriesFEEDBACK_mean = make_ensemble_mean_timeseries(hiTimeseriesFEEDBACK_ens, 10)
    
    hiTimeseriesFEEDBACK_LOWER_ens = make_timeseries(10, 'hi', latIce, lonIce, 
                                                     92, 66.5, 360, 0, hiFEEDBACK_LOWER)
    hiTimeseriesFEEDBACK_LOWER_mean = make_ensemble_mean_timeseries(hiTimeseriesFEEDBACK_LOWER_ens, 10)
    
    ## Arctic Circle  
    hiTimseriesCONTROL_ens = make_timeseries(10, 'sithick', latIce, lonIce, 91, 66.5, 360, 0, hiCONTROL)
    hiTimseriesFEEDBACK_ens = make_timeseries(10, 'sithick', latIce, lonIce, 91, 66.5, 360, 0, hiFEEDBACK)
    hiTimseriesFEEDBACK_LOWER_ens = make_timeseries(10, 'sithick', latIce, lonIce, 91, 66.5, 360, 0, hiFEEDBACK_LOWER)
    hiTimseriesCONTROL_mean = make_ensemble_mean_timeseries(hiTimseriesCONTROL_ens, 10)
    hiTimseriesFEEDBACK_mean = make_ensemble_mean_timeseries(hiTimseriesFEEDBACK_ens, 10)
    hiTimseriesFEEDBACK_LOWER_mean = make_ensemble_mean_timeseries(hiTimseriesFEEDBACK_LOWER_ens, 10)
    print("% change SSP: ", (np.mean(hiTimseriesCONTROL_mean[50:]) - np.mean(hiTimseriesCONTROL_mean[20:25]))/(np.mean(hiTimseriesCONTROL_mean[20:25])) * 100.)
    print("% change ARISE: ", (np.mean(hiTimseriesFEEDBACK_mean[30:]) - np.mean(hiTimseriesFEEDBACK_mean[:5]))/(np.mean(hiTimseriesFEEDBACK_mean[:5])) * 100.)
    print("% change ARISE: ", (np.mean(hiTimseriesFEEDBACK_LOWER_mean[30:]) - np.mean(hiTimseriesFEEDBACK_LOWER_mean[:5]))/(np.mean(hiTimseriesFEEDBACK_LOWER_mean[:5])) * 100.)
    
    fig, ax = membersAndMeanTimeSeries(hiTimseriesCONTROL_ens, hiTimseriesFEEDBACK_ens, hiTimseriesFEEDBACK_LOWER_ens,
                                       hiTimseriesCONTROL_mean, hiTimseriesFEEDBACK_mean, hiTimseriesFEEDBACK_LOWER_mean,
                                       0, 1.5,'thickness (m)', 'b) Sea ice thickness, above Arctic Circle', 
                                       '/Users/arielmor/Desktop/SAI/ArcticShipping/figures/', 
                                       'Fig3a_hi_timeseries_annual_entire_AO')
    
    ## Canada coast
    hiTimseriesCONTROL_canada_ens = make_timeseries(10, 'sithick', latIce, lonIce, 79, 61, 330, 190, hiCONTROL)
    hiTimseriesFEEDBACK_canada_ens = make_timeseries(10, 'sithick', latIce, lonIce, 79, 61, 330, 190, hiFEEDBACK)
    hiTimseriesFEEDBACK_LOWER_canada_ens = make_timeseries(10, 'sithick', latIce, lonIce, 79, 61, 330, 190, hiFEEDBACK_LOWER)
    hiTimseriesCONTROL_canada_mean = make_ensemble_mean_timeseries(hiTimseriesCONTROL_canada_ens, 10)
    hiTimseriesFEEDBACK_canada_mean = make_ensemble_mean_timeseries(hiTimseriesFEEDBACK_canada_ens, 10)
    hiTimseriesFEEDBACK_LOWER_canada_mean = make_ensemble_mean_timeseries(hiTimseriesFEEDBACK_LOWER_canada_ens, 10)
    print("% change SSP: ", (np.mean(hiTimseriesCONTROL_canada_mean[50:]) - np.mean(hiTimseriesCONTROL_canada_mean[20:25]))/(np.mean(hiTimseriesCONTROL_canada_mean[20:25])) * 100.)
    print("% change ARISE: ", (np.mean(hiTimseriesFEEDBACK_canada_mean[30:]) - np.mean(hiTimseriesFEEDBACK_canada_mean[:5]))/(np.mean(hiTimseriesFEEDBACK_canada_mean[:5])) * 100.)
    print("% change ARISE: ", (np.mean(hiTimseriesFEEDBACK_LOWER_canada_mean[30:]) - np.mean(hiTimseriesFEEDBACK_LOWER_canada_mean[:5]))/(np.mean(hiTimseriesFEEDBACK_LOWER_canada_mean[:5])) * 100.)
    
    
    fig, ax = membersAndMeanTimeSeries(hiTimseriesCONTROL_canada_ens, hiTimseriesFEEDBACK_canada_ens, hiTimseriesFEEDBACK_LOWER_canada_ens,
                                       hiTimseriesCONTROL_canada_mean, hiTimseriesFEEDBACK_canada_mean, hiTimseriesFEEDBACK_LOWER_canada_mean, 
                                       0, 1.5,'thickness (m)', 'b) Northwest Passage', 
                                       '/Users/arielmor/Desktop/SAI/ArcticShipping/figures/', 
                                       'Fig3b_hi_timeseries_annual_canada_coast')
    
    ## Russia coast
    hiTimseriesCONTROL_russia_ens = make_timeseries(10, 'sithick', latIce, lonIce, 80, 64, 190, 0, hiCONTROL)
    hiTimseriesFEEDBACK_russia_ens = make_timeseries(10, 'sithick', latIce, lonIce, 80, 64, 190, 0, hiFEEDBACK)
    hiTimseriesFEEDBACK_LOWER_russia_ens = make_timeseries(10, 'sithick', latIce, lonIce, 80, 64, 190, 0, hiFEEDBACK_LOWER)
    hiTimseriesCONTROL_russia_mean = make_ensemble_mean_timeseries(hiTimseriesCONTROL_russia_ens, 10)
    hiTimseriesFEEDBACK_russia_mean = make_ensemble_mean_timeseries(hiTimseriesFEEDBACK_russia_ens, 10)
    hiTimseriesFEEDBACK_LOWER_russia_mean = make_ensemble_mean_timeseries(hiTimseriesFEEDBACK_LOWER_russia_ens, 10)
    print("% change SSP: ", (np.mean(hiTimseriesCONTROL_russia_mean[50:]) - np.mean(hiTimseriesCONTROL_russia_mean[20:25]))/(np.mean(hiTimseriesCONTROL_russia_mean[20:25])) * 100.)
    print("% change ARISE: ", (np.mean(hiTimseriesFEEDBACK_russia_mean[30:]) - np.mean(hiTimseriesFEEDBACK_russia_mean[:5]))/(np.mean(hiTimseriesFEEDBACK_russia_mean[:5])) * 100.)
    print("% change ARISE: ", (np.mean(hiTimseriesFEEDBACK_LOWER_russia_mean[30:]) - np.mean(hiTimseriesFEEDBACK_LOWER_russia_mean[:5]))/(np.mean(hiTimseriesFEEDBACK_LOWER_russia_mean[:5])) * 100.)
    
    
    fig, ax = membersAndMeanTimeSeries(hiTimseriesCONTROL_russia_ens, hiTimseriesFEEDBACK_russia_ens, hiTimseriesFEEDBACK_LOWER_russia_ens, 
                                       hiTimseriesCONTROL_russia_mean, hiTimseriesFEEDBACK_russia_mean, hiTimseriesFEEDBACK_LOWER_russia_mean,
                                       0, 1.5,'thickness (m)', 'c) Northern Sea Route', 
                                       '/Users/arielmor/Desktop/SAI/ArcticShipping/figures/', 
                                       'Fig3c_hi_timeseries_annual_russia_coast')
    
    return



def ice_numeral_PC6(icethic, cat3, cat4, cat5):
    import numpy as np
    IN = np.zeros_like(icethic)
    totalIceConc = cat3 + cat4 + cat5

    mask_no_ice = np.isnan(icethic)
    mask_thin_ice = (icethic <= 1.2) & (~mask_no_ice)
    mask_thick_ice = (icethic > 1.2) & (~mask_no_ice)
    mask_little_multi_year_ice = (totalIceConc <= 0.05) & mask_thin_ice

    IN[mask_no_ice] = 1  # no ice
    IN[mask_thin_ice & mask_little_multi_year_ice] = 1  # very little multi-year ice
    IN[mask_thin_ice & ~mask_little_multi_year_ice] = 0
    IN[mask_thick_ice] = 0
    return IN

def ice_numeral_OW(icethic,cat2,cat3,cat4,cat5):
    import numpy as np
    IN = np.zeros_like(icethic)
    totalIceConc = cat2+cat3+cat4+cat5
    
    mask_no_ice = np.isnan(icethic)
    mask_thin_ice = (icethic <= 0.15) & (~mask_no_ice)
    mask_thick_ice = (icethic > 0.15) & (~mask_no_ice)
    mask_little_multi_year_ice = (totalIceConc <= 0.005) & mask_thin_ice
    
    IN[mask_no_ice] = 1  # no ice
    IN[mask_thin_ice & mask_little_multi_year_ice] = 1  # very little multi-year ice
    IN[mask_thin_ice & ~mask_little_multi_year_ice] = 0
    IN[mask_thick_ice] = 0
           
    return IN



def readDailyIce(simulation):
    import xarray as xr
    import numpy  as np
    
    dataDir = '/Users/arielmor/Desktop/SAI/ArcticShipping/data/'
    ens = ['001','002','003','004','005','006','007','008','009','010']
    
    years = ['2035','2036','2037','2038','2039','2040',
             '2041','2042','2043','2044','2045','2046','2047','2048','2049','2050',
             '2051','2052','2053','2054','2055','2056','2057','2058','2059','2060',
             '2061','2062','2063','2064','2065','2066','2067','2068','2069']
     
    if simulation == 'arise-1.5': 
        simName = 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.'
        print(simName)
        timePeriod = '20350101-20691231'
        startDate = '2035-01-01'
    elif simulation == 'ssp245':
        simName = 'b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.'
        print(simName)
        timePeriod = '20150101-20691231'
        startDate = '2015-01-01'
    elif simulation == 'arise-1.0':
        simName = 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-LOWER-0.5.'
        print(simName)
        timePeriod = '20350101-20691231'
        startDate = '2035-01-01'
        
    
    #### ice thickness
    dailyIceThick           = {}
    dailyIceThickAll        = {}
    for i in range(len(ens)):
        ds = xr.open_dataset(dataDir + str(simName) + str(ens[i]) + '.cice.h1.hi_d_only.' + str(timePeriod) + '_RG_NH.nc',
                             decode_times=False)
        ds['time'] = xr.cftime_range(start = startDate, 
                                     periods = ds.sizes['time'], freq = "D", calendar = "noleap")
        latIce = ds.lat; lonIce = ds.lon
        dailyIceThickAll[ens[i]] = ds.hi_d 
        
        
        dailyIceThick[ens[i]] = np.full((len(years),365,len(latIce),len(lonIce)), np.nan)
        for iyear in range(len(years)):
            dailyIceThick[ens[i]][iyear,:,:,:] = dailyIceThickAll[ens[i]].sel(
                                                time=slice(str(years[iyear] + '-01-01'), str(years[iyear] + '-12-31')))
        ds.close()
    
    #### ice fraction 
    dailyIceFrac           = {}
    dailyIceFracAll        = {}
    for i in range(len(ens)):
        ds = xr.open_dataset(dataDir + str(simName) + str(ens[i]) + '.cice.h1.aice_d_only.' + str(timePeriod) + '_RG_NH.nc',
                             decode_times=False)
        ds['time'] = xr.cftime_range(start = startDate, 
                                     periods = ds.sizes['time'], freq = "D", calendar = "noleap")
        dailyIceFracAll[ens[i]] = ds.aice_d
        
        dailyIceFrac[ens[i]] = np.full((len(years),365,len(latIce),len(lonIce)), np.nan)
        for iyear in range(len(years)):
            dailyIceFrac[ens[i]][iyear,:,:,:] = dailyIceFracAll[ens[i]].sel(
                                                time=slice(str(years[iyear] + '-01-01'), str(years[iyear] + '-12-31')))
        ds.close()
    
    print(dailyIceFrac[ens[2]].shape)         
    return latIce,lonIce,dailyIceThick,dailyIceFrac



def readDailyIceForNavigableDays(simulation):
    import xarray as xr
    
    dataDir = '/Users/arielmor/Desktop/SAI/ArcticShipping/data/'
    ens =  ens = ['001','002','003','004','005','006','007','008','009','010']
    
        
    if simulation == 'arise-1.5': 
        simName = 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.'
        print(simName)
        timePeriod = '20350101-20691231'
        startDate = '2035-01-01'
    elif simulation == 'ssp245':
        simName = 'b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.'
        print(simName)
        timePeriod = '20150101-20691231'
        startDate = '2015-01-01'
    elif simulation == 'arise-1.0':
        simName = 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-LOWER-0.5.'
        print(simName)
        timePeriod = '20350101-20691231'
        startDate = '2035-01-01'
        
    
    dailyIceThick_20352039 = {}
    dailyIceThick_20652069 = {}
    for i in range(len(ens)):
        ds = xr.open_dataset(dataDir + str(simName) + str(ens[i]) + '.cice.h1.hi_d_only.' + str(timePeriod) + '_RG_NH.nc',
                             decode_times=False)
        ds['time'] = xr.cftime_range(start = startDate, 
                                     periods = ds.sizes['time'], freq = "D", calendar = "noleap")
        latIce = ds.lat; lonIce = ds.lon
        
        if simulation == 'ssp245':
            dailyIceThick_20352039[ens[i]] = (ds.hi_d[7300:9125,:,:]).groupby('time.dayofyear').mean('time', skipna=True)
        else:    
            dailyIceThick_20352039[ens[i]] = (ds.hi_d[:1825,:,:]).groupby('time.dayofyear').mean('time', skipna=True)
        
        dailyIceThick_20652069[ens[i]] = (ds.hi_d[-1825:,:,:]).groupby('time.dayofyear').mean('time', skipna=True)
        ds.close()
        
    dailyIceThick_20352040 = {}
    dailyIceThick_20642069 = {}
    for i in range(len(ens)):
        ds = xr.open_dataset(dataDir + str(simName) + str(ens[i]) + '.cice.h1.hi_d_only.' + str(timePeriod) + '_RG_NH.nc',
                             decode_times=False)
        ds['time'] = xr.cftime_range(start = startDate, 
                                     periods = ds.sizes['time'], freq = "D", calendar = "noleap")
        latIce = ds.lat; lonIce = ds.lon
        
        if simulation == 'ssp245':
            dailyIceThick_20352040[ens[i]] = (ds.hi_d[7300:9490,:,:]).groupby('time.dayofyear').mean('time', skipna=True)
        else:    
            dailyIceThick_20352040[ens[i]] = (ds.hi_d[:2190,:,:]).groupby('time.dayofyear').mean('time', skipna=True)
        
        dailyIceThick_20642069[ens[i]] = (ds.hi_d[-2190:,:,:]).groupby('time.dayofyear').mean('time', skipna=True)
        ds.close()
    
    
    #### season length - April to following March
    years = ['2035','2036','2037','2038','2039','2040']
    years2 = ['2064','2065','2066','2067','2068','2069']
             
    
    dailyIceThickForPersistence_20352040 = {}
    dailyIceThickForPersistence_20642069 = {}
   
    for i in range(len(ens)):
        ds = xr.open_dataset(dataDir + str(simName) + str(ens[i]) + '.cice.h1.hi_d_only.' + str(timePeriod) + '_RG_NH.nc',
                             decode_times=False)
        ds['time'] = xr.cftime_range(start = startDate, 
                                     periods = ds.sizes['time'], freq = "D", calendar = "noleap")
        latIce = ds.lat; lonIce = ds.lon
        
        for iyear in range(len(years)-1):
            if simulation == 'ssp245':
                dailyIceThickForPersistence_20352040[ens[i]] = (ds.hi_d[7300:9490,:,:]).sel(
                                        time=slice(str(years[iyear])+ "-03-01", str(years[iyear+1]) + "-02-28")).groupby(
                                            'time.dayofyear').mean('time', skipna=True)
            else:    
                dailyIceThickForPersistence_20352040[ens[i]] = (ds.hi_d[:2190,:,:]).sel(
                                        time=slice(str(years[iyear])+ "-03-01", str(years[iyear+1]) + "-02-28")).groupby(
                                            'time.dayofyear').mean('time', skipna=True)
            
            dailyIceThickForPersistence_20642069[ens[i]] = (ds.hi_d[-2190:,:,:]).sel(
                                    time=slice(str(years2[iyear])+ "-03-01", str(years2[iyear+1]) + "-02-28")).groupby(
                                        'time.dayofyear').mean('time', skipna=True)
        ds.close()
        
    
    return latIce,lonIce,dailyIceThick_20352039,dailyIceThick_20652069,dailyIceThickForPersistence_20352040,dailyIceThickForPersistence_20642069



def interpolateDailyIceForNavigableDays(simulation):
    import numpy as np
    
    #### read daily ice
    latIce,lonIce,dailyIceThick_20352039,dailyIceThick_20652069,dailyIceThickForPersistence_20352040,dailyIceThickForPersistence_20642069 = readDailyIceForNavigableDays(simulation)
    
    
    #### ENS MEAN: 2035-2039 interpolate ice thickness across small channels
    dailyIceThickMean = np.nanmean((np.stack(dailyIceThick_20352039.values())),axis=0) 
    
    a = np.full((365,43,288),np.nan)
    for iday in range(365): 
        nan_indices = np.argwhere(np.isnan(dailyIceThickMean[iday,:,:]))
        dailyIceThickMeanInterp = dailyIceThickMean[iday,:,:].copy()
        # Iterate through the NaN indices
        for nan_index in nan_indices:
            row, col = nan_index
            neighbors = []
            
            # Check the surrounding 8 elements (or less at the edges)
            for i in range(max(0, row - 1), min(dailyIceThickMeanInterp.shape[0], row + 2)):
                for j in range(max(0, col - 1), min(dailyIceThickMeanInterp.shape[1], col + 2)):
                    if not np.isnan(dailyIceThickMeanInterp[i, j]):
                        neighbors.append(dailyIceThickMeanInterp[i, j])
            
            # If there are valid neighbors, fill the NaN with the average of all valid neighbors
            if neighbors:
                dailyIceThickMeanInterp[row, col] = np.nanmean(neighbors)
        a[iday,:,:] = dailyIceThickMeanInterp
    
    
    if simulation == 'ssp245':
        np.save('/Users/arielmor/Desktop/SAI/ArcticShipping/data/dailyIceThickMeanInterpForNav_20352039_ssp.npy',a)
    elif simulation == 'arise-1.5':
        np.save('/Users/arielmor/Desktop/SAI/ArcticShipping/data/dailyIceThickMeanInterpForNav_20352039_arise.npy',a)
    elif simulation == 'arise-1.0':
        np.save('/Users/arielmor/Desktop/SAI/ArcticShipping/data/dailyIceThickMeanInterpForNav_20352039_arise1.0.npy',a)
    del a, dailyIceThickMean, dailyIceThickMeanInterp
        
    
    #### ENS MEAN: 2065-2069 interpolate ice thickness across small channels
    dailyIceThickMean = np.nanmean((np.stack(dailyIceThick_20652069.values())),axis=0) 
    
    a = np.full((365,43,288),np.nan)
    for iday in range(365): 
        nan_indices = np.argwhere(np.isnan(dailyIceThickMean[iday,:,:]))
        dailyIceThickMeanInterp = dailyIceThickMean[iday,:,:].copy()
        # Iterate through the NaN indices
        for nan_index in nan_indices:
            row, col = nan_index
            neighbors = []
            
            # Check the surrounding 8 elements (or less at the edges)
            for i in range(max(0, row - 1), min(dailyIceThickMeanInterp.shape[0], row + 2)):
                for j in range(max(0, col - 1), min(dailyIceThickMeanInterp.shape[1], col + 2)):
                    if not np.isnan(dailyIceThickMeanInterp[i, j]):
                        neighbors.append(dailyIceThickMeanInterp[i, j])
            
            # If there are valid neighbors, fill the NaN with the average of all valid neighbors
            if neighbors:
                dailyIceThickMeanInterp[row, col] = np.nanmean(neighbors)
        a[iday,:,:] = dailyIceThickMeanInterp
     
    
    if simulation == 'ssp245':
        np.save('/Users/arielmor/Desktop/SAI/ArcticShipping/data/dailyIceThickMeanInterpForNav_20652069_ssp.npy',a)
    elif simulation == 'arise-1.5':
        np.save('/Users/arielmor/Desktop/SAI/ArcticShipping/data/dailyIceThickMeanInterpForNav_20652069_arise.npy',a)
    elif simulation == 'arise-1.0':
        np.save('/Users/arielmor/Desktop/SAI/ArcticShipping/data/dailyIceThickMeanInterpForNav_20652069_arise1.0.npy',a)
    
        
    #### ENS MEAN: interp for persistence, 2035-2040
    dailyIceThickMeanPersistent = np.nanmean((np.stack(dailyIceThickForPersistence_20352040.values())),axis=0) 
    
    a = np.full((365,43,288),np.nan)
    for iday in range(365): 
        nan_indices = np.argwhere(np.isnan(dailyIceThickMeanPersistent[iday,:,:]))
        dailyIceThickMeanPersistentInterp = dailyIceThickMeanPersistent[iday,:,:].copy()
        # Iterate through the NaN indices
        for nan_index in nan_indices:
            row, col = nan_index
            neighbors = []
            
            # Check the surrounding 8 elements (or less at the edges)
            for i in range(max(0, row - 1), min(dailyIceThickMeanPersistentInterp.shape[0], row + 2)):
                for j in range(max(0, col - 1), min(dailyIceThickMeanPersistentInterp.shape[1], col + 2)):
                    if not np.isnan(dailyIceThickMeanPersistentInterp[i, j]):
                        neighbors.append(dailyIceThickMeanPersistentInterp[i, j])
            
            # If there are valid neighbors, fill the NaN with the average of all valid neighbors
            if neighbors:
                dailyIceThickMeanPersistentInterp[row, col] = np.nanmean(neighbors)
        a[iday,:,:] = dailyIceThickMeanPersistentInterp
    
    
    if simulation == 'ssp245':
        np.save('/Users/arielmor/Desktop/SAI/ArcticShipping/data/dailyIceThickMeanInterpForPersistence_20352040_ssp.npy',a)
    elif simulation == 'arise-1.5':
        np.save('/Users/arielmor/Desktop/SAI/ArcticShipping/data/dailyIceThickMeanInterpForPersistence_20352040_arise.npy',a)
    elif simulation == 'arise-1.0':
        np.save('/Users/arielmor/Desktop/SAI/ArcticShipping/data/dailyIceThickMeanInterpForPersistence_20352040_arise1.0.npy',a)
    del a, dailyIceThickMeanPersistent, dailyIceThickMeanPersistentInterp
    
    
    #### ENS MEAN: interp for persistence, 2064-2069
    dailyIceThickMeanPersistent = np.nanmean((np.stack(dailyIceThickForPersistence_20642069.values())),axis=0) 
    
    a = np.full((365,43,288),np.nan)
    for iday in range(365): 
        nan_indices = np.argwhere(np.isnan(dailyIceThickMeanPersistent[iday,:,:]))
        dailyIceThickMeanPersistentInterp = dailyIceThickMeanPersistent[iday,:,:].copy()
        # Iterate through the NaN indices
        for nan_index in nan_indices:
            row, col = nan_index
            neighbors = []
            
            # Check the surrounding 8 elements (or less at the edges)
            for i in range(max(0, row - 1), min(dailyIceThickMeanPersistentInterp.shape[0], row + 2)):
                for j in range(max(0, col - 1), min(dailyIceThickMeanPersistentInterp.shape[1], col + 2)):
                    if not np.isnan(dailyIceThickMeanPersistentInterp[i, j]):
                        neighbors.append(dailyIceThickMeanPersistentInterp[i, j])
            
            # If there are valid neighbors, fill the NaN with the average of all valid neighbors
            if neighbors:
                dailyIceThickMeanPersistentInterp[row, col] = np.nanmean(neighbors)
        a[iday,:,:] = dailyIceThickMeanPersistentInterp
    
    
    if simulation == 'ssp245':
        np.save('/Users/arielmor/Desktop/SAI/ArcticShipping/data/dailyIceThickMeanInterpForPersistence_20642069_ssp.npy',a)
    elif simulation == 'arise-1.5':
        np.save('/Users/arielmor/Desktop/SAI/ArcticShipping/data/dailyIceThickMeanInterpForPersistence_20642069_arise.npy',a)
    elif simulation == 'arise-1.0':
        np.save('/Users/arielmor/Desktop/SAI/ArcticShipping/data/dailyIceThickMeanInterpForPersistence_20642069_arise1.0.npy',a)
    del a, dailyIceThickMeanPersistent, dailyIceThickMeanPersistentInterp
    
    
    #### ENS MEMBERS: 2035-2039 interpolate ice thickness across small channels
    ens = ['001','002','003','004','005','006','007','008','009','010']
    
    for ensMem in range(len(ens)):
        print(ensMem)
        a = np.full((365,43,288),np.nan)
        for iday in range(365): 
            nan_indices = np.argwhere(np.isnan(dailyIceThick_20352039[ens[ensMem]][iday,:,:].values))
            dailyIceThickInterp = dailyIceThick_20352039[ens[ensMem]][iday,:,:].values.copy()
            # Iterate through the NaN indices
            for nan_index in nan_indices:
                row, col = nan_index
                neighbors = []
                
                # Check the surrounding 8 elements (or less at the edges)
                for i in range(max(0, row - 1), min(dailyIceThickInterp.shape[0], row + 2)):
                    for j in range(max(0, col - 1), min(dailyIceThickInterp.shape[1], col + 2)):
                        if not np.isnan(dailyIceThickInterp[i, j]):
                            neighbors.append(dailyIceThickInterp[i, j])
                
                # If there are valid neighbors, fill the NaN with the average of all valid neighbors
                if neighbors:
                    dailyIceThickInterp[row, col] = np.nanmean(neighbors)
            a[iday,:,:] = dailyIceThickInterp
        
        
        np.save('/Users/arielmor/Desktop/SAI/ArcticShipping/data/dailyIceThickMeanInterpForNav_20352039_' + str(simulation) + '_' + str(ens[ensMem]) + '.npy',a)
        del a
        
        #### ENS MEMBERS: 2065-2069 interpolate ice thickness across small channels
        a = np.full((365,43,288),np.nan)
        for iday in range(365): 
            nan_indices = np.argwhere(np.isnan(dailyIceThick_20652069[ens[ensMem]][iday,:,:].values))
            dailyIceThickMeanInterp = dailyIceThick_20652069[ens[ensMem]][iday,:,:].copy()
            # Iterate through the NaN indices
            for nan_index in nan_indices:
                row, col = nan_index
                neighbors = []
                
                # Check the surrounding 8 elements (or less at the edges)
                for i in range(max(0, row - 1), min(dailyIceThickMeanInterp.shape[0], row + 2)):
                    for j in range(max(0, col - 1), min(dailyIceThickMeanInterp.shape[1], col + 2)):
                        if not np.isnan(dailyIceThickMeanInterp[i, j]):
                            neighbors.append(dailyIceThickMeanInterp[i, j])
                
                # If there are valid neighbors, fill the NaN with the average of all valid neighbors
                if neighbors:
                    dailyIceThickMeanInterp[row, col] = np.nanmean(neighbors)
            a[iday,:,:] = dailyIceThickMeanInterp
         
        
        np.save('/Users/arielmor/Desktop/SAI/ArcticShipping/data/dailyIceThickMeanInterpForNav_20652069_' + str(simulation) + '_' + str(ens[ensMem]) + '.npy',a)
        del a
        
    return        
        

        
def interpolateDailyIce(simulation):
    import xarray as xr
    import numpy as np
    
    #### read daily ice
    latIce,lonIce,dailyIceThick,dailyIceThickMayOctober,dailyIceFrac,dailyIceFracMayOctober = readDailyIce(simulation)
    
    
    #### land mask for maps
    ds = xr.open_dataset('/Users/arielmor/Desktop/SAI/data/ARISE/data/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001.clm2.h0.ALT.203501-206912_NH.nc')
    landmask = ds.landmask[42:,:]
    landMask = landmask.copy() + 2
    landMask = np.where(~np.isnan(landmask),landMask, 1)  
    landMask = np.where(landMask == 3, np.nan, 1)
    ds.close()
    
    #### interpolate ice thickness across small channels
    dailyIceThickMean = np.nanmean((np.stack(dailyIceThick.values())),axis=0) 
    
    a = np.full((365,43,288),np.nan)
    b = np.full((35,365,43,288),np.nan)
    for iyear in range(35): 
        for iday in range(365): 
            nan_indices = np.argwhere(np.isnan(dailyIceThickMean[iyear,iday,:,:]))
            dailyIceThickMeanInterp = dailyIceThickMean[iyear,iday,:,:].copy()
            # Iterate through the NaN indices
            for nan_index in nan_indices:
                row, col = nan_index
                neighbors = []
                
                # Check the surrounding 8 elements (or less at the edges)
                for i in range(max(0, row - 1), min(dailyIceThickMeanInterp.shape[0], row + 2)):
                    for j in range(max(0, col - 1), min(dailyIceThickMeanInterp.shape[1], col + 2)):
                        if not np.isnan(dailyIceThickMeanInterp[i, j]):
                            neighbors.append(dailyIceThickMeanInterp[i, j])
                
                # If there are valid neighbors, fill the NaN with the average of all valid neighbors
                if neighbors:
                    dailyIceThickMeanInterp[row, col] = np.nanmean(neighbors)
            a[iday,:,:] = dailyIceThickMeanInterp
        b[iyear,:,:,:] = a
 
    landMask = np.load('/Users/arielmor/Desktop/SAI/ArcticShipping/data/landMask_from_global.npy')
    
    b = np.where(landMask == 1, np.nan, b)
    
    if simulation == 'ssp245':
        np.save('/Users/arielmor/Desktop/SAI/ArcticShipping/data/dailyIceThickMeanInterp_ssp.npy',b)
    elif simulation == 'arise-1.5':
        np.save('/Users/arielmor/Desktop/SAI/ArcticShipping/data/dailyIceThickMeanInterp_arise.npy',b)
    elif simulation == 'arise-1.0':
        np.save('/Users/arielmor/Desktop/SAI/ArcticShipping/data/dailyIceThickMeanInterp_arise1.0.npy',b)
        
        
    #### interpolate ice fraction across small channels
    dailyIceFracMean = np.nanmean((np.stack(dailyIceFrac.values())),axis=0)    
    
    a = np.full((365,43,288),np.nan)
    b = np.full((35,365,43,288),np.nan)
    for iyear in range(35): 
        for iday in range(365): 
            nan_indices = np.argwhere(np.isnan(dailyIceFracMean[iyear,iday,:,:]))
            dailyIceFracMeanInterp = dailyIceFracMean[iyear,iday,:,:].copy()
            # Iterate through the NaN indices
            for nan_index in nan_indices:
                row, col = nan_index
                neighbors = []
                
                # Check the surrounding 8 elements (or less at the edges)
                for i in range(max(0, row - 1), min(dailyIceFracMeanInterp.shape[0], row + 2)):
                    for j in range(max(0, col - 1), min(dailyIceFracMeanInterp.shape[1], col + 2)):
                        if not np.isnan(dailyIceFracMeanInterp[i, j]):
                            neighbors.append(dailyIceFracMeanInterp[i, j])
                
                # If there are valid neighbors, fill the NaN with the average of all valid neighbors
                if neighbors:
                    dailyIceFracMeanInterp[row, col] = np.nanmean(neighbors)
            a[iday,:,:] = dailyIceFracMeanInterp
        b[iyear,:,:,:] = a
    
    b = np.where(landMask == 1, np.nan, b)
    
    if simulation == 'ssp245':
        np.save('/Users/arielmor/Desktop/SAI/ArcticShipping/data/dailyIceFracMeanInterp_ssp.npy',b)
    elif simulation == 'arise-1.5':
        np.save('/Users/arielmor/Desktop/SAI/ArcticShipping/data/dailyIceFracMeanInterp_arise.npy',b)
    elif simulation == 'arise-1.0':
        np.save('/Users/arielmor/Desktop/SAI/ArcticShipping/data/dailyIceFracMeanInterp_arise1.0.npy',b)    
    
    return



def readMonthlyIce():
    import xarray as xr
    import numpy as np
    import pandas as pd
    
    dataDir = '/Users/arielmor/Desktop/SAI/ArcticShipping/data/'
    ens = ['001','002','003','004','005','006','007','008','009','010']
    
    #### land mask for maps
    ds = xr.open_dataset('/Users/arielmor/Desktop/SAI/data/ARISE/data/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001.clm2.h0.ALT.203501-206912_NH.nc')
    landmask = ds.landmask[42:,:]
    landMask = landmask.copy() + 2
    landMask = np.where(~np.isnan(landmask),landMask, 1)  
    landMask = np.where(landMask == 3, np.nan, 1)
    ds.close()
    
    iceFracSSP      = {}
    iceThickSSP     = {}
    iceFracARISE    = {}
    iceThickARISE   = {}
    iceFracARISE_1  = {}
    iceThickARISE_1 = {}
    
    for i in range(len(ens)):
        ## SSP2-4.5
        ds = xr.open_dataset(dataDir + 'b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.' + str(ens[i]) + '.cice.h.aice_only.201501-206912_RG_NH.nc',
                             decode_times = False)
        ds['time'] = pd.date_range(start=pd.to_datetime("2015-01-01"), 
                                               periods=ds.sizes['time'], freq='MS')
        iceFracSSP[ens[i]] = (ds.aice[240:,:,:] * 100.)
        iceFracSSP[ens[i]] = xr.where(landMask == 1, iceFracSSP[ens[i]], np.nan)
        latIce = ds.lat; lonIce = ds.lon
        ds.close()
        
        ds = xr.open_dataset(dataDir + 'b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.' + str(ens[i]) + '.cice.h.hi_only.201501-206912_RG_NH.nc')
        ds['time'] = pd.date_range(start=pd.to_datetime("2015-01-01"), 
                                               periods=ds.sizes['time'], freq='MS')
        iceThickSSP[ens[i]] = ds.hi[240:,:,:]
        iceThickSSP[ens[i]] = xr.where(landMask == 1, iceThickSSP[ens[i]], np.nan)
        ds.close()
        
        ## ARISE-1.6
        ds = xr.open_dataset(dataDir + 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.' + str(ens[i]) + '.cice.h.aice_only.203501-206912_RG_NH.nc',
                             decode_times = False)
        ds['time'] = pd.date_range(start=pd.to_datetime("2035-01-01"), 
                                               periods=ds.sizes['time'], freq='MS')
        iceFracARISE[ens[i]] = (ds.aice * 100.)
        iceFracARISE[ens[i]] = xr.where(landMask == 1, iceFracARISE[ens[i]], np.nan)
        ds.close()
        
        ds = xr.open_dataset(dataDir + 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.' + str(ens[i]) + '.cice.h.hi_only.203501-206912_RG_NH.nc')
        ds['time'] = pd.date_range(start=pd.to_datetime("2035-01-01"), 
                                               periods=ds.sizes['time'], freq='MS')
        iceThickARISE[ens[i]] = ds.hi
        iceThickARISE[ens[i]] = xr.where(landMask == 1, iceThickARISE[ens[i]], np.nan)
        ds.close()  
        
        ## ARISE-1.0
        ds = xr.open_dataset(dataDir + 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-LOWER-0.5.' + str(ens[i]) + '.cice.h.aice_only.203501-206912_RG_NH.nc',
                             decode_times = False)
        ds['time'] = pd.date_range(start=pd.to_datetime("2035-01-01"), 
                                               periods=ds.sizes['time'], freq='MS')
        iceFracARISE_1[ens[i]] = (ds.aice * 100.)
        iceFracARISE_1[ens[i]] = xr.where(landMask == 1, iceFracARISE_1[ens[i]], np.nan)
        ds.close()
        
        ds = xr.open_dataset(dataDir + 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-LOWER-0.5.' + str(ens[i]) + '.cice.h.hi_only.203501-206912_RG_NH.nc')
        ds['time'] = pd.date_range(start=pd.to_datetime("2035-01-01"), 
                                               periods=ds.sizes['time'], freq='MS')
        iceThickARISE_1[ens[i]] = ds.hi 
        iceThickARISE_1[ens[i]] = xr.where(landMask == 1, iceThickARISE_1[ens[i]], np.nan)
        ds.close()  
    
    return latIce, lonIce, iceFracSSP, iceThickSSP, iceFracARISE, iceThickARISE, iceFracARISE_1, iceThickARISE_1



def averageIceCategories(dataDict):
    import numpy as np
    ens = ['001','002','003','004','005','006','007','008','009','010']
    cats = ['1', '2', '3', '4', '5']
    
    avgCat = {}
    for c in range(len(cats)):
        avgCat[cats[c]] = np.nanmean(
            np.stack((dataDict[cats[c],ens[0]].values, dataDict[cats[c],ens[1]].values,
                    dataDict[cats[c],ens[2]].values, dataDict[cats[c],ens[3]].values,
                    dataDict[cats[c],ens[4]].values, dataDict[cats[c],ens[5]].values,
                    dataDict[cats[c],ens[6]].values, dataDict[cats[c],ens[7]].values,
                    dataDict[cats[c],ens[8]].values, dataDict[cats[c],ens[9]].values), 
                     axis = 0),
                    axis = 0)
        
    return avgCat



def iceCategories():
    import xarray as xr
    import numpy as np
    import pandas as pd
    from readSeaIce import averageIceCategories
    
    dataDir = '/Users/arielmor/Desktop/SAI/ArcticShipping/data/'
    ens = ['001','002','003','004','005','006','007','008','009','010']
    cats = ['1', '2', '3', '4', '5']
    
    landMask = np.load('/Users/arielmor/Desktop/SAI/ArcticShipping/data/landMask_from_global_edited.npy')
    
    #### read ice categories
    cat_SSP    = {}
    cat_ARISE  = {}
    cat_ARISE1 = {}
    
    for c in range(len(cats)):
        for i in range(len(ens)):
            ## SSP2-4.5
            ds = xr.open_dataset(dataDir + 'b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.' 
                                 + str(ens[i]) + '.cice.h.aicen_only.201501-206912_cat' + str(cats[c]) + '_RG_NH.nc', 
                                 decode_times = False)
            ds['time'] = xr.cftime_range(start = '2015-01', periods = ds.sizes['time'], freq = "M", calendar = "noleap")
            cat_SSP[cats[c], ens[i]] = ds.aicen[240:,:,:]
            latIce = ds.lat; lonIce = ds.lon
            ds.close()
        
            ds = xr.open_dataset(dataDir + 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.' 
                                 + str(ens[i]) + '.cice.h.aicen_only.203501-206912_cat' + str(cats[c]) + '_RG_NH.nc',
                                 decode_times = False)
            ds['time'] = xr.cftime_range(start = '2035-01', periods = ds.sizes['time'], freq = "M", calendar = "noleap")
            cat_ARISE[cats[c], ens[i]] = ds.aicen
            ds.close()
            
            ds = xr.open_dataset(dataDir + 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-LOWER-0.5.' 
                                 + str(ens[i]) + '.cice.h.aicen_only.203501-206912_cat' + str(cats[c]) + '_RG_NH.nc',
                                 decode_times = False)
            ds['time'] = xr.cftime_range(start = '2035-01', periods = ds.sizes['time'], freq = "M", calendar = "noleap")
            cat_ARISE1[cats[c], ens[i]] = ds.aicen
            ds.close()

    #### ens mean ice categories
    avgCat_SSP = averageIceCategories(cat_SSP)
    avgCat_ARISE = averageIceCategories(cat_ARISE)
    avgCat_ARISE1 = averageIceCategories(cat_ARISE1)
    
    #### interpolate over small channels
    '''SSP'''
    for c in range(len(cats)):
        print("category " + str(c))
        a = np.full((420,43,288),np.nan)
        for imonth in range(420): 
            nan_indices = np.argwhere(np.isnan(avgCat_SSP[cats[c]][imonth,:,:]))
            monthlyIceThickMeanInterp = avgCat_SSP[cats[c]][imonth,:,:].copy()
            # Iterate through the NaN indices
            for nan_index in nan_indices:
                row, col = nan_index
                neighbors = []
                
                # Check the surrounding 8 elements (or less at the edges)
                for i in range(max(0, row - 1), min(monthlyIceThickMeanInterp.shape[0], row + 2)):
                    for j in range(max(0, col - 1), min(monthlyIceThickMeanInterp.shape[1], col + 2)):
                        if not np.isnan(monthlyIceThickMeanInterp[i, j]):
                            neighbors.append(monthlyIceThickMeanInterp[i, j])
                
                # If there are valid neighbors, fill the NaN with the average of all valid neighbors
                if neighbors:
                    monthlyIceThickMeanInterp[row, col] = np.nanmean(neighbors)
            a[imonth,:,:] = monthlyIceThickMeanInterp
        
        a = np.where(landMask == 1, np.nan, a)
        np.save(dataDir + 'iceCategory' + str(cats[c]) + '_SSP.npy',a)
        
    '''ARISE default'''    
    for c in range(len(cats)):
        print("category " + str(c))
        a = np.full((420,43,288),np.nan)
        for imonth in range(420): 
            nan_indices = np.argwhere(np.isnan(avgCat_ARISE[cats[c]][imonth,:,:]))
            monthlyIceThickMeanInterp = avgCat_ARISE[cats[c]][imonth,:,:].copy()
            # Iterate through the NaN indices
            for nan_index in nan_indices:
                row, col = nan_index
                neighbors = []
                
                # Check the surrounding 8 elements (or less at the edges)
                for i in range(max(0, row - 1), min(monthlyIceThickMeanInterp.shape[0], row + 2)):
                    for j in range(max(0, col - 1), min(monthlyIceThickMeanInterp.shape[1], col + 2)):
                        if not np.isnan(monthlyIceThickMeanInterp[i, j]):
                            neighbors.append(monthlyIceThickMeanInterp[i, j])
                
                # If there are valid neighbors, fill the NaN with the average of all valid neighbors
                if neighbors:
                    monthlyIceThickMeanInterp[row, col] = np.nanmean(neighbors)
            a[imonth,:,:] = monthlyIceThickMeanInterp
        
        a = np.where(landMask == 1, np.nan, a)
        np.save(dataDir + 'iceCategory' + str(cats[c]) + '_ARISE.npy',a)
        del a, monthlyIceThickMeanInterp
        
    '''ARISE lower'''   
    for c in range(len(cats)):
        print("category " + str(c))
        a = np.full((420,43,288),np.nan)
        for imonth in range(420): 
            nan_indices = np.argwhere(np.isnan(avgCat_ARISE1[cats[c]][imonth,:,:]))
            monthlyIceThickMeanInterp = avgCat_ARISE1[cats[c]][imonth,:,:].copy()
            # Iterate through the NaN indices
            for nan_index in nan_indices:
                row, col = nan_index
                neighbors = []
                
                # Check the surrounding 8 elements (or less at the edges)
                for i in range(max(0, row - 1), min(monthlyIceThickMeanInterp.shape[0], row + 2)):
                    for j in range(max(0, col - 1), min(monthlyIceThickMeanInterp.shape[1], col + 2)):
                        if not np.isnan(monthlyIceThickMeanInterp[i, j]):
                            neighbors.append(monthlyIceThickMeanInterp[i, j])
                
                # If there are valid neighbors, fill the NaN with the average of all valid neighbors
                if neighbors:
                    monthlyIceThickMeanInterp[row, col] = np.nanmean(neighbors)
            a[imonth,:,:] = monthlyIceThickMeanInterp
        
        a = np.where(landMask == 1, np.nan, a)
        np.save(dataDir + 'iceCategory' + str(cats[c]) + '_ARISE1.npy',a)
        
    return 



def iceCategoriesDaily():
    import xarray as xr
    import numpy as np
    from readSeaIce import averageIceCategories
    
    dataDir = '/Users/arielmor/Desktop/SAI/ArcticShipping/data/'
    ens = ['001','002','003','004','005','006','007','008','009','010']
    cats = ['1', '2', '3', '4', '5']
    
    landMask = np.load('/Users/arielmor/Desktop/SAI/ArcticShipping/data/landMask_from_global_edited.npy')
    
    #### read ice categories
    cat_SSP_20352039    = {}
    cat_SSP_20652069    = {}
    cat_ARISE_20352039  = {}
    cat_ARISE_20652069  = {}
    cat_ARISE1_20352039 = {}
    cat_ARISE1_20652069 = {}
    
    for c in range(len(cats)):
        for i in range(len(ens)):
            ## SSP2-4.5
            ds = xr.open_dataset(dataDir + 'b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.' 
                                 + str(ens[i]) + '.cice.h1.aicen_d_only.20350101-20391231_cat' + str(cats[c]) + '_RG_NH.nc', 
                                 decode_times = False)
            ds['time'] = xr.cftime_range(start = '2035-01-01', periods = ds.sizes['time'], freq = "D", calendar = "noleap")
            cat_SSP_20352039[cats[c], ens[i]] = (ds.aicen_d).groupby('time.dayofyear').mean('time', skipna = True)
            ds.close()
            
            ds = xr.open_dataset(dataDir + 'b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.' 
                                 + str(ens[i]) + '.cice.h1.aicen_d_only.20650101-20691231_cat' + str(cats[c]) + '_RG_NH.nc', 
                                 decode_times = False)
            ds['time'] = xr.cftime_range(start = '2065-01-01', periods = ds.sizes['time'], freq = "D", calendar = "noleap")
            cat_SSP_20652069[cats[c], ens[i]] = (ds.aicen_d).groupby('time.dayofyear').mean('time', skipna = True)
            ds.close()
        
            ## ARISE-1.5
            ds = xr.open_dataset(dataDir + 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.' 
                                 + str(ens[i]) + '.cice.h1.aicen_d_only.20350101-20391231_cat' + str(cats[c]) + '_RG_NH.nc',
                                 decode_times = False)
            ds['time'] = xr.cftime_range(start = '2035-01-01', periods = ds.sizes['time'], freq = "D", calendar = "noleap")
            cat_ARISE_20352039[cats[c], ens[i]] = (ds.aicen_d).groupby('time.dayofyear').mean('time', skipna = True)
            ds.close()
            
            ds = xr.open_dataset(dataDir + 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.' 
                                 + str(ens[i]) + '.cice.h1.aicen_d_only.20650101-20691231_cat' + str(cats[c]) + '_RG_NH.nc',
                                 decode_times = False)
            ds['time'] = xr.cftime_range(start = '2065-01-01', periods = ds.sizes['time'], freq = "D", calendar = "noleap")
            cat_ARISE_20652069[cats[c], ens[i]] = (ds.aicen_d).groupby('time.dayofyear').mean('time', skipna = True)
            ds.close()
            
            ## ARISE-1.0
            ds = xr.open_dataset(dataDir + 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-LOWER-0.5.' 
                                 + str(ens[i]) + '.cice.h1.aicen_d_only.20350101-20391231_cat' + str(cats[c]) + '_RG_NH.nc',
                                 decode_times = False)
            ds['time'] = xr.cftime_range(start = '2035-01-01', periods = ds.sizes['time'], freq = "D", calendar = "noleap")
            cat_ARISE1_20352039[cats[c], ens[i]] = (ds.aicen_d).groupby('time.dayofyear').mean('time', skipna = True)
            ds.close()
            
            ds = xr.open_dataset(dataDir + 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-LOWER-0.5.' 
                                 + str(ens[i]) + '.cice.h1.aicen_d_only.20650101-20691231_cat' + str(cats[c]) + '_RG_NH.nc',
                                 decode_times = False)
            ds['time'] = xr.cftime_range(start = '2065-01-01', periods = ds.sizes['time'], freq = "D", calendar = "noleap")
            cat_ARISE1_20652069[cats[c], ens[i]] = (ds.aicen_d).groupby('time.dayofyear').mean('time', skipna = True)
            ds.close()

    
    #### ens mean ice categories
    avgCat_SSP_20352039    = averageIceCategories(cat_SSP_20352039)
    avgCat_ARISE_20352039  = averageIceCategories(cat_ARISE_20352039)
    avgCat_ARISE1_20352039 = averageIceCategories(cat_ARISE1_20352039)
    
    avgCat_SSP_20652069    = averageIceCategories(cat_SSP_20652069)
    avgCat_ARISE_20652069  = averageIceCategories(cat_ARISE_20652069)
    avgCat_ARISE1_20652069 = averageIceCategories(cat_ARISE1_20652069)
    
        
    #### interpolate over small channels
    '''SSP'''
    for c in range(len(cats)):
        print("category " + str(c))
        a = np.full((365,43,288),np.nan)
        for imonth in range(365): 
            nan_indices = np.argwhere(np.isnan(avgCat_SSP_20352039[cats[c]][imonth,:,:]))
            monthlyIceThickMeanInterp = avgCat_SSP_20352039[cats[c]][imonth,:,:].copy()
            # Iterate through the NaN indices
            for nan_index in nan_indices:
                row, col = nan_index
                neighbors = []
                
                # Check the surrounding 8 elements (or less at the edges)
                for i in range(max(0, row - 1), min(monthlyIceThickMeanInterp.shape[0], row + 2)):
                    for j in range(max(0, col - 1), min(monthlyIceThickMeanInterp.shape[1], col + 2)):
                        if not np.isnan(monthlyIceThickMeanInterp[i, j]):
                            neighbors.append(monthlyIceThickMeanInterp[i, j])
                
                # If there are valid neighbors, fill the NaN with the average of all valid neighbors
                if neighbors:
                    monthlyIceThickMeanInterp[row, col] = np.nanmean(neighbors)
            a[imonth,:,:] = monthlyIceThickMeanInterp
        
        np.save(dataDir + 'iceCategoryDaily' + str(cats[c]) + '_SSP_20352039.npy',a)
    del a, monthlyIceThickMeanInterp
        
    for c in range(len(cats)):
        print("category " + str(c))
        a = np.full((365,43,288),np.nan)
        for imonth in range(365): 
            nan_indices = np.argwhere(np.isnan(avgCat_SSP_20652069[cats[c]][imonth,:,:]))
            monthlyIceThickMeanInterp = avgCat_SSP_20652069[cats[c]][imonth,:,:].copy()
            # Iterate through the NaN indices
            for nan_index in nan_indices:
                row, col = nan_index
                neighbors = []
                
                # Check the surrounding 8 elements (or less at the edges)
                for i in range(max(0, row - 1), min(monthlyIceThickMeanInterp.shape[0], row + 2)):
                    for j in range(max(0, col - 1), min(monthlyIceThickMeanInterp.shape[1], col + 2)):
                        if not np.isnan(monthlyIceThickMeanInterp[i, j]):
                            neighbors.append(monthlyIceThickMeanInterp[i, j])
                
                # If there are valid neighbors, fill the NaN with the average of all valid neighbors
                if neighbors:
                    monthlyIceThickMeanInterp[row, col] = np.nanmean(neighbors)
            a[imonth,:,:] = monthlyIceThickMeanInterp
        
        np.save(dataDir + 'iceCategoryDaily' + str(cats[c]) + '_SSP_20652069.npy',a)
    del a, monthlyIceThickMeanInterp
        
    '''ARISE default'''    
    for c in range(len(cats)):
        print("category " + str(c))
        a = np.full((365,43,288),np.nan)
        for imonth in range(365): 
            nan_indices = np.argwhere(np.isnan(avgCat_ARISE_20352039[cats[c]][imonth,:,:]))
            monthlyIceThickMeanInterp = avgCat_ARISE_20352039[cats[c]][imonth,:,:].copy()
            # Iterate through the NaN indices
            for nan_index in nan_indices:
                row, col = nan_index
                neighbors = []
                
                # Check the surrounding 8 elements (or less at the edges)
                for i in range(max(0, row - 1), min(monthlyIceThickMeanInterp.shape[0], row + 2)):
                    for j in range(max(0, col - 1), min(monthlyIceThickMeanInterp.shape[1], col + 2)):
                        if not np.isnan(monthlyIceThickMeanInterp[i, j]):
                            neighbors.append(monthlyIceThickMeanInterp[i, j])
                
                # If there are valid neighbors, fill the NaN with the average of all valid neighbors
                if neighbors:
                    monthlyIceThickMeanInterp[row, col] = np.nanmean(neighbors)
            a[imonth,:,:] = monthlyIceThickMeanInterp
        
        np.save(dataDir + 'iceCategoryDaily' + str(cats[c]) + '_ARISE_20352039.npy',a)
    del a, monthlyIceThickMeanInterp
        
    for c in range(len(cats)):
        print("category " + str(c))
        a = np.full((365,43,288),np.nan)
        for imonth in range(365): 
            nan_indices = np.argwhere(np.isnan(avgCat_ARISE_20652069[cats[c]][imonth,:,:]))
            monthlyIceThickMeanInterp = avgCat_ARISE_20652069[cats[c]][imonth,:,:].copy()
            # Iterate through the NaN indices
            for nan_index in nan_indices:
                row, col = nan_index
                neighbors = []
                
                # Check the surrounding 8 elements (or less at the edges)
                for i in range(max(0, row - 1), min(monthlyIceThickMeanInterp.shape[0], row + 2)):
                    for j in range(max(0, col - 1), min(monthlyIceThickMeanInterp.shape[1], col + 2)):
                        if not np.isnan(monthlyIceThickMeanInterp[i, j]):
                            neighbors.append(monthlyIceThickMeanInterp[i, j])
                
                # If there are valid neighbors, fill the NaN with the average of all valid neighbors
                if neighbors:
                    monthlyIceThickMeanInterp[row, col] = np.nanmean(neighbors)
            a[imonth,:,:] = monthlyIceThickMeanInterp
        
        a = np.where(landMask == 1, np.nan, a)
        np.save(dataDir + 'iceCategoryDaily' + str(cats[c]) + '_ARISE_20652069.npy',a)
    del a, monthlyIceThickMeanInterp
    
        
    '''ARISE lower'''   
    for c in range(len(cats)):
        print("category " + str(c))
        a = np.full((365,43,288),np.nan)
        for imonth in range(365): 
            nan_indices = np.argwhere(np.isnan(avgCat_ARISE1_20352039[cats[c]][imonth,:,:]))
            monthlyIceThickMeanInterp = avgCat_ARISE1_20352039[cats[c]][imonth,:,:].copy()
            # Iterate through the NaN indices
            for nan_index in nan_indices:
                row, col = nan_index
                neighbors = []
                
                # Check the surrounding 8 elements (or less at the edges)
                for i in range(max(0, row - 1), min(monthlyIceThickMeanInterp.shape[0], row + 2)):
                    for j in range(max(0, col - 1), min(monthlyIceThickMeanInterp.shape[1], col + 2)):
                        if not np.isnan(monthlyIceThickMeanInterp[i, j]):
                            neighbors.append(monthlyIceThickMeanInterp[i, j])
                
                # If there are valid neighbors, fill the NaN with the average of all valid neighbors
                if neighbors:
                    monthlyIceThickMeanInterp[row, col] = np.nanmean(neighbors)
            a[imonth,:,:] = monthlyIceThickMeanInterp
        
        np.save(dataDir + 'iceCategoryDaily' + str(cats[c]) + '_ARISE1_20352039.npy',a)
    del a, monthlyIceThickMeanInterp
        
    for c in range(len(cats)):
        print("category " + str(c))
        a = np.full((365,43,288),np.nan)
        for imonth in range(365): 
            nan_indices = np.argwhere(np.isnan(avgCat_ARISE1_20652069[cats[c]][imonth,:,:]))
            monthlyIceThickMeanInterp = avgCat_ARISE1_20652069[cats[c]][imonth,:,:].copy()
            # Iterate through the NaN indices
            for nan_index in nan_indices:
                row, col = nan_index
                neighbors = []
                
                # Check the surrounding 8 elements (or less at the edges)
                for i in range(max(0, row - 1), min(monthlyIceThickMeanInterp.shape[0], row + 2)):
                    for j in range(max(0, col - 1), min(monthlyIceThickMeanInterp.shape[1], col + 2)):
                        if not np.isnan(monthlyIceThickMeanInterp[i, j]):
                            neighbors.append(monthlyIceThickMeanInterp[i, j])
                
                # If there are valid neighbors, fill the NaN with the average of all valid neighbors
                if neighbors:
                    monthlyIceThickMeanInterp[row, col] = np.nanmean(neighbors)
            a[imonth,:,:] = monthlyIceThickMeanInterp
        
        # a = np.where(landMask == 1, np.nan, a)
        np.save(dataDir + 'iceCategoryDaily' + str(cats[c]) + '_ARISE1_20652069.npy',a)
    del a, monthlyIceThickMeanInterp
        
    return 


def iceCategoriesDailyForPersistence():
    import xarray as xr
    import numpy as np
    
    dataDir = '/Users/arielmor/Desktop/SAI/ArcticShipping/data/'
    ens = ['001','002','003','004','005','006','007','008','009','010']
    cats = ['1', '2', '3', '4', '5']
    
    landMask = np.load('/Users/arielmor/Desktop/SAI/ArcticShipping/data/landMask_from_global_edited.npy')
    
    #### read ice categories
    cat_SSP_20352040    = {}
    cat_SSP_20642069    = {}
    cat_ARISE_20352040  = {}
    cat_ARISE_20642069  = {}
    cat_ARISE1_20352040 = {}
    cat_ARISE1_20642069 = {}
    
    for c in range(len(cats)):
        for i in range(len(ens)):
            ## SSP2-4.5 - 2035-2040
            ds = xr.open_dataset(dataDir + 'b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.' 
                                 + str(ens[i]) + '.cice.h1.aicen_d_only.20350101-20391231_cat' + str(cats[c]) + '_RG_NH.nc', 
                                 decode_times = False)
            ds['time'] = xr.cftime_range(start = '2035-01-01', periods = ds.sizes['time'], freq = "D", calendar = "noleap")
            aicen_20352039 = ds.aicen_d
            
            ds = xr.open_dataset(dataDir + 'b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.' 
                                 + str(ens[i]) + '.cice.h1.aicen_d_only.20400101-20401231.cat' + str(cats[c]) + '_RG_NH.nc', 
                                 decode_times = False)
            ds['time'] = xr.cftime_range(start = '2040-01-01', periods = ds.sizes['time'], freq = "D", calendar = "noleap")
            aicen_2040 = ds.aicen_d
            
            aicen_20352040 = xr.concat([aicen_20352039,aicen_2040], dim='time')
            
            cat_SSP_20352040[cats[c], ens[i]] = aicen_20352040.groupby('time.dayofyear').mean('time', skipna = True)
            ds.close()
            
            ## SSP2-4.5 - 2064-2069
            ds = xr.open_dataset(dataDir + 'b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.' 
                                 + str(ens[i]) + '.cice.h1.aicen_d_only.20650101-20691231_cat' + str(cats[c]) + '_RG_NH.nc', 
                                 decode_times = False)
            ds['time'] = xr.cftime_range(start = '2065-01-01', periods = ds.sizes['time'], freq = "D", calendar = "noleap")
            aicen_20652069 = ds.aicen_d
            
            ds = xr.open_dataset(dataDir + 'b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.' 
                                 + str(ens[i]) + '.cice.h1.aicen_d_only.20640101-20641231.cat' + str(cats[c]) + '_RG_NH.nc', 
                                 decode_times = False)
            ds['time'] = xr.cftime_range(start = '2064-01-01', periods = ds.sizes['time'], freq = "D", calendar = "noleap")
            aicen_2064 = ds.aicen_d
            
            aicen_20642069 = xr.concat([aicen_2064,aicen_20652069], dim='time')
            
            cat_SSP_20642069[cats[c], ens[i]] = aicen_20642069.groupby('time.dayofyear').mean('time', skipna = True)
            ds.close()
        
        
            ## ARISE-1.5 - 20342039
            ds = xr.open_dataset(dataDir + 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.' 
                                 + str(ens[i]) + '.cice.h1.aicen_d_only.20350101-20391231_cat' + str(cats[c]) + '_RG_NH.nc',
                                 decode_times = False)
            ds['time'] = xr.cftime_range(start = '2035-01-01', periods = ds.sizes['time'], freq = "D", calendar = "noleap")
            aicen_20352039 = ds.aicen_d
            
            ds = xr.open_dataset(dataDir + 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.' 
                                 + str(ens[i]) + '.cice.h1.aicen_d_only.20400101-20401231.cat' + str(cats[c]) + '_RG_NH.nc',
                                 decode_times = False)
            ds['time'] = xr.cftime_range(start = '2040-01-01', periods = ds.sizes['time'], freq = "D", calendar = "noleap")
            aicen_2040 = ds.aicen_d
            
            aicen_20352040 = xr.concat([aicen_20352039,aicen_2040], dim='time')
            
            cat_ARISE_20352040[cats[c], ens[i]] = aicen_20352040.groupby('time.dayofyear').mean('time', skipna = True)
            ds.close()
            
            ## ARISE-1.5 - 20642069
            ds = xr.open_dataset(dataDir + 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.' 
                                 + str(ens[i]) + '.cice.h1.aicen_d_only.20650101-20691231_cat' + str(cats[c]) + '_RG_NH.nc',
                                 decode_times = False)
            ds['time'] = xr.cftime_range(start = '2065-01-01', periods = ds.sizes['time'], freq = "D", calendar = "noleap")
            aicen_20652069 = ds.aicen_d
            
            ds = xr.open_dataset(dataDir + 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.' 
                                 + str(ens[i]) + '.cice.h1.aicen_d_only.20640101-20641231.cat' + str(cats[c]) + '_RG_NH.nc',
                                 decode_times = False)
            ds['time'] = xr.cftime_range(start = '2064-01-01', periods = ds.sizes['time'], freq = "D", calendar = "noleap")
            aicen_2064 = ds.aicen_d
            
            aicen_20642069 = xr.concat([aicen_2064,aicen_20652069], dim='time')
            
            cat_ARISE_20642069[cats[c], ens[i]] = aicen_20642069.groupby('time.dayofyear').mean('time', skipna = True)
            ds.close()
            
            ## ARISE-1.0 - 20352040
            ds = xr.open_dataset(dataDir + 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-LOWER-0.5.' 
                                 + str(ens[i]) + '.cice.h1.aicen_d_only.20350101-20391231_cat' + str(cats[c]) + '_RG_NH.nc',
                                 decode_times = False)
            ds['time'] = xr.cftime_range(start = '2035-01-01', periods = ds.sizes['time'], freq = "D", calendar = "noleap")
            aicen_20352039 = ds.aicen_d
            
            ds = xr.open_dataset(dataDir + 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-LOWER-0.5.' 
                                 + str(ens[i]) + '.cice.h1.aicen_d_only.20400101-20401231.cat' + str(cats[c]) + '_RG_NH.nc',
                                 decode_times = False)
            ds['time'] = xr.cftime_range(start = '2040-01-01', periods = ds.sizes['time'], freq = "D", calendar = "noleap")
            aicen_2040 = ds.aicen_d
            
            aicen_20352040 = xr.concat([aicen_20352039,aicen_2040], dim='time')
            
            cat_ARISE1_20352040[cats[c], ens[i]] = aicen_20352040.groupby('time.dayofyear').mean('time', skipna = True)
            ds.close()
            
            
            ## ARISE-1.0 - 20642069
            ds = xr.open_dataset(dataDir + 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-LOWER-0.5.' 
                                 + str(ens[i]) + '.cice.h1.aicen_d_only.20650101-20691231_cat' + str(cats[c]) + '_RG_NH.nc',
                                 decode_times = False)
            ds['time'] = xr.cftime_range(start = '2065-01-01', periods = ds.sizes['time'], freq = "D", calendar = "noleap")
            aicen_20652069 = ds.aicen_d
            
            ds = xr.open_dataset(dataDir + 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-LOWER-0.5.' 
                                 + str(ens[i]) + '.cice.h1.aicen_d_only.20640101-20641231.cat' + str(cats[c]) + '_RG_NH.nc',
                                 decode_times = False)
            ds['time'] = xr.cftime_range(start = '2064-01-01', periods = ds.sizes['time'], freq = "D", calendar = "noleap")
            aicen_2064 = ds.aicen_d
            
            aicen_20642069 = xr.concat([aicen_2064,aicen_20652069], dim='time')
            
            cat_ARISE1_20642069[cats[c], ens[i]] = aicen_20642069.groupby('time.dayofyear').mean('time', skipna = True)
            ds.close()

    
    #### ens mean ice categories
    avgCat_SSP_20352040   = averageIceCategories(cat_SSP_20352040)
    avgCat_ARISE_20352040  = averageIceCategories(cat_ARISE_20352040)
    avgCat_ARISE1_20352040 = averageIceCategories(cat_ARISE1_20352040)
    
    avgCat_SSP_20642069    = averageIceCategories(cat_SSP_20642069)
    avgCat_ARISE_20642069  = averageIceCategories(cat_ARISE_20642069)
    avgCat_ARISE1_20642069 = averageIceCategories(cat_ARISE1_20642069)
    
        
    #### interpolate over small channels
    '''SSP'''
    for c in range(len(cats)):
        print("category " + str(c))
        a = np.full((365,43,288),np.nan)
        for imonth in range(365): 
            nan_indices = np.argwhere(np.isnan(avgCat_SSP_20352040[cats[c]][imonth,:,:]))
            monthlyIceThickMeanInterp = avgCat_SSP_20352040[cats[c]][imonth,:,:].copy()
            # Iterate through the NaN indices
            for nan_index in nan_indices:
                row, col = nan_index
                neighbors = []
                
                # Check the surrounding 8 elements (or less at the edges)
                for i in range(max(0, row - 1), min(monthlyIceThickMeanInterp.shape[0], row + 2)):
                    for j in range(max(0, col - 1), min(monthlyIceThickMeanInterp.shape[1], col + 2)):
                        if not np.isnan(monthlyIceThickMeanInterp[i, j]):
                            neighbors.append(monthlyIceThickMeanInterp[i, j])
                
                # If there are valid neighbors, fill the NaN with the average of all valid neighbors
                if neighbors:
                    monthlyIceThickMeanInterp[row, col] = np.nanmean(neighbors)
            a[imonth,:,:] = monthlyIceThickMeanInterp
        
        np.save(dataDir + 'iceCategoryDailyForPersistence' + str(cats[c]) + '_SSP_20352040.npy',a)
    del a, monthlyIceThickMeanInterp
        
    for c in range(len(cats)):
        print("category " + str(c))
        a = np.full((365,43,288),np.nan)
        for imonth in range(365): 
            nan_indices = np.argwhere(np.isnan(avgCat_SSP_20642069[cats[c]][imonth,:,:]))
            monthlyIceThickMeanInterp = avgCat_SSP_20642069[cats[c]][imonth,:,:].copy()
            # Iterate through the NaN indices
            for nan_index in nan_indices:
                row, col = nan_index
                neighbors = []
                
                # Check the surrounding 8 elements (or less at the edges)
                for i in range(max(0, row - 1), min(monthlyIceThickMeanInterp.shape[0], row + 2)):
                    for j in range(max(0, col - 1), min(monthlyIceThickMeanInterp.shape[1], col + 2)):
                        if not np.isnan(monthlyIceThickMeanInterp[i, j]):
                            neighbors.append(monthlyIceThickMeanInterp[i, j])
                
                # If there are valid neighbors, fill the NaN with the average of all valid neighbors
                if neighbors:
                    monthlyIceThickMeanInterp[row, col] = np.nanmean(neighbors)
            a[imonth,:,:] = monthlyIceThickMeanInterp
        
        np.save(dataDir + 'iceCategoryDailyForPersistence' + str(cats[c]) + '_SSP_20642069.npy',a)
    del a, monthlyIceThickMeanInterp
        
    '''ARISE default'''    
    for c in range(len(cats)):
        print("category " + str(c))
        a = np.full((365,43,288),np.nan)
        for imonth in range(365): 
            nan_indices = np.argwhere(np.isnan(avgCat_ARISE_20352040[cats[c]][imonth,:,:]))
            monthlyIceThickMeanInterp = avgCat_ARISE_20352040[cats[c]][imonth,:,:].copy()
            # Iterate through the NaN indices
            for nan_index in nan_indices:
                row, col = nan_index
                neighbors = []
                
                # Check the surrounding 8 elements (or less at the edges)
                for i in range(max(0, row - 1), min(monthlyIceThickMeanInterp.shape[0], row + 2)):
                    for j in range(max(0, col - 1), min(monthlyIceThickMeanInterp.shape[1], col + 2)):
                        if not np.isnan(monthlyIceThickMeanInterp[i, j]):
                            neighbors.append(monthlyIceThickMeanInterp[i, j])
                
                # If there are valid neighbors, fill the NaN with the average of all valid neighbors
                if neighbors:
                    monthlyIceThickMeanInterp[row, col] = np.nanmean(neighbors)
            a[imonth,:,:] = monthlyIceThickMeanInterp
        
        np.save(dataDir + 'iceCategoryDailyForPersistence' + str(cats[c]) + '_ARISE_20352040.npy',a)
    del a, monthlyIceThickMeanInterp
        
    for c in range(len(cats)):
        print("category " + str(c))
        a = np.full((365,43,288),np.nan)
        for imonth in range(365): 
            nan_indices = np.argwhere(np.isnan(avgCat_ARISE_20642069[cats[c]][imonth,:,:]))
            monthlyIceThickMeanInterp = avgCat_ARISE_20642069[cats[c]][imonth,:,:].copy()
            # Iterate through the NaN indices
            for nan_index in nan_indices:
                row, col = nan_index
                neighbors = []
                
                # Check the surrounding 8 elements (or less at the edges)
                for i in range(max(0, row - 1), min(monthlyIceThickMeanInterp.shape[0], row + 2)):
                    for j in range(max(0, col - 1), min(monthlyIceThickMeanInterp.shape[1], col + 2)):
                        if not np.isnan(monthlyIceThickMeanInterp[i, j]):
                            neighbors.append(monthlyIceThickMeanInterp[i, j])
                
                # If there are valid neighbors, fill the NaN with the average of all valid neighbors
                if neighbors:
                    monthlyIceThickMeanInterp[row, col] = np.nanmean(neighbors)
            a[imonth,:,:] = monthlyIceThickMeanInterp
        
        a = np.where(landMask == 1, np.nan, a)
        np.save(dataDir + 'iceCategoryDailyForPersistence' + str(cats[c]) + '_ARISE_20642069.npy',a)
    del a, monthlyIceThickMeanInterp
    
        
    '''ARISE lower'''   
    for c in range(len(cats)):
        print("category " + str(c))
        a = np.full((365,43,288),np.nan)
        for imonth in range(365): 
            nan_indices = np.argwhere(np.isnan(avgCat_ARISE1_20352040[cats[c]][imonth,:,:]))
            monthlyIceThickMeanInterp = avgCat_ARISE1_20352040[cats[c]][imonth,:,:].copy()
            # Iterate through the NaN indices
            for nan_index in nan_indices:
                row, col = nan_index
                neighbors = []
                
                # Check the surrounding 8 elements (or less at the edges)
                for i in range(max(0, row - 1), min(monthlyIceThickMeanInterp.shape[0], row + 2)):
                    for j in range(max(0, col - 1), min(monthlyIceThickMeanInterp.shape[1], col + 2)):
                        if not np.isnan(monthlyIceThickMeanInterp[i, j]):
                            neighbors.append(monthlyIceThickMeanInterp[i, j])
                
                # If there are valid neighbors, fill the NaN with the average of all valid neighbors
                if neighbors:
                    monthlyIceThickMeanInterp[row, col] = np.nanmean(neighbors)
            a[imonth,:,:] = monthlyIceThickMeanInterp
        
        np.save(dataDir + 'iceCategoryDailyForPersistence' + str(cats[c]) + '_ARISE1_20352040.npy',a)
    del a, monthlyIceThickMeanInterp
        
    for c in range(len(cats)):
        print("category " + str(c))
        a = np.full((365,43,288),np.nan)
        for imonth in range(365): 
            nan_indices = np.argwhere(np.isnan(avgCat_ARISE1_20642069[cats[c]][imonth,:,:]))
            monthlyIceThickMeanInterp = avgCat_ARISE1_20642069[cats[c]][imonth,:,:].copy()
            # Iterate through the NaN indices
            for nan_index in nan_indices:
                row, col = nan_index
                neighbors = []
                
                # Check the surrounding 8 elements (or less at the edges)
                for i in range(max(0, row - 1), min(monthlyIceThickMeanInterp.shape[0], row + 2)):
                    for j in range(max(0, col - 1), min(monthlyIceThickMeanInterp.shape[1], col + 2)):
                        if not np.isnan(monthlyIceThickMeanInterp[i, j]):
                            neighbors.append(monthlyIceThickMeanInterp[i, j])
                
                # If there are valid neighbors, fill the NaN with the average of all valid neighbors
                if neighbors:
                    monthlyIceThickMeanInterp[row, col] = np.nanmean(neighbors)
            a[imonth,:,:] = monthlyIceThickMeanInterp
        
        np.save(dataDir + 'iceCategoryDailyForPersistence' + str(cats[c]) + '_ARISE1_20642069.npy',a)
    del a, monthlyIceThickMeanInterp
        
    return 



def ensembleMemsPersistence(simulation, threshold):
    import xarray as xr
    import numpy as np
    
    dataDir = '/Users/arielmor/Desktop/SAI/ArcticShipping/data/'
    ens = ['001','002','003','004','005','006','007','008','009','010']
    cats = ['1', '2', '3', '4', '5']
    
    #### read ice categories
    cat_SSP_20352040    = {}
    cat_SSP_20642069    = {}
    cat_ARISE_20352040  = {}
    cat_ARISE_20642069  = {}
    cat_ARISE1_20352040 = {}
    cat_ARISE1_20642069 = {}
    
    for c in range(len(cats)):
        print(c)
        for i in range(len(ens)):
            ## SSP2-4.5 - 2035-2040
            ds = xr.open_dataset(dataDir + 'b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.' 
                                 + str(ens[i]) + '.cice.h1.aicen_d_only.20350101-20391231_cat' + str(cats[c]) + '_RG_NH.nc', 
                                 decode_times = False)
            ds['time'] = xr.cftime_range(start = '2035-01-01', periods = ds.sizes['time'], freq = "D", calendar = "noleap")
            aicen_20352039 = ds.aicen_d
            
            ds = xr.open_dataset(dataDir + 'b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.' 
                                 + str(ens[i]) + '.cice.h1.aicen_d_only.20400101-20401231.cat' + str(cats[c]) + '_RG_NH.nc', 
                                 decode_times = False)
            ds['time'] = xr.cftime_range(start = '2040-01-01', periods = ds.sizes['time'], freq = "D", calendar = "noleap")
            aicen_2040 = ds.aicen_d
            
            aicen_20352040 = xr.concat([aicen_20352039,aicen_2040], dim='time')
            
            cat_SSP_20352040[cats[c], ens[i]] = aicen_20352040.groupby('time.dayofyear').mean('time', skipna = True)
            ds.close()
            
            ## SSP2-4.5 - 2064-2069
            ds = xr.open_dataset(dataDir + 'b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.' 
                                 + str(ens[i]) + '.cice.h1.aicen_d_only.20650101-20691231_cat' + str(cats[c]) + '_RG_NH.nc', 
                                 decode_times = False)
            ds['time'] = xr.cftime_range(start = '2065-01-01', periods = ds.sizes['time'], freq = "D", calendar = "noleap")
            aicen_20652069 = ds.aicen_d
            
            ds = xr.open_dataset(dataDir + 'b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.' 
                                 + str(ens[i]) + '.cice.h1.aicen_d_only.20640101-20641231.cat' + str(cats[c]) + '_RG_NH.nc', 
                                 decode_times = False)
            ds['time'] = xr.cftime_range(start = '2064-01-01', periods = ds.sizes['time'], freq = "D", calendar = "noleap")
            aicen_2064 = ds.aicen_d
            
            aicen_20642069 = xr.concat([aicen_2064,aicen_20652069], dim='time')
            
            cat_SSP_20642069[cats[c], ens[i]] = aicen_20642069.groupby('time.dayofyear').mean('time', skipna = True)
            ds.close()
        
        
            ## ARISE-1.5 - 20342039
            ds = xr.open_dataset(dataDir + 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.' 
                                 + str(ens[i]) + '.cice.h1.aicen_d_only.20350101-20391231_cat' + str(cats[c]) + '_RG_NH.nc',
                                 decode_times = False)
            ds['time'] = xr.cftime_range(start = '2035-01-01', periods = ds.sizes['time'], freq = "D", calendar = "noleap")
            aicen_20352039 = ds.aicen_d
            
            ds = xr.open_dataset(dataDir + 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.' 
                                 + str(ens[i]) + '.cice.h1.aicen_d_only.20400101-20401231.cat' + str(cats[c]) + '_RG_NH.nc',
                                 decode_times = False)
            ds['time'] = xr.cftime_range(start = '2040-01-01', periods = ds.sizes['time'], freq = "D", calendar = "noleap")
            aicen_2040 = ds.aicen_d
            
            aicen_20352040 = xr.concat([aicen_20352039,aicen_2040], dim='time')
            
            cat_ARISE_20352040[cats[c], ens[i]] = aicen_20352040.groupby('time.dayofyear').mean('time', skipna = True)
            ds.close()
            
            ## ARISE-1.5 - 20642069
            ds = xr.open_dataset(dataDir + 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.' 
                                 + str(ens[i]) + '.cice.h1.aicen_d_only.20650101-20691231_cat' + str(cats[c]) + '_RG_NH.nc',
                                 decode_times = False)
            ds['time'] = xr.cftime_range(start = '2065-01-01', periods = ds.sizes['time'], freq = "D", calendar = "noleap")
            aicen_20652069 = ds.aicen_d
            
            ds = xr.open_dataset(dataDir + 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.' 
                                 + str(ens[i]) + '.cice.h1.aicen_d_only.20640101-20641231.cat' + str(cats[c]) + '_RG_NH.nc',
                                 decode_times = False)
            ds['time'] = xr.cftime_range(start = '2064-01-01', periods = ds.sizes['time'], freq = "D", calendar = "noleap")
            aicen_2064 = ds.aicen_d
            
            aicen_20642069 = xr.concat([aicen_2064,aicen_20652069], dim='time')
            
            cat_ARISE_20642069[cats[c], ens[i]] = aicen_20642069.groupby('time.dayofyear').mean('time', skipna = True)
            ds.close()
            
            ## ARISE-1.0 - 20352040
            ds = xr.open_dataset(dataDir + 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-LOWER-0.5.' 
                                 + str(ens[i]) + '.cice.h1.aicen_d_only.20350101-20391231_cat' + str(cats[c]) + '_RG_NH.nc',
                                 decode_times = False)
            ds['time'] = xr.cftime_range(start = '2035-01-01', periods = ds.sizes['time'], freq = "D", calendar = "noleap")
            aicen_20352039 = ds.aicen_d
            
            ds = xr.open_dataset(dataDir + 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-LOWER-0.5.' 
                                 + str(ens[i]) + '.cice.h1.aicen_d_only.20400101-20401231.cat' + str(cats[c]) + '_RG_NH.nc',
                                 decode_times = False)
            ds['time'] = xr.cftime_range(start = '2040-01-01', periods = ds.sizes['time'], freq = "D", calendar = "noleap")
            aicen_2040 = ds.aicen_d
            
            aicen_20352040 = xr.concat([aicen_20352039,aicen_2040], dim='time')
            
            cat_ARISE1_20352040[cats[c], ens[i]] = aicen_20352040.groupby('time.dayofyear').mean('time', skipna = True)
            ds.close()
            
            
            ## ARISE-1.0 - 20642069
            ds = xr.open_dataset(dataDir + 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-LOWER-0.5.' 
                                 + str(ens[i]) + '.cice.h1.aicen_d_only.20650101-20691231_cat' + str(cats[c]) + '_RG_NH.nc',
                                 decode_times = False)
            ds['time'] = xr.cftime_range(start = '2065-01-01', periods = ds.sizes['time'], freq = "D", calendar = "noleap")
            aicen_20652069 = ds.aicen_d
            
            ds = xr.open_dataset(dataDir + 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-LOWER-0.5.' 
                                 + str(ens[i]) + '.cice.h1.aicen_d_only.20640101-20641231.cat' + str(cats[c]) + '_RG_NH.nc',
                                 decode_times = False)
            ds['time'] = xr.cftime_range(start = '2064-01-01', periods = ds.sizes['time'], freq = "D", calendar = "noleap")
            aicen_2064 = ds.aicen_d
            
            aicen_20642069 = xr.concat([aicen_2064,aicen_20652069], dim='time')
            
            cat_ARISE1_20642069[cats[c], ens[i]] = aicen_20642069.groupby('time.dayofyear').mean('time', skipna = True)
            ds.close()
            
            
    #### read daily ice thickness
    if simulation == 'arise-1.5': 
        simName = 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.'
        print(simName)
        timePeriod = '20350101-20691231'
        startDate = '2035-01-01'
    elif simulation == 'ssp245':
        simName = 'b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.'
        print(simName)
        timePeriod = '20150101-20691231'
        startDate = '2015-01-01'
    elif simulation == 'arise-1.0':
        simName = 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-LOWER-0.5.'
        print(simName)
        timePeriod = '20350101-20691231'
        startDate = '2035-01-01'
        
    dailyIceThick_20352040 = {}
    dailyIceThick_20642069 = {}
    for i in range(len(ens)):
        ds = xr.open_dataset(dataDir + str(simName) + str(ens[i]) + '.cice.h1.hi_d_only.' + str(timePeriod) + '_RG_NH.nc',
                             decode_times=False)
        ds['time'] = xr.cftime_range(start = startDate, 
                                     periods = ds.sizes['time'], freq = "D", calendar = "noleap")
        latIce = ds.lat; lonIce = ds.lon
        
        if simulation == 'ssp245':
            dailyIceThick_20352040[ens[i]] = (ds.hi_d[7300:9125,:,:]).groupby('time.dayofyear').mean('time', skipna=True)
        else:    
            dailyIceThick_20352040[ens[i]] = (ds.hi_d[:1825,:,:]).groupby('time.dayofyear').mean('time', skipna=True)
        
        dailyIceThick_20642069[ens[i]] = (ds.hi_d[-1825:,:,:]).groupby('time.dayofyear').mean('time', skipna=True)
        ds.close()
        
    
        
    #### ice numeral ----------------------
    iceNumeralSumPC6_ens_20352040 = {}
    iceNumeralDaily_ens_20352040 = {}
    iceNumeralSumPC6_ens_20352040 = {}
    for i in range(len(ens)):
        print(i)
        if simulation == 'ssp245':
            category2 = cat_SSP_20352040[cats[1],ens[i]]
            category3 = cat_SSP_20352040[cats[2],ens[i]]
            category4 = cat_SSP_20352040[cats[3],ens[i]]
            category5 = cat_SSP_20352040[cats[4],ens[i]]
        elif simulation == 'arise-1.5':
            category2 = cat_ARISE_20352040[cats[1],ens[i]]
            category3 = cat_ARISE_20352040[cats[2],ens[i]]
            category4 = cat_ARISE_20352040[cats[3],ens[i]]
            category5 = cat_ARISE_20352040[cats[4],ens[i]]
        elif simulation == 'arise-1.0':
            category2 = cat_ARISE1_20352040[cats[1],ens[i]]
            category3 = cat_ARISE1_20352040[cats[2],ens[i]]
            category4 = cat_ARISE1_20352040[cats[3],ens[i]]
            category5 = cat_ARISE1_20352040[cats[4],ens[i]]
            
        iceNumeralDaily_ens_20352040[ens[i]]  = np.zeros((365,43,288)) # 1 or 0 for each day in 5 year average
        iceNumeralSumPC6_ens_20352040[ens[i]] = np.zeros((43,288)) # total sum for 5 year average  
        
        for iday in range(365):
            iceNumeralDaily_ens_20352040[ens[i]][iday,:,:] = ice_numeral_PC6(
                dailyIceThick_20352040[ens[i]][iday,:,:],
                category3[iday,:,:],
                category4[iday,:,:],
                category5[iday,:,:])
        iceNumeralSumPC6_ens_20352040[ens[i]] = np.nansum(iceNumeralDaily_ens_20352040[ens[i]], axis=0)
        np.save(dataDir + '/persistenceIceNumeralPC6_ens_20352040_' + str(simulation) + '_' + str(ens[i]) + '.npy', iceNumeralDaily_ens_20352040[ens[i]])
    
    
    ## 2065-2069
    iceNumeralSumPC6_ens_20642069 = {}
    iceNumeralDaily_ens_20642069 = {}
    iceNumeralSumPC6_ens_20642069 = {}
    for i in range(len(ens)):
        print(i)
        if simulation == 'ssp245':
            category3 = cat_SSP_20642069[cats[2],ens[i]]
            category4 = cat_SSP_20642069[cats[3],ens[i]]
            category5 = cat_SSP_20642069[cats[4],ens[i]]
        elif simulation == 'arise-1.5':
            category3 = cat_ARISE_20642069[cats[2],ens[i]]
            category4 = cat_ARISE_20642069[cats[3],ens[i]]
            category5 = cat_ARISE_20642069[cats[4],ens[i]]
        elif simulation == 'arise-1.0':
            category3 = cat_ARISE1_20642069[cats[2],ens[i]]
            category4 = cat_ARISE1_20642069[cats[3],ens[i]]
            category5 = cat_ARISE1_20642069[cats[4],ens[i]]
            
        iceNumeralDaily_ens_20642069[ens[i]]  = np.zeros((365,43,288)) # 1 or 0 for each day in 5 year average
        iceNumeralSumPC6_ens_20642069[ens[i]] = np.zeros((43,288)) # total sum for 5 year average  
        
        for iday in range(365):
            iceNumeralDaily_ens_20642069[ens[i]][iday,:,:] = ice_numeral_PC6(
                dailyIceThick_20642069[ens[i]][iday,:,:],
                category3[iday,:,:],
                category4[iday,:,:],
                category5[iday,:,:])
        iceNumeralSumPC6_ens_20642069[ens[i]] = np.nansum(iceNumeralDaily_ens_20642069[ens[i]], axis=0)
        np.save(dataDir + '/persistenceIceNumeralPC6_ens_20642069_' + str(simulation) + '_' + str(ens[i]) + '.npy', iceNumeralDaily_ens_20642069[ens[i]])
    
        
    
    #### difference
    diffIceNumeralPC6_pos = {}
    diffIceNumeralPC6_neg = {}
    for i in range(len(ens)):
        diffIceNumeralPC6_pos[ens[i]] = np.where((
            iceNumeralSumPC6_ens_20642069[ens[i]] - iceNumeralSumPC6_ens_20352040[ens[i]]) > 0., 1, np.nan)
        diffIceNumeralPC6_neg[ens[i]] = np.where((
            iceNumeralSumPC6_ens_20642069[ens[i]] - iceNumeralSumPC6_ens_20352040[ens[i]]) < 0., 1, np.nan)
        

        
    diffIceNumeralPC6_pos_agree = np.where((np.nansum(np.stack(diffIceNumeralPC6_pos.values()),axis=0)) >= 8., 1., np.nan)
    diffIceNumeralPC6_neg_agree = np.where((np.nansum(np.stack(diffIceNumeralPC6_neg.values()),axis=0)) >= 8., 1., np.nan)
    diffIceNumeralPC6_agreement = np.nansum(np.stack((diffIceNumeralPC6_pos_agree,
                                                  diffIceNumeralPC6_neg_agree),axis=0), axis=0)
    diffIceNumeralPC6_agreement[diffIceNumeralPC6_agreement == 0.] = np.nan
    if simulation == 'ssp245': np.save(dataDir + '/persistenceDiffIceNumeralPC6_agreement_ssp.npy', diffIceNumeralPC6_agreement)
    elif simulation == 'arise-1.5': np.save(dataDir + '/persistenceDiffIceNumeralPC6_agreement_arise.npy', diffIceNumeralPC6_agreement)
    elif simulation == 'arise-1.0': np.save(dataDir + '/persistenceDiffIceNumeralPC6_agreement_arise1.npy', diffIceNumeralPC6_agreement)
    
    
    #### OW ice numeral ----------------------
    iceNumeralSumOW_ens_20352040 = {}
    iceNumeralDaily_ens_20352040 = {}
    iceNumeralSumOW_ens_20352040 = {}
    for i in range(len(ens)):
        print(i)
        if simulation == 'ssp245':
            category2 = cat_SSP_20352040[cats[1],ens[i]]
            category3 = cat_SSP_20352040[cats[2],ens[i]]
            category4 = cat_SSP_20352040[cats[3],ens[i]]
            category5 = cat_SSP_20352040[cats[4],ens[i]]
        elif simulation == 'arise-1.5':
            category2 = cat_ARISE_20352040[cats[1],ens[i]]
            category3 = cat_ARISE_20352040[cats[2],ens[i]]
            category4 = cat_ARISE_20352040[cats[3],ens[i]]
            category5 = cat_ARISE_20352040[cats[4],ens[i]]
        elif simulation == 'arise-1.0':
            category2 = cat_ARISE1_20352040[cats[1],ens[i]]
            category3 = cat_ARISE1_20352040[cats[2],ens[i]]
            category4 = cat_ARISE1_20352040[cats[3],ens[i]]
            category5 = cat_ARISE1_20352040[cats[4],ens[i]]
            
        iceNumeralDaily_ens_20352040[ens[i]]  = np.zeros((365,43,288)) # 1 or 0 for each day in 5 year average
        iceNumeralSumOW_ens_20352040[ens[i]] = np.zeros((43,288)) # total sum for 5 year average  
        
        for iday in range(365):
            iceNumeralDaily_ens_20352040[ens[i]][iday,:,:] = ice_numeral_OW(
                dailyIceThick_20352040[ens[i]][iday,:,:],
                category2[iday,:,:],
                category3[iday,:,:],
                category4[iday,:,:],
                category5[iday,:,:])
        iceNumeralSumOW_ens_20352040[ens[i]] = np.nansum(iceNumeralDaily_ens_20352040[ens[i]], axis=0)
        np.save(dataDir + '/persistenceIceNumeralOW_ens_20352040_' + str(simulation) + '_' + str(ens[i]) + '.npy', iceNumeralDaily_ens_20352040[ens[i]])
    
    del category2, category3, category4, category5
    
    ## 2065-2069
    iceNumeralSumOW_ens_20642069 = {}
    iceNumeralDaily_ens_20642069 = {}
    iceNumeralSumOW_ens_20642069 = {}
    for i in range(len(ens)):
        print(i)
        if simulation == 'ssp245':
            category2 = cat_SSP_20642069[cats[1],ens[i]]
            category3 = cat_SSP_20642069[cats[2],ens[i]]
            category4 = cat_SSP_20642069[cats[3],ens[i]]
            category5 = cat_SSP_20642069[cats[4],ens[i]]
        elif simulation == 'arise-1.5':
            category2 = cat_ARISE_20642069[cats[1],ens[i]]
            category3 = cat_ARISE_20642069[cats[2],ens[i]]
            category4 = cat_ARISE_20642069[cats[3],ens[i]]
            category5 = cat_ARISE_20642069[cats[4],ens[i]]
        elif simulation == 'arise-1.0':
            category2 = cat_ARISE1_20642069[cats[1],ens[i]]
            category3 = cat_ARISE1_20642069[cats[2],ens[i]]
            category4 = cat_ARISE1_20642069[cats[3],ens[i]]
            category5 = cat_ARISE1_20642069[cats[4],ens[i]]
            
        iceNumeralDaily_ens_20642069[ens[i]]  = np.zeros((365,43,288)) # 1 or 0 for each day in 5 year average
        iceNumeralSumOW_ens_20642069[ens[i]] = np.zeros((43,288)) # total sum for 5 year average  
        
        for iday in range(365):
            iceNumeralDaily_ens_20642069[ens[i]][iday,:,:] = ice_numeral_OW(
                dailyIceThick_20642069[ens[i]][iday,:,:],
                category2[iday,:,:],
                category3[iday,:,:],
                category4[iday,:,:],
                category5[iday,:,:])
        iceNumeralSumOW_ens_20642069[ens[i]] = np.nansum(iceNumeralDaily_ens_20642069[ens[i]], axis=0)
        np.save(dataDir + '/persistenceIceNumeralOW_ens_20642069_' + str(simulation) + '_' + str(ens[i]) + '.npy', iceNumeralDaily_ens_20642069[ens[i]])      
    
    #### difference
    diffIceNumeralOW_pos = {}
    diffIceNumeralOW_neg = {}
    for i in range(len(ens)):
        diffIceNumeralOW_pos[ens[i]] = np.where((
            iceNumeralSumOW_ens_20642069[ens[i]] - iceNumeralSumOW_ens_20352040[ens[i]]) > 0., 1, np.nan)
        diffIceNumeralOW_neg[ens[i]] = np.where((
            iceNumeralSumOW_ens_20642069[ens[i]] - iceNumeralSumOW_ens_20352040[ens[i]]) < 0., 1, np.nan)
        

        
    diffIceNumeralOW_pos_agree = np.where((np.nansum(np.stack(diffIceNumeralOW_pos.values()),axis=0)) >= 8., 1., np.nan)
    diffIceNumeralOW_neg_agree = np.where((np.nansum(np.stack(diffIceNumeralOW_neg.values()),axis=0)) >= 8., 1., np.nan)
    diffIceNumeralOW_agreement = np.nansum(np.stack((diffIceNumeralOW_pos_agree,
                                                  diffIceNumeralOW_neg_agree),axis=0), axis=0)
    diffIceNumeralOW_agreement[diffIceNumeralOW_agreement == 0.] = np.nan
    if simulation == 'ssp245': np.save(dataDir + '/persistenceDiffIceNumeralOW_agreement_ssp.npy', diffIceNumeralOW_agreement)
    elif simulation == 'arise-1.5': np.save(dataDir + '/persistenceDiffIceNumeralOW_agreement_arise.npy', diffIceNumeralOW_agreement)
    elif simulation == 'arise-1.0': np.save(dataDir + '/persistenceDiffIceNumeralOW_agreement_arise1.npy', diffIceNumeralOW_agreement)
    
    
    #### calculate persistence
    iceNumeralSumPC6_2035 = {}
    iceNumeralSumPC6_2065 = {}
    iceNumeralSumOW_2035 = {}
    iceNumeralSumOW_2065 = {}
    
    diffPC6_pos = {}
    diffPC6_neg = {}
    diffOW_pos = {}
    diffOW_neg = {}
    
    for i in range(len(ens)):
        iceNumeralSumPC6_2035[ens[i]] = np.load(dataDir + '/persistenceIceNumeralPC6_ens_20352040_' + str(simulation) + '_' + str(ens[i]) + '.npy', allow_pickle = True)
        iceNumeralSumPC6_2065[ens[i]] = np.load(dataDir + '/persistenceIceNumeralPC6_ens_20642069_' + str(simulation) + '_' + str(ens[i]) + '.npy', allow_pickle = True)
        iceNumeralSumOW_2035[ens[i]] = np.load(dataDir + '/persistenceIceNumeralOW_ens_20352040_' + str(simulation) + '_' + str(ens[i]) + '.npy', allow_pickle = True)
        iceNumeralSumOW_2065[ens[i]] = np.load(dataDir + '/persistenceIceNumeralOW_ens_20642069_' + str(simulation) + '_' + str(ens[i]) + '.npy', allow_pickle = True)
    
        ## ens persistence 
        firstDayThinIcePC6_20352040 = np.empty((43,288))
        firstDayThinIcePC6_20642069 = np.empty((43,288))
        firstDayThinIceOW_20352040  = np.empty((43,288))
        firstDayThinIceOW_20642069  = np.empty((43,288))
        for ilat in range(len(latIce)):
            for ilon in range(len(lonIce)):
                firstDayThinIcePC6_20352040[ilat,ilon] = thinIceConditions_30days(iceNumeralSumPC6_2035[ens[i]][:,ilat,ilon],0.5,threshold)
                firstDayThinIcePC6_20642069[ilat,ilon] = thinIceConditions_30days(iceNumeralSumPC6_2065[ens[i]][:,ilat,ilon],0.5,threshold)
                firstDayThinIceOW_20352040[ilat,ilon]  = thinIceConditions_30days(iceNumeralSumOW_2035[ens[i]][:,ilat,ilon],0.5,threshold)
                firstDayThinIceOW_20642069[ilat,ilon]  = thinIceConditions_30days(iceNumeralSumOW_2065[ens[i]][:,ilat,ilon],0.5,threshold)      
    
        np.save(dataDir + '/firstDayThinIcePC6_20352040_' + str(simulation) + '_' + str(ens[i]) + '.npy', firstDayThinIcePC6_20352040)
        np.save(dataDir + '/firstDayThinIcePC6_20642069_' + str(simulation) + '_' + str(ens[i]) + '.npy', firstDayThinIcePC6_20642069)
        np.save(dataDir + '/firstDayThinIceOW_20352040_' + str(simulation) + '_' + str(ens[i]) + '.npy', firstDayThinIceOW_20352040)
        np.save(dataDir + '/firstDayThinIceOW_20642069_' + str(simulation) + '_' + str(ens[i]) + '.npy', firstDayThinIceOW_20642069)
    
        #### difference
        diffPC6_pos[ens[i]] = np.where((firstDayThinIcePC6_20642069 - firstDayThinIcePC6_20352040) > 0., 1, np.nan)
        diffPC6_neg[ens[i]] = np.where((firstDayThinIcePC6_20642069 - firstDayThinIcePC6_20352040) < 0., 1, np.nan)
        diffOW_pos[ens[i]] = np.where((firstDayThinIceOW_20642069 - firstDayThinIceOW_20352040) > 0., 1, np.nan)
        diffOW_neg[ens[i]] = np.where((firstDayThinIceOW_20642069 - firstDayThinIceOW_20352040) < 0., 1, np.nan)
        
        diffPC6_pos_agree = np.where((np.nansum(np.stack(diffPC6_pos.values()),axis=0)) >= 8., 1., np.nan)
        diffPC6_neg_agree = np.where((np.nansum(np.stack(diffPC6_neg.values()),axis=0)) >= 8., 1., np.nan)
        diffPC6_agreement = np.nansum(np.stack((diffPC6_pos_agree,
                                                      diffPC6_neg_agree),axis=0), axis=0)
        diffPC6_agreement[diffPC6_agreement == 0.] = np.nan
        
        diffOW_pos_agree = np.where((np.nansum(np.stack(diffOW_pos.values()),axis=0)) >= 8., 1., np.nan)
        diffOW_neg_agree = np.where((np.nansum(np.stack(diffOW_neg.values()),axis=0)) >= 8., 1., np.nan)
        diffOW_agreement = np.nansum(np.stack((diffOW_pos_agree,
                                                      diffOW_neg_agree),axis=0), axis=0)
        diffOW_agreement[diffOW_agreement == 0.] = np.nan


    np.save(dataDir + '/persistenceDiffIceNumeralPC6_agreement_' + str(simulation) + '.npy', diffPC6_agreement)
    np.save(dataDir + '/persistenceDiffIceNumeralOW_agreement_' + str(simulation) + '.npy', diffOW_agreement)
    
    return



def interpolateMonthlyIce(simulation, testMap):
    from readSeaIce import readMonthlyIce
    import numpy as np
    from plottingFunctions import make_maps
    
    landMask = np.load('/Users/arielmor/Desktop/SAI/ArcticShipping/data/landMask_from_global.npy')
    
    latIce, lonIce, iceFracSSP, iceThickSSP, iceFracARISE, iceThickARISE, iceFracARISE_1, iceThickARISE_1 = readMonthlyIce()
    
    if simulation == 'ssp245':
        monthlyIceThickMean = np.nanmean((np.stack(iceThickSSP.values())),axis=0) 
        monthlyIceFracMean  = np.nanmean((np.stack(iceFracSSP.values())),axis=0)
    elif simulation == 'arise-1.5':
        monthlyIceThickMean = np.nanmean((np.stack(iceThickARISE.values())),axis=0) 
        monthlyIceFracMean  = np.nanmean((np.stack(iceFracARISE.values())),axis=0)
    elif simulation == 'arise-1.0':
        monthlyIceThickMean = np.nanmean((np.stack(iceThickARISE_1.values())),axis=0) 
        monthlyIceFracMean  = np.nanmean((np.stack(iceFracARISE_1.values())),axis=0)
    
    #### interpolate over small channels
    a = np.full((420,43,288),np.nan)
    for imonth in range(420): 
        nan_indices = np.argwhere(np.isnan(monthlyIceThickMean[imonth,:,:]))
        monthlyIceThickMeanInterp = monthlyIceThickMean[imonth,:,:].copy()
        # Iterate through the NaN indices
        for nan_index in nan_indices:
            row, col = nan_index
            neighbors = []
            
            # Check the surrounding 8 elements (or less at the edges)
            for i in range(max(0, row - 1), min(monthlyIceThickMeanInterp.shape[0], row + 2)):
                for j in range(max(0, col - 1), min(monthlyIceThickMeanInterp.shape[1], col + 2)):
                    if not np.isnan(monthlyIceThickMeanInterp[i, j]):
                        neighbors.append(monthlyIceThickMeanInterp[i, j])
            
            # If there are valid neighbors, fill the NaN with the average of all valid neighbors
            if neighbors:
                monthlyIceThickMeanInterp[row, col] = np.nanmean(neighbors)
        a[imonth,:,:] = monthlyIceThickMeanInterp
    
    a = np.where(landMask == 1, np.nan, a)
    
    if simulation == 'ssp245':
        np.save('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthlyIceThickMeanInterp_ssp.npy',a)
    elif simulation == 'arise-1.5':
        np.save('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthlyceThickMeanInterp_arise.npy',a)
    elif simulation == 'arise-1.0':
        np.save('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthlyIceThickMeanInterp_arise1.0.npy',a)
        
    del a, monthlyIceThickMeanInterp
        
    #### interpolate    
    a = np.full((420,43,288),np.nan)
    for imonth in range(420): 
        nan_indices = np.argwhere(np.isnan(monthlyIceFracMean[imonth,:,:]))
        monthlyIceFracMeanInterp = monthlyIceFracMean[imonth,:,:].copy()
        # Iterate through the NaN indices
        for nan_index in nan_indices:
            row, col = nan_index
            neighbors = []
            
            # Check the surrounding 8 elements (or less at the edges)
            for i in range(max(0, row - 1), min(monthlyIceFracMeanInterp.shape[0], row + 2)):
                for j in range(max(0, col - 1), min(monthlyIceFracMeanInterp.shape[1], col + 2)):
                    if not np.isnan(monthlyIceFracMeanInterp[i, j]):
                        neighbors.append(monthlyIceFracMeanInterp[i, j])
            
            # If there are valid neighbors, fill the NaN with the average of all valid neighbors
            if neighbors:
                monthlyIceFracMeanInterp[row, col] = np.nanmean(neighbors)
        a[imonth,:,:] = monthlyIceFracMeanInterp
    
    a = np.where(landMask == 1, np.nan, a)
    
    if simulation == 'ssp245':
        np.save('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthlyIceFracMeanInterp_ssp.npy',a)
    elif simulation == 'arise-1.5':
        np.save('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthlyceFracMeanInterp_arise.npy',a)
    elif simulation == 'arise-1.0':
        np.save('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthlyIceFracMeanInterp_arise1.0.npy',a)
        
    #### test map
    if testMap:
        fig, ax = make_maps(a[4,:,:],
                            latIce,lonIce,0,100,10,
                            'viridis',' ', str(simulation) + ' test',
                            'test_map','neither',False,True)
        
    return



def navigableDays(simulation, yearRange, navigation):
    import numpy as np
    dataDir = '/Users/arielmor/Desktop/SAI/ArcticShipping/data/'
        
    if simulation == 'ssp245':
        print("ssp245")
        if yearRange == '20352039':
            category2 = np.load(dataDir + 'iceCategoryDaily2_SSP_20352039.npy')
            category3 = np.load(dataDir + 'iceCategoryDaily3_SSP_20352039.npy')
            category4 = np.load(dataDir + 'iceCategoryDaily4_SSP_20352039.npy')
            category5 = np.load(dataDir + 'iceCategoryDaily5_SSP_20352039.npy')
        elif yearRange == '20352040':
            category2 = np.load(dataDir + 'iceCategoryDaily2_SSP_20352039.npy')
            category3 = np.load(dataDir + 'iceCategoryDaily3_SSP_20352039.npy')
            category4 = np.load(dataDir + 'iceCategoryDaily4_SSP_20352039.npy')
            category5 = np.load(dataDir + 'iceCategoryDaily5_SSP_20352039.npy')
        elif yearRange == '20642069':
            category2 = np.load(dataDir + 'iceCategoryDaily2_SSP_20652069.npy')
            category3 = np.load(dataDir + 'iceCategoryDaily3_SSP_20652069.npy')
            category4 = np.load(dataDir + 'iceCategoryDaily4_SSP_20652069.npy')
            category5 = np.load(dataDir + 'iceCategoryDaily5_SSP_20652069.npy')
        elif yearRange == '20652069':
            category2 = np.load(dataDir + 'iceCategoryDaily2_SSP_20652069.npy')
            category3 = np.load(dataDir + 'iceCategoryDaily3_SSP_20652069.npy')
            category4 = np.load(dataDir + 'iceCategoryDaily4_SSP_20652069.npy')
            category5 = np.load(dataDir + 'iceCategoryDaily5_SSP_20652069.npy')
            
    elif simulation == 'arise-1.5':
        print("arise-1.5")
        if yearRange == '20352039':
            category2 = np.load(dataDir + 'iceCategoryDaily2_ARISE_20352039.npy')
            category3 = np.load(dataDir + 'iceCategoryDaily3_ARISE_20352039.npy')
            category4 = np.load(dataDir + 'iceCategoryDaily4_ARISE_20352039.npy')
            category5 = np.load(dataDir + 'iceCategoryDaily5_ARISE_20352039.npy')
        elif yearRange == '20352040':
            category2 = np.load(dataDir + 'iceCategoryDaily2_ARISE_20352039.npy')
            category3 = np.load(dataDir + 'iceCategoryDaily3_ARISE_20352039.npy')
            category4 = np.load(dataDir + 'iceCategoryDaily4_ARISE_20352039.npy')
            category5 = np.load(dataDir + 'iceCategoryDaily5_ARISE_20352039.npy')
        elif yearRange == '20642069':
            category2 = np.load(dataDir + 'iceCategoryDaily2_ARISE_20652069.npy')
            category3 = np.load(dataDir + 'iceCategoryDaily3_ARISE_20652069.npy')
            category4 = np.load(dataDir + 'iceCategoryDaily4_ARISE_20652069.npy')
            category5 = np.load(dataDir + 'iceCategoryDaily5_ARISE_20652069.npy')
        elif yearRange == '20652069':
            category2 = np.load(dataDir + 'iceCategoryDaily2_ARISE_20652069.npy')
            category3 = np.load(dataDir + 'iceCategoryDaily3_ARISE_20652069.npy')
            category4 = np.load(dataDir + 'iceCategoryDaily4_ARISE_20652069.npy')
            category5 = np.load(dataDir + 'iceCategoryDaily5_ARISE_20652069.npy')
            
    elif simulation == 'arise-1.0':
        print("arise-1.0")
        if yearRange == '20352039':
            category2 = np.load(dataDir + 'iceCategoryDaily2_ARISE1_20352039.npy')
            category3 = np.load(dataDir + 'iceCategoryDaily3_ARISE1_20352039.npy')
            category4 = np.load(dataDir + 'iceCategoryDaily4_ARISE1_20352039.npy')
            category5 = np.load(dataDir + 'iceCategoryDaily5_ARISE1_20352039.npy')
        elif yearRange == '20352040':
            category2 = np.load(dataDir + 'iceCategoryDaily2_ARISE1_20352039.npy')
            category3 = np.load(dataDir + 'iceCategoryDaily3_ARISE1_20352039.npy')
            category4 = np.load(dataDir + 'iceCategoryDaily4_ARISE1_20352039.npy')
            category5 = np.load(dataDir + 'iceCategoryDaily5_ARISE1_20352039.npy')
        elif yearRange == '20642069':
            category2 = np.load(dataDir + 'iceCategoryDaily2_ARISE1_20652069.npy')
            category3 = np.load(dataDir + 'iceCategoryDaily3_ARISE1_20652069.npy')
            category4 = np.load(dataDir + 'iceCategoryDaily4_ARISE1_20652069.npy')
            category5 = np.load(dataDir + 'iceCategoryDaily5_ARISE1_20652069.npy')
        elif yearRange == '20652069':
            category2 = np.load(dataDir + 'iceCategoryDaily2_ARISE1_20652069.npy')
            category3 = np.load(dataDir + 'iceCategoryDaily3_ARISE1_20652069.npy')
            category4 = np.load(dataDir + 'iceCategoryDaily4_ARISE1_20652069.npy')
            category5 = np.load(dataDir + 'iceCategoryDaily5_ARISE1_20652069.npy')
            
        
    category2[np.isnan(category2)] = 0
    category3[np.isnan(category3)] = 0
    category4[np.isnan(category4)] = 0
    category5[np.isnan(category5)] = 0
    
    if navigation: dailyIceThickMeanInterp = np.load(dataDir + 'dailyIceThickMeanInterpForNav_' + str(yearRange) + '_' + str(simulation) + '.npy')
    else: dailyIceThickMeanInterp = np.load(dataDir + 'dailyIceThickMeanInterpForPersistence_' + str(yearRange) + '_' + str(simulation) + '.npy')
        
        
    #### PC6
    # including categories:
    iceNumeralDaily  = np.zeros((365,43,288)) # 1 or 0 for each day in 5 year average
    iceNumeralSumPC6 = np.zeros((43,288)) # total sum for 5 year average  
    
    # start March 1, go to Feb 28 following year
    for iday in range(365):
        for ilat in range(43):
            for ilon in range(288):
                iceNumeralDaily[iday,ilat,ilon] = ice_numeral_PC6(
                    dailyIceThickMeanInterp[iday,ilat,ilon],
                    category3[iday,ilat,ilon],
                    category4[iday,ilat,ilon],
                    category5[iday,ilat,ilon])
    iceNumeralSumPC6 = np.nansum(iceNumeralDaily, axis=0) # sums along the day dimension
    if navigation:
        np.save(dataDir + 'PC6IceNumSum_' + str(simulation) + '_full_year_updated_with_categories_' + str(yearRange) + '.npy',iceNumeralSumPC6)
        np.save(dataDir + 'PC6IceNum_' + str(simulation) + '_full_year_updated_with_categories_' + str(yearRange) + '.npy',iceNumeralDaily)
    else:
        np.save(dataDir + 'PC6IceNumSum_' + str(simulation) + '_full_year_updated_with_categories_for_persistence_' + str(yearRange) + '.npy',iceNumeralSumPC6)
        np.save(dataDir + 'PC6IceNum_' + str(simulation) + '_full_year_updated_with_categories_for_persistence_' + str(yearRange) + '.npy',iceNumeralDaily)
    del iceNumeralDaily    
        
    #### OW
    iceNumeralDaily  = np.zeros((365,43,288)) # 1 or 0 for each day in 5 year average
    iceNumeralSumOW = np.zeros((43,288)) # total sum for each year

    for iday in range(365):
        for ilat in range(43):
            for ilon in range(288):
                iceNumeralDaily[iday,ilat,ilon] = ice_numeral_OW(
                    dailyIceThickMeanInterp[iday,ilat,ilon],
                    category2[iday,ilat,ilon],
                    category3[iday,ilat,ilon],
                    category4[iday,ilat,ilon],
                    category5[iday,ilat,ilon])
    iceNumeralSumOW = np.nansum(iceNumeralDaily, axis=0) # sums along the day dimension
    if navigation:
        np.save(dataDir + 'OWIceNumSum_' + str(simulation) + '_full_year_updated_with_categories_' + str(yearRange) + '.npy',iceNumeralSumOW)
        np.save(dataDir + 'OWIceNum_' + str(simulation) + '_full_year_updated_with_categories_' + str(yearRange) + '.npy',iceNumeralDaily)
    else:
        np.save(dataDir + 'OWIceNumSum_' + str(simulation) + '_full_year_updated_with_categories_for_persistence_' + str(yearRange) + '.npy',iceNumeralSumOW)
        np.save(dataDir + 'OWIceNum_' + str(simulation) + '_full_year_updated_with_categories_for_persistence_' + str(yearRange) + '.npy',iceNumeralDaily)
    
    return



def navigableDaysEnsMembers(simulation, yearRange):
    
    def ice_numeral_PC6(icethic, cat3, cat4, cat5):
        IN = np.zeros_like(icethic)
        totalIceConc = cat3 + cat4 + cat5
    
        mask_no_ice = np.isnan(icethic)
        mask_thin_ice = (icethic <= 1.2) & (~mask_no_ice)
        mask_thick_ice = (icethic > 1.2) & (~mask_no_ice)
        mask_little_multi_year_ice = (totalIceConc <= 0.05) & mask_thin_ice
    
        IN[mask_no_ice] = 1  # no ice
        IN[mask_thin_ice & mask_little_multi_year_ice] = 1  # very little multi-year ice
        IN[mask_thin_ice & ~mask_little_multi_year_ice] = 0
        IN[mask_thick_ice] = 0
        return IN
    
    def ice_numeral_OW(icethic,cat2,cat3,cat4,cat5):
        IN = np.zeros_like(icethic)
        totalIceConc = cat2+cat3+cat4+cat5
        
        mask_no_ice = np.isnan(icethic)
        mask_thin_ice = (icethic <= 0.15) & (~mask_no_ice)
        mask_thick_ice = (icethic > 0.15) & (~mask_no_ice)
        mask_little_multi_year_ice = (totalIceConc <= 0.005) & mask_thin_ice
        
        IN[mask_no_ice] = 1  # no ice
        IN[mask_thin_ice & mask_little_multi_year_ice] = 1  # very little multi-year ice
        IN[mask_thin_ice & ~mask_little_multi_year_ice] = 0
        IN[mask_thick_ice] = 0
               
        return IN
        
    

    import numpy as np
    import xarray as xr
    dataDir = '/Users/arielmor/Desktop/SAI/ArcticShipping/data/'

    
    dataDir = '/Users/arielmor/Desktop/SAI/ArcticShipping/data/'
    ens = ['001','002','003','004','005','006','007','008','009','010']
    cats = ['1', '2', '3', '4', '5']
    
    
    #### read ice categories
    cat_SSP_20352039    = {}
    cat_SSP_20652069    = {}
    cat_ARISE_20352039  = {}
    cat_ARISE_20652069  = {}
    cat_ARISE1_20352039 = {}
    cat_ARISE1_20652069 = {}
    
    for c in range(len(cats)):
        for i in range(len(ens)):
            ## SSP2-4.5
            ds = xr.open_dataset(dataDir + 'b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.' 
                                 + str(ens[i]) + '.cice.h1.aicen_d_only.20350101-20391231_cat' + str(cats[c]) + '_RG_NH.nc', 
                                 decode_times = False)
            ds['time'] = xr.cftime_range(start = '2035-01-01', periods = ds.sizes['time'], freq = "D", calendar = "noleap")
            cat_SSP_20352039[cats[c], ens[i]] = (ds.aicen_d).groupby('time.dayofyear').mean('time', skipna = True)
            ds.close()
            
            ds = xr.open_dataset(dataDir + 'b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.' 
                                 + str(ens[i]) + '.cice.h1.aicen_d_only.20650101-20691231_cat' + str(cats[c]) + '_RG_NH.nc', 
                                 decode_times = False)
            ds['time'] = xr.cftime_range(start = '2065-01-01', periods = ds.sizes['time'], freq = "D", calendar = "noleap")
            cat_SSP_20652069[cats[c], ens[i]] = (ds.aicen_d).groupby('time.dayofyear').mean('time', skipna = True)
            ds.close()
        
            ## ARISE-1.5
            ds = xr.open_dataset(dataDir + 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.' 
                                 + str(ens[i]) + '.cice.h1.aicen_d_only.20350101-20391231_cat' + str(cats[c]) + '_RG_NH.nc',
                                 decode_times = False)
            ds['time'] = xr.cftime_range(start = '2035-01-01', periods = ds.sizes['time'], freq = "D", calendar = "noleap")
            cat_ARISE_20352039[cats[c], ens[i]] = (ds.aicen_d).groupby('time.dayofyear').mean('time', skipna = True)
            ds.close()
            
            ds = xr.open_dataset(dataDir + 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.' 
                                 + str(ens[i]) + '.cice.h1.aicen_d_only.20650101-20691231_cat' + str(cats[c]) + '_RG_NH.nc',
                                 decode_times = False)
            ds['time'] = xr.cftime_range(start = '2065-01-01', periods = ds.sizes['time'], freq = "D", calendar = "noleap")
            cat_ARISE_20652069[cats[c], ens[i]] = (ds.aicen_d).groupby('time.dayofyear').mean('time', skipna = True)
            ds.close()
            
            ## ARISE-1.0
            ds = xr.open_dataset(dataDir + 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-LOWER-0.5.' 
                                 + str(ens[i]) + '.cice.h1.aicen_d_only.20350101-20391231_cat' + str(cats[c]) + '_RG_NH.nc',
                                 decode_times = False)
            ds['time'] = xr.cftime_range(start = '2035-01-01', periods = ds.sizes['time'], freq = "D", calendar = "noleap")
            cat_ARISE1_20352039[cats[c], ens[i]] = (ds.aicen_d).groupby('time.dayofyear').mean('time', skipna = True)
            ds.close()
            
            ds = xr.open_dataset(dataDir + 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-LOWER-0.5.' 
                                 + str(ens[i]) + '.cice.h1.aicen_d_only.20650101-20691231_cat' + str(cats[c]) + '_RG_NH.nc',
                                 decode_times = False)
            ds['time'] = xr.cftime_range(start = '2065-01-01', periods = ds.sizes['time'], freq = "D", calendar = "noleap")
            cat_ARISE1_20652069[cats[c], ens[i]] = (ds.aicen_d).groupby('time.dayofyear').mean('time', skipna = True)
            ds.close()
            
    
    #### read ice thickness
    if simulation == 'arise-1.5': 
        simName = 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.'
        print(simName)
        timePeriod = '20350101-20691231'
        startDate = '2035-01-01'
    elif simulation == 'ssp245':
        simName = 'b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.'
        print(simName)
        timePeriod = '20150101-20691231'
        startDate = '2015-01-01'
    elif simulation == 'arise-1.0':
        simName = 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-LOWER-0.5.'
        print(simName)
        timePeriod = '20350101-20691231'
        startDate = '2035-01-01'
        
    
    dailyIceThick_20352039 = {}
    dailyIceThick_20652069 = {}
    for i in range(len(ens)):
        ds = xr.open_dataset(dataDir + str(simName) + str(ens[i]) + '.cice.h1.hi_d_only.' + str(timePeriod) + '_RG_NH.nc',
                             decode_times=False)
        ds['time'] = xr.cftime_range(start = startDate, 
                                     periods = ds.sizes['time'], freq = "D", calendar = "noleap")
        latIce = ds.lat; lonIce = ds.lon
        
        if simulation == 'ssp245':
            dailyIceThick_20352039[ens[i]] = (ds.hi_d[7300:9125,:,:]).groupby('time.dayofyear').mean('time', skipna=True)
        else:    
            dailyIceThick_20352039[ens[i]] = (ds.hi_d[:1825,:,:]).groupby('time.dayofyear').mean('time', skipna=True)
        
        dailyIceThick_20652069[ens[i]] = (ds.hi_d[-1825:,:,:]).groupby('time.dayofyear').mean('time', skipna=True)
        ds.close()
        
    
        
    #### ice numeral ----------------------
    iceNumeralSumPC6_ens_20352039 = {}
    iceNumeralDaily_ens_20352039 = {}
    iceNumeralSumPC6_ens_20352039 = {}
    for i in range(len(ens)):
        print(i)
        if simulation == 'ssp245':
            category3 = cat_SSP_20352039[cats[2],ens[i]]
            category4 = cat_SSP_20352039[cats[3],ens[i]]
            category5 = cat_SSP_20352039[cats[4],ens[i]]
        elif simulation == 'arise-1.5':
            category3 = cat_ARISE_20352039[cats[2],ens[i]]
            category4 = cat_ARISE_20352039[cats[3],ens[i]]
            category5 = cat_ARISE_20352039[cats[4],ens[i]]
        elif simulation == 'arise-1.0':
            category3 = cat_ARISE1_20352039[cats[2],ens[i]]
            category4 = cat_ARISE1_20352039[cats[3],ens[i]]
            category5 = cat_ARISE1_20352039[cats[4],ens[i]]
            
        iceNumeralDaily_ens_20352039[ens[i]]  = np.zeros((365,43,288)) # 1 or 0 for each day in 5 year average
        iceNumeralSumPC6_ens_20352039[ens[i]] = np.zeros((43,288)) # total sum for 5 year average  
        
        for iday in range(365):
            iceNumeralDaily_ens_20352039[ens[i]][iday,:,:] = ice_numeral_PC6(
                dailyIceThick_20352039[ens[i]][iday,:,:],
                category3[iday,:,:],
                category4[iday,:,:],
                category5[iday,:,:])
        iceNumeralSumPC6_ens_20352039[ens[i]] = np.nansum(iceNumeralDaily_ens_20352039[ens[i]], axis=0)
        np.save(dataDir + '/iceNumeralSumPC6_ens_20352039_' + str(simulation) + '_' + str(ens[i]) + '.npy', iceNumeralSumPC6_ens_20352039)
    
    
    ## 2065-2069
    iceNumeralSumPC6_ens_20652069 = {}
    iceNumeralDaily_ens_20652069 = {}
    iceNumeralSumPC6_ens_20652069 = {}
    for i in range(len(ens)):
        print(i)
        if simulation == 'ssp245':
            category3 = cat_SSP_20652069[cats[2],ens[i]]
            category4 = cat_SSP_20652069[cats[3],ens[i]]
            category5 = cat_SSP_20652069[cats[4],ens[i]]
        elif simulation == 'arise-1.5':
            category3 = cat_ARISE_20652069[cats[2],ens[i]]
            category4 = cat_ARISE_20652069[cats[3],ens[i]]
            category5 = cat_ARISE_20652069[cats[4],ens[i]]
        elif simulation == 'arise-1.0':
            category3 = cat_ARISE1_20652069[cats[2],ens[i]]
            category4 = cat_ARISE1_20652069[cats[3],ens[i]]
            category5 = cat_ARISE1_20652069[cats[4],ens[i]]
            
        iceNumeralDaily_ens_20652069[ens[i]]  = np.zeros((365,43,288)) # 1 or 0 for each day in 5 year average
        iceNumeralSumPC6_ens_20652069[ens[i]] = np.zeros((43,288)) # total sum for 5 year average  
        
        for iday in range(365):
            iceNumeralDaily_ens_20652069[ens[i]][iday,:,:] = ice_numeral_PC6(
                dailyIceThick_20652069[ens[i]][iday,:,:],
                category3[iday,:,:],
                category4[iday,:,:],
                category5[iday,:,:])
        iceNumeralSumPC6_ens_20652069[ens[i]] = np.nansum(iceNumeralDaily_ens_20652069[ens[i]], axis=0)
        np.save(dataDir + '/iceNumeralSumPC6_ens_20652069_' + str(simulation) + '_' + str(ens[i]) + '.npy', iceNumeralSumPC6_ens_20652069)
    
        
    
    #### difference
    diffIceNumeralPC6_pos = {}
    diffIceNumeralPC6_neg = {}
    for i in range(len(ens)):
        diffIceNumeralPC6_pos[ens[i]] = np.where((
            iceNumeralSumPC6_ens_20652069[ens[i]] - iceNumeralSumPC6_ens_20352039[ens[i]]) > 0., 1, np.nan)
        diffIceNumeralPC6_neg[ens[i]] = np.where((
            iceNumeralSumPC6_ens_20652069[ens[i]] - iceNumeralSumPC6_ens_20352039[ens[i]]) < 0., 1, np.nan)
        

        
    diffIceNumeralPC6_pos_agree = np.where((np.nansum(np.stack(diffIceNumeralPC6_pos.values()),axis=0)) >= 8., 1., np.nan)
    diffIceNumeralPC6_neg_agree = np.where((np.nansum(np.stack(diffIceNumeralPC6_neg.values()),axis=0)) >= 8., 1., np.nan)
    diffIceNumeralPC6_agreement = np.nansum(np.stack((diffIceNumeralPC6_pos_agree,
                                                  diffIceNumeralPC6_neg_agree),axis=0), axis=0)
    diffIceNumeralPC6_agreement[diffIceNumeralPC6_agreement == 0.] = np.nan
    if simulation == 'ssp245': np.save(dataDir + '/diffIceNumeralPC6_agreement_ssp.npy', diffIceNumeralPC6_agreement)
    elif simulation == 'arise-1.5': np.save(dataDir + '/diffIceNumeralPC6_agreement_arise.npy', diffIceNumeralPC6_agreement)
    elif simulation == 'arise-1.0': np.save(dataDir + '/diffIceNumeralPC6_agreement_arise1.npy', diffIceNumeralPC6_agreement)
    
    
    #### OW ice numeral ----------------------
    iceNumeralSumOW_ens_20352039 = {}
    iceNumeralDaily_ens_20352039 = {}
    iceNumeralSumOW_ens_20352039 = {}
    for i in range(len(ens)):
        print(i)
        if simulation == 'ssp245':
            category2 = cat_SSP_20352039[cats[1],ens[i]]
            category3 = cat_SSP_20352039[cats[2],ens[i]]
            category4 = cat_SSP_20352039[cats[3],ens[i]]
            category5 = cat_SSP_20352039[cats[4],ens[i]]
        elif simulation == 'arise-1.5':
            category2 = cat_ARISE_20352039[cats[1],ens[i]]
            category3 = cat_ARISE_20352039[cats[2],ens[i]]
            category4 = cat_ARISE_20352039[cats[3],ens[i]]
            category5 = cat_ARISE_20352039[cats[4],ens[i]]
        elif simulation == 'arise-1.0':
            category2 = cat_ARISE1_20352039[cats[1],ens[i]]
            category3 = cat_ARISE1_20352039[cats[2],ens[i]]
            category4 = cat_ARISE1_20352039[cats[3],ens[i]]
            category5 = cat_ARISE1_20352039[cats[4],ens[i]]
            
        iceNumeralDaily_ens_20352039[ens[i]]  = np.zeros((365,43,288)) # 1 or 0 for each day in 5 year average
        iceNumeralSumOW_ens_20352039[ens[i]] = np.zeros((43,288)) # total sum for 5 year average  
        
        for iday in range(365):
            iceNumeralDaily_ens_20352039[ens[i]][iday,:,:] = ice_numeral_OW(
                dailyIceThick_20352039[ens[i]][iday,:,:],
                category2[iday,:,:],
                category3[iday,:,:],
                category4[iday,:,:],
                category5[iday,:,:])
        iceNumeralSumOW_ens_20352039[ens[i]] = np.nansum(iceNumeralDaily_ens_20352039[ens[i]], axis=0)
        np.save(dataDir + '/iceNumeralSumOW_ens_20352039_' + str(simulation) + '_' + str(ens[i]) + '.npy', iceNumeralSumOW_ens_20352039)
    
    del category2, category3, category4, category5
    
    ## 2065-2069
    iceNumeralSumOW_ens_20652069 = {}
    iceNumeralDaily_ens_20652069 = {}
    iceNumeralSumOW_ens_20652069 = {}
    for i in range(len(ens)):
        print(i)
        if simulation == 'ssp245':
            category2 = cat_SSP_20652069[cats[1],ens[i]]
            category3 = cat_SSP_20652069[cats[2],ens[i]]
            category4 = cat_SSP_20652069[cats[3],ens[i]]
            category5 = cat_SSP_20652069[cats[4],ens[i]]
        elif simulation == 'arise-1.5':
            category2 = cat_ARISE_20652069[cats[1],ens[i]]
            category3 = cat_ARISE_20652069[cats[2],ens[i]]
            category4 = cat_ARISE_20652069[cats[3],ens[i]]
            category5 = cat_ARISE_20652069[cats[4],ens[i]]
        elif simulation == 'arise-1.0':
            category2 = cat_ARISE1_20652069[cats[1],ens[i]]
            category3 = cat_ARISE1_20652069[cats[2],ens[i]]
            category4 = cat_ARISE1_20652069[cats[3],ens[i]]
            category5 = cat_ARISE1_20652069[cats[4],ens[i]]
            
        iceNumeralDaily_ens_20652069[ens[i]]  = np.zeros((365,43,288)) # 1 or 0 for each day in 5 year average
        iceNumeralSumOW_ens_20652069[ens[i]] = np.zeros((43,288)) # total sum for 5 year average  
        
        for iday in range(365):
            iceNumeralDaily_ens_20652069[ens[i]][iday,:,:] = ice_numeral_OW(
                dailyIceThick_20652069[ens[i]][iday,:,:],
                category2[iday,:,:],
                category3[iday,:,:],
                category4[iday,:,:],
                category5[iday,:,:])
        iceNumeralSumOW_ens_20652069[ens[i]] = np.nansum(iceNumeralDaily_ens_20652069[ens[i]], axis=0)
        np.save(dataDir + '/iceNumeralSumOW_ens_20652069_' + str(simulation) + '_' + str(ens[i]) + '.npy', iceNumeralSumOW_ens_20652069)      
    
    #### difference
    diffIceNumeralOW_pos = {}
    diffIceNumeralOW_neg = {}
    for i in range(len(ens)):
        diffIceNumeralOW_pos[ens[i]] = np.where((
            iceNumeralSumOW_ens_20652069[ens[i]] - iceNumeralSumOW_ens_20352039[ens[i]]) > 0., 1, np.nan)
        diffIceNumeralOW_neg[ens[i]] = np.where((
            iceNumeralSumOW_ens_20652069[ens[i]] - iceNumeralSumOW_ens_20352039[ens[i]]) < 0., 1, np.nan)
        

        
    diffIceNumeralOW_pos_agree = np.where((np.nansum(np.stack(diffIceNumeralOW_pos.values()),axis=0)) >= 8., 1., np.nan)
    diffIceNumeralOW_neg_agree = np.where((np.nansum(np.stack(diffIceNumeralOW_neg.values()),axis=0)) >= 8., 1., np.nan)
    diffIceNumeralOW_agreement = np.nansum(np.stack((diffIceNumeralOW_pos_agree,
                                                  diffIceNumeralOW_neg_agree),axis=0), axis=0)
    diffIceNumeralOW_agreement[diffIceNumeralOW_agreement == 0.] = np.nan
    if simulation == 'ssp245': np.save(dataDir + '/diffIceNumeralOW_agreement_ssp.npy', diffIceNumeralOW_agreement)
    elif simulation == 'arise-1.5': np.save(dataDir + '/diffIceNumeralOW_agreement_arise.npy', diffIceNumeralOW_agreement)
    elif simulation == 'arise-1.0': np.save(dataDir + '/diffIceNumeralOW_agreement_arise1.npy', diffIceNumeralOW_agreement)


    return 



def navigableDaysCategoriesMaps(latIce,lonIce,vessel):
    print("navigable days for " + str(vessel))
    dataDir = '/Users/arielmor/Desktop/SAI/ArcticShipping/data/'
    import numpy as np
    import xarray as xr
    from matplotlib import cm
    import cmasher as cmr
    
    iceNumeralSumFEEDBACK_20352039 = np.load(dataDir + str(vessel) + 'IceNumSum_arise-1.5_full_year_updated_with_categories_20352039.npy')
    iceNumeralSumFEEDBACK_LOWER_20352039 = np.load(dataDir + str(vessel) + 'IceNumSum_arise-1.0_full_year_updated_with_categories_20352039.npy')
    iceNumeralSumCONTROL_20352039 = np.load(dataDir + str(vessel) + 'IceNumSum_ssp245_full_year_updated_with_categories_20352039.npy')
    
    iceNumeralSumFEEDBACK_20652069 = np.load(dataDir + str(vessel) + 'IceNumSum_arise-1.5_full_year_updated_with_categories_20652069.npy')
    iceNumeralSumFEEDBACK_LOWER_20652069 = np.load(dataDir + str(vessel) + 'IceNumSum_arise-1.0_full_year_updated_with_categories_20652069.npy')
    iceNumeralSumCONTROL_20652069 = np.load(dataDir + str(vessel) + 'IceNumSum_ssp245_full_year_updated_with_categories_20652069.npy')
    
    diffCONTROL = (iceNumeralSumCONTROL_20652069-iceNumeralSumCONTROL_20352039)
    diffFEEDBACK = (iceNumeralSumFEEDBACK_20652069-iceNumeralSumFEEDBACK_20352039)
    diffFEEDBACK_LOWER = (iceNumeralSumFEEDBACK_LOWER_20652069-iceNumeralSumFEEDBACK_LOWER_20352039)
    
    
    diffCONTROL_ens = np.load(dataDir + '/diffIceNumeral' + str(vessel) + '_agreement_ssp.npy')
    diffFEEDBACK_ens = np.load(dataDir + '/diffIceNumeral' + str(vessel) + '_agreement_arise.npy')
    diffFEEDBACK_LOWER_ens = np.load(dataDir + '//diffIceNumeral' + str(vessel) + '_agreement_arise1.npy')
    
    
    landMask = np.load(dataDir + 'landMask_from_global_edited.npy')
    iceNumeralSumFEEDBACK_20352039 = np.where(landMask == 1, np.nan, iceNumeralSumFEEDBACK_20352039)
    iceNumeralSumFEEDBACK_20652069 = np.where(landMask == 1, np.nan, iceNumeralSumFEEDBACK_20652069)
    iceNumeralSumFEEDBACK_LOWER_20352039 = np.where(landMask == 1, np.nan, iceNumeralSumFEEDBACK_LOWER_20352039)
    iceNumeralSumFEEDBACK_LOWER_20652069 = np.where(landMask == 1, np.nan, iceNumeralSumFEEDBACK_LOWER_20652069)
    iceNumeralSumCONTROL_20352039 = np.where(landMask == 1, np.nan, iceNumeralSumCONTROL_20352039)
    iceNumeralSumCONTROL_20652069 = np.where(landMask == 1, np.nan, iceNumeralSumCONTROL_20652069)
    
    
    
    #### maps
    if vessel == 'PC6': vesselName = 'Polar Class 6'
    elif vessel == 'OW': vesselName = 'Open Water'

    fig, ax = make_maps(iceNumeralSumCONTROL_20352039,
                        latIce,lonIce,30,365,31,
                        'navigableDays','No. of ' + str(vesselName) + ' navigable days',
                        'a) SSP2-4.5, 2035-2039','a_control_navigable_days_from_cat_'+str(vessel)+'_2035_2039_updated',
                        'min',True)
    fig, ax = make_maps(iceNumeralSumCONTROL_20652069,
                        latIce,lonIce,30,365,31,
                        'navigableDays','No. of ' + str(vesselName) + ' navigable days',
                        'b) SSP2-4.5, 2065-2069','b_control_navigable_days_from_cat_'+str(vessel)+'_2065_2069_updated',
                        'min',True)
    fig, ax = make_maps(diffCONTROL,
                        latIce,lonIce,-90,90,31,
                        'PiYG','Change in no. of ' + str(vesselName) + ' navigable days',
                        'c) SSP2-4.5, 2065-2069 minus 2035-2039','c_control_navigable_days_from_cat_'+str(vessel)+'2065_2069_minus_2035_2039_updated',
                        'both',True, True, diffCONTROL_ens)
    fig, ax = make_maps(iceNumeralSumFEEDBACK_20352039,
                        latIce,lonIce,30,365,31,
                        'navigableDays','No. of ' + str(vesselName) + ' navigable days',
                        'd) ARISE-1.5, 2035-2039','d_feedback_navigable_days_from_cat_'+str(vessel)+'_2035_2039_updated',
                        'min',True)
    fig, ax = make_maps(iceNumeralSumFEEDBACK_20652069,
                        latIce,lonIce,30,365,31,
                        'navigableDays','No. of ' + str(vesselName) + ' navigable days',
                        'e) ARISE-1.5, 2065-2069','e_feedback_navigable_days_from_cat_'+str(vessel)+'_2065_2069_updated',
                        'min',True, False, None)
    fig, ax = make_maps(diffFEEDBACK,
                        latIce,lonIce,-90,90,31,
                        'PiYG','Change in no. of ' + str(vesselName) + ' navigable days',
                        'f) ARISE-1.5, 2065-2069 minus 2035-2039','f_feedback_navigable_days_from_cat_'+str(vessel)+'_2065_2069_minus_2035_2039_updated',
                        'both',True, True, diffFEEDBACK_ens)
    fig, ax = make_maps(iceNumeralSumFEEDBACK_LOWER_20352039,
                        latIce,lonIce,30,365,31,
                        'navigableDays','No. of ' + str(vesselName) + ' navigable days',
                        'g) ARISE-1.0, 2035-2039','g_feedback_navigable_days_from_cat_'+str(vessel)+'_2035_2039_updated',
                        'min',True, False, None)
    fig, ax = make_maps(iceNumeralSumFEEDBACK_LOWER_20652069,
                        latIce,lonIce,30,365,31,
                        'navigableDays','No. of ' + str(vesselName) + ' navigable days',
                        'h) ARISE-1.0, 2065-2069','h_feedback_navigable_days_from_cat_'+str(vessel)+'_2065_2069_updated',
                        'min',True, False, None)
    fig, ax = make_maps(diffFEEDBACK_LOWER,
                        latIce,lonIce,-90,90,31,
                        'PiYG','Change in no. of ' + str(vesselName) + ' navigable days',
                        'i) ARISE-1.0, 2065-2069 minus 2035-2039','i_feedback_navigable_days_from_cat_'+str(vessel)+'_2065_2069_minus_2035_2039_updated',
                        'both',True, True, diffFEEDBACK_LOWER_ens)
    
   
    
    ds = xr.open_dataset(dataDir + 'gridareaNH.nc')
    gridArea = ds.cell_area[31:,:]
    ds.close()
    
    ga_SSP_2035 = (np.nansum(np.where(iceNumeralSumCONTROL_20352039[10:,:] > 179, gridArea, 0)))/1e6
    print(str(np.round(ga_SSP_2035/1e6,decimals=3)) + " million sq km open for 6 months above 60N, SSP 2035")
    ga_SSP_2065 = (np.nansum(np.where(iceNumeralSumCONTROL_20652069[10:,:] > 179, gridArea, 0)))/1e6
    print(str(np.round(ga_SSP_2065/1e6,decimals=3)) + " million sq km open for 6 months above 60N, SSP 2065")
    diff = ga_SSP_2065 - ga_SSP_2035
    print("diff in SSP: ", str(diff/1e6))
    
    ga_ARISE_2035 = (np.nansum(np.where(iceNumeralSumFEEDBACK_20352039[10:,:] > 179, gridArea, 0)))/1e6
    print(str(np.round(ga_ARISE_2035/1e6,decimals=3)) + " million sq km open for 6 months above 60N, ARISE 2035")
    ga_ARISE_2065 = (np.nansum(np.where(iceNumeralSumFEEDBACK_20652069[10:,:] > 179, gridArea, 0)))/1e6
    print(str(np.round(ga_ARISE_2065/1e6,decimals=3)) + " million sq km open for 6 months above 60N, ARISE 2065")
    diff = ga_ARISE_2065 - ga_ARISE_2035
    print("diff in ARISE: ", str(diff/1e6))
    
    ga_ARISE1_2035 = (np.nansum(np.where(iceNumeralSumFEEDBACK_LOWER_20352039[10:,:] > 179, gridArea, 0)))/1e6
    print(str(np.round(ga_ARISE1_2035/1e6,decimals=3)) + " million sq km open for 6 months above 60N, ARISE1 2035")
    ga_ARISE1_2065 = (np.nansum(np.where(iceNumeralSumFEEDBACK_LOWER_20652069[10:,:] > 179, gridArea, 0)))/1e6
    print(str(np.round(ga_ARISE1_2065/1e6,decimals=3)) + " million sq km open for 6 months above 60N, ARISE1 2065")
    diff = ga_ARISE1_2065 - ga_ARISE1_2035
    print("diff in ARISE1: ", str(diff/1e6))
    
    
    ## entire year
    ga_SSP_2035 = (np.nansum(np.where(iceNumeralSumCONTROL_20352039[10:,:] == 365, gridArea, 0)))/1e6
    print(str(np.round(ga_SSP_2035/1e6,decimals=3)) + " million sq km open for whole year, SSP 2035")
    ga_SSP_2065 = (np.nansum(np.where(iceNumeralSumCONTROL_20652069[10:,:] == 365, gridArea, 0)))/1e6
    print(str(np.round(ga_SSP_2065/1e6,decimals=3)) + " million sq km open for whole year, SSP 2065")
    diff = ga_SSP_2065 - ga_SSP_2035
    print("diff in SSP: ", str(diff/1e6))
    
    ga_ARISE_2035 = (np.nansum(np.where(iceNumeralSumFEEDBACK_20352039[10:,:] == 365, gridArea, 0)))/1e6
    print(str(ga_ARISE_2035/1e6) + " million sq km open for whole year, ARISE 2035")
    ga_ARISE_2065 = (np.nansum(np.where(iceNumeralSumFEEDBACK_20652069[10:,:] == 365, gridArea, 0)))/1e6
    print(str(ga_ARISE_2065/1e6) + " million sq km open for whole year, ARISE 2065")
    diff = ga_ARISE_2065 - ga_ARISE_2035
    print("diff in ARISE: ", str(diff/1e6))
    
    ga_ARISE1_2035 = (np.nansum(np.where(iceNumeralSumFEEDBACK_LOWER_20352039[10:,:] == 365, gridArea, 0)))/1e6
    print(str(ga_ARISE1_2035/1e6) + " million sq km open for whole year, ARISE1 2035")
    ga_ARISE1_2065 = (np.nansum(np.where(iceNumeralSumFEEDBACK_LOWER_20652069[10:,:] == 365, gridArea, 0)))/1e6
    print(str(ga_ARISE1_2065/1e6) + " million sq km open for whole year, ARISE1 2065")
    diff = ga_ARISE1_2065 - ga_ARISE1_2035
    print("diff in ARISE1: ", str(diff/1e6))
    
    ## not at all navigable
    ga_SSP_2035 = (np.nansum(np.where(iceNumeralSumCONTROL_20352039[10:,:] == 0, gridArea, np.nan)))/1e6
    print(str(ga_SSP_2035/1e6) + " million sq km closed, SSP 2035")
    ga_SSP_2065 = (np.nansum(np.where(iceNumeralSumCONTROL_20652069[10:,:] == 0, gridArea, np.nan)))/1e6
    print(str(ga_SSP_2065/1e6) + " million sq km closed, SSP 2065")
    diff = ga_SSP_2065 - ga_SSP_2035
    print("diff in SSP: ", str(diff/1e6))
    
    ga_ARISE_2035 = (np.nansum(np.where(iceNumeralSumFEEDBACK_20352039[10:,:] == 0, gridArea, np.nan)))/1e6
    print(str(ga_ARISE_2035/1e6) + " million sq km closed, ARISE 2035")
    ga_ARISE_2065 = (np.nansum(np.where(iceNumeralSumFEEDBACK_20652069[10:,:] == 0, gridArea, np.nan)))/1e6
    print(str(ga_ARISE_2065/1e6) + " million sq km closed, ARISE 2065")
    diff = ga_ARISE_2065 - ga_ARISE_2035
    print("diff in ARISE: ", str(diff/1e6))
    
    ga_ARISE1_2035 = (np.nansum(np.where(iceNumeralSumFEEDBACK_LOWER_20352039[10:,:] == 0, gridArea, np.nan)))/1e6
    print(str(ga_ARISE1_2035/1e6) + " million sq km closed, ARISE1 2035")
    ga_ARISE1_2065 = (np.nansum(np.where(iceNumeralSumFEEDBACK_LOWER_20652069[10:,:] == 0, gridArea, np.nan)))/1e6
    print(str(ga_ARISE1_2065/1e6) + " million sq km closed, ARISE1 2065")
    diff = ga_ARISE1_2065 - ga_ARISE1_2035
    print("diff in ARISE1: ", str(diff/1e6))
    
    print(" ")
    print(" ")
    #### average above 60N:
    latIceStart = 10
    lonmesh,latmesh = np.meshgrid(lonIce,latIce[latIceStart:])
    weights = np.cos(np.deg2rad(latmesh))
    
    arrayVar = iceNumeralSumCONTROL_20352039
    weights[np.isnan(arrayVar[latIceStart:,:])] = np.nan
    weights = np.ma.asanyarray(weights)
    ensMasked_grouped = np.ma.MaskedArray(arrayVar[latIceStart:,:], 
        mask=np.isnan(arrayVar[latIceStart:,:]))
    weights.mask = ensMasked_grouped.mask
    weighted_ND1 = np.array([np.ma.average(ensMasked_grouped,weights=weights)])
    print("SSP 2035 average ND north of " + str(np.round(latIce[latIceStart].values,decimals=1)), ": ",  weighted_ND1)
    
    arrayVar = iceNumeralSumCONTROL_20652069
    weights[np.isnan(arrayVar[latIceStart:,:])] = np.nan
    weights = np.ma.asanyarray(weights)
    ensMasked_grouped = np.ma.MaskedArray(arrayVar[latIceStart:,:], 
        mask=np.isnan(arrayVar[latIceStart:,:]))
    weights.mask = ensMasked_grouped.mask
    weighted_ND2 = np.array([np.ma.average(ensMasked_grouped,weights=weights)])
    print("SSP 2065 average ND north of " + str(np.round(latIce[latIceStart].values,decimals=1)), ": ",  weighted_ND2)
    print("SSP 2065 minus 2035 ND north of " + str(np.round(latIce[latIceStart].values,decimals=1)), ": ",  (weighted_ND2 - weighted_ND1))
    
    arrayVar = iceNumeralSumFEEDBACK_20352039
    weights[np.isnan(arrayVar[latIceStart:,:])] = np.nan
    weights = np.ma.asanyarray(weights)
    ensMasked_grouped = np.ma.MaskedArray(arrayVar[latIceStart:,:], 
        mask=np.isnan(arrayVar[latIceStart:,:]))
    weights.mask = ensMasked_grouped.mask
    weighted_ND1 = np.array([np.ma.average(ensMasked_grouped,weights=weights)])
    print("ARISE 2035 average ND north of " + str(np.round(latIce[latIceStart].values,decimals=1)), ": ",  weighted_ND1)
    
    arrayVar = iceNumeralSumFEEDBACK_20652069
    weights[np.isnan(arrayVar[latIceStart:,:])] = np.nan
    weights = np.ma.asanyarray(weights)
    ensMasked_grouped = np.ma.MaskedArray(arrayVar[latIceStart:,:], 
        mask=np.isnan(arrayVar[latIceStart:,:]))
    weights.mask = ensMasked_grouped.mask
    weighted_ND2 = np.array([np.ma.average(ensMasked_grouped,weights=weights)])
    print("ARISE 2065 average ND north of " + str(np.round(latIce[latIceStart].values,decimals=1)), ": ",  weighted_ND2)
    print("ARISE 2065 minus 2035 ND north of " + str(np.round(latIce[latIceStart].values,decimals=1)), ": ",  (weighted_ND2 - weighted_ND1))
    
    arrayVar = iceNumeralSumFEEDBACK_LOWER_20352039
    weights[np.isnan(arrayVar[latIceStart:,:])] = np.nan
    weights = np.ma.asanyarray(weights)
    ensMasked_grouped = np.ma.MaskedArray(arrayVar[latIceStart:,:], 
        mask=np.isnan(arrayVar[latIceStart:,:]))
    weights.mask = ensMasked_grouped.mask
    weighted_ND1 = np.array([np.ma.average(ensMasked_grouped,weights=weights)])
    print("ARISE1 2035 average ND north of " + str(np.round(latIce[latIceStart].values,decimals=1)), ": ",  weighted_ND1)
    
    arrayVar = iceNumeralSumFEEDBACK_LOWER_20652069
    weights[np.isnan(arrayVar[latIceStart:,:])] = np.nan
    weights = np.ma.asanyarray(weights)
    ensMasked_grouped = np.ma.MaskedArray(arrayVar[latIceStart:,:], 
        mask=np.isnan(arrayVar[latIceStart:,:]))
    weights.mask = ensMasked_grouped.mask
    weighted_ND2 = np.array([np.ma.average(ensMasked_grouped,weights=weights)])
    print("ARISE1 2065 average ND north of " + str(np.round(latIce[latIceStart].values,decimals=1)), ": ",  weighted_ND2)    
    print("ARISE1 2065 minus 2035 ND north of " + str(np.round(latIce[latIceStart].values,decimals=1)), ": ",  (weighted_ND2 - weighted_ND1))
    
    return 







def thinIceConditions(lat_lon_series, threshold, N):
     import numpy as np
     # first day when sit < threshold
     mask = np.convolve(np.less(lat_lon_series,threshold),np.ones(N,dtype=int))>=N
     # if sit < threshold, go through rest of time series to see if remains below threshold
     if mask.any():
         firstInd = mask.argmax() - N + 1
         mask2 = np.convolve(np.less(lat_lon_series[firstInd:],threshold),np.ones(184-firstInd,dtype=int))>=(184-firstInd)
         # if it does remain below threshold, return that first index
         if (mask2.argmax()) == (184-firstInd-1):
             return firstInd
         # if it doesn't remain below threshold, keep going through time series until next day when below threshold
         else:# (mask2.argmax()) != (184-firstInd-1): # else
             mask3 = np.convolve(np.less(lat_lon_series[firstInd+1:],threshold),np.ones(N,dtype=int))>=N
             # if there's another day below threshold, find that index
             if mask3.any():
                 secondInd = mask3.argmax() + 1
                 mask4 = np.convolve(np.less(lat_lon_series[(firstInd+secondInd):],threshold),np.ones(184-firstInd-secondInd-1,dtype=int))>=(184-firstInd-secondInd-1)
                 if (mask4.argmax() + 1) == (184-firstInd-secondInd-1):
                     # if it does remain below threshold, return that second index
                     return secondInd+firstInd 
                
     else:
         return None
     


def thinIceConditions_30days(lat_lon_series, threshold, N):
     import numpy as np
     mask = np.convolve(np.greater(lat_lon_series,threshold),np.ones(N,dtype=int))>=N
     if mask.any():
         return mask.argmax() - N + 1
     else:
         return None
     
        
     
def persistentThinIce(latIce,lonIce,simulation,threshold):
    dataDir = '/Users/arielmor/Desktop/SAI/ArcticShipping/data/'
    
    import numpy as np
    from matplotlib.colors import ListedColormap
    from matplotlib import cm
    import matplotlib as mpl
    from matplotlib import colors as c
    import matplotlib.colors as mcolors
    import cartopy.crs as ccrs
    from cartopy.util import add_cyclic_point
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    import cartopy.feature as cfeature
    import matplotlib.ticker as mticker
    import matplotlib.pyplot as plt
    import matplotlib.path as mpath
    pri = cm.get_cmap('PiYG_r', (40))
    newcolors = pri(np.linspace(0, 1, 256))
    newcolors[122:134, :] = np.array([1, 1, 1, 1])
    iceburn = ListedColormap(newcolors)
    
        
    #### read data
    dailyIceThickMean_20352040 = np.load(dataDir + 'PC6IceNum_' + str(simulation) + '_full_year_updated_with_categories_for_persistence_20352040.npy')
    dailyIceThickMean_20642069 = np.load(dataDir + 'PC6IceNum_' + str(simulation) + '_full_year_updated_with_categories_for_persistence_20642069.npy')
    dailyIceThickMeanOW_20352040 = np.load(dataDir + 'OWIceNum_' + str(simulation) + '_full_year_updated_with_categories_for_persistence_20352040.npy')
    dailyIceThickMeanOW_20642069 = np.load(dataDir + 'OWIceNum_' + str(simulation) + '_full_year_updated_with_categories_for_persistence_20642069.npy')
    
    
    
    firstDayThinIcePC6_20352040 = np.empty((43,288))
    firstDayThinIcePC6_20642069 = np.empty((43,288))
    firstDayThinIceOW_20352040  = np.empty((43,288))
    firstDayThinIceOW_20642069  = np.empty((43,288))
    for ilat in range(len(latIce)):
        for ilon in range(len(lonIce)):
            firstDayThinIcePC6_20352040[ilat,ilon] = thinIceConditions_30days(dailyIceThickMean_20352040[:,ilat,ilon],0.5,threshold)
            firstDayThinIcePC6_20642069[ilat,ilon] = thinIceConditions_30days(dailyIceThickMean_20642069[:,ilat,ilon],0.5,threshold)
            firstDayThinIceOW_20352040[ilat,ilon]  = thinIceConditions_30days(dailyIceThickMeanOW_20352040[:,ilat,ilon],0.5,threshold)
            firstDayThinIceOW_20642069[ilat,ilon]  = thinIceConditions_30days(dailyIceThickMeanOW_20642069[:,ilat,ilon],0.5,threshold)
    
    
    # #### area that is persistent
    # polarCodeArea = np.load(dataDir + 'polarCodeGridArea.npy')
    # pa_SSP_2035 = (np.nansum(np.where(~np.isnan(firstDayThinIcePC6_20352040[9:,:]), polarCodeArea, 0)))/1e6
    # print("persistent area " + str(simulation) + " 2035, PC6: ", (pa_SSP_2035)/1e6)
    # pa_SSP_2065 = (np.nansum(np.where(~np.isnan(firstDayThinIcePC6_20642069[9:,:]), polarCodeArea, 0)))/1e6
    # print("persistent area " + str(simulation) + " 2065, PC6: ", (pa_SSP_2065)/1e6)
    # print("diff in persistent area " + str(simulation) + " PC6: ", (pa_SSP_2065-pa_SSP_2035)/1e6)
    # oa_SSP_2035 = (np.nansum(np.where(~np.isnan(firstDayThinIceOW_20352040[9:,:]), polarCodeArea, 0)))/1e6
    # print("persistent area " + str(simulation) + " 2035, OW: ", (oa_SSP_2035)/1e6)
    # oa_SSP_2065 = (np.nansum(np.where(~np.isnan(firstDayThinIceOW_20642069[9:,:]), polarCodeArea, 0)))/1e6
    # print("persistent area " + str(simulation) + " 2065, OW: ", (oa_SSP_2065)/1e6)
    # print("diff in persistent area " + str(simulation) + " OW: ", (oa_SSP_2065-oa_SSP_2035)/1e6)

    
    #### maps
    if simulation == 'arise-1.5':
        fig, ax = make_maps(firstDayThinIcePC6_20352040,latIce,lonIce,
                            0,365,24,'freeze_r',' ',
                            'd) ARISE-1.5, 2035-2040',
                            'persistent_thin_ice_arise_pc6_2035-2040_' + str(threshold) + '_days','neither',True, False, None)
        fig, ax = make_maps(firstDayThinIcePC6_20642069,latIce,lonIce,
                            0,365,24,'freeze_r',' ',
                            'e) ARISE-1.5, 2064-2069',
                            'persistent_thin_ice_arise_pc6_2064-2069_' + str(threshold) + '_days','neither',True, False, None)
        fig, ax = make_maps(firstDayThinIceOW_20352040,latIce,lonIce,
                            0,365,24,'freeze_r',' ',
                            'd) ARISE-1.5, 2035-2040',
                            'persistent_thin_ice_arise_ow_from_cats_2035-2040_' + str(threshold) + '_days','neither',True, False, None)
        fig, ax = make_maps(firstDayThinIceOW_20642069,latIce,lonIce,
                            0,365,24,'freeze_r',' ',
                            'e) ARISE-1.5, 2064-2069',
                            'persistent_thin_ice_arise_ow_from_cats_2064-2069_' + str(threshold) + '_days','neither',True, False, None)
    elif simulation == 'ssp245':
        fig, ax = make_maps(firstDayThinIcePC6_20352040,latIce,lonIce,
                            0,365,24,'freeze_r',' ',
                            'a) SSP2-4.5, 2035-2040',
                            'persistent_thin_ice_ssp_pc6_from_cats_2035-2040_' + str(threshold) + '_days','neither',True, False, None)
        fig, ax = make_maps(firstDayThinIcePC6_20642069,latIce,lonIce,
                            0,365,24,'freeze_r',' ',
                            'b) SSP2-4.5, 2064-2069',
                            'persistent_thin_ice_ssp_pc6_from_cats_2064-2069_' + str(threshold) + '_days','neither',True, False, None)
        fig, ax = make_maps(firstDayThinIceOW_20352040,latIce,lonIce,
                            0,365,24,'freeze_r',' ',
                            'a) SSP2-4.5, 2035-2040',
                            'persistent_thin_ice_ssp_ow_from_cats_2035-2040_' + str(threshold) + '_days','neither',True, False, None)
        fig, ax = make_maps(firstDayThinIceOW_20642069,latIce,lonIce,
                            0,365,24,'freeze_r',' ',
                            'b) SSP2-4.5, 2064-2069',
                            'persistent_thin_ice_ssp_ow_from_cats_2064-2069_' + str(threshold) + '_days','neither',True, False, None)
    elif simulation == 'arise-1.0':
        fig, ax = make_maps(firstDayThinIcePC6_20352040,latIce,lonIce,
                            0,365,24,'freeze_r',' ',
                            'g) ARISE-1.0, 2035-2040',
                            'persistent_thin_ice_arise_1.0_pc6_from_cats_2035-2040_' + str(threshold) + '_days','neither',True, False, None)
        fig, ax = make_maps(firstDayThinIcePC6_20642069,latIce,lonIce,
                            0,365,24,'freeze_r',' ',
                            'h) ARISE-1.0, 2064-2069',
                            'persistent_thin_ice_arise_1.0_pc6_from_cats_2064-2069_' + str(threshold) + '_days','neither',True, False, None)
        fig, ax = make_maps(firstDayThinIceOW_20352040,latIce,lonIce,
                            0,365,24,'freeze_r',' ',
                            'g) ARISE-1.0, 2035-2040',
                            'persistent_thin_ice_arise_1.0_ow_from_cats_2035-2040_' + str(threshold) + '_days','neither',True, False, None)
        fig, ax = make_maps(firstDayThinIceOW_20642069,latIce,lonIce,
                            0,365,24,'freeze_r',' ',
                            'h) ARISE-1.0, 2064-2069',
                            'persistent_thin_ice_arise_1.0_ow_from_cats_2064-2069_' + str(threshold) + '_days','neither',True, False, None)
    
    
    agreementPC6 = np.load(dataDir + '/persistenceDiffIceNumeralPC6_agreement_' + str(simulation) + '.npy')
    agreementOW = np.load(dataDir + '/persistenceDiffIceNumeralOW_agreement_' + str(simulation) + '.npy')
    
    #### difference figure        
    cmap_nav60sNot30s = c.ListedColormap(['w'])
    cmap_nav30sNot60s = c.ListedColormap(['w'])
    none_map = ListedColormap(['none'])
    
    diffPC6 = firstDayThinIcePC6_20642069 - firstDayThinIcePC6_20352040
    nav60sNot30s = firstDayThinIcePC6_20642069.copy()
    nav60sNot30s[
        (np.isnan(firstDayThinIcePC6_20352040)) & 
        (~np.isnan(firstDayThinIcePC6_20642069))] = -99 # nan in feedback mean = didn't thaw in FB
    nav60sNot30s[nav60sNot30s != -99] = np.nan
    
    nav30sNot60s = firstDayThinIcePC6_20352040.copy()
    nav30sNot60s[
        (np.isnan(firstDayThinIcePC6_20642069)) & 
        (~np.isnan(firstDayThinIcePC6_20352040))] = -99 # nan in feedback mean = didn't thaw in FB
    nav30sNot60s[nav30sNot60s != -99] = np.nan
    
    
    fig = plt.figure(figsize=(10,8))
    norm = mcolors.TwoSlopeNorm(vmin=-90, vcenter=0, vmax=90)
    longitude = np.linspace(0,360,288) 
    plottingVarMean,lon2 = add_cyclic_point(diffPC6,coord=longitude)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
    ax.set_extent([180, -180, 60, 90], crs=ccrs.PlateCarree())
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes); ax.set_facecolor('0.8')
    
    land_50m = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                    edgecolor='k',
                                    facecolor='0.8',
                                    linewidth=0.4)
    ax.add_feature(land_50m)
    
    ## thawed in control but not feedback
    ax.pcolormesh(lonIce,latIce,nav60sNot30s,transform=ccrs.PlateCarree(),
                        cmap=cmap_nav60sNot30s)
    hatch1 = ax.pcolor(lonIce, latIce, nav60sNot30s, transform=ccrs.PlateCarree(), cmap=none_map,
                    hatch='\\ \\', edgecolor='k', lw=0, zorder=2)
    ax.pcolormesh(lonIce,latIce,nav30sNot60s,transform=ccrs.PlateCarree(),
                        cmap=cmap_nav30sNot60s)
    hatch2 = ax.pcolor(lonIce, latIce, nav30sNot60s, transform=ccrs.PlateCarree(), cmap=none_map,
                   hatch='o o', edgecolor='k', lw=0, zorder=2)
    hatch3 = ax.pcolor(lonIce, latIce, agreementPC6, transform=ccrs.PlateCarree(), cmap=none_map,
                   hatch='X X X', edgecolor='k', lw=0, zorder=2)
    
    ## difference in thaw timing
    cf1 = ax.pcolormesh(lon2,latIce,plottingVarMean,transform=ccrs.PlateCarree(), 
                  norm=norm, cmap=iceburn) # seismic
    ax.coastlines(linewidth=0.7)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), 
                      draw_labels=True,
                      linewidth=1, color='C7', 
                  alpha=0.8, linestyle=':',
                  x_inline=False,
                  y_inline=True,
                  rotate_labels=False)
    gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120])
    gl.ylocator = mticker.FixedLocator([60, 70, 80])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.ylabel_style = {'size': 9, 'rotation':20}
    plt.draw()  # Enable the use of `gl._labels`
    for ea in gl.label_artists:
        # return ea of mpl.text type, e.g. Text(135, 30, '30°N') 
        pos = ea.get_position()
        if pos[0] == 150:
            ea.set_position([0, pos[1]])
            
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=iceburn),
         ax=ax, orientation='horizontal',
         extend='both', fraction=0.051) # change orientation if needed
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Change in days\n \u2190 earlier            later \u2192', fontsize=13)
    
    if simulation == 'arise-1.5':
        plt.title('f) ARISE-1.5, 2064-2069 minus 2035-2040', fontsize=16, fontweight='bold', y=1.07)
        plt.savefig('/Users/arielmor/Desktop/SAI/ArcticShipping/figures/diff_persistent_thin_ice_arise_pc6_from_cats_20352040_20642069_' + str(threshold) + '_days.png', 
                    dpi=1200, bbox_inches='tight')
    elif simulation == 'ssp245':
        plt.title('c) SSP2-4.5, 2064-2069 minus 2035-2040', fontsize=16, fontweight='bold', y=1.07)
        plt.savefig('/Users/arielmor/Desktop/SAI/ArcticShipping/figures/diff_persistent_thin_ice_ssp_pc6_from_cats_20352040_20642069_' + str(threshold) + '_days.png', 
                    dpi=1200, bbox_inches='tight')
    elif simulation == 'arise-1.0':
        plt.title('i) ARISE-1.0, 2064-2069 minus 2035-2040', fontsize=16, fontweight='bold', y=1.07)
        plt.savefig('/Users/arielmor/Desktop/SAI/ArcticShipping/figures/diff_persistent_thin_ice_arise_1.0_from_cats_pc6_20352040_20642069_' + str(threshold) + '_days.png', 
                    dpi=1200, bbox_inches='tight')
    del fig, ax, cf1, nav60sNot30s, nav30sNot60s
    
    
    ## open water
    diffOW = firstDayThinIceOW_20642069 - firstDayThinIceOW_20352040
    nav60sNot30s = firstDayThinIceOW_20642069.copy()
    nav60sNot30s[
        (np.isnan(firstDayThinIceOW_20352040)) & 
        (~np.isnan(firstDayThinIceOW_20642069))] = -99 # nan in feedback mean = didn't thaw in FB
    nav60sNot30s[nav60sNot30s != -99] = np.nan
    
    nav30sNot60s = firstDayThinIceOW_20352040.copy()
    nav30sNot60s[
        (np.isnan(firstDayThinIceOW_20642069)) & 
        (~np.isnan(firstDayThinIceOW_20352040))] = -99 # nan in feedback mean = didn't thaw in FB
    nav30sNot60s[nav30sNot60s != -99] = np.nan
    
    fig = plt.figure(figsize=(10,8))
    norm = mcolors.TwoSlopeNorm(vmin=-90, vcenter=0, vmax=90)
    longitude = np.linspace(0,360,288) 
    plottingVarMean,lon2 = add_cyclic_point(diffOW,coord=longitude)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
    ax.set_extent([180, -180, 60, 90], crs=ccrs.PlateCarree())
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes); ax.set_facecolor('0.8')
    
    land_50m = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                    edgecolor='k',
                                    facecolor='0.8',
                                    linewidth=0.4)
    ax.add_feature(land_50m)
    
    ## thawed in control but not feedback
    ax.pcolormesh(lonIce,latIce,nav60sNot30s,transform=ccrs.PlateCarree(),
                        cmap=cmap_nav60sNot30s)
    hatch1 = ax.pcolor(lonIce, latIce, nav60sNot30s, transform=ccrs.PlateCarree(), cmap=none_map,
                    hatch='\\ \\', edgecolor='k', lw=0, zorder=2)
    ax.pcolormesh(lonIce,latIce,nav30sNot60s,transform=ccrs.PlateCarree(),
                        cmap=cmap_nav30sNot60s)
    hatch2 = ax.pcolor(lonIce, latIce, nav30sNot60s, transform=ccrs.PlateCarree(), cmap=none_map,
                   hatch='o o', edgecolor='k', lw=0, zorder=2)
    hatch3 = ax.pcolor(lonIce, latIce, agreementOW, transform=ccrs.PlateCarree(), cmap=none_map,
                   hatch='X X', edgecolor='k', lw=0, zorder=2)
    
    ## difference in thaw timing
    cf1 = ax.pcolormesh(lon2,latIce,plottingVarMean,transform=ccrs.PlateCarree(), 
                  norm=norm, cmap=iceburn) # seismic
    ax.coastlines(linewidth=0.7)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), 
                      draw_labels=True,
                      linewidth=1, color='C7', 
                  alpha=0.8, linestyle=':',
                  x_inline=False,
                  y_inline=True,
                  rotate_labels=False)
    gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120])
    gl.ylocator = mticker.FixedLocator([60, 70, 80])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.ylabel_style = {'size': 9, 'rotation':20}
    plt.draw()  # Enable the use of `gl._labels`
    for ea in gl.label_artists:
        # return ea of mpl.text type, e.g. Text(135, 30, '30°N') 
        pos = ea.get_position()
        if pos[0] == 150:
            ea.set_position([0, pos[1]])
            
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=iceburn),
         ax=ax, orientation='horizontal',
         extend='both', fraction=0.051) # change orientation if needed
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Change in days\n \u2190 earlier            later \u2192', fontsize=13)
    
    if simulation == 'arise-1.5':
        plt.title('f) ARISE-1.5, 2064-2069 minus 2035-2040', fontsize=16, fontweight='bold', y=1.07)
        plt.savefig('/Users/arielmor/Desktop/SAI/ArcticShipping/figures/diff_persistent_thin_ice_arise_ow_from_cats_20352040_20642069_' + str(threshold) + '_days.png', 
                    dpi=1200, bbox_inches='tight')
    elif simulation == 'ssp245':
        plt.title('c) SSP2-4.5, 2064-2069 minus 2035-2040', fontsize=16, fontweight='bold', y=1.07)
        plt.savefig('/Users/arielmor/Desktop/SAI/ArcticShipping/figures/diff_persistent_thin_ice_ssp_ow_from_cats_20352040_20642069_' + str(threshold) + '_days.png', 
                    dpi=1200, bbox_inches='tight')
    elif simulation == 'arise-1.0':
        plt.title('i) ARISE-1.0, 2064-2069 minus 2035-2040', fontsize=16, fontweight='bold', y=1.07)
        plt.savefig('/Users/arielmor/Desktop/SAI/ArcticShipping/figures/diff_persistent_thin_ice_arise_1.0_ow_from_cats_20352040_20642069_' + str(threshold) + '_days.png', 
                    dpi=1200, bbox_inches='tight')
    del fig, ax, cf1, nav60sNot30s, nav30sNot60s
    
    return 


def iceMultipliers(cat1,cat2,cat3,cat4,cat5):
    '''
    ## following Copland et al., 2021 (https://www.facetsjournal.com/doi/10.1139/facets-2020-0096#eq1)
    ## and AIRSS (Transport Canada): https://tc.canada.ca/en/marine-transportation/marine-safety/tp-12259e-arctic-ice-regime-shipping-system-airss-standard
    ## concentration = tenths, not fraction
    ## 0.5 concentration = 5 tenths
    ## category max thickness (m):
    ## NCAT = 0.6445072, 1.391433, 2.470179, 4.567288, 1e+08
    ## cat 1 = FY (2)
    ## cat 2 = MFY (1)
    ## cat 3 = TFY (-1)
    ## cat 4 = MY (-4) --> captains avoid
    ## cat 5 = MY (-4) --> captains avoid at all costs
    ## IN = A1*IM1 + A2*IM2 + A3*IM3 + A4*IM4 + A5*IM5
    ## if A of any ice regime is >=0.6, IM is reduced by 1 (Transport Canada)
    ## include open water as an "ice category"
    '''
    
    import numpy as np
    
    #IM1 = 2; IM2 = 1; IM3 = -1; IM4 = -4; IM5 = -4
    
    if 0 <= cat1 < 6: IM1 = 2
    elif cat1 >= 6: IM1 = 1
    
    if 0 <= cat2 < 6: IM2 = 1
    elif cat2 >= 6: IM2 = 0
    
    if 0 <= cat3 < 6: IM3 = -1
    elif cat3 >= 6: IM3 = -2
    
    if 0 <= cat4 < 6: IM4 = -4
    elif cat4 >= 6: IM4 = -5
    
    if 0 <= cat5 < 6: IM5 = -4
    elif cat5 >= 6: IM5 = -5
    
    openWaterConc = 10 - (cat1 + cat2 + cat3 + cat4 + cat5)
    IM6 = 2
    
    iceNumeral = cat1*IM1 + cat2*IM2 + cat3*IM3 + cat4*IM4 + cat5*IM5 + openWaterConc*IM6
    IN = np.round(iceNumeral, decimals = 0)
    
    S3 = []
    ## from Akensov et al. 2017
    if IN < 0:           S3 = 0
    elif 0 <= IN <= 8:    S3 = 4
    elif 9 <= IN <= 13:    S3 = 5
    elif 14 <= IN <= 15:   S3 = 6
    elif IN == 16:         S3 = 7
    elif IN == 17:         S3 = 8
    elif IN == 18:         S3 = 9
    elif IN == 19:         S3 = 10
    elif IN == 20:         S3 = 11
    
    IN = np.where(iceNumeral < 0, 0, 1)
    
    return IN, S3



def iceMultipliersOW(cat1,cat2,cat3,cat4,cat5):
    ## following Copland et al., 2021 (https://www.facetsjournal.com/doi/10.1139/facets-2020-0096#eq1)
    
    import numpy as np
    
    #IM1 = 2; IM2 = 1; IM3 = -1; IM4 = -4; IM5 = -4
    if 0 <= cat1 < 6: IM1 = 1
    elif cat1 >= 6: IM1 = 0
    
    if 0 <= cat2 < 6: IM2 = -2
    elif cat2 >= 6: IM2 = -3
    
    if 0 <= cat3 < 6: IM3 = -4
    elif cat3 >= 6: IM3 = -5
    
    if 0 <= cat4 < 6: IM4 = -4
    elif cat4 >= 6: IM4 = -5
    
    if 0 <= cat5 < 6: IM5 = -4
    elif cat5 >= 6: IM5 = -5
    
    openWaterConc = 10 - (cat1 + cat2 + cat3 + cat4 + cat5)
    IM6 = 2
    
    IN = np.round((cat1*IM1 + cat2*IM2 + cat3*IM3 + cat4*IM4 + cat5*IM5 + openWaterConc*IM6), decimals = 0)
    
    IN = np.where(IN < 0, 0, 1)
    
    return IN



def safeShipSpeedPC6(simulation,make_route_maps):
    dataDir = '/Users/arielmor/Desktop/SAI/ArcticShipping/data/'
    
    import numpy as np
    # import readDailyIce, iceMultipliers
    
    landMask = np.load('/Users/arielmor/Desktop/SAI/ArcticShipping/data/landMask_from_global_edited.npy')
    landMask[18,36] = 1.
    landMask[20,40] = 0.
    landMask[20:23,211] = 0.
    landMask[29,213] = 0.
    landMask[33,272:275] = 1.
    
    
    latIce,lonIce,dailyIceThick,dailyIceFrac = readDailyIce(simulation)
    
    if simulation == 'ssp245':
        print("ssp245")
        category1 = np.load(dataDir + 'iceCategory1_SSP.npy')
        category2 = np.load(dataDir + 'iceCategory2_SSP.npy')
        category3 = np.load(dataDir + 'iceCategory3_SSP.npy')
        category4 = np.load(dataDir + 'iceCategory4_SSP.npy')
        category5 = np.load(dataDir + 'iceCategory5_SSP.npy')
    elif simulation == 'arise-1.5':
        print("arise-1.5")
        category1 = np.load(dataDir + 'iceCategory1_ARISE.npy')
        category2 = np.load(dataDir + 'iceCategory2_ARISE.npy')
        category3 = np.load(dataDir + 'iceCategory3_ARISE.npy')
        category4 = np.load(dataDir + 'iceCategory4_ARISE.npy')
        category5 = np.load(dataDir + 'iceCategory5_ARISE.npy')
    elif simulation == 'arise-1.0':
        category1 = np.load(dataDir + 'iceCategory1_ARISE1.npy')
        category2 = np.load(dataDir + 'iceCategory2_ARISE1.npy')
        category3 = np.load(dataDir + 'iceCategory3_ARISE1.npy')
        category4 = np.load(dataDir + 'iceCategory4_ARISE1.npy')
        category5 = np.load(dataDir + 'iceCategory5_ARISE1.npy')
        
    category1[np.isnan(category1)] = 0
    category2[np.isnan(category2)] = 0
    category3[np.isnan(category3)] = 0
    category4[np.isnan(category4)] = 0
    category5[np.isnan(category5)] = 0
    
    years = ['2035','2036','2037','2038','2039','2040',
             '2041','2042','2043','2044','2045','2046','2047','2048','2049','2050',
             '2051','2052','2053','2054','2055','2056','2057','2058','2059','2060',
             '2061','2062','2063','2064','2065','2066','2067','2068','2069']
    
    #### safe ship speed for monthly data
    S3 = np.full((35, 12, 43, 288), np.nan)
    for iyear in range(len(years)):
        firstMonth = 12 * iyear 
        lastMonth  = (12 * iyear) + 12
        print(years[iyear], firstMonth, lastMonth)
        cat1 = (category1[firstMonth:lastMonth, :, :]) * 10
        cat2 = (category2[firstMonth:lastMonth, :, :]) * 10
        cat3 = (category3[firstMonth:lastMonth, :, :]) * 10
        cat4 = (category4[firstMonth:lastMonth, :, :]) * 10
        cat5 = (category5[firstMonth:lastMonth, :, :]) * 10
        for imonth in range(12):
            for ilat in range(len(latIce)):
                for ilon in range(len(lonIce)):
                    IN, S3[iyear, imonth, ilat, ilon] = iceMultipliers(cat1[imonth, ilat, ilon], cat2[imonth, ilat, ilon], 
                                                                   cat3[imonth, ilat, ilon], cat4[imonth, ilat, ilon], 
                                                                   cat5[imonth, ilat, ilon])
        
    S3 = np.where(landMask == 1, np.nan, S3)
   
               
    if simulation == 'ssp245':
        np.save('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_safe_ship_speed_from_categories_pc6_ssp.npy', S3)
    elif simulation == 'arise-1.5':
        np.save('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_safe_ship_speed_from_categories_pc6_arise.npy', S3)
    elif simulation == 'arise-1.0':
        np.save('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_safe_ship_speed_from_categories_pc6_arise1.0.npy', S3)
    del cat1, cat2, cat3, cat4, cat5    
    
   
    return S3
    
    
    
def safeShipSpeedAllEns():
    import xarray as xr
    import numpy as np

    dataDir = '/Users/arielmor/Desktop/SAI/ArcticShipping/data/'
        
    ens = ['001','002','003','004','005','006','007','008','009','010']
    cats = ['1', '2', '3', '4', '5']
    years = ['2035','2036','2037','2038','2039','2040',
             '2041','2042','2043','2044','2045','2046','2047','2048','2049','2050',
             '2051','2052','2053','2054','2055','2056','2057','2058','2059','2060',
             '2061','2062','2063','2064','2065','2066','2067','2068','2069']
    
    landMask = np.load('/Users/arielmor/Desktop/SAI/ArcticShipping/data/landMask_from_global_edited.npy')
    landMask[18,36] = 1.
    landMask[20,40] = 0.
    landMask[20:23,211] = 0.
    landMask[29,213] = 0.
    landMask[33,272:275] = 1.
    np.save('/Users/arielmor/Desktop/SAI/ArcticShipping/data/landMask_from_global_edited.npy', landMask)
    
    #### read ice categories
    cat_SSP    = {}
    cat_ARISE  = {}
    cat_ARISE1 = {}
    
    for c in range(len(cats)):
        print("category " + str(c))
        for i in range(len(ens)):
            ## SSP2-4.5
            ds = xr.open_dataset(dataDir + 'b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.' 
                                 + str(ens[i]) + '.cice.h.aicen_only.201501-206912_cat' + str(cats[c]) + '_RG_NH.nc', 
                                 decode_times = False)
            ds['time'] = xr.cftime_range(start = '2015-01', periods = ds.sizes['time'], freq = "M", calendar = "noleap")
            cat_SSP[cats[c], ens[i]] = ds.aicen[240:,:,:]
            latIce = ds.lat; lonIce = ds.lon
            ds.close()
            
            ### ARISE-1.5
            ds = xr.open_dataset(dataDir + 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.' 
                                 + str(ens[i]) + '.cice.h.aicen_only.203501-206912_cat' + str(cats[c]) + '_RG_NH.nc',
                                 decode_times = False)
            ds['time'] = xr.cftime_range(start = '2035-01', periods = ds.sizes['time'], freq = "M", calendar = "noleap")
            cat_ARISE[cats[c], ens[i]] = ds.aicen
            ds.close()
            
            
            ### ARISE-1.0
            ds = xr.open_dataset(dataDir + 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-LOWER-0.5.' 
                                 + str(ens[i]) + '.cice.h.aicen_only.203501-206912_cat' + str(cats[c]) + '_RG_NH.nc',
                                 decode_times = False)
            ds['time'] = xr.cftime_range(start = '2035-01', periods = ds.sizes['time'], freq = "M", calendar = "noleap")
            cat_ARISE1[cats[c], ens[i]] = ds.aicen
            ds.close()
    
    #### SSP       
    for numEns in range(len(ens)):        
        for c in range(len(cats)):
            print("category " + str(c))
            a = np.full((420, 43, 288), np.nan)
            for imonth in range(420): 
                nan_indices = np.argwhere(np.isnan(cat_SSP[cats[c], ens[numEns]][imonth,:,:].values))
                monthlyIceThickMeanInterp = cat_SSP[cats[c], ens[numEns]][imonth,:,:].values.copy()
                # Iterate through the NaN indices
                for nan_index in nan_indices:
                    row, col = nan_index
                    neighbors = []
                    
                    # Check the surrounding 8 elements (or less at the edges)
                    for i in range(max(0, row - 1), min(monthlyIceThickMeanInterp.shape[0], row + 2)):
                        for j in range(max(0, col - 1), min(monthlyIceThickMeanInterp.shape[1], col + 2)):
                            if not np.isnan(monthlyIceThickMeanInterp[i, j]):
                                neighbors.append(monthlyIceThickMeanInterp[i, j])
                    
                    # If there are valid neighbors, fill the NaN with the average of all valid neighbors
                    if neighbors:
                        monthlyIceThickMeanInterp[row, col] = np.nanmean(neighbors)
                a[imonth,:,:] = monthlyIceThickMeanInterp
            np.save(dataDir + 'iceCategory' + str(cats[c]) + '_SSP_ens' + str(ens[numEns]) + '.npy',a)
        
        
    #### safe ship speed for monthly data
    for numEns in range(len(ens)):   
        print(numEns)
        S3 = np.full((35, 12, 43, 288), np.nan)
        IN = []
        category1 = np.load(dataDir + 'iceCategory1_SSP_ens' + str(ens[numEns]) + '.npy')
        category2 = np.load(dataDir + 'iceCategory2_SSP_ens' + str(ens[numEns]) + '.npy')
        category3 = np.load(dataDir + 'iceCategory3_SSP_ens' + str(ens[numEns]) + '.npy')
        category4 = np.load(dataDir + 'iceCategory4_SSP_ens' + str(ens[numEns]) + '.npy')
        category5 = np.load(dataDir + 'iceCategory5_SSP_ens' + str(ens[numEns]) + '.npy')
        
        category1[np.isnan(category1)] = 0
        category2[np.isnan(category2)] = 0
        category3[np.isnan(category3)] = 0
        category4[np.isnan(category4)] = 0
        category5[np.isnan(category5)] = 0
        for iyear in range(len(years)):
            firstMonth = 12 * iyear 
            lastMonth  = (12 * iyear) + 12
            cat1 = (category1[firstMonth:lastMonth, :, :]) * 10
            cat2 = (category2[firstMonth:lastMonth, :, :]) * 10
            cat3 = (category3[firstMonth:lastMonth, :, :]) * 10
            cat4 = (category4[firstMonth:lastMonth, :, :]) * 10
            cat5 = (category5[firstMonth:lastMonth, :, :]) * 10
            for imonth in range(12):
                for ilat in range(len(latIce)):
                    for ilon in range(len(lonIce)):
                        IN, S3[iyear, imonth, ilat, ilon] = iceMultipliers(cat1[imonth, ilat, ilon], cat2[imonth, ilat, ilon], 
                                                                       cat3[imonth, ilat, ilon], cat4[imonth, ilat, ilon], 
                                                                       cat5[imonth, ilat, ilon])
            
        S3 = np.where(landMask == 1, np.nan, S3)
        np.save(dataDir + '/safeShipSpeed_SSP_ens' + str(ens[numEns]) + '.npy', S3)
        del S3
    
        
    #### ARISE-1.5    
    for numEns in range(len(ens)):        
        for c in range(len(cats)):
            print("category " + str(c))
            a = np.full((420, 43, 288), np.nan)
            for imonth in range(420): 
                nan_indices = np.argwhere(np.isnan(cat_ARISE[cats[c], ens[numEns]][imonth,:,:].values))
                monthlyIceThickMeanInterp = cat_ARISE[cats[c], ens[numEns]][imonth,:,:].values.copy()
                # Iterate through the NaN indices
                for nan_index in nan_indices:
                    row, col = nan_index
                    neighbors = []
                    
                    # Check the surrounding 8 elements (or less at the edges)
                    for i in range(max(0, row - 1), min(monthlyIceThickMeanInterp.shape[0], row + 2)):
                        for j in range(max(0, col - 1), min(monthlyIceThickMeanInterp.shape[1], col + 2)):
                            if not np.isnan(monthlyIceThickMeanInterp[i, j]):
                                neighbors.append(monthlyIceThickMeanInterp[i, j])
                    
                    # If there are valid neighbors, fill the NaN with the average of all valid neighbors
                    if neighbors:
                        monthlyIceThickMeanInterp[row, col] = np.nanmean(neighbors)
                a[imonth,:,:] = monthlyIceThickMeanInterp
            np.save(dataDir + 'iceCategory' + str(cats[c]) + '_ARISE_ens' + str(ens[numEns]) + '.npy',a)
        
        
    #### safe ship speed for monthly data
    for numEns in range(len(ens)):   
        print(numEns)
        S3 = np.full((35, 12, 43, 288), np.nan)
        IN = []
        category1 = np.load(dataDir + 'iceCategory1_ARISE_ens' + str(ens[numEns]) + '.npy')
        category2 = np.load(dataDir + 'iceCategory2_ARISE_ens' + str(ens[numEns]) + '.npy')
        category3 = np.load(dataDir + 'iceCategory3_ARISE_ens' + str(ens[numEns]) + '.npy')
        category4 = np.load(dataDir + 'iceCategory4_ARISE_ens' + str(ens[numEns]) + '.npy')
        category5 = np.load(dataDir + 'iceCategory5_ARISE_ens' + str(ens[numEns]) + '.npy')
        
        category1[np.isnan(category1)] = 0
        category2[np.isnan(category2)] = 0
        category3[np.isnan(category3)] = 0
        category4[np.isnan(category4)] = 0
        category5[np.isnan(category5)] = 0
        for iyear in range(len(years)):
            firstMonth = 12 * iyear 
            lastMonth  = (12 * iyear) + 12
            cat1 = (category1[firstMonth:lastMonth, :, :]) * 10
            cat2 = (category2[firstMonth:lastMonth, :, :]) * 10
            cat3 = (category3[firstMonth:lastMonth, :, :]) * 10
            cat4 = (category4[firstMonth:lastMonth, :, :]) * 10
            cat5 = (category5[firstMonth:lastMonth, :, :]) * 10
            for imonth in range(12):
                for ilat in range(len(latIce)):
                    for ilon in range(len(lonIce)):
                        IN, S3[iyear, imonth, ilat, ilon] = iceMultipliers(cat1[imonth, ilat, ilon], cat2[imonth, ilat, ilon], 
                                                                       cat3[imonth, ilat, ilon], cat4[imonth, ilat, ilon], 
                                                                       cat5[imonth, ilat, ilon])
            
        # S3 = np.where(landMask == 1, np.nan, S3)
        np.save(dataDir + '/safeShipSpeed_ARISE_ens' + str(ens[numEns]) + '.npy', S3)
        del S3, cat1, cat2, cat3, cat4, cat5, category1, category2, category3, category4, category5
        
        
    #### ARISE-1.0    
    for numEns in range(len(ens)):        
        for c in range(len(cats)):
            print("category " + str(c))
            a = np.full((420, 43, 288), np.nan)
            for imonth in range(420): 
                nan_indices = np.argwhere(np.isnan(cat_ARISE1[cats[c], ens[numEns]][imonth,:,:].values))
                monthlyIceThickMeanInterp = cat_ARISE1[cats[c], ens[numEns]][imonth,:,:].values.copy()
                # Iterate through the NaN indices
                for nan_index in nan_indices:
                    row, col = nan_index
                    neighbors = []
                    
                    # Check the surrounding 8 elements (or less at the edges)
                    for i in range(max(0, row - 1), min(monthlyIceThickMeanInterp.shape[0], row + 2)):
                        for j in range(max(0, col - 1), min(monthlyIceThickMeanInterp.shape[1], col + 2)):
                            if not np.isnan(monthlyIceThickMeanInterp[i, j]):
                                neighbors.append(monthlyIceThickMeanInterp[i, j])
                    
                    # If there are valid neighbors, fill the NaN with the average of all valid neighbors
                    if neighbors:
                        monthlyIceThickMeanInterp[row, col] = np.nanmean(neighbors)
                a[imonth,:,:] = monthlyIceThickMeanInterp
            np.save(dataDir + 'iceCategory' + str(cats[c]) + '_ARISE1_ens' + str(ens[numEns]) + '.npy',a)
        
        
    #### safe ship speed for monthly data
    for numEns in range(len(ens)):   
        print(numEns)
        S3 = np.full((35, 12, 43, 288), np.nan)
        IN = []
        category1 = np.load(dataDir + 'iceCategory1_ARISE1_ens' + str(ens[numEns]) + '.npy')
        category2 = np.load(dataDir + 'iceCategory2_ARISE1_ens' + str(ens[numEns]) + '.npy')
        category3 = np.load(dataDir + 'iceCategory3_ARISE1_ens' + str(ens[numEns]) + '.npy')
        category4 = np.load(dataDir + 'iceCategory4_ARISE1_ens' + str(ens[numEns]) + '.npy')
        category5 = np.load(dataDir + 'iceCategory5_ARISE1_ens' + str(ens[numEns]) + '.npy')
        
        category1[np.isnan(category1)] = 0
        category2[np.isnan(category2)] = 0
        category3[np.isnan(category3)] = 0
        category4[np.isnan(category4)] = 0
        category5[np.isnan(category5)] = 0
        for iyear in range(len(years)):
            firstMonth = 12 * iyear 
            lastMonth  = (12 * iyear) + 12
            cat1 = (category1[firstMonth:lastMonth, :, :]) * 10
            cat2 = (category2[firstMonth:lastMonth, :, :]) * 10
            cat3 = (category3[firstMonth:lastMonth, :, :]) * 10
            cat4 = (category4[firstMonth:lastMonth, :, :]) * 10
            cat5 = (category5[firstMonth:lastMonth, :, :]) * 10
            for imonth in range(12):
                for ilat in range(len(latIce)):
                    for ilon in range(len(lonIce)):
                        IN, S3[iyear, imonth, ilat, ilon] = iceMultipliers(cat1[imonth, ilat, ilon], cat2[imonth, ilat, ilon], 
                                                                       cat3[imonth, ilat, ilon], cat4[imonth, ilat, ilon], 
                                                                       cat5[imonth, ilat, ilon])
            
        S3 = np.where(landMask == 1, np.nan, S3)
        np.save(dataDir + '/safeShipSpeed_ARISE1_ens' + str(ens[numEns]) + '.npy', S3)
        del S3
        
    return
    
    
    
def transitTime(simulation, latIce, lonIce, routeName):
    dataDir = '/Users/arielmor/Desktop/SAI/ArcticShipping/data/'
    figureDir = '/Users/arielmor/Desktop/SAI/ArcticShipping/figures/'
    import numpy as np
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from matplotlib.lines import Line2D
    import cartopy.feature as cfeature
    
    def is_valid(coord_value):
        return coord_value > 0.0
    
    def bfs_shortest_distance(coordinates, start, end):
        rows, cols = len(coordinates), len(coordinates[0])
        visited = set()
        queue = [(0, start, [])]  # Start node and its distance
        visited.add(start)
    
        while queue:
            #(row, col), distance, path = queue.popleft()
            distance, (row, col), path = heapq.heappop(queue)

    
            if (row, col) == end:
                return distance, path  # Reached the destination
    
            for dr, dc in [(1,1),(1,0),(1,-1),(0,1),(0,-1),(-1,1),(-1,0),(-1,-1)]:
                new_row, new_col = row + dr, col + dc
    
                if 0 <= new_row < rows and 0 <= new_col < cols and (new_row, new_col) not in visited:
                    if is_valid(coordinates[new_row][new_col]):  # Check validity using is_valid function
                        visited.add((new_row, new_col))
                        new_coords = (latIce[new_row + latOffset].values, lonIce[new_col + lonOffset].values)
                        old_coords = (latIce[row + latOffset].values, lonIce[col + lonOffset].values)
                        step_distance = geodesic(new_coords, old_coords).nautical
                        heapq.heappush(queue, (distance + step_distance, (new_row, new_col), path+[(row,col)]))
    
        return -1  # No path found

    
    
    def bfs_shortest_time(coordinates, start, end):
        rows, cols = len(coordinates), len(coordinates[0])
        visited = set()
        queue = [(0, start, [])]  # Start node and its distance
        visited.add(start)
        prev_speed = None  # Variable to store the previous speed
        complete_paths = []
        while queue:
            time, (row, col), path = heapq.heappop(queue)
    
            if (row, col) == end:
                complete_paths.append((time, (row, col), path))
                
                #assume shortest path by time is in first 4k returned paths
                if len(complete_paths) > 4000:
                    complete_paths = sorted(complete_paths, key=lambda x: x[1])  
                    fastest_path = complete_paths[0]
                    return fastest_path[0], fastest_path[2]  # Reached the destination
    
            for dr, dc in [(1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1)]:
                new_row, new_col = row + dr, col + dc
    
                if 0 <= new_row < rows and 0 <= new_col < cols and (new_row, new_col) not in visited:
                    if is_valid(coordinates[new_row][new_col]):  # Check validity using is_valid function
                        visited.add((new_row, new_col))
                        current_speed = coordinates[new_row][new_col]  # Speed at this new step
                        if prev_speed is not None:
                            speed_at_step = (prev_speed + current_speed) / 2  # Average of previous and current speeds
                        else:
                            speed_at_step = current_speed
                        
                        new_coords = (latIce[new_row + latOffset].values, lonIce[new_col + lonOffset].values)
                        old_coords = (latIce[row + latOffset].values, lonIce[col + lonOffset].values)
                        
                        step_distance = (geodesic(new_coords, old_coords).nautical) / 2
                        time_at_step = (step_distance / speed_at_step) + (step_distance / current_speed)
                        heapq.heappush(queue, (time + time_at_step, (new_row, new_col), path + [(row, col)]))  # Add time to the queue
                        prev_speed = current_speed  # Update previous speed
        
        
        if len(complete_paths) > 0:
            # print("all complete paths found")
            complete_paths = sorted(complete_paths, key=lambda x: x[1])  
            fastest_path = complete_paths[0]
            return fastest_path[0], fastest_path[2]
                
        return -1  # No path found
    
    
    #### Northern Sea Route start/end
    latAmderma = 21#int(np.abs(amderma[0]-latIce).argmin())
    latUlen    = 15#int(np.abs(ulen[0]-latIce).argmin())
    lonAmderma = 49#int(np.abs(amderma[1]-lonIce).argmin())
    lonUlen    = 153#int(np.abs(ulen[1]-lonIce).argmin())
    latRotterdam = 2
    lonRotterdam = 3
    latLawrence = 1
    lonLawrence = 242
    latChurchill = 9
    lonChurchill = 213
    print(latAmderma, lonAmderma)
    print(latChurchill, lonChurchill)
    print(latRotterdam, lonRotterdam)
    print(latUlen, lonUlen)
    
    #### shortest route
    from geopy.distance import geodesic
    import heapq  
    
    
    latOffset = 0 # where lat array starts in bounding box
    lonOffset = 0 # where lon array starts in bounding box
    
    #### route name    
    if routeName == 'NSR':
        origin_lat_lon = (latRotterdam-latOffset,lonRotterdam-lonOffset)
        dest_lat_lon = (latUlen-latOffset,lonUlen-lonOffset)
        print(origin_lat_lon, dest_lat_lon)
    elif routeName == 'NWP':
        origin_lat_lon = (latLawrence-latOffset,lonLawrence-lonOffset)
        dest_lat_lon = (latUlen-latOffset,lonUlen-lonOffset)
        print(origin_lat_lon, dest_lat_lon)
    elif routeName == 'trueNWP':
        origin_lat_lon = (latChurchill-latOffset,lonChurchill-lonOffset)
        dest_lat_lon = (latUlen-latOffset,lonUlen-lonOffset)
        print(origin_lat_lon, dest_lat_lon)
    
    
    #### load safe ship speed
    if simulation == 'ssp245':
        S3 = np.load('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_safe_ship_speed_from_categories_pc6_ssp.npy')
    elif simulation == 'arise-1.5':
        S3 = np.load('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_safe_ship_speed_from_categories_pc6_arise.npy')
    elif simulation == 'arise-1.0':
        S3 = np.load('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_safe_ship_speed_from_categories_pc6_arise1.0.npy')
        
    landMask = np.load('/Users/arielmor/Desktop/SAI/ArcticShipping/data/landMask_from_global_edited.npy')
    landMask[18,36] = 1.
    landMask[20,40] = 0.
    landMask[20:23,211] = 0.
    landMask[29,213] = 0.
    landMask[33,272:275] = 1.
    
    S3 = np.where(landMask == 1, np.nan, S3)
    
    
    #### single year to test
    yearForCalc = 0
    lat_lon_speed_arr = S3[yearForCalc,8,:,:]
    
    shortest_time2 = bfs_shortest_time(lat_lon_speed_arr, origin_lat_lon, dest_lat_lon)
    if shortest_time2 != -1:
        print("Shortest time following navigable cells:", np.round(shortest_time2[0]/24.,decimals=2), "days")
    else: print("no path found")
    
    
    #### loop through months
    print(simulation, routeName)
    # shortest_distance = {}
    shortest_time = {}
    lonPath       = {}
    latPath       = {}
    for iyear in range(35):
        print(iyear)
        # shortest_distance1 = np.array([]) 
        shortest_time1     = np.array([])
        lonPaths = []
        latPaths = []
        
        for imonth in range(12):
            lat_lon_speed_arr = S3[iyear,imonth,:,:]
            time = bfs_shortest_time(lat_lon_speed_arr, origin_lat_lon, dest_lat_lon)
            
            if time == -1: #or dist == -1:
                shortest_time1 = np.append(shortest_time1, [0])
                
            else:
                shortest_time1 = np.append(shortest_time1, [time[0]])
                
                lon1 = np.zeros((len(time[1])))
                lat1 = np.zeros((len(time[1])))
                for istep in range(len(time[1])):
                    lon1[istep] = lonIce[time[1][istep][1] + lonOffset]
                    lat1[istep] = latIce[time[1][istep][0] + latOffset]
                lonPaths.append(lon1)
                latPaths.append(lat1)
                
        shortest_time[iyear]     = shortest_time1  
        lonPath[iyear] = lonPaths
        latPath[iyear] = latPaths
    
    #### save paths/times
    import pickle 
    with open('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_shortest_time_' + str(routeName) + '_from_categories_' + str(simulation) + '.pickle', 'wb') as handle:
        pickle.dump(shortest_time, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_shortest_time_'+ str(routeName) + '_lonPath_from_categories_' + str(simulation) + '.pickle', 'wb') as handle:
        pickle.dump(lonPath, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_shortest_time_'+ str(routeName) + '_latPath_from_categories_' + str(simulation) + '.pickle', 'wb') as handle:
        pickle.dump(latPath, handle, protocol=pickle.HIGHEST_PROTOCOL)
     
    del S3, shortest_time, lonPath, latPath 


    ## -------------------------------- ##
    #### ensembles - load safe ship speed - 
    ens = ['001','002','003','004','005','006','007','008','009','010']
    S3 = {}
    for ensNum in range(len(ens)):
        print(ensNum)
        if simulation == 'ssp245':
            S3[ens[ensNum]] = np.load(dataDir + 'safeShipSpeed_SSP_ens' + str(ens[ensNum]) + '.npy')
        elif simulation == 'arise-1.5':
            S3[ens[ensNum]] = np.load(dataDir + 'safeShipSpeed_ARISE_ens' + str(ens[ensNum]) + '.npy')
        elif simulation == 'arise-1.0':
            S3[ens[ensNum]] = np.load(dataDir + 'safeShipSpeed_ARISE1_ens' + str(ens[ensNum]) + '.npy')
       
        landMask = np.load('/Users/arielmor/Desktop/SAI/ArcticShipping/data/landMask_from_global_edited.npy')
        S3[ens[ensNum]] = np.where(landMask == 1, np.nan, S3[ens[ensNum]])
        

        ## loop through months
        shortest_time = {}
        for iyear in range(35):
            shortest_time1 = np.array([])
            
            for imonth in range(12):
                lat_lon_speed_arr = S3[ens[ensNum]][iyear,imonth,:,:]
                time = bfs_shortest_time(lat_lon_speed_arr, origin_lat_lon, dest_lat_lon)
                
                if time == -1: 
                    shortest_time1 = np.append(shortest_time1, [np.nan])
                    
                else:
                    shortest_time1 = np.append(shortest_time1, [time[0]])
                    
            shortest_time[iyear] = shortest_time1  
    
        ## save paths/times
        print("saving " + str(simulation) + ", " + str(ens[ensNum]))
        with open('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_shortest_time_' + str(routeName) + '_from_categories_' + str(simulation) + '_ens' + str(ens[ensNum]) + '.pickle', 'wb') as handle:
            pickle.dump(shortest_time, handle, protocol=pickle.HIGHEST_PROTOCOL)
             
    
    #### range for NSR
    sspNSR    = {}
    sspNWP    = {}
    ariseNSR  = {}
    ariseNWP  = {}
    arise1NSR = {}
    arise1NWP = {}
    for ensNum in range(len(ens)):
        with open(dataDir + 'monthly_shortest_time_NSR_from_categories_ssp245_ens' + str(ens[ensNum]) + '.pickle', 'rb') as handle:
            sspNSR[ens[ensNum]] = pickle.load(handle)
        with open(dataDir + 'monthly_shortest_time_NWP_from_categories_ssp245_ens' + str(ens[ensNum]) + '.pickle', 'rb') as handle:
            sspNWP[ens[ensNum]] = pickle.load(handle)
        del handle
        
        with open(dataDir + 'monthly_shortest_time_NSR_from_categories_arise-1.5_ens' + str(ens[ensNum]) + '.pickle', 'rb') as handle:
            ariseNSR[ens[ensNum]] = pickle.load(handle)
        with open(dataDir + 'monthly_shortest_time_NWP_from_categories_arise-1.5_ens' + str(ens[ensNum]) + '.pickle', 'rb') as handle:
            ariseNWP[ens[ensNum]] = pickle.load(handle)
        del handle
        
        with open(dataDir + 'monthly_shortest_time_NSR_from_categories_arise-1.0_ens' + str(ens[ensNum]) + '.pickle', 'rb') as handle:
            arise1NSR[ens[ensNum]] = pickle.load(handle)
        with open(dataDir + 'monthly_shortest_time_NWP_from_categories_arise-1.0_ens' + str(ens[ensNum]) + '.pickle', 'rb') as handle:
            arise1NWP[ens[ensNum]] = pickle.load(handle)    
        del handle
        
    
    rangeAllMonths_ssp = np.full((len(ens),12,35), np.nan)
    minAllMonths_ssp = np.full((12,35), np.nan)
    maxAllMonths_ssp = np.full((12,35), np.nan) 
    for ensNum in range(len(ens)): 
        for iMonth in range(12):
            for iyear in range(35):
                rangeAllMonths_ssp[ensNum,iMonth,iyear] = (sspNSR[ens[ensNum]][iyear][iMonth])/24.
                if np.count_nonzero(~np.isnan(rangeAllMonths_ssp[:,iMonth,iyear])) > 7:
                    minAllMonths_ssp[iMonth,iyear] = np.nanmin(rangeAllMonths_ssp[:,iMonth,iyear])
                    maxAllMonths_ssp[iMonth,iyear] = np.nanmax(rangeAllMonths_ssp[:,iMonth,iyear])
                else:
                    minAllMonths_ssp[iMonth,iyear] = np.nan
                    maxAllMonths_ssp[iMonth,iyear] = np.nan
                    
    
    rangeAllMonths_arise = np.full((len(ens),12,35),np.nan)
    minAllMonths_arise = np.full((12,35),np.nan)
    maxAllMonths_arise = np.full((12,35),np.nan) 
    for ensNum in range(len(ens)): 
        print(ensNum)
        for iMonth in range(12):
            for iyear in range(35):
                rangeAllMonths_arise[ensNum,iMonth,iyear] = (ariseNSR[ens[ensNum]][iyear][iMonth])/24.
                if np.count_nonzero(~np.isnan(rangeAllMonths_arise[:,iMonth,iyear])) > 7:
                    minAllMonths_arise[iMonth,iyear] = np.nanmin(rangeAllMonths_arise[:,iMonth,iyear])
                    maxAllMonths_arise[iMonth,iyear] = np.nanmax(rangeAllMonths_arise[:,iMonth,iyear])
                else:
                    minAllMonths_arise[iMonth,iyear] = np.nan
                    maxAllMonths_arise[iMonth,iyear] = np.nan
    
                    
    
    rangeAllMonths_arise1 = np.full((len(ens),12,35),np.nan)
    minAllMonths_arise1 = np.full((12,35),np.nan)
    maxAllMonths_arise1 = np.full((12,35),np.nan)  
    for ensNum in range(len(ens)): 
        for iMonth in range(12):
            for iyear in range(35):
                rangeAllMonths_arise1[ensNum,iMonth,iyear] = (arise1NSR[ens[ensNum]][iyear][iMonth])/24.
                if np.count_nonzero(~np.isnan(rangeAllMonths_arise1[:,iMonth,iyear])) > 7:
                    minAllMonths_arise1[iMonth,iyear] = np.nanmin(rangeAllMonths_arise1[:,iMonth,iyear])
                    maxAllMonths_arise1[iMonth,iyear] = np.nanmax(rangeAllMonths_arise1[:,iMonth,iyear])
                else:
                    minAllMonths_arise1[iMonth,iyear] = np.nan
                    maxAllMonths_arise1[iMonth,iyear] = np.nan
                    
                    
    #### range for NWP
    rangeAllMonths_NWP_ssp = np.full((len(ens),12,35), np.nan)
    minAllMonths_NWP_ssp = np.full((12,35), np.nan)
    maxAllMonths_NWP_ssp = np.full((12,35), np.nan) 
    for ensNum in range(len(ens)): 
        for iMonth in range(12):
            for iyear in range(35):
                rangeAllMonths_NWP_ssp[ensNum,iMonth,iyear] = (sspNWP[ens[ensNum]][iyear][iMonth])/24.
                if np.count_nonzero(~np.isnan(rangeAllMonths_NWP_ssp[:,iMonth,iyear])) > 7:
                    minAllMonths_NWP_ssp[iMonth,iyear] = np.nanmin(rangeAllMonths_NWP_ssp[:,iMonth,iyear])
                    maxAllMonths_NWP_ssp[iMonth,iyear] = np.nanmax(rangeAllMonths_NWP_ssp[:,iMonth,iyear])
                else:
                    minAllMonths_NWP_ssp[iMonth,iyear] = np.nan
                    maxAllMonths_NWP_ssp[iMonth,iyear] = np.nan
                    
    
    rangeAllMonths_NWP_arise = np.full((len(ens),12,35),np.nan)
    minAllMonths_NWP_arise = np.full((12,35),np.nan)
    maxAllMonths_NWP_arise = np.full((12,35),np.nan) 
    for ensNum in range(len(ens)): 
        for iMonth in range(12):
            for iyear in range(35):
                rangeAllMonths_NWP_arise[ensNum,iMonth,iyear] = (ariseNWP[ens[ensNum]][iyear][iMonth])/24.
                if np.count_nonzero(~np.isnan(rangeAllMonths_NWP_arise[:,iMonth,iyear])) > 7:
                    minAllMonths_NWP_arise[iMonth,iyear] = np.nanmin(rangeAllMonths_NWP_arise[:,iMonth,iyear])
                    maxAllMonths_NWP_arise[iMonth,iyear] = np.nanmax(rangeAllMonths_NWP_arise[:,iMonth,iyear])
                else:
                    minAllMonths_NWP_arise[iMonth,iyear] = np.nan
                    maxAllMonths_NWP_arise[iMonth,iyear] = np.nan
                    
    
    rangeAllMonths_NWP_arise1 = np.full((len(ens),12,35),np.nan)
    minAllMonths_NWP_arise1 = np.full((12,35),np.nan)
    maxAllMonths_NWP_arise1 = np.full((12,35),np.nan)  
    for ensNum in range(len(ens)): 
        for iMonth in range(12):
            for iyear in range(35):
                rangeAllMonths_NWP_arise1[ensNum,iMonth,iyear] = (arise1NWP[ens[ensNum]][iyear][iMonth])/24.
                if np.count_nonzero(~np.isnan(rangeAllMonths_NWP_arise1[:,iMonth,iyear])) > 7:
                    minAllMonths_NWP_arise1[iMonth,iyear] = np.nanmin(rangeAllMonths_NWP_arise1[:,iMonth,iyear])
                    maxAllMonths_NWP_arise1[iMonth,iyear] = np.nanmax(rangeAllMonths_NWP_arise1[:,iMonth,iyear])
                else:
                    minAllMonths_NWP_arise1[iMonth,iyear] = np.nan
                    maxAllMonths_NWP_arise1[iMonth,iyear] = np.nan            
        
        
    ## NSR
    # with open('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_shortest_time_NSR_from_categories_ssp245_updated.pickle', 'rb') as handle:
    #     shortest_time_ssp = pickle.load(handle)
    # with open('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_shortest_time_NSR_from_categories_arise-1.5_updated.pickle', 'rb') as handle:
    #     shortest_time_arise = pickle.load(handle)
    # with open('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_shortest_time_NSR_from_categories_arise-1.0_updated.pickle', 'rb') as handle:
    #     shortest_time_arise1 = pickle.load(handle)
        
    # with open('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_shortest_time_NSR_latPath_from_categories_ssp245_updated.pickle', 'rb') as handle:
    #     latPath_ssp = pickle.load(handle)
    # with open('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_shortest_time_NSR_latPath_from_categories_arise-1.5_updated.pickle', 'rb') as handle:
    #     latPath_arise = pickle.load(handle)
    # with open('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_shortest_time_NSR_latPath_from_categories_arise-1.0_updated.pickle', 'rb') as handle:
    #     latPath_arise1 = pickle.load(handle)
        
    # with open('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_shortest_time_NSR_lonPath_from_categories_ssp245_updated.pickle', 'rb') as handle:
    #      lonPath_ssp = pickle.load(handle)
    # with open('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_shortest_time_NSR_lonPath_from_categories_arise-1.5_updated.pickle', 'rb') as handle:
    #      lonPath_arise = pickle.load(handle)
    # with open('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_shortest_time_NSR_lonPath_from_categories_arise-1.0_updated.pickle', 'rb') as handle:
    #      lonPath_arise1 = pickle.load(handle)
    
    # ## NWP
    # with open('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_shortest_time_NWP_from_categories_ssp245_updated.pickle', 'rb') as handle:
    #     shortest_time_NWP_ssp = pickle.load(handle)
    # with open('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_shortest_time_NWP_from_categories_arise-1.5_updated.pickle', 'rb') as handle:
    #     shortest_time_NWP_arise = pickle.load(handle)
    # with open('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_shortest_time_NWP_from_categories_arise-1.0_updated.pickle', 'rb') as handle:
    #     shortest_time_NWP_arise1 = pickle.load(handle)
            
    # with open('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_shortest_time_NWP_latPath_from_categories_ssp245_updated.pickle', 'rb') as handle:
    #     latPath_NWP_ssp = pickle.load(handle)
    # with open('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_shortest_time_NWP_latPath_from_categories_arise-1.5_updated.pickle', 'rb') as handle:
    #     latPath_NWP_arise = pickle.load(handle)
    # with open('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_shortest_time_NWP_latPath_from_categories_arise-1.0_updated.pickle', 'rb') as handle:
    #     latPath_NWP_arise1 = pickle.load(handle)
                 
    # with open('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_shortest_time_NWP_lonPath_from_categories_ssp245_updated.pickle', 'rb') as handle:
    #      lonPath_NWP_ssp = pickle.load(handle)
    # with open('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_shortest_time_NWP_lonPath_from_categories_arise-1.5_updated.pickle', 'rb') as handle:
    #      lonPath_NWP_arise = pickle.load(handle)
    # with open('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_shortest_time_NWP_lonPath_from_categories_arise-1.0_updated.pickle', 'rb') as handle:
    #      lonPath_NWP_arise1 = pickle.load(handle)
    
    #### OPEN paths/times - all sims
    with open('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_shortest_time_NSR_from_categories_ssp245.pickle', 'rb') as handle:
        shortest_time_ssp = pickle.load(handle)
    with open('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_shortest_time_NSR_from_categories_arise-1.5.pickle', 'rb') as handle:
        shortest_time_arise = pickle.load(handle)
    with open('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_shortest_time_NSR_from_categories_arise-1.0.pickle', 'rb') as handle:
        shortest_time_arise1 = pickle.load(handle)
        
    with open('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_shortest_time_NSR_latPath_from_categories_ssp245.pickle', 'rb') as handle:
        latPath_ssp = pickle.load(handle)
    with open('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_shortest_time_NSR_latPath_from_categories_arise-1.5.pickle', 'rb') as handle:
        latPath_arise = pickle.load(handle)
    with open('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_shortest_time_NSR_latPath_from_categories_arise-1.0.pickle', 'rb') as handle:
        latPath_arise1 = pickle.load(handle)
        
    with open('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_shortest_time_NSR_lonPath_from_categories_ssp245.pickle', 'rb') as handle:
         lonPath_ssp = pickle.load(handle)
    with open('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_shortest_time_NSR_lonPath_from_categories_arise-1.5.pickle', 'rb') as handle:
         lonPath_arise = pickle.load(handle)
    with open('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_shortest_time_NSR_lonPath_from_categories_arise-1.0.pickle', 'rb') as handle:
         lonPath_arise1 = pickle.load(handle)
    
    ## NWP
    with open('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_shortest_time_NWP_from_categories_ssp245.pickle', 'rb') as handle:
        shortest_time_NWP_ssp = pickle.load(handle)
    with open('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_shortest_time_NWP_from_categories_arise-1.5.pickle', 'rb') as handle:
        shortest_time_NWP_arise = pickle.load(handle)
    with open('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_shortest_time_NWP_from_categories_arise-1.0.pickle', 'rb') as handle:
        shortest_time_NWP_arise1 = pickle.load(handle)
            
    with open('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_shortest_time_NWP_latPath_from_categories_ssp245.pickle', 'rb') as handle:
        latPath_NWP_ssp = pickle.load(handle)
    with open('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_shortest_time_NWP_latPath_from_categories_arise-1.5.pickle', 'rb') as handle:
        latPath_NWP_arise = pickle.load(handle)
    with open('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_shortest_time_NWP_latPath_from_categories_arise-1.0.pickle', 'rb') as handle:
        latPath_NWP_arise1 = pickle.load(handle)
                 
    with open('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_shortest_time_NWP_lonPath_from_categories_ssp245.pickle', 'rb') as handle:
         lonPath_NWP_ssp = pickle.load(handle)
    with open('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_shortest_time_NWP_lonPath_from_categories_arise-1.5.pickle', 'rb') as handle:
         lonPath_NWP_arise = pickle.load(handle)
    with open('/Users/arielmor/Desktop/SAI/ArcticShipping/data/monthly_shortest_time_NWP_lonPath_from_categories_arise-1.0.pickle', 'rb') as handle:
         lonPath_NWP_arise1 = pickle.load(handle)
        
    
    #### number of trips in each region
    ## max = 24/year 
    ## choose latitude and longitude boundaries for each region (NWP, TPR, NSR)
    ## TPR => any longitude, latitude > 84
    ## NWP => 200 < lon < 300, latitude < 84
    ## NSR => 0 < lon < 170, latitude < 84
    ## length of latPath/lonPath = 35 (number of years in sim) 
    ##     --> for each year, count how many trips fall within the lat/lon bounds
    ##          of each region
    ''' SSP2-4.5 '''
    TPR_BS_count_ssp        = np.zeros((35))
    percent_TPR_from_BS_ssp = np.zeros((35))
    NWP_count_ssp           = np.zeros((35))
    percent_NWP_ssp         = np.zeros((35))
    percent_NSR_ssp         = np.zeros((35))
    TPR_R_count_ssp         = np.zeros((35))
    percent_TPR_from_R_ssp  = np.zeros((35))
    NSR_count_ssp           = np.zeros((35))
    
    for iyear in range(35):
        ## from Rotterdam
        for imonth in range(len(latPath_ssp[iyear])):
            if (latPath_ssp[iyear][imonth] >= 85.).any():
                TPR_R_count_ssp[iyear] = TPR_R_count_ssp[iyear] + 1
            elif (latPath_ssp[iyear][imonth] < 85.).all() and (lonPath_ssp[iyear][imonth] > 200.).any():
                NWP_count_ssp[iyear] = NWP_count_ssp[iyear] + 1
            elif (latPath_ssp[iyear][imonth] < 85.).all() and (lonPath_ssp[iyear][imonth] < 170.).any():
                NSR_count_ssp[iyear] = NSR_count_ssp[iyear] + 1
        percent_TPR_from_R_ssp[iyear] = TPR_R_count_ssp[iyear]/12
        percent_NSR_ssp[iyear] = NSR_count_ssp[iyear]/12
                
        ## from Blanc-Sablon        
        for imonth in range(len(latPath_NWP_ssp[iyear])):
            if (latPath_NWP_ssp[iyear][imonth] >= 85.).any():
                TPR_BS_count_ssp[iyear] = TPR_BS_count_ssp[iyear] + 1
            elif (latPath_NWP_ssp[iyear][imonth] < 85.).all() and (lonPath_NWP_ssp[iyear][imonth] > 200.).any():
                NWP_count_ssp[iyear] = NWP_count_ssp[iyear] + 1
            elif (latPath_NWP_ssp[iyear][imonth] < 85.).all() and (lonPath_NWP_ssp[iyear][imonth] < 170.).any():
                NSR_count_ssp[iyear] = NSR_count_ssp[iyear] + 1
        percent_TPR_from_BS_ssp[iyear] = TPR_BS_count_ssp[iyear]/12
        percent_NWP_ssp[iyear] = NWP_count_ssp[iyear]/12
   
                
    ''' ARISE-1.5 '''
    TPR_BS_count_arise        = np.zeros((35))
    percent_TPR_from_BS_arise = np.zeros((35))
    NWP_count_arise           = np.zeros((35))
    percent_NWP_arise         = np.zeros((35))
    percent_NSR_arise         = np.zeros((35))
    TPR_R_count_arise         = np.zeros((35))
    percent_TPR_from_R_arise  = np.zeros((35))
    NSR_count_arise           = np.zeros((35))
    
    for iyear in range(35):
        ## from Rotterdam
        for imonth in range(len(latPath_arise[iyear])):
            if (latPath_arise[iyear][imonth] >= 85.).any():
                TPR_R_count_arise[iyear] = TPR_R_count_arise[iyear] + 1
            elif (latPath_arise[iyear][imonth] < 85.).all() and (lonPath_arise[iyear][imonth] > 200.).any():
                NWP_count_arise[iyear] = NWP_count_arise[iyear] + 1
            elif (latPath_arise[iyear][imonth] < 85.).all() and (lonPath_arise[iyear][imonth] < 170.).any():
                NSR_count_arise[iyear] = NSR_count_arise[iyear] + 1
        percent_TPR_from_R_arise[iyear] = TPR_R_count_arise[iyear]/12
        percent_NSR_arise[iyear] = NSR_count_arise[iyear]/12
                
        ## from Blanc-Sablon        
        for imonth in range(len(latPath_NWP_arise[iyear])):
            if (latPath_NWP_arise[iyear][imonth] >= 85.).any():
                TPR_BS_count_arise[iyear] = TPR_BS_count_arise[iyear] + 1
            elif (latPath_NWP_arise[iyear][imonth] < 85.).all() and (lonPath_NWP_arise[iyear][imonth] > 200.).any():
                NWP_count_arise[iyear] = NWP_count_arise[iyear] + 1
            elif (latPath_NWP_arise[iyear][imonth] < 85.).all() and (lonPath_NWP_arise[iyear][imonth] < 170.).any():
                NSR_count_arise[iyear] = NSR_count_arise[iyear] + 1
        percent_TPR_from_BS_arise[iyear] = TPR_BS_count_arise[iyear]/12
        percent_NWP_arise[iyear] = NWP_count_arise[iyear]/12
    
    
    ''' ARISE-1.0 '''
    TPR_BS_count_arise1        = np.zeros((35))
    percent_TPR_from_BS_arise1 = np.zeros((35))
    NWP_count_arise1           = np.zeros((35))
    percent_NWP_arise1         = np.zeros((35))
    percent_NSR_arise1         = np.zeros((35))
    TPR_R_count_arise1         = np.zeros((35))
    percent_TPR_from_R_arise1  = np.zeros((35))
    NSR_count_arise1           = np.zeros((35))
    
    for iyear in range(35):
        ## from Rotterdam
        for imonth in range(len(latPath_arise1[iyear])):
            if (latPath_arise1[iyear][imonth] >= 85.).any():
                TPR_R_count_arise1[iyear] = TPR_R_count_arise1[iyear] + 1
            elif (latPath_arise1[iyear][imonth] < 85.).all() and (lonPath_arise1[iyear][imonth] > 200.).any():
                NWP_count_arise1[iyear] = NWP_count_arise1[iyear] + 1
            elif (latPath_arise1[iyear][imonth] < 85.).all() and (lonPath_arise1[iyear][imonth] < 170.).any():
                NSR_count_arise1[iyear] = NSR_count_arise1[iyear] + 1
        percent_TPR_from_R_arise1[iyear] = TPR_R_count_arise1[iyear]/12
        percent_NSR_arise1[iyear] = NSR_count_arise1[iyear]/12
                
        ## from Blanc-Sablon        
        for imonth in range(len(latPath_NWP_arise1[iyear])):
            if (latPath_NWP_arise1[iyear][imonth] >= 85.).any():
                TPR_BS_count_arise1[iyear] = TPR_BS_count_arise1[iyear] + 1
            elif (latPath_NWP_arise1[iyear][imonth] < 85.).all() and (lonPath_NWP_arise1[iyear][imonth] > 200.).any():
                NWP_count_arise1[iyear] = NWP_count_arise1[iyear] + 1
            elif (latPath_NWP_arise1[iyear][imonth] < 85.).all() and (lonPath_NWP_arise1[iyear][imonth] < 170.).any():
                NSR_count_arise1[iyear] = NSR_count_arise1[iyear] + 1
        percent_TPR_from_BS_arise1[iyear] = TPR_BS_count_arise1[iyear]/12
        percent_NWP_arise1[iyear] = NWP_count_arise1[iyear]/12
    
    
    print("ssp NWP: ", NWP_count_ssp)
    print("ssp NSR: ", NSR_count_ssp)
    print("ssp TSR from Canada", TPR_BS_count_ssp)
    print("ssp TSR from Neth", TPR_R_count_ssp)
    print("ssp total NWP: ", NWP_count_ssp.sum())
    print("ssp total NSR: ", NSR_count_ssp.sum())
    print("ssp total TSR from Canada: ", TPR_BS_count_ssp.sum())
    print("ssp total TSR from Neth: ", TPR_R_count_ssp.sum())
    print(" ")
    print("arise NWP: ", NWP_count_arise)
    print("arise NSR: ", NSR_count_arise)
    print("arise TSR from Canada", TPR_BS_count_arise)
    print("arise TSR from Neth", TPR_R_count_arise)
    print("arise total NWP: ", NWP_count_arise.sum())
    print("arise total NSR: ", NSR_count_arise.sum())
    print("arise total TSR from Canada: ", TPR_BS_count_arise.sum())
    print("arise total TSR from Neth: ", TPR_R_count_arise.sum())
    print(" ")
    print("arise1 NWP: ", NWP_count_arise1)
    print("arise1 NSR: ", NSR_count_arise1)
    print("arise1 TSR from Canada", TPR_BS_count_arise1)
    print("arise1 TSR from Neth", TPR_R_count_arise1)
    print("arise1 total NWP: ", NWP_count_arise1.sum())
    print("arise1 total NSR: ", NSR_count_arise1.sum())
    print("arise1 total TSR from Canada: ", TPR_BS_count_arise1.sum())
    print("arise1 total TSR from Neth: ", TPR_R_count_arise1.sum())
    
        
    #### scatterplot shortest time by month
    shortest_time_arise_by_month = np.zeros((35,12)) 
    shortest_time_ssp_by_month = np.zeros((35,12)) 
    shortest_time_arise1_by_month = np.zeros((35,12)) 
    for iyear in range(35):
        for imonth in range(12):
            shortest_time_arise_by_month[iyear,imonth] = shortest_time_arise[iyear][imonth]/24.
            shortest_time_arise1_by_month[iyear,imonth] = shortest_time_arise1[iyear][imonth]/24.
            shortest_time_ssp_by_month[iyear,imonth] = shortest_time_ssp[iyear][imonth]/24.
            
    shortest_time_arise_by_month_NWP = np.zeros((35,12)) 
    shortest_time_ssp_by_month_NWP = np.zeros((35,12)) 
    shortest_time_arise1_by_month_NWP = np.zeros((35,12)) 
    for iyear in range(35):
        for imonth in range(12):
            shortest_time_arise_by_month_NWP[iyear,imonth] = shortest_time_NWP_arise[iyear][imonth]/24.
            shortest_time_arise1_by_month_NWP[iyear,imonth] = shortest_time_NWP_arise1[iyear][imonth]/24.
            shortest_time_ssp_by_month_NWP[iyear,imonth] = shortest_time_NWP_ssp[iyear][imonth]/24.
   
    #### error bars for all ens
    ## for plotting purposes, change 0 to 12 (below data's minimum value)
    shortest_time_arise_by_month[shortest_time_arise_by_month == 0.] = np.nan
    shortest_time_ssp_by_month[shortest_time_ssp_by_month == 0.] = np.nan
    shortest_time_arise1_by_month[shortest_time_arise1_by_month == 0.] = np.nan
    
    shortest_time_arise_by_month_NWP[shortest_time_arise_by_month_NWP == 0.] = np.nan
    shortest_time_ssp_by_month_NWP[shortest_time_ssp_by_month_NWP == 0.] = np.nan
    shortest_time_arise1_by_month_NWP[shortest_time_arise1_by_month_NWP == 0.] = np.nan
    
    lower_error_ssp = np.empty((12,35))
    upper_error_ssp = np.empty((12,35))
    lower_error_arise = np.empty((12,35))
    upper_error_arise = np.empty((12,35))
    lower_error_arise1 = np.empty((12,35))
    upper_error_arise1 = np.empty((12,35))
    for iMonth in range(12):
        lower_error_ssp[iMonth,:] = np.abs(shortest_time_ssp_by_month[:,iMonth] - minAllMonths_ssp[iMonth,:])
        upper_error_ssp[iMonth,:] = np.abs(shortest_time_ssp_by_month[:,iMonth] - maxAllMonths_ssp[iMonth,:])
        lower_error_arise[iMonth,:] = np.abs(shortest_time_arise_by_month[:,iMonth] - minAllMonths_arise[iMonth,:])
        upper_error_arise[iMonth,:] = np.abs(shortest_time_arise_by_month[:,iMonth] - maxAllMonths_arise[iMonth,:])
        lower_error_arise1[iMonth,:] = np.abs(shortest_time_arise1_by_month[:,iMonth] - minAllMonths_arise1[iMonth,:])
        upper_error_arise1[iMonth,:] = np.abs(shortest_time_arise1_by_month[:,iMonth] - maxAllMonths_arise1[iMonth,:])
        
    
    
    lower_error_NWP_ssp = np.empty((12,35))
    upper_error_NWP_ssp = np.empty((12,35))
    lower_error_NWP_arise = np.empty((12,35))
    upper_error_NWP_arise = np.empty((12,35))
    lower_error_NWP_arise1 = np.empty((12,35))
    upper_error_NWP_arise1 = np.empty((12,35))
    for iMonth in range(12):
        lower_error_NWP_ssp[iMonth,:] = np.abs(shortest_time_ssp_by_month_NWP[:,iMonth] - minAllMonths_NWP_ssp[iMonth,:])
        upper_error_NWP_ssp[iMonth,:] = np.abs(shortest_time_ssp_by_month_NWP[:,iMonth] - maxAllMonths_NWP_ssp[iMonth,:])
        lower_error_NWP_arise[iMonth,:] = np.abs(shortest_time_arise_by_month_NWP[:,iMonth] - minAllMonths_NWP_arise[iMonth,:])
        upper_error_NWP_arise[iMonth,:] = np.abs(shortest_time_arise_by_month_NWP[:,iMonth] - maxAllMonths_NWP_arise[iMonth,:])
        lower_error_NWP_arise1[iMonth,:] = np.abs(shortest_time_arise1_by_month_NWP[:,iMonth] - minAllMonths_NWP_arise1[iMonth,:])
        upper_error_NWP_arise1[iMonth,:] = np.abs(shortest_time_arise1_by_month_NWP[:,iMonth] - maxAllMonths_NWP_arise1[iMonth,:])
        
        
    ## for plotting purposes, change 0 to 12.3 (below data's minimum value)
    shortest_time_ssp_by_month = np.where(np.isnan(shortest_time_ssp_by_month), 12, shortest_time_ssp_by_month)
    shortest_time_arise_by_month = np.where(np.isnan(shortest_time_arise_by_month), 12, shortest_time_arise_by_month)
    shortest_time_arise1_by_month = np.where(np.isnan(shortest_time_arise1_by_month), 12, shortest_time_arise1_by_month)
    
    shortest_time_ssp_by_month_NWP = np.where(np.isnan(shortest_time_ssp_by_month_NWP), 12, shortest_time_ssp_by_month_NWP)
    shortest_time_arise_by_month_NWP = np.where(np.isnan(shortest_time_arise_by_month_NWP), 12, shortest_time_arise_by_month_NWP)
    shortest_time_arise1_by_month_NWP = np.where(np.isnan(shortest_time_arise1_by_month_NWP), 12, shortest_time_arise1_by_month_NWP)

    
    monthNames = ['January', 'February', 'March', 'April','May','June','July','August',
                  'September','October','November','December']
    saveMonth = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']  
    
    #### for main text
    for i in range(12):
        monthToPlot = i
        
        asymmetric_error_ssp = [lower_error_ssp[i,:], upper_error_ssp[i,:]]
        asymmetric_error_arise1 = [lower_error_arise1[i,:], upper_error_arise1[i,:]]
        asymmetric_error_arise = [lower_error_arise[i,:], upper_error_arise[i,:]]
        
        fig = plt.figure(figsize=(8.5,4), dpi=1200)
        plt.errorbar(np.linspace(0,34,35), shortest_time_ssp_by_month[:,monthToPlot], yerr=asymmetric_error_ssp, capsize=2.5, color='r', fmt = 'o', label='SSP2-4.5')#, ms=37)
        plt.errorbar(np.linspace(0,34,35), shortest_time_arise1_by_month[:,monthToPlot], yerr=asymmetric_error_arise1, capsize=2.5, color='purple', fmt='s', label='ARISE-1.0')#, ms=35)
        plt.errorbar(np.linspace(0,34,35), shortest_time_arise_by_month[:,monthToPlot], yerr=asymmetric_error_arise, capsize=2.5, color=(0.02,0.65,0.93), fmt = 'x', label='ARISE-1.5', ms=7)
        plt.xticks([0, 4, 9, 14, 19, 24, 29, 34], 
                    ['2035', '2040', '2045', '2050', '2055', '2060', '2065', '2069'], 
                    fontsize = 12)
        plt.ylabel('Time to cross (days)', fontsize=11, fontweight='bold')
        plt.ylim((11,37.5)) 
        plt.yticks([12, 16, 20, 24, 28, 32, 36],
                ['No\ntrips', 16, 20, 24, 28, 32, 36])
        
        # ## for supplement: 
        # plt.ylim((11.3,75))
        # plt.yticks([12.3, 16, 20, 24, 28, 32, 36, 40, 44],
        #         ['No\ntrips', 16, 20, 24, 28, 32, 36, 40, 44])
        
        plt.axhline(y = 13.3, color = 'xkcd:grey', linestyle = '--') 
        plt.axhline(y = 30, color = 'k', linestyle = '--', linewidth=0.5) 
        if i == 0:
            handles, labels = plt.gca().get_legend_handles_labels()
            order = [0,2,1]
            plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
                       loc='upper left',bbox_to_anchor=(0.1, 0.5, 0.5, 0.5)) #bbox_to_anchor=(0., 0.09, 0.5, 0.5))
        plt.title(str(monthNames[monthToPlot]), fontsize=14, fontweight='bold')
        plt.savefig(figureDir + 
                    '/shortest_time_to_cross_NSR_scatter_' + str(saveMonth[monthToPlot]) + 
                    '_all_categories_updated.jpg', 
                    dpi=1200, bbox_inches='tight')
        
        
        
        #### scatterplot NWP
        fig = plt.figure(figsize=(8.5,4), dpi=1200)
        plt.scatter(np.linspace(0,34,35), shortest_time_ssp_by_month_NWP[:,monthToPlot], color='r', marker = 'o', label='SSP2-4.5', s=37)
        plt.scatter(np.linspace(0,34,35), shortest_time_arise1_by_month_NWP[:,monthToPlot], color='purple', marker = 's', label='ARISE-1.0', s=35)
        plt.scatter(np.linspace(0,34,35), shortest_time_arise_by_month_NWP[:,monthToPlot], color=(0.02,0.65,0.93), marker = 'x', label='ARISE-1.5', s=44)
        plt.xticks([0, 4, 9, 14, 19, 24, 29, 34], 
                    ['2035', '2040', '2045', '2050', '2055', '2060', '2065', '2069'], 
                    fontsize = 12)
        plt.ylim((11.3,37.5)) 
        plt.yticks([12.3, 16, 20, 24, 28, 32, 36],
                ['No\ntrips', 16, 20, 24, 28, 32, 36])
        
        # ## for supplement: 
        # plt.ylim((11.3,45))
        # plt.yticks([12.3, 16, 20, 24, 28, 32, 36, 40, 44],
        #         ['No\ntrips', 16, 20, 24, 28, 32, 36, 40, 44])
        
        plt.axhline(y = 13.3, color = 'xkcd:grey', linestyle = '--') 
        plt.axhline(y = 30, color = 'k', linestyle = '--', linewidth=0.5) 
        if i == 0:
            handles, labels = plt.gca().get_legend_handles_labels()
            order = [0,2,1]
            plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
                       loc='lower left',bbox_to_anchor=(0., 0.09, 0.5, 0.5))
        plt.ylabel('Time to cross (days)', fontsize=11, fontweight='bold')
        plt.title(str(monthNames[monthToPlot]), fontsize=14, fontweight='bold')
        plt.savefig(
            '/Users/arielmor/Desktop/SAI/ArcticShipping/figures/SUPP_shortest_time_to_cross_NWP_scatter_' + str(saveMonth[monthToPlot]) + '_all_categories_updated.jpg', 
                                dpi=1200, bbox_inches='tight')
        
        
    #### supplement
    for i in range(12):
        monthToPlot = i
        
        asymmetric_error_ssp = [lower_error_ssp[i,:], upper_error_ssp[i,:]]
        asymmetric_error_arise1 = [lower_error_arise1[i,:], upper_error_arise1[i,:]]
        asymmetric_error_arise = [lower_error_arise[i,:], upper_error_arise[i,:]]
        
        fig = plt.figure(figsize=(8.5,4), dpi=1200)
        plt.errorbar(np.linspace(0,34,35), shortest_time_ssp_by_month[:,monthToPlot], yerr=asymmetric_error_ssp, capsize=2.5, color='r', fmt = 'o', label='SSP2-4.5')#, ms=37)
        plt.errorbar(np.linspace(0,34,35), shortest_time_arise1_by_month[:,monthToPlot], yerr=asymmetric_error_arise1, capsize=2.5, color='purple', fmt='s', label='ARISE-1.0')#, ms=35)
        plt.errorbar(np.linspace(0,34,35), shortest_time_arise_by_month[:,monthToPlot], yerr=asymmetric_error_arise, capsize=2.5, color=(0.02,0.65,0.9), fmt = '^', label='ARISE-1.5', ms=6)
        plt.xticks([0, 4, 9, 14, 19, 24, 29, 34], 
                    ['2035', '2040', '2045', '2050', '2055', '2060', '2065', '2069'], 
                    fontsize = 12)
        plt.ylabel('Time to cross (days)', fontsize=11, fontweight='bold')
        plt.xlim([-1,35])
        plt.ylim((10,75))
        plt.yticks([12, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
                ['No\ntrips', 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75])
        
        plt.axhline(y = 14, color = 'xkcd:grey', linestyle = '--') 
        plt.axhline(y = 30, color = 'k', linestyle = '--', linewidth=0.5) 
        if i == 0:
            handles, labels = plt.gca().get_legend_handles_labels()
            order = [0,2,1]
            plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
                        loc='upper left',bbox_to_anchor=(0.06, 0.5, 0.5, 0.5), fancybox=True, fontsize=12) #bbox_to_anchor=(0., 0.09, 0.5, 0.5))
        plt.title(str(monthNames[monthToPlot]), fontsize=14, fontweight='bold')
        
        plt.savefig(
            '/Users/arielmor/Desktop/SAI/ArcticShipping/figures/SUPP_shortest_time_to_cross_NSR_scatter_' + str(saveMonth[monthToPlot]) + '_all_categories_with_range.jpg', 
                                dpi=1200, bbox_inches='tight')
        
        
        #### scatterplot NWP
        asymmetric_error_ssp_NWP = [lower_error_NWP_ssp[i,:], upper_error_NWP_ssp[i,:]]
        asymmetric_error_arise1_NWP = [lower_error_NWP_arise1[i,:], upper_error_NWP_arise1[i,:]]
        asymmetric_error_arise_NWP = [lower_error_NWP_arise[i,:], upper_error_NWP_arise[i,:]]
        
        fig = plt.figure(figsize=(8.5,4), dpi=1200)
        plt.errorbar(np.linspace(0,34,35), shortest_time_ssp_by_month_NWP[:,monthToPlot], yerr=asymmetric_error_ssp_NWP, capsize=2.5, color='r', fmt = 'o', label='SSP2-4.5')#, ms=37)
        plt.errorbar(np.linspace(0,34,35), shortest_time_arise1_by_month_NWP[:,monthToPlot], yerr=asymmetric_error_arise1_NWP, capsize=2.5, color='purple', fmt='s', label='ARISE-1.0')#, ms=35)
        plt.errorbar(np.linspace(0,34,35), shortest_time_arise_by_month_NWP[:,monthToPlot], yerr=asymmetric_error_arise_NWP, capsize=2.5, color=(0.02,0.65,0.9), fmt = '^', label='ARISE-1.5', ms=6)
        plt.xticks([0, 4, 9, 14, 19, 24, 29, 34], 
                    ['2035', '2040', '2045', '2050', '2055', '2060', '2065', '2069'], 
                    fontsize = 12)
        plt.ylabel('Time to cross (days)', fontsize=11, fontweight='bold')
        
        plt.xlim([-1,35])
        plt.ylim((10,75))
        plt.yticks([12, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
                ['No\ntrips', 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75])
        
        plt.axhline(y = 14, color = 'xkcd:grey', linestyle = '--') 
        plt.axhline(y = 30, color = 'k', linestyle = '--', linewidth=0.5) 
        if i == 0:
            handles, labels = plt.gca().get_legend_handles_labels()
            order = [0,2,1]
            plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
                       loc='upper left',bbox_to_anchor=(0, 0.5, 0.5, 0.5), fancybox=True, fontsize=12) #bbox_to_anchor=(0., 0.09, 0.5, 0.5))
        plt.title(str(monthNames[monthToPlot]), fontsize=14, fontweight='bold')
        plt.savefig(
            '/Users/arielmor/Desktop/SAI/ArcticShipping/figures/SUPP_shortest_time_to_cross_NWP_scatter_' + str(saveMonth[monthToPlot]) + '_all_categories_with_range.jpg', 
                                dpi=1200, bbox_inches='tight')
     
        
    #### lat/lon path
    ## if there's more than one day in shortest_time:
    lonPath = {}
    latPath = {}
    for iday in range(len(shortest_time2[1])):
        for istep in range(len(shortest_time2[1])):
            lonPath[iday,istep] = lonIce[shortest_time2[1][istep][1] + lonOffset]
            latPath[iday,istep] = latIce[shortest_time2[1][istep][0] + latOffset]
            
            
    #### heat map for all trips
    import collections
    ## SSP
    totalCountsYear = np.zeros((43, 288))
    for iyear in range(35):
        totalCounts = np.zeros((43, 288))
        for imonth in range(len(latPath_ssp[iyear])):  
            coordinates = list(zip(latPath_ssp[iyear][imonth], lonPath_ssp[iyear][imonth]))
            count_dict = collections.Counter(coordinates)
    
            counts = np.zeros((43, 288))
            for ilat, ilon in np.ndindex(counts.shape):
                key = (latIce.values[ilat], lonIce.values[ilon])
                if key in count_dict:
                    counts[ilat, ilon] = count_dict[key]
    
            totalCounts += counts
        totalCountsYear += totalCounts
    print("SSP total trips from Rotterdam: ", np.max(totalCountsYear))
    del counts, totalCounts 
    
    totalCountsYearNWP = np.zeros((43, 288))
    for iyear in range(35):
        totalCountsNWP = np.zeros((43, 288))
        for imonth in range(len(latPath_NWP_ssp[iyear])):  
            coordinates = list(zip(latPath_NWP_ssp[iyear][imonth], lonPath_NWP_ssp[iyear][imonth]))
            count_dict = collections.Counter(coordinates)
    
            counts = np.zeros((43, 288))
            for ilat, ilon in np.ndindex(counts.shape):
                key = (latIce.values[ilat], lonIce.values[ilon])
                if key in count_dict:
                    counts[ilat, ilon] = count_dict[key]
    
            totalCountsNWP += counts
        totalCountsYearNWP += totalCountsNWP
    print("SSP total trips from Blanc-Sablon: ", np.max(totalCountsYearNWP))
    
    del counts, totalCountsNWP
        
    totalCounts = totalCountsYear + totalCountsYearNWP
    
    fig = plt.figure(figsize=(10,8), dpi=1200)        
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(central_longitude=180,central_latitude=40))
    ax.coastlines()
    ax.set_facecolor('k')
    ax.coastlines(linewidth=0.7);
    land_50m = cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='k',
                                    facecolor='0.7', linewidth=0.4)
    ax.add_feature(land_50m)    
    for i in range(len(lonPath_ssp)):
        for j in range(len(lonPath_ssp[i])):
            ax.plot(lonPath_ssp[i][j], latPath_ssp[i][j], transform=ccrs.PlateCarree(), alpha = 0.4, 
                      color='xkcd:bright red', linewidth=1.5)
    for i in range(len(lonPath_NWP_ssp)):
        for j in range(len(lonPath_NWP_ssp[i])):
            plt.plot(lonPath_NWP_ssp[i][j], latPath_NWP_ssp[i][j], transform=ccrs.PlateCarree(), alpha = 0.4, 
                      color='xkcd:bright red', linewidth=1.5)
    plt.title("a) SSP2-4.5", fontsize=13, fontweight='bold')
    plt.savefig('/Users/arielmor/Desktop/SAI/ArcticShipping/figures/poster_shortest_shipping_routes_all_categories_ssp_updated.jpg', 
                dpi = 1800, bbox_inches = 'tight')
    del totalCounts, totalCountsYear 
    
    
    ## ARISE-1.0
    totalCountsYear = np.zeros((43, 288))
    for iyear in range(35):
        totalCounts = np.zeros((43, 288))
        for imonth in range(len(latPath_arise1[iyear])):  
            coordinates = list(zip(latPath_arise1[iyear][imonth], lonPath_arise1[iyear][imonth]))
            count_dict = collections.Counter(coordinates)
    
            counts = np.zeros((43, 288))
            for ilat, ilon in np.ndindex(counts.shape):
                key = (latIce.values[ilat], lonIce.values[ilon])
                if key in count_dict:
                    counts[ilat, ilon] = count_dict[key]
    
            totalCounts += counts
        totalCountsYear += totalCounts
    print("ARISE-1.0 total trips from Rotterdam: ", np.max(totalCountsYear))
    del counts, totalCounts, count_dict
    
    totalCountsYearNWP = np.zeros((43, 288))
    for iyear in range(35):
        totalCountsNWP = np.zeros((43, 288))
        for imonth in range(len(latPath_NWP_arise1[iyear])):  
            coordinates = list(zip(latPath_NWP_arise1[iyear][imonth], lonPath_NWP_arise1[iyear][imonth]))
            count_dict = collections.Counter(coordinates)
    
            counts = np.zeros((43, 288))
            for ilat, ilon in np.ndindex(counts.shape):
                key = (latIce.values[ilat], lonIce.values[ilon])
                if key in count_dict:
                    counts[ilat, ilon] = count_dict[key]
    
            totalCountsNWP += counts
        totalCountsYearNWP += totalCountsNWP
        
    print("ARISE-1.0 total trips from Blanc-Sablon: ", np.max(totalCountsYearNWP))
    totalCounts = totalCountsYear + totalCountsYearNWP
    
        
    fig = plt.figure(figsize=(10,8), dpi=1200)        
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
    ax.set_extent([0, 360, 63, 90], crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_facecolor('k')
    ax.coastlines(linewidth=0.7);
    land_50m = cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='k',
                                    facecolor='0.7', linewidth=0.4)
    ax.add_feature(land_50m)    
    
    for i in range(len(lonPath_arise1)):
        for j in range(len(lonPath_arise1[i])):
            plt.plot(lonPath_arise1[i][j], latPath_arise1[i][j], transform=ccrs.PlateCarree(), alpha = 0.4, 
                      color='xkcd:bright purple', linewidth=1.5)
    for i in range(len(lonPath_NWP_arise1)):
        for j in range(len(lonPath_NWP_arise1[i])):
            plt.plot(lonPath_NWP_arise1[i][j], latPath_NWP_arise1[i][j], transform=ccrs.PlateCarree(), alpha = 0.4, 
                      color='xkcd:bright purple', linewidth=1.5)
    
    plt.title("c) ARISE-1.0", fontsize=13, fontweight='bold')
    plt.savefig('/Users/arielmor/Desktop/SAI/ArcticShipping/figures/poster_shortest_shipping_routes_all_categories_arise1_updated.jpg', 
                dpi = 1800, bbox_inches = 'tight')
    del counts, totalCounts, totalCountsNWP, totalCountsYear 
    
    
    ## ARISE-1.5 
    totalCountsYear = np.zeros((43, 288))
    for iyear in range(35):
        totalCounts = np.zeros((43, 288))
        for imonth in range(len(latPath_arise[iyear])):  
            coordinates = list(zip(latPath_arise[iyear][imonth], lonPath_arise[iyear][imonth]))
            count_dict = collections.Counter(coordinates)
    
            counts = np.zeros((43, 288))
            for ilat, ilon in np.ndindex(counts.shape):
                key = (latIce.values[ilat], lonIce.values[ilon])
                if key in count_dict:
                    counts[ilat, ilon] = count_dict[key]
    
            totalCounts += counts
        totalCountsYear += totalCounts
    print("ARISE-1.5 total trips from Rotterdam: ", np.max(totalCountsYear))
    del counts, totalCounts 
    
    totalCountsYearNWP = np.zeros((43, 288))
    for iyear in range(35):
        totalCountsNWP = np.zeros((43, 288))
        for imonth in range(len(latPath_NWP_arise[iyear])):  
            coordinates = list(zip(latPath_NWP_arise[iyear][imonth], lonPath_NWP_arise[iyear][imonth]))
            count_dict = collections.Counter(coordinates)
    
            counts = np.zeros((43, 288))
            for ilat, ilon in np.ndindex(counts.shape):
                key = (latIce.values[ilat], lonIce.values[ilon])
                if key in count_dict:
                    counts[ilat, ilon] = count_dict[key]
    
            totalCountsNWP += counts
        totalCountsYearNWP += totalCountsNWP
        
    print("ARISE-1.5 total trips from Blanc-Sablon: ", np.max(totalCountsYearNWP))
    totalCounts = totalCountsYear + totalCountsYearNWP
    
        
    fig = plt.figure(figsize=(10,8), dpi=1200)        
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
    ax.set_extent([0, 360, 63, 90], crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_facecolor('k')
    ax.coastlines(linewidth=0.7);
    land_50m = cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='k',
                                    facecolor='0.7', linewidth=0.4)
    ax.add_feature(land_50m)    
    ## field to be plotted
    # cf1 = ax.pcolormesh(lonIce,latIce, totalCounts, vmin=0, vmax=150, 
    #                     transform=ccrs.PlateCarree(), cmap='cmr.freeze')
    for i in range(len(lonPath_arise)):
        for j in range(len(lonPath_arise[i])):
            plt.plot(lonPath_arise[i][j], latPath_arise[i][j], transform=ccrs.PlateCarree(), alpha = 0.4, 
                      color="xkcd:azure", linewidth=1.5) 
    for i in range(len(lonPath_NWP_arise)):
        for j in range(len(lonPath_NWP_arise[i])):
            plt.plot(lonPath_NWP_arise[i][j], latPath_NWP_arise[i][j], transform=ccrs.PlateCarree(), alpha = 0.4, 
                      color="xkcd:azure", linewidth=1.5)
    plt.title("b) ARISE-1.5", fontsize=13, fontweight='bold')
    plt.savefig('/Users/arielmor/Desktop/SAI/ArcticShipping/figures/poster_shortest_shipping_routes_all_categories_arise_updated.jpg', 
                dpi = 1800, bbox_inches = 'tight') 

        
    return