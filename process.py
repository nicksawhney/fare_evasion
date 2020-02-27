import pandas as pd 
import geopandas as gpd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

import shapely

'''
Author: Nick Sawhney 

This is a set of utility functions I wrote during the process of exploring, analyzing, and visualizing Fare Evasion Arrests in New York City

Documentation will become more detailed as the project expands
'''


def load_tos_arrests(hist_path='data/NYPD_Arrests_Data__Historic_.csv', 
                     ytd_path='data/NYPD_Arrest_Data__Year_to_Date_.csv',
                     save=False):

    historic = gpd_from_csv(hist_path)
    ytd = gpd_from_csv(ytd_path)

    all_arrests = pd.concat([historic, ytd], ignore_index=True)

    all_arrests['ARREST_DATE'] = pd.to_datetime(all_arrests['ARREST_DATE'])

    theft_of_services = all_arrests.groupby('LAW_CODE').get_group('PL 1651503')

    if save:
        theft_of_services.to_csv(save)

    return theft_of_services



def load_all_data(station_file='data/stations.csv', 
                  evasions_file='data/evasions.csv',
                  acs_file='data/2013_2017_acs.dta',
                  tract_file = 'data/nyu_2451_34505',
                  lines_file='data/Subway Lines.geojson',
                  city_shapefile = 'data/new-york-city-boroughs.geojson',
                  include_arrests=True, 
                  start_date=pd.datetime(2010, 1, 1),
                  end_date=pd.datetime(2030, 1, 1)):

    '''
        Loads datasets and conforms to uniform schemas when necessary
        Important for quick merging and exploring down the line

        returns: stations, evasions, census, lines, nyc
    '''

    # Load the station data and type the buffer column properly
    stations = gpd_from_csv(station_file, long='stop_lon', lat='stop_lat')

    if 'buff' in stations.columns:
        stations['buff'] = stations['buff'].apply(shapely.wkt.loads)

    # Load the evasion data, ensure proper columns
    evasions = gpd_from_csv(evasions_file)
    evasions['ARREST_DATE'] = pd.to_datetime(evasions['ARREST_DATE'])

    # populate with fare evasion arrests if needed
    if include_arrests:
        stations = populate_arrests(stations, evasions, start_date=start_date, end_date=end_date)

    # Get shape files!
    lines = gpd.read_file(lines_file)
    nyc = gpd.read_file(city_shapefile)

    # Load demographic and census tract information
    acs_data = pd.read_stata(acs_file)
    census_tracts = gpd.read_file(tract_file)


    # Merge itno one census data set
    census = gpd.GeoDataFrame(acs_data.merge(
        census_tracts[['tractid', 'nta_name', 'namelsad', 'geometry']],
        left_on='fips',
        right_on='tractid'))


    # set the crs in the census data and then reproject to Latitude Longitude
    census.crs = dict(init='epsg:2263')
    census = census.to_crs({'init':'epsg:4326'})

    return stations, evasions, census, lines, nyc


def gpd_from_csv(filename, crs={'init':'epsg:4326'}, long='Longitude', lat='Latitude'):
    '''
        Helper function for load_all_data
        Reads and returns a GeoDataFrame with geometry from "lat/long" columns (really can be any xy coordinates)
    '''

    file_pd = pd.read_csv(filename)
    
    return gpd.GeoDataFrame(
        file_pd,
        geometry=gpd.points_from_xy(
            file_pd[long], file_pd[lat]
        ),
        crs=crs
    )


def populate_arrests(stations, 
                     evasions, 
                     start_date=pd.datetime(2010, 1, 1), 
                     end_date=pd.datetime(2030, 1, 1), 
                     arr_column='arrests'):
    '''
        Populate the subway station DataFrame with the number of arrests at each station according to the evasions DataFrame
        over a specified time period.
    '''

    ev = evasions[evasions['ARREST_DATE'] >= start_date]
    ev = ev[evasions['ARREST_DATE'] <= end_date]
    
    st = stations.copy()

    if arr_column in stations.columns:
        st = st.drop(arr_column, axis=1)

    if 'complex_id' in st.columns:
        st = st.set_index('complex_id')

    st[arr_column] = ev.groupby('station_id').count()['ARREST_DATE']

    st[arr_column] = st[arr_column].fillna(value=0)
    
    return st
    

def plot_num_arrests_station(stations, 
                             lines, 
                             nyc,
                             census_data,
                             figsize=(25, 25),
                             title='Number of Fare Evasion Arrests Per Station',
                             arr_col='arrests',
                             plot_race=True,
                             mult=2,
                             save=True):

    fig, ax = plt.subplots(figsize=figsize)

    '''
        Makes an awesome map with all the available data!
    '''

    ax.set_ylim((40.55, 40.95))
    ax.set_xlim((-74.05, -73.70))
    
    if plot_race:
        
        divider = make_axes_locatable(ax)

        cax = divider.append_axes("top", size="5%", pad=0.1, )
        
        cd = census_data \
            .dissolve(by='nta_name', aggfunc='mean')\
            .plot(column='percentnonwhite', 
                  ax=ax,
#                   legend=True, 
                  alpha=0.5,
                  cax=cax,
                  legend_kwds={'label': "Percent Nonwhite",
                               'orientation': "horizontal"},
                  cmap='BuPu')
        cd.axis('off')
        
        
        
    else:
        nyc.plot(ax=ax, alpha=0.2, color='purple')

    stations.plot(markersize=stations[arr_col]*mult, ax=ax, color='black')
    lines.plot(ax=ax, color='black')
    
    

    plt.axis('off')

    plt.title(title, size=30)
#     ax.legend(prop={'size':20})

    if save:
        plt.savefig(f'figs/{title}.png')

    plt.show()


def get_intersecting_fips(station_buffer_row, census):
    '''
        Gets all census tracts within station_buffer_row
        Applies a weight to each census tract based on the proportion of the buffer 
        that is within the census tract

        Used in get_demo to provide weighted demographic information for each station.

        station_buffer: a list of station buffers to use for areas
        census: census tracts and their corresponding fips
    '''
    
    intersections = []
    
    st_buff = station_buffer_row['buff']
    
    st_area = st_buff.area
    
    if 'fips' in census.columns:
        census = census.set_index('fips')
        
        # make sure the index is fips
        
    for fips, tract in census.iterrows():
        census_geom = tract.geometry
        
        
        if st_buff.intersects(census_geom):
            # if the station buffer intersects with the census tract,
            # ad d the fips to the intersections and calculate its geometry
            # might as well add the weight too, if we can calculate it in the same function
            
            un = st_buff.intersection(census_geom)
            weight = un.area / st_area
            
            intersections.append((fips, weight)) 
    
    return intersections


def get_demo(station_row, census, indicator):
    '''
        Get the weighted demographic information from the `indicator` column of `census` 
        for station_row

        station.apply(get_demo, axis=1, args=(census, 'percentnonwhite'))
    '''

    if 'fips' in census.columns:
        census = census.set_index('fips')
        
    int_fips = station_row['int_fips']
    
    ind_val = 0
    
    for fips, weight in int_fips:
        ind_val += census[indicator].loc[fips] * weight

    return ind_val

def split_arrest_data(evasions, cutoff=pd.datetime(2018, 2, 2), include_pre=False):
    '''
        Split the arrests into two DataFrames, one before and one after `cutoff`

        Useful for finding stats before and after a specific cutoff
    '''

    if include_pre:    
        pre_cutoff = evasions[evasions['ARREST_DATE'] <= cutoff]
        post_cutoff = evasions[evasions['ARREST_DATE'] > cutoff]

    else:
        pre_cutoff = evasions[evasions['ARREST_DATE'] < cutoff]
        post_cutoff = evasions[evasions['ARREST_DATE'] >= cutoff]


    return pre_cutoff, post_cutoff


def get_buffer(stations, feet):
    '''
        Creates a buffer of size feet (in feet) around each subway station, returns the 
        list of buffers (with the same index as the station)
    '''
    crs = stations.crs

    return stations.to_crs(epsg=2263).buffer(feet).to_crs(crs)



def get_station_arrests(evasions, stations, subway_radius):
    '''
        Find the station in which each arrest occured, provided the location of a station,
        the location of an arrest, and a buffer around the station. 

        #TODO: remove spaghetti
        #TODO: do something with no_station currently unused
        #This is basically a jupyer notebook thrown into some functions.
    '''
    station_arrests = {}
    no_station = []
    count = 0
    num_evasions = len(evasions)
    
    if not 'complex_id' in stations.columns:
        stations = stations.reset_index()

    def get_nearest_station(row):
        '''
            Find the closest station to the arrest in the row

            #TODO: This should not be a nested function.
        '''
        nonlocal count, num_evasions, station_arrests

        print(f'{round(count/num_evasions*100, ndigits=2)}% complete', end='\r')
        
        geom = row.geometry
        
        in_stations = []
        
        for idx, buffer in enumerate(subway_radius):
            if buffer.contains(geom):
                in_stations.append(idx)

        if len(in_stations) == 0:
            no_station.append(row.ARREST_KEY)
        
        elif len(in_stations) == 1:
            station_arrests[row.ARREST_KEY] = {
                'name' : stations['complex_nm'].iloc[in_stations[0]],
                'id' : stations['complex_id'].iloc[in_stations[0]]
            }
            
        elif len(in_stations) > 1:
    #         print(in_stations)
            pts = np.array(
                [geom.distance(stations.loc[station]['geometry']) for station in in_stations]
            )
    #         print(pts)
            nearest_idx = in_stations[pts.argmin()]
            
    #         print(nearest_idx)
            nm = stations.loc[nearest_idx]['complex_nm']

            i_d = stations.loc[nearest_idx]['complex_id']
        
            station_arrests[row.ARREST_KEY] = {
                'name' : nm,
                'id' : i_d
            }
                  
        else:
            print('something broke')
            return
        
        count += 1
        
        return in_stations

    evasions.apply(get_nearest_station, axis=1)

    arr_id_stat = pd.DataFrame(station_arrests).T

    evasion_arrests = evasions.set_index('ARREST_KEY')

    evasion_arrests[['station_name', 'station_id']] = arr_id_stat

    return evasion_arrests





