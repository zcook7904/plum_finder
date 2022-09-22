"""SF_plum_finder receives an address or intersection (i.e. 16th and Mission)
and returns the nearest plum tree to that location"""

import json
import os
from time import time
import argparse
import pandas as pd
from numpy import nan
from warnings import warn
from math import sqrt
from SF_plum_finder.API_handling import get_key, get_geolocation, create_client
data_dir = os.path.join(os.path.dirname(__file__), 'data')


# TODO get data to work in package

error_dict = {
    490: 'Address should hold at least 3 components',
    491: 'Invalid street number - should be integer',
    492: 'SF Street Name List was unable to load',
    493: 'The street entered is not valid in San Francisco',
    494: 'Google Maps geolocation API not used correctly',
    495: 'Google Maps direction matrix API not used correctly',

    590: 'API Key file not found',
    591: 'Invalid Google Maps API Key',
    592: 'Google Maps timed out',
    593: 'Tree data file not found',
    594: 'Tree data not loaded correctly'

}


def timer_func(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2 - t1):.4f}s')
        return result

    return wrap_func


def get_cli_args():
    """Argument parser for CLI"""
    parser = argparse.ArgumentParser(description='Get the location of the closest plum tree!')
    parser.add_argument('-n', type=int, default=5,
                        help='the number of potential closest trees to be sent to googlemaps. Max allowed: 25')
    parser.add_argument('address', nargs='+',
                        help='a street address located in San Francisco, CA')
    args = parser.parse_args()
    return args.address, args.n


def check_address_arg_length(input_address) -> bool:
    """ Returns false if the input address doesn't have at least 3 components"""
    arg_length = len(input_address)
    # make sure input is correct length
    if arg_length < 3 or arg_length > 10:
        # Address should contain {number} {street_name} {street_suffix} at minimum
        return False
    return True


def check_street_number(street_number) -> bool:
    """ Returns True if the street number is valid"""
    # ensure first input is an integer
    try:
        int(street_number)
        return True
    except ValueError:
        return False


def format_street_name(street_name: str) -> str:
    """adds '0' to street name if street name is numeric, <10, and doesn't contain '0' (ex 9th st -> 09th st)"""
    if street_name[0].isnumeric() and not street_name[1].isnumeric():
        street_name = '0' + street_name
    return street_name


def load_SF_streets(street_name_path: str = os.path.join(data_dir, 'Street_Names.json')):
    """Loads a json file containing all street names in SF and returns as a dict.
    Returns False if file cannot be loaded"""
    print(os.getcwd())

    try:
        fp = open(street_name_path, 'r')
        acceptable_street_names = json.load(fp)
        fp.close()
        return acceptable_street_names
    except FileNotFoundError:
        return False


def check_if_street_in_SF(acceptable_street_names, street: str) -> bool:
    """Returns True if the street name is in the SF street name list"""
    return street.casefold() in (name.casefold() for name in acceptable_street_names)


def process_address(input_address):
    """ verify the user has given a real address and process the address into something manageable by
    the Google Maps api.
    Accepts list in format [number, street_name, stree_type] and returns string of full postal address"""
    if not check_address_arg_length(input_address):
        return 490

    street_number = input_address[0]
    if not check_street_number(street_number):
        return 491

    # clean up street name
    street_name = input_address[1:-1]
    street_name = ' '.join(street_name)
    street_name = format_street_name(street_name)

    # join name and type make street
    street_type = input_address[-1]
    street = ' '.join([street_name, street_type])

    # load json containing all street names in SF
    acceptable_streets = load_SF_streets()
    if not acceptable_streets:
        # file unable to load
        return 492

    if not check_if_street_in_SF(acceptable_streets, street):
        # street
        return 493

    # join everything together and return
    city_address = 'SAN FRANCISCO, CA'
    address = street_number + ' ' + street + ', ' + city_address
    return address


def approximate_distance(data: pd.DataFrame, latitude: float, longitude: float) -> pd.DataFrame:
    """Adds the geometric distance from a given latitude and longitude to the street_tree_list data frame."""

    # function to map geometric distance
    def geometric_distance_map(row):
        row.geometric_distance = sqrt((row.Longitude - geometric_distance_map.longitude) ** 2
                                      + (row.Latitude - geometric_distance_map.latitude) ** 2)
        return row

    assigned_data = data.assign(geometric_distance=nan)
    del data
    geometric_distance_map.longitude = longitude
    geometric_distance_map.latitude = latitude

    mapped_data = assigned_data.apply(geometric_distance_map, axis='columns')

    return mapped_data


def verify_closest(client, origin_address, destinations: list, mode: str = 'walking', sort: bool = True) -> list:
    """verify which tree is actually closest walking-wise from Google Maps api call.
    Accepts Google Maps API key, a single origin, and a list of destinations. Origin and destinations can be in form
    of address or latitude and longitude. See Google Maps api docs.
    Mode are route modes per Google Maps api (walking, driving, transit, biking).
    If sort is true, returned list will be sorted by distance in with the smallest distance first.
    Returns list of dictionary in form {'address': {address}, 'distance': {distance in m}}'"""

    # ensure only one origin is passed to gmaps
    if type(origin_address) == list:
        origin_address = list[0]
        warn('This function typically only accepts one origin. '
             'Response will only be distances from first origin in list')

    # request google and get response
    gmaps = client
    response = gmaps.distance_matrix(origin_address, destinations, mode=mode)

    # create list of distances
    distances = []
    for i, tree in enumerate(response['rows'][0]['elements']):
        distances.append({'address': destinations[i], 'distance': tree['distance']['value']})

    # sorting, if sort==True
    if sort:
        # sort function
        def by_distance(e): return e['distance']

        distances.sort(key=by_distance)

    return distances


def return_tree(closest_tree):
    # return the closest tree
    address = closest_tree.qAddress.iloc[0]
    species = closest_tree.qSpecies.iloc[0]
    distance = closest_tree.street_distance

    response = """The closest tree:
Species: {}
Address: {}
Distance: {}m""".format(species, address, distance)
    return response


def open_last_response(path):
    # opens the last response for testing
    try:
        with open(path, 'r') as file:
            return json.load(file)[0]
    except FileNotFoundError:
        print("Couldn't find file")


def find_closest_plum(user_address, n=10):

    # only allow up to 25 trees
    if n > 25:
        print('Too many trees have been requested to query. Setting the number to 25')
        n = 25

    start_time = time()
    processed_address = process_address(user_address)

    # if process_address raises an error code, return the code
    if type(processed_address) == int:
        return processed_address

    key_path = 'keys.txt'
    try:
        keys = get_key(key_path)
        gmaps = create_client(keys['GoogleAPIkey'])
    except FileNotFoundError:
        # Keys file not found
        return 590
    except ValueError:
        # invalid API key used
        return 591

    # user's geographic location
    try:
        latitude, longitude = get_geolocation(gmaps, processed_address)
    except ValueError:
        # API used incorrectly
        return 494
    except RuntimeError:
        # timed out
        return 592

    # response = open_last_response('test.json')
    # latitude = float(response['geometry']['location']['lat'])
    # longitude = float(response['geometry']['location']['lng'])

    # load tree data
    tree_data_path = os.path.join(data_dir, 'Plum_Street_Tree_List.csv')

    try:
        data = pd.read_csv(tree_data_path).set_index(['TreeID'])
    except FileNotFoundError:
        # tree data not found
        return 593

    if data.empty:
        # tree data loaded incorrectly
        return 594

    # add distance from inputted address to data frame
    data.drop_duplicates('qAddress', inplace=True)
    data = approximate_distance(data, latitude, longitude)

    # get the n shortest distances and create a dict containing the addresses to send to gmaps
    closest_trees = data.nsmallest(n, 'geometric_distance')

    destinations = closest_trees.qAddress.to_list()
    city_suffix = ', San Francisco, CA'
    len_city_suffix = len(city_suffix)

    for i, address in enumerate(destinations):
        full_address = address + city_suffix
        destinations[i] = full_address

    try:
        distances = verify_closest(gmaps, processed_address, destinations)
    except ValueError:
        # API used incorrectly
        return 494
    except RuntimeError:
        # timed out
        return 592

    closest_address = distances[0]['address'][0:-len_city_suffix]
    closest_tree = data.loc[data.qAddress == closest_address]
    closest_tree.street_distance = distances[0]['distance']
    response = return_tree(closest_tree)
    time_taken = time() - start_time
    return response


def command_line_runner():
    CLI_address, CLI_n = get_cli_args()
    closest_plum = find_closest_plum(CLI_address, n=CLI_n)

    if type(closest_plum) == int:
        print(error_dict[closest_plum])
    else:
        print(closest_plum)


if __name__ == '__main__':
    command_line_runner()
