"""SF_plum_finder receives an address or intersection (i.e. 16th and Mission)
and returns the nearest plum tree to that location"""

import json
import configparser
import os
from time import time
import argparse
import csv
import pandas as pd
import numpy as np
from warnings import warn
from SF_plum_finder.API_handling import get_geolocation, create_client
from init_config import init_config_file
data_dir = os.path.join(os.path.dirname(__file__), 'data')


def load_config():
    configparse = configparser.ConfigParser()
    try:
        configparse.read('config.ini')
        return configparse
    except FileNotFoundError:
        print('Config file not found, creating one')
        init_config_file()
        return None
    except configparser.Error as e:
        print(f'Error with config file: {e}')
        return None


# load config file and set config vars
config = load_config()
_n = int(config['Settings']['n'])
performance_log = config['Settings'].getboolean('performancelog')
use_SQL = config['Settings'].getboolean('usesql')

_test_address = '1468 Valencia St'
_test_address = None


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


def create_performance_log():
    """Creates the header row of the performance log and saves the file in the working directory"""
    header = ['InputAddress', 'InputLatitude', 'InputLongitude',
              'ApproxTreeID', 'ApproxAddress', 'ApproxLatitude', 'ApproxLongitude',
              'ActualTreeID', 'ActualAddress', 'ActualLatitude', 'ActualLongitude',
              'DistanceDifference', 'SQLUsed', 'TotalTime']
    try:
        with open('performance_log.csv', 'x', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header)

    except FileExistsError:
        print('Performance log already exists')


def write_to_log(row: list):
    """Appends the performance log row to the performance log csv file"""
    if not os.path.exists('performance_log.csv'):
        print('No performance log found, creating one')
        create_performance_log()
        print(f'Performance log created at {os.path.abspath("performance_log.csv")}')

    try:
        with open('performance_log.csv', 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(row)
    except FileNotFoundError:
        print('Something has gone wrong with the performance logger')


# TODO Implement geocoding through SF address list
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


def _get_cli_args():
    """Argument parser for CLI"""
    parser = argparse.ArgumentParser(description='Get the location of the closest plum tree!')
    parser.add_argument('-n', type=int, default=_n,
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


def load_SF_streets(street_name_path: str = os.path.join(data_dir, 'Street_Names.json')) -> list | None:
    """Loads a json file containing all street names in SF and returns as a dict.
    Returns False if file cannot be loaded"""

    try:
        fp = open(street_name_path, 'r')
        acceptable_street_names = json.load(fp)
        fp.close()
        return acceptable_street_names
    except FileNotFoundError:
        return None


def check_if_street_in_SF(acceptable_street_names, street: str) -> bool:
    """Returns True if the street name is in the SF street name list"""
    return street.casefold() in (name.casefold() for name in acceptable_street_names)


def _convert_address_to_list(input_address: str | list) -> list:
    """Converts the inputted address to a list if it is in string form. Returns the address as a list"""
    if type(input_address) == str:
        input_address = input_address.split(' ')
    return input_address


def process_address(input_address: str | list, add_city: bool = True) -> str | int:
    """ verify the user has given a real address and process the address into something manageable by
    the Google Maps api.
    Accepts list in format [number, street_name, stree_type] and returns string of full postal address"""
    input_address = _convert_address_to_list(input_address)

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
    address = street_number + ' ' + street

    # adds city suffix by default
    if add_city:
        city_address = 'SAN FRANCISCO, CA'
        address = address + ', ' + city_address

    return address


@timer_func
def get_geocode_from_db(user_address):
    """Gets the users geocode from the SF address database."""
    pass


@timer_func
def get_user_geolocation(gmaps, processed_address: str):
    try:
        latitude, longitude = get_geolocation(gmaps, processed_address)
        return latitude, longitude
    except ValueError:
        # API used incorrectly
        return 494
    except RuntimeError:
        # timed out
        return 592


def approximate_distance(data: pd.DataFrame, latitude: float, longitude: float) -> pd.DataFrame:
    """Adds the geometric distance from a given latitude and longitude to the street_tree_list data frame."""

    # create numpy arrays for latitudes, longitude, and then compute distances from given latitude and longitude
    latitudes = data.loc[:, 'Latitude'].values
    longitudes = data.loc[:, 'Longitude'].values
    geometric_distances = np.sqrt((latitudes - latitude) ** 2 + (longitudes - longitude) ** 2)

    data.insert(7, 'geometric_distance', geometric_distances)

    return data


def verify_closest(client, origin_address, destinations: list, mode: str = 'walking',
                   sort: bool = True, distance_diff: bool = False) -> list | tuple:
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

    original_shortest_distance = distances[0]['distance']

    # sorting, if sort==True
    if sort:
        # sort function
        def by_distance(e): return e['distance']

        distances.sort(key=by_distance)

    distance_difference = original_shortest_distance - distances[0]['distance']

    # returns the difference between the closest tree geometrically and the
    # geographically closest in geographical distance
    if distance_diff:
        return distances, distance_difference

    return distances


def return_tree(closest_tree, input_address):
    # return the closest tree
    address = closest_tree['qAddress'][0]
    species = closest_tree['qSpecies'][0]
    distance = closest_tree['street_distance'][0]

    response = """Closest tree to {}:
Species: {}
Address: {}
Distance: {}m""".format(input_address, species, address, distance)
    return response


def open_last_response(path):
    # opens the last response for testing
    try:
        with open(path, 'r') as file:
            return json.load(file)[0]
    except FileNotFoundError:
        print("Couldn't find file")


def find_closest_plum(user_address: str | list, key: str, n: int):

    # only allow up to 25 trees
    if n > 25:
        print('Too many trees have been requested to query. Setting the number to 25')
        n = 25
    if performance_log:
        start_time = time()

    processed_address = process_address(user_address)

    # if process_address raises an error code, return the code
    if type(processed_address) == int:
        return processed_address

    try:
        gmaps = create_client(key)
    except ValueError:
        # invalid API key used
        return 591

    # user's geographic location
    geocode = get_user_geolocation(gmaps, processed_address)

    # return error if occurs
    if isinstance(geocode, int):
        return geocode

    latitude, longitude = geocode

    # response = open_last_response('test.json')
    # latitude = float(response['geometry']['location']['lat'])
    # longitude = float(response['geometry']['location']['lng'])

    # load tree data
    tree_data_path = os.path.join(data_dir, 'Plum_Street_Tree_List.csv')

    def load_data(path):
        try:
            tree_data = pd.read_csv(path).set_index(['TreeID'])
            return tree_data
        except FileNotFoundError:
            # tree data not found
            return 593

    data = load_data(tree_data_path)

    if type(data) == int:
        return data

    if data.empty:
        # tree data loaded incorrectly
        return 594

    # add distance from inputted address to data frame
    data.drop_duplicates('qAddress', inplace=True)
    data = approximate_distance(data, latitude, longitude)

    # get the n shortest distances and create a dict containing the addresses to send to gmaps
    approx_closest_trees = data.nsmallest(n, 'geometric_distance')

    destinations = approx_closest_trees.qAddress.to_list()
    city_suffix = ', San Francisco, CA'
    len_city_suffix = len(city_suffix)

    for i, address in enumerate(destinations):
        full_address = address + city_suffix
        destinations[i] = full_address

    # query gmaps for sorted distance matrix of possible closest trees
    # distance difference ives difference between approximate closest and actual closest
    try:
        distances, distances_difference = verify_closest(gmaps, processed_address, destinations, distance_diff=True)
    except ValueError:
        # API used incorrectly
        return 494
    except RuntimeError:
        # timed out
        return 592

    closest_address = distances[0]['address'][0:-len_city_suffix]
    closest_tree = data.loc[data.qAddress == closest_address]
    response = closest_tree.to_dict('list')
    response['street_distance'] = [distances[0]['distance']]

    # logging data for performance and usage monitoring
    if performance_log:
        total_time = time() - start_time
        approx_closest_tree = approx_closest_trees.iloc[0, :]
        closest_tree = closest_tree.iloc[0]
        if isinstance(user_address, list):
            user_address = ' '.join(user_address)

        # creates row and appends it to performance_log.csv
        log_row = [user_address, latitude, longitude,
                   approx_closest_tree.name, approx_closest_tree.loc['Latitude'], approx_closest_tree.loc['Longitude'],
                   closest_tree.name, closest_tree.loc['Latitude'], closest_tree.loc['Longitude'],
                   distances_difference, use_SQL, total_time]

        write_to_log(log_row)

    return response


def command_line_runner(text_input=None):
    """Runs the command line interface. Prints out the results from find_closest_plum"""
    if text_input:
        CLI_address = text_input
        CLI_n = _n
    else:
        CLI_address, CLI_n = _get_cli_args()

    CLI_address = _convert_address_to_list(CLI_address)
    closest_plum = find_closest_plum(CLI_address, config['Keys']['GoogleMaps'], n=CLI_n)

    if type(closest_plum) == int:
        print(error_dict[closest_plum])
    else:
        address = ' '.join(CLI_address)
        response = return_tree(closest_plum, address)
        print(response)


if __name__ == '__main__':
    if config:
        if _test_address:
            cheese = _test_address
            command_line_runner(text_input=cheese)
        else:
            command_line_runner()
