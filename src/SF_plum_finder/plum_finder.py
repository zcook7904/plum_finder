"""SF_plum_finder receives an address or intersection (i.e. 16th and Mission)
and returns the nearest plum tree to that location"""

import json
import configparser
import os
from dataclasses import dataclass
from time import time
import argparse
from csv import writer as csv_writer
import pandas as pd
from numpy import sqrt
from SF_plum_finder.API_handling import get_geolocation, create_client
from SF_plum_finder.init_config import init_config_file

data_dir = os.path.join(os.path.dirname(__file__), 'data')


@dataclass
class Address:
    """Dataclass to represent address. str returns the street address as a string and
    len the number of elements in the street address."""
    street_address: str
    city: str = 'San Francisco, CA'
    latitude: float = None
    longitude: float = None

    def __str__(self):
        return self.street_address

    def __len__(self):
        return len(self.get_address_as_list())

    def get_address_as_list(self):
        """Converts the inputted address to a list if it is in string form. Returns the address as a list"""
        address_list = self.street_address.split(' ')
        return address_list

    def get_number(self):
        return int(self.get_address_as_list()[0])

    def get_street_name(self):
        return ' '.join(self.get_address_as_list()[1:])

    def get_full_address(self):
        return f'{self.street_address}, {self.city}'

    def set_number(self, new_number: int):
        street_name = self.get_street_name()
        self.street_address = f'{new_number} {street_name}'

    def set_street_name(self, new_name: str):
        street_number = self.get_number()
        self.street_address = f'{street_number} {new_name}'

    def set_geolocation(self, gmaps):
        """Get the geolocation of the address"""
        try:
            latitude, longitude = get_geolocation(gmaps, self.get_full_address())
            self.latitude = latitude
            self.longitude = longitude
        except ValueError:
            # API used incorrectly
            return 494
        except RuntimeError:
            # timed out
            return 592

    def set_geocode_from_addresses(self, address_df: pd.DataFrame):
        """Gets the users geocode from the SF address database."""
        if self.street_address.upper() in address_df.loc[:, 'Address'].to_list():
            self.latitude = address_df.loc[address_df.Address == self.street_address.upper()].Latitude.iloc[0]
            self.longitude = address_df.loc[address_df.Address == self.street_address.upper()].Longitude.iloc[0]
            return True
        else:
            return None


def load_config():
    configparse = configparser.ConfigParser()
    if not os.path.exists('config.ini'):
        init_config_file()
        path = os.path.abspath('config.ini')
        print(f'Config file not found, creating new one at {path}.')
        print('You will need to add your google maps key to the config file')
        return None

    try:
        configparse.read('config.ini')
        return configparse
    except FileNotFoundError:
        print('Config file not found')
        return None
    except configparser.Error as e:
        print(f'Error with config file: {e}')
        return None


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
    594: 'Tree data not loaded correctly',
    595: 'Config file not found'
}


def create_performance_log():
    """Creates the header row of the performance log and saves the file in the working directory"""
    header = ['InputAddress', 'InputLatitude', 'InputLongitude',
              'ApproxTreeID', 'ApproxAddress', 'ApproxLatitude', 'ApproxLongitude',
              'ActualTreeID', 'ActualAddress', 'ActualLatitude', 'ActualLongitude',
              'DistanceDifference', 'SQLUsed', 'TotalTime']
    try:
        with open('performance_log.csv', 'x', newline='') as csv_file:
            writer = csv_writer(csv_file)
            writer.writerow(header)
            return True

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
            writer = csv_writer(csv_file)
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
    """Argument parser for CLI. Returns address args as a string"""
    parser = argparse.ArgumentParser(description='Get the location of the closest plum tree!')
    parser.add_argument('address', nargs='+',
                        help='a street address located in San Francisco, CA')
    args = parser.parse_args()
    address = ' '.join(args.address)
    return address


def load_addresses(path: str):
    try:
        address_data = pd.read_csv(path)
    except FileNotFoundError:
        # tree data not found
        return 593
    return address_data


def check_address_arg_length(address: Address) -> bool:
    """ Returns false if the input address doesn't have at least 3 components"""
    arg_length = len(address)
    # make sure input is correct length, max 10 is arb...
    if arg_length < 3 or arg_length > 10:
        # Address should contain {number} {street_name} {street_suffix} at minimum
        return False
    return True


def check_street_number(address: Address) -> bool:
    """ Returns True if the street number is valid"""
    # ensure first input is an integer
    try:
        # should throw value error if street number cannot be returned
        address.get_number()
        return True
    except ValueError:
        return False


def format_street_name(address: Address) -> Address:
    """adds '0' to street name if street name is numeric, <10, and doesn't contain '0' (ex 9th st -> 09th st)"""
    street_name = address.get_street_name()
    if street_name[0].isnumeric() and not street_name[1].isnumeric():
        new_street_name = '0' + street_name
        address.set_street_name(new_street_name)
    return address


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


def check_if_street_in_SF(acceptable_street_names, address: Address) -> bool:
    """Returns True if the street name is in the SF street name list"""

    return address.get_street_name().casefold() in (name.casefold() for name in acceptable_street_names)


def process_address(address: Address) -> Address | int:
    """ verify the user has given a real address and process the address into something manageable by
    the Google Maps api.
    Accepts list in format [number, street_name, stree_type] and returns string of full postal address"""

    if not check_address_arg_length(address):
        return 490

    if not check_street_number(address):
        return 491

    # clean up street name
    address = format_street_name(address)

    # load json containing all street names in SF
    acceptable_streets = load_SF_streets()
    if not acceptable_streets:
        # file unable to load
        return 492

    if not check_if_street_in_SF(acceptable_streets, address):
        # street
        return 493

    return address


def load_tree_data(path: str):
    """Loads the Plum_Street_Tree_List.csv file into a pandas dataframe and returns it"""
    try:
        tree_data = pd.read_csv(path).set_index(['TreeID'])
    except FileNotFoundError:
        # tree data not found
        return 593

    if tree_data.empty:
        # tree data loaded incorrectly
        return 594

    # only need one tree per address
    tree_data.drop_duplicates('qAddress', inplace=True)

    return tree_data


def approximate_distance(data: pd.DataFrame, origin_address: Address) -> pd.DataFrame:
    """Adds the geometric distance from a given latitude and longitude to the street_tree_list data frame."""

    # create numpy arrays for latitudes, longitude, and then compute distances from given latitude and longitude
    latitudes = data.loc[:, 'Latitude'].values
    longitudes = data.loc[:, 'Longitude'].values
    geometric_distances = sqrt((latitudes - origin_address.latitude) ** 2
                               + (longitudes - origin_address.longitude) ** 2)

    data.insert(7, 'geometric_distance', geometric_distances)

    return data


def verify_closest(gmaps, origin_address: Address, destinations: list, mode: str = 'walking',
                   sort: bool = True, distance_diff: bool = False) -> list | tuple:
    """verify which tree is actually closest walking-wise from Google Maps api call.
    Accepts Google Maps API key, a single origin, and a list of destinations. Origin and destinations can be in form
    of address or latitude and longitude. See Google Maps api docs.
    Mode are route modes per Google Maps api (walking, driving, transit, biking).
    If sort is true, returned list will be sorted by distance in with the smallest distance first.
    Returns list of dictionary in form {'address': {address}, 'distance': {distance in m}}'"""

    # request google and get response
    response = gmaps.distance_matrix(origin_address.get_full_address(), destinations, mode=mode)

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

    # the difference between the closest tree geometrically and the
    # geographically closest in geographical distance
    distance_difference = original_shortest_distance - distances[0]['distance']

    # if desired, returns a tuple containing the distances and the distance difference
    if distance_diff:
        return distances, distance_difference

    return distances


def open_last_response(path):
    # opens the last response for testing
    try:
        with open(path, 'r') as file:
            return json.load(file)[0]
    except FileNotFoundError:
        print("Couldn't find file")


def find_closest_plum(input_address: str, config):
    # Set config vars
    key = config['Keys']['GoogleMaps']

    n = int(config['Settings']['n'])
    performance_log = config['Settings'].getboolean('performancelog')
    use_SQL = config['Settings'].getboolean('usesql')

    # only allow up to 25 trees
    if n > 25:
        print('Too many trees have been requested to query. Setting the number to 25')
        n = 25
    if performance_log:
        start_time = time()

    user_address = Address(input_address)

    user_address = process_address(user_address)

    # if process_address raises an error code, return the code
    if type(user_address) == int:
        return user_address

    try:
        gmaps = create_client(key)
    except ValueError:
        # invalid API key used
        return 591

    address_data_path = os.path.join(data_dir, 'Addresses.csv')
    addresses = load_addresses(address_data_path)
    geocode_set = user_address.set_geocode_from_addresses(addresses)

    if not geocode_set:
        print('Failed to set geocode from db')
        # sets user's geographic location
        try:
            user_address.set_geolocation(gmaps)

        except ValueError:
            # API used incorrectly
            return 494
        except RuntimeError:
            # timed out
            return 592

    # load tree data
    # may make this async in future
    tree_data_path = os.path.join(data_dir, 'Plum_Street_Tree_List.csv')
    tree_data = load_tree_data(tree_data_path)

    if type(tree_data) == int:
        return tree_data

    # add distance from inputted address to data frame
    tree_data = approximate_distance(tree_data, user_address)

    # get the n shortest distances and create a dict containing the addresses to send to gmaps
    approx_closest_trees = tree_data.nsmallest(n, 'geometric_distance')

    destinations = approx_closest_trees.qAddress.to_list()
    city_suffix = user_address.city
    len_city_suffix = len(city_suffix)

    for i, address in enumerate(destinations):
        full_address = address + city_suffix
        destinations[i] = full_address

    # query gmaps for sorted distance matrix of possible closest trees
    # distance difference gives difference between approximate closest and actual closest
    try:
        distances, distances_difference = verify_closest(gmaps, user_address, destinations, distance_diff=True)
    except ValueError:
        # API used incorrectly
        return 494
    except RuntimeError:
        # timed out
        return 592

    # get address from sorted list
    closest_address = distances[0]['address'][0:-len_city_suffix]
    closest_tree = tree_data.loc[tree_data.qAddress == closest_address]
    response = closest_tree.to_dict('list')

    # adding the walking distance
    response['street_distance'] = [distances[0]['distance']]

    # logging data for performance and usage monitoring
    if performance_log:
        total_time = time() - start_time
        approx_closest_tree = approx_closest_trees.iloc[0]
        closest_tree = closest_tree.iloc[0]

        # creates row and appends it to performance_log.csv
        log_row = [user_address.street_address, user_address.latitude, user_address.longitude,
                   approx_closest_tree.name, approx_closest_tree.loc['Latitude'], approx_closest_tree.loc['Longitude'],
                   closest_tree.name, closest_tree.loc['Latitude'], closest_tree.loc['Longitude'],
                   distances_difference, use_SQL, total_time]

        write_to_log(log_row)

    return response


def command_line_runner(config, text_input=None):
    """Runs the command line interface. Prints out the results from find_closest_plum"""
    if text_input:
        CLI_address = text_input
    else:
        CLI_address = _get_cli_args()

    closest_plum = find_closest_plum(CLI_address, config)

    if type(closest_plum) == int:
        print(error_dict[closest_plum])
    else:
        response = return_tree(closest_plum, CLI_address)
        print(response)


def return_tree(closest_tree, input_address: str):
    # return the closest tree
    address = closest_tree['qAddress'][0]
    species = closest_tree['qSpecies'][0]
    distance = closest_tree['street_distance'][0]

    response = """Closest tree to {}:
Species: {}
Address: {}
Distance: {}m""".format(input_address, species, address, distance)
    return response


if __name__ == '__main__':
    plum_finder_config = load_config()
    if plum_finder_config:
        command_line_runner(plum_finder_config)
    else:
        print(error_dict[plum_finder_config])
