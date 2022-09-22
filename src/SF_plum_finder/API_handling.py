import json
import os
import googlemaps

# creating a class that holds all API keys...


def get_key(path: str):
    """ accepts the path as a string and returns a dictionary containing all of the keys for the program
    in the format: {"CompanyAPIkey": "key_as_string"}"""

    try:
        with open(path, 'r') as key_file:
            return json.load(key_file)
    except FileNotFoundError:
        raise FileNotFoundError('Key file not found')


def create_client(key: str):
    # creates the googlemaps client
    try:
        # probably not efficient to create googlemaps client everytime this function is called but...
        gmaps = googlemaps.Client(key=key)
        return gmaps
    except ValueError:
        raise ValueError('Key is invalid')


def get_geolocation(client, address: str):
    # post address to google maps and returns geolocation as latitude and longitude
    gmaps = client
    try:
        response = gmaps.geocode(address)
        response = response[0]

        # response = open_last_response('test.json')
        latitude = float(response['geometry']['location']['lat'])
        longitude = float(response['geometry']['location']['lng'])
        return latitude, longitude

    except googlemaps.exceptions.ApiError:
        raise ValueError('Something is wrong with the API usage')

    except googlemaps.exceptions.Timeout:
        raise RuntimeError('Google maps timed out')


def get_address(client, latitude: float, longitude: float) -> str:
    # post address to google maps and returns geolocation as latitude and longitude

    gmaps = client
    latlng = (latitude, longitude)
    response = gmaps.reverse_geocode(latlng)
    with open('reverse.json', 'w') as file:
        json.dump(response, file)

    response = response[0]['address_components']

    street_address = response[0]['short_name'] + ' ' + response[1]['short_name']

    return street_address


