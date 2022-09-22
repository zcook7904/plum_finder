import json


def init_key_file():

    keys = {
        'GoogleAPIkey': '{KEY}',
        'TwilioAccountSID': '{KEY}',
        'TwilioAccountToken': '{KEY}',
        "TwilioNumber": "+19475006300"
    }
    path = 'keys.txt'

    with open(path, 'x') as file:
        json.dump(keys, file)


if __name__ == '__main__':
    init_key_file()
