import json
import argparse
import os.path


def command_line_runner():
    """Argument parser for CLI"""
    parser = argparse.ArgumentParser(description='Creates a keys.txt file to be used as a template')
    parser.add_argument('-d',
                        help='An alternative directory to store the keys.txt file (not recommended)')
    args = parser.parse_args()

    if args.d:
        path = os.path.join(args.d, 'keys.txt')
        return init_key_file(path)
    else:
        return init_key_file()


def init_key_file(path: str = 'keys.txt') -> bool:

    keys = {
        'GoogleAPIkey': '{KEY}',
        'TwilioAccountSID': '{KEY}',
        'TwilioAccountToken': '{KEY}',
        "TwilioNumber": "+19475006300"
    }

    # Don't overwrite if the file already exists
    if os.path.exists(path):
        print('Path already exists, check: {}'.format(os.path.abspath(path)))
        return False

    try:
        with open(path, 'x') as file:
            json.dump(keys, file)
        return True
    except FileNotFoundError:
        FileNotFoundError('Cannot create keys.txt file')


if __name__ == '__main__':
    command_line_runner()
