import argparse
import os.path
import configparser


def command_line_runner():
    """Argument parser for CLI"""
    parser = argparse.ArgumentParser(description='Creates an initial config file for SF_plum_finder')
    args = parser.parse_args()
    return init_config_file()


def init_config_file(path: str = 'config.ini') -> bool:
    config = configparser.ConfigParser()
    config['Keys'] = {'GoogleMaps': '{key}'}
    config['Settings'] = {'n': '10',
                          'UseSql': 'True',
                          'PerformanceLog': 'True'}

    # Don't overwrite if the file already exists
    absolute_path = os.path.abspath(path)
    if os.path.exists(path):
        print(f'Path already exists, check: {absolute_path}')
        return False
    try:
        with open(path, 'x') as file:
            config.write(file)
        print(f'Config file successfully created at {absolute_path}')
        return True
    except FileNotFoundError:
        FileNotFoundError('Cannot create config.ini file')


if __name__ == '__main__':
    command_line_runner()
