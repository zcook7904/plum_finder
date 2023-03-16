SF_plum_finder
======

A program that finds the nearest plum trees to a given address in San Francisco

Installation
------------

To install with pip, run:

    pip install SF_plum_finder

Config
------------
To use the plum finder, a config file must be generated in the working directory. 
This is generated automatically by running the program the first time. 

### Keys
Google maps: an appropriate google maps API key must be stored here

### Settings
n: number of possible closest trees to be sent to gmaps; max = 25  
useSQL: attempt to query a SQLite db for address geocode before making an API call to Google Maps  
performance_log: logs parameters regarding the performance and entered data for each call to find_closest_plum if
yes or True


Usage
----------
### Cloned Repository:
CLI usage can be achieved by entering:

```python src/SF_plum_finder [street address in San Francisco]```

in the working directory.

### As a module
The CLI can also be accessed using the following python code:
```
from SF_plum_finder import plum_finder

plum_finder.command_line_runner()
```



Contribute
----------
If you'd like to contribute to SF_plum_finder, check out https://github.com/zcook7904/plum_finder

Tree Locations
----------
DataSF graciously hosts the entire data set of trees maintained by the Department of Public Works here: 
https://data.sfgov.org/City-Infrastructure/Street-Tree-List/tkzw-k3nqq

This data was filtered for plum trees on side walks for more manageable search and geolocations were added to trees 
with addresses that were missing them