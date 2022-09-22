SF_plum_finder
======

A program that finds the nearest plum trees to a given address in San Francisco

Installation
------------

To install with pip, run:

    pip install SF_plum_finder


Usage
----------
### Cloned Repository:
CLI usage can be achieved by entering:

```python src/SF_plum_finder [street address in San Francisco]```

in the working directory.

### As a module
Simple CLI use can be acheived through the following script:
```
'from SF_plum_finder import plum_finder

plum_finder.command_line_runner()'
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