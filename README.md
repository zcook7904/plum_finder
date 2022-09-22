plumr
======

A program that finds the nearest plum trees in San Francisco

Installation
------------

To install with pip, run:

    pip install plumr

TODO
----------------
Need method to automatically download data base from DataSF 
and update the repo's db


Data processing
----------------
data_process.py holds most tools to do basic querying of data. Other scripts in the data_processing folder were used to 
originally further process the data or summarize it.



Contribute
----------
If you'd like to contribute to plumr, check out https://github.com/zcook7904/plumr

Tree Locations
----------
DataSF graciously hosts the entire data set of trees maintained by the Department of Public Works here: 
https://data.sfgov.org/City-Infrastructure/Street-Tree-List/tkzw-k3nqq

This data was filtered for plum trees on side walks for more manageable search and geolocations were added to trees 
with addresses that were missing them