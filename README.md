# aes770hw2

Using hyperspectral DESIS data retrieve aerosol optical depth and
validate your results.

### CoordSystem.py

Prototype of a class system intended to provide a framework for
specifying arbitrary embeddings of data in an infinite but discrete
coordinate space.

The IntAxis expands on the concept of a Python slice. An IntAxis 
can be specified by any 3 of 4 int values (start, stop, step, size).

The CoordAxis is a child class of an IntAxis that assigns each
element in the IntAxis to a monotonic series of coordinate values,
which may be floats.

The CoordSystem is an ordered collection of CoordAxis objects that
allows for a value to be searched for in coordinate space, and will
hopefully soon allow for interpolation between coordinate values. It
is intended to be applied to an arbitrary-dimensional array with a
shape corresponding to each coordinate axis.

This project didn't allow enough time for this concept to mature,
however in the long run I would like to adapt this into a
string-encoded system for representing a collection of
arbitrary-dimensional datasets which are sparsely embedded in
overlapping coordinate spaces.

### desis.py

Contains methods for parsing reflectance data, band information, and
data quality masks from a directory associated with a DESIS
radiometer granule, and for dumping them into a zarr file with fields
"meta", "info", "flabels", and "clabels" for reference, and with
coordinate arrays associated with the (y,x,wavelength) dimensions of
the data grid.

### get\_lut\_aod.py

Uses a pool of processes executing `krttdkit.acquire.sbdart` to
generate spectral radiance as well as layerwise spectral fluxes given
one or more SBDART parameters to iterate over. The default
coordinates are the final 3, ordered like (vza, raa, wavelength),
and any other coordinates (ie COD, TBAER, IAER, etc) are prepended as
dimensions prior to the default coordinates in the order they are
provided.

The consequent lookup table is stored in a zarr file alongside the
coordinate values (in order of the grid shape).

### HyperGrid.py

The HyperGrid is an expansion on the concept of the 2D FeatureGrid
in that it facilitates labeled arrays, searching by coordinates, and
easy subgridding. It assigns a CoordSystem and IntAxis to each
dimension to facilitate easy linear interpolation.

There are still some idiosyncracies with the current approach, and
the recipe evaluation system isn't yet implemented, so as of yet it's
best to just use the HyperGrid as a medium for generating
FeatureGrids. Eventually I want to switch to an optional HDF5 backend
for HyperGrids, and externalize the recipe evaluation system using
structures defined in terms of IntAxis or CoordSystem objects.

## MemGrid.py

This is just an unfinished attempt to memory-map HyperGrid objects
using zarr files. Not a thread I want to pull again.


[A Hyperspectral Crawl over San Francisco][1]
[https://youtube.com/shorts/P0V-THPp-uw?si=kWr_31_nylP_zrSq]:1
