import zarr
import numpy as np
from pathlib import Path
import shlex
from subprocess import Popen, PIPE

from krttdkit.products import FeatureGrid
from krttdkit.visualize import guitools as gt
from krttdkit.visualize import geoplot as gp
from krttdkit.operate import enhance as enh

from CoordSystem import CoordSystem, IntAxis, CoordAxis

class HyperGrid:
    """
    The HyperGrid expands the FeatureGrid concept to an N-dimensional
    array like (C1, C2, ... CN, F) for N coordinate axes and F labeled
    features. It consists of the following attributes, which are all
    serializable to an ndarray like the shape above and a JSON string,

    - A data array shaped like above
    - A list of 2-tuples mapping string names to lists of data coordinate
      values, each list having monotonically increasing or decreasing values,
      and the same length as the corresponding data dimension.
    - A list of strings with the same length as the final axis in the ndarray
      which label the features axis

    :TODO: masks, info dicts, meta dict,
    """
    attr_labels = frozenset(["meta", "info", "labels"])
    @staticmethod
    def from_store(zarr_store:Path):
        """
        Open a Zarr storage format path (ie .zip, .zarr directory) that
        minimally has a "flabels" arguments as attributes including
        """
        Z = zarr.open(zarr_store, mode="r")
        assert "flabels" in Z.attrs.keys()
        assert len(Z.attrs["flabels"]) == Z.shape[-1]
        return HyperGrid(data=Z, **Z.attrs)

    def __init__(self, data:zarr.core.Array,
                 flabels:list, clabels:list=None, coord_arrays:list=None,
                 info:dict=None, meta:dict=None, masks:dict=None,
                 recipes:dict=None):
        """
        Minimally last dimension corresponds to feature_labels.
        """
        self._data = data

        ## The final dimension is always the feature dimension,
        ## so must have a length equal to the number of features.
        self._flabels = list(flabels)
        assert len(self._flabels)==data.shape[-1]

        self._clabels = [] if clabels is None else list(clabels)
        self._coords = [] if coord_arrays is None else list(coord_arrays)
        ## There must be a coordinate label for every dimension prior to
        ## the feature dimension
        coords_shape = tuple(len(c) for c in self._coords)
        assert coords_shape == data.shape[:-1]
        ## There must be a label for every coordinate dimension
        assert len(self._clabels)==len(self._coords)
        ## Initialize a CoordSystem for data indexing
        self._cs = CoordSystem(self._coords)

        ## Info stores information on data, masks, or coords by unique label.
        self._info = {} if info is None else dict(info)
        ## Meta stores information pertaining to the entire dataset
        self._meta = {} if meta is None else dict(meta)
        ## Masks stores boolean masks for one or more feature/coordinate dims
        self._masks = {} if masks is None else dict(masks)
        ## recipes stores Recipe objects for data operations
        self._recipes = {} if recipes is None else dict(recipes)

        ## Full collection of labels must be unique
        assert len(self.vocab)==len(set(self.vocab))

    def validate_label(self, label:str):
        """
        Ensure that a string label is in the vocab, and return a sanitized
        version of it.
        """
        label = str(label)
        assert label.lower() in self.vocab
        return label

    def around(self, as_slice=False, bound_err=True, **coords):
        """
        Returns a tuple of IntAxis surrounding the provided coordinate values.
        """
        assert all(k in self._clabels for k in coords.keys())
        bounds = []
        for cl in self._clabels:
            if cl in coords.keys():
                v = coords[cl]
                bounds.append(self.coords(cl).around(v, bound_err=bound_err))
            else:
                bounds.append(IntAxis((0, None)))
        if as_slice:
            bounds = [b.as_slice for b in bounds]
        return tuple(bounds)

    def subgrid(self, flabels=None, **coords):
        """
        Returns a new HyperGrid with the provided flabels (in order)
        """
        ## Make sure all feature labels are valid and in the appropriate order
        if flabels is None:
            flabels = self._flabels
        elif hasattr(flabels, "__iter__") and not type(flabels) is str:
            assert all(str(l).lower() in self._flabels for l in flabels)
        else:
            flabels = [str(flabels)]
        feat_idxs = [self._flabels.index(str(l).lower()) for l in flabels]

        slices = []
        ## kwargs must map a coordinate label c to an argument in data
        ## coordinates: either None, a number, or a range (start, stop)
        for c in self._clabels:
            ## If the coordinate exists,
            if c in coords.keys():
                cidx = self._clabels.index(c)
                ## Handle the default case
                if coords[c] is None:
                    slices.append(IntAxis(None).as_slice)
                ## If an IntAxis is provided, use it to generate a slice
                if type(coords[c]) is IntAxis:
                    slices.append(coords[c].as_slice)
                    continue
                ## Otherwise assume its a 2-tuple or number
                if type(coords[c]) is str:
                    raise ValueError(f"Coord constraints for {c} must be one "
                                     "of IntAxis, float, or 2-tuple")
                elif hasattr(coords[c], "__iter__"):
                    ## data coord range arguments must be (start, stop)
                    assert len(coords[c])==2
                    slices.append(self._cs.axis(cidx).range(
                        *tuple(coords[c])).as_slice)
                else:
                    slices.append(self.coords(c).index(
                        float(coords[c])).as_slice)
            else:
                slices.append(IntAxis(None).as_slice)

        if type(self._data) is zarr.Array:
            data = self._data.oindex[*slices,feat_idxs]
        else:
            data = self._data[*slices,feat_idxs]
        new_coords = [self._coords[i][slices[i]]
                      for i in range(len(self._coords))]
        ## TODO: to subgrid masks and remove stale keys from info dict
        return HyperGrid(
                data=data,
                flabels=flabels,
                clabels=self._clabels,
                coord_arrays=new_coords,
                info=self._info,
                meta=self._meta,
                masks=self._masks,
                )

    def data(self, flabels=None, **kwargs):
        """
        Returns a data view given constraints on flabels, and either IntAxis
        or CoordAxis constraints on coordinate dimensions
        """
        return self.subgrid(flabels, **kwargs)._data

    def coords(self, clabel:str=None):
        if clabel is None:
            return self._cs
        return self._cs.axis(self._clabels.index(self.validate_label(clabel)))
    @property
    def clabels(self):
        """ return the coordinate labels """
        return self._clabels
    @property
    def flabels(self):
        """ return the feature labels """
        return self._flabels
    @property
    def shape(self):
        return self._data.shape
    @property
    def vocab(self):
        """
        The HyperGrid's vocab is the list of all unique labels it can use,
        including feature, coordinate, and mask labels.
        """
        vocab = list(self._flabels) + list(self._clabels) + \
                list(self._recipes.keys()) + list(self._masks.keys())
        return list(map(str.lower, set(vocab)))
    def info(self, label:str=None):
        """ return info dict entry for a provided label, or the whole dict. """
        return self._info[self.validate_label(label)] if not label is None \
                else self._info
    def meta(self, key:str=None):
        """ return the meta dict, or the entry for a specific key"""
        return self._meta[key] if not key is None else self._meta

    def to_fg(self, labels:list=None, vrange=(None, None),
              hrange=(None,None)):
        X = self.to_array(vrange,hrange,labels)
        return FeatureGrid(
                data=[X[...,i] for i in range(len(labels))],
                labels=labels,
                info=[self.info(i) for i in range(len(labels))],
                meta=self._meta)

if __name__=="__main__":
    #zarr_path = Path("data/DESIS-HSI-L2A-DT0865788448_017-20230607T153935.zarr")
    zarr_path = Path("data/DESIS-HSI-L2A-DT0865788448_015-20230607T153935.zarr")
    MG = HyperGrid.from_store(zarr_path)
    hrange=(400,1000)
    vrange=(400,1000)
    window = 5
    get_wl = lambda l:400+l*2.551
    for i in range(len(MG.labels)-window):
        tmp_fg = MG.to_fg(vrange=vrange, hrange=hrange,
                          labels=MG.labels[i:i+window])
        X = np.average(tmp_fg.data(), axis=-1)
