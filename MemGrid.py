import zarr
import numpy as np
from pathlib import Path
import shlex
from subprocess import Popen, PIPE

from krttdkit.products import FeatureGrid
from krttdkit.visualize import guitools as gt
from krttdkit.visualize import geoplot as gp
from krttdkit.operate import enhance as enh

class MemGrid:
    """
    MemGrid expands on the FeatureGrid functionality by providing a readonly
    interface for extracting FeatureGrids from a compressed file (currently
    just a zarr zip archive).
    """
    attr_labels = frozenset(["meta", "info", "labels"])

    @staticmethod
    def from_store(zarr_store:Path):
        """
        Open a Zarr storage format path (ie .zip, .zarr directory) that
        minimally has a "labels" arguments as attributes including
        """
        Z = zarr.open(zarr_store, mode="r")
        assert "labels" in Z.attrs.keys()
        assert len(Z.shape)==3
        assert len(Z.attrs["labels"]) == Z.shape[2]
        return MemGrid(
                data=Z,
                labels=Z.attrs.get("labels"),
                info=Z.attrs.get("info"),
                meta=Z.attrs.get("meta"),
                )

    def __init__(self, data:zarr.core.Array, labels:list, info=None, meta=None):
        self._data = data
        self._labels = list(labels)
        self._info = [] if info is None else list(info)
        self._meta = {} if meta is None else dict(meta)

    def data(self, label):
        return self._data[:,:,self.labels.index(label)]
    @property
    def labels(self):
        return self._labels
    @property
    def shape(self):
        self._data.shape[:2]
    def info(self, label:str=None):
        return self._info[label] if not label is None else self._info
    def meta(self, key:str=None):
        return self._meta[key] if not key is None else self._meta

    def to_array(self, labels:list=None, vrange=(None,None),
                 hrange=(None,None), mask=None):
        """
        Return a data view fitting the above constraints
        """
        # Assume single numbers are indeces rather than ranges
        if type(vrange) is int:
            vrange = (vrange, vrange+1)
        if type(hrange) is int:
            hrange = (hrange, hrange+1)
        labels = self.labels if labels is None else labels
        idxs = (self.labels.index(l) for l in labels)
        C = np.dstack([self._data.oindex[slice(*vrange),slice(*hrange),idx]
                       for idx in idxs])
        return C if mask is None else C[mask]

    def to_fg(self, labels:list=None, vrange=(None, None),
              hrange=(None,None)):
        X = self.to_array(vrange,hrange,labels)
        return FeatureGrid(
                data=[X[...,i] for i in range(len(labels))],
                labels=labels,
                info=[self.info(i) for i in range(len(labels))],
                meta=self._meta)

if __name__=="__main__":
    pass
