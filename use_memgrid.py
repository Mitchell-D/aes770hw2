from pathlib import Path
import zarr
import numpy as np
from typing import Callable

#from sklearn.cluster import KMeans

from krttdkit.operate import enhance as enh
from krttdkit.operate import recipe_book as rb
from krttdkit.visualize import guitools as gt
from krttdkit.visualize import geoplot as gp

from HyperGrid import HyperGrid
from CoordSystem import CoordSystem, IntAxis, CoordAxis

def map_to_chist(X:np.array, nbins:int=256):
    """
    Returns an array with the same shape as X, but with brightness values
    mapped to their [0,1] cumulative distribution function value
    """
    ## Normalize
    H = enh.get_cumulative_hist(fg.data(), nbins=nbins)[0]
    Y = np.array(enh.linear_gamma_stretch(fg.data())*bins, dtype=int)
    ## Saturate brightest pixel to get correct scale
    Y[np.where(Y==bins)] = bins-1
    _to_chval = lambda b:H[j][b]
    return _to_chval(Y)

def rgb_dchist(X:np.array, nbins:int=256, alt=False):
    """
    Make an RGB by averaging the change in cumulative histogram position
    over  the third axis.
    """
    # Must be able to iterate over axis 3
    assert len(X.shape)==3 and X.shape[2]>1
    chist = map_to_chist(X, nbins=nbins)
    mdiff = np.diff(chist, axis=2)
    mmean = np.average(mdiff, axis=2)
    mstdev = np.std(mdiff, axis=2)
    if not alt:
        return gt.scal_to_rgb(enh.linear_gamma_stretch(mmean))
    RGB = [# decreasing wavelengths are redshifted
           np.where(mmean<0, np.zeros_like(mmean), -mmean)/mstdev,
           # bright -> low standard deviation
           np.amax(mstdev)-mstdev,
           #np.full_like(mmean, 0),
           np.where(mmean>0, np.zeros_like(mmean), mmean)/mstdev
           ]
    return np.dstack(list(map( rb.gaussnorm, RGB)))



def _none(mg:HyperGrid, apply_across):
    total_mean = np.zeros_like(HG.shape)
    total_stdev =None
    for i in range(len(HG.labels)):
        total_mean += HG.data
    for i in range(len(HG.labels)-window):
        img_template = Path(
                f"buffer/desis_17_histdiff/histdiff-3-nogreen_{i:03}.png")

def convert_zarr_fg_to_hg(in_store:Path, out_store:Path):
    ## Add singular feature dimension
    Z = zarr.open(in_store, mode="r")
    wl = [inf["wl"] for inf in Z.attrs["info"]]
    y,x = tuple(range(Z.shape[0])),tuple(range(Z.shape[1]))
    attrs = {
            "flabels":("srad",),
            "clabels":("y", "x", "wl"),
            "coord_arrays":(y,x,wl),
            "info":{"wl":Z.attrs["info"]},
            "meta":Z.attrs["meta"],
            }

    store = zarr.ZipStore(out_store, mode="w")
    A = zarr.create((*Z.shape,1), store=store)
    A[...,0] = Z[:]
    A.attrs.update(attrs)
    store.flush()
    store.close()

def gen_hg_tiles(hg:HyperGrid, tile:dict):
    """
    Returns a generator that tiles over a HyperGrid's shape with a kernel
    mapping string coordinate labels to a rooted IntAxis range

    :@return: generator for combos and tuple showing final shape
    """
    assert all(v.bounded and v.start==0 for v in tile.values())
    t_labels,t_axes = list(zip(*tile.items()))
    t_shape = np.array([s.size for s in t_axes])
    combos = list(np.where(np.full(
        [hg.coords(t_labels[i]).size//t_axes[i].size
         for i in range(len(t_labels))], True)))
    combos = [combos[i]*t_axes[i].size for i in range(len(combos))]
    combos = list(map(np.asarray, zip(*combos)))
    combos = [[IntAxis((c[i], c[i]+t_shape[i],1)) for i in range(len(c))]
              for c in combos]
    combos = ((c,hg.subgrid(**dict(zip(t_labels, c)))) for c in combos)
    return combos

def cloud_mask(hg:HyperGrid):
    ndwi = (hg.data(wl=550)-hg.data(wl=850)) / \
            (hg.data(wl=550)+hg.data(wl=850))
    gt.quick_render(enh.linear_gamma_stretch(np.squeeze(ndwi)))

if __name__=="__main__":
    #zarr_path = Path("data/DESIS-HSI-L2A-DT0865788448_014-20230607T153935.zarr")
    #zarr_path = Path("data/DESIS-HSI-L2A-DT0865788448_015-20230607T153935.zarr")
    #zarr_path = Path("data/DESIS-HSI-L2A-DT0865788448_016-20230607T153935.zarr")
    #zarr_path = Path("data/DESIS-HSI-L2A-DT0865788448_017-20230607T153935.zarr")
    new_path = Path("buffer/tmp_hg.zip")

    hrange=(400,1000)
    vrange=(400,1000)
    window = 12
    bins = 1024
    hg = HyperGrid.from_store(new_path).subgrid(x=IntAxis(hrange),
                                                y=IntAxis(vrange))
    cloud_test(hg)
    exit(0)

    kernel = {"x":IntAxis((0,20)), "y":IntAxis((0,20))}
    Y = np.zeros((hg.shape[0]//20, hg.shape[1]//20))
    for combo,hg in gen_hg_tiles(hg, kernel):
        slices = [a.as_slice for a in combo]
        Y[*slices] = cloud_test(hg)
