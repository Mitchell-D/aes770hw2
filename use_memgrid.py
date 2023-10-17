import pickle as pkl
import zarr
import numpy as np
from pathlib import Path
from typing import Callable
from matplotlib import pyplot as plt

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

def water_mask(hg:HyperGrid, thresh=.1):
    ndwi = (hg.data(wl=550)-hg.data(wl=850)) / \
            (hg.data(wl=550)+hg.data(wl=850))
    wmask = np.squeeze(ndwi) > thresh
    return wmask

def get_rgb(hg, rgb_wls:tuple):
    rgb = [np.squeeze(hg.data(wl=wl)) for wl in rgb_wls]
    return enh.linear_gamma_stretch(np.dstack(rgb))

def plot_stats(hg:np.array, mask:np.array, model:np.array=None,
               model_domain:np.array=None):
    """
    Plots mean value and standard deviation of HyperGrid data
    where masked values are True
    """
    valid = np.squeeze(hg._data[np.where(mask)])
    name = "Spectral Radiance"
    means = np.average(valid, axis=0)
    stdevs = np.std(valid, axis=0)
    wls = hg.coords("wl").coords
    plt.plot(wls, means, label="Reflectance $\mu$",)
    plt.plot(wls, stdevs, label="Reflectance $\sigma$",)
    if not model is None:
        plt.plot(model_domain, model,
                 label="Model reflectance over ocean (no aerosol)")
    plt.xlim((wls[0],wls[-1]))
    plt.title("Model vs observed over-ocean spectral response")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.grid()
    plt.legend()
    plt.show()

def get_model_ref(flux_file:Path):
    """ returns ocean reflectance """
    flabels, fluxes = pkl.load(flux_file.open("rb"))
    # Just use the first one since sfc reflectance should be the same.
    fluxes = fluxes[0]
    sfcup, sfcdn = fluxes[-2:]
    wl = fluxes[0]
    sfcref = (sfcup/sfcdn)
    return wl, sfcref

def plot_hists(arrays:list, labels:list=None, nbins=256):
    hists,coords = zip(*[enh.get_nd_hist([a], bin_counts=nbins)
                         for a in arrays])
    if labels is None:
        labels = [None for i in range(len(coords))]
    for i in range(len(coords)):
        plt.plot(np.squeeze(coords[i]), hists[i], label=labels[i])
    plt.legend()
    plt.title("DESIS reflectance histograms over ocean water")
    plt.ylabel("Frequency")
    plt.xlabel("Reflectance")
    plt.show()

if __name__=="__main__":
    #hg_path = Path("data/DESIS-HSI-L2A-DT0865788448_017-20230607T153935.zip")
    #hg_path = Path("data/DESIS-HSI-L2A-DT0865788448_016-20230607T153935.zip")
    #hg_path = Path("data/DESIS-HSI-L2A-DT0865788448_015-20230607T153935.zip")
    #hg_path = Path("data/DESIS-HSI-L2A-DT0865788448_014-20230607T153935.zip")
    #hg_path = Path("data/DESIS-HSI-L2A-DT0865788448_017-20230607T153935.zip")
    hg_path = Path("data/DESIS-HSI-L2A-DT0314290188_005-20190503T164053.zip")

    hrange=(400,1000)
    vrange=(400,1000)

    hg = HyperGrid.from_store(hg_path)
    hg = hg.subgrid(x=IntAxis(hrange), y=IntAxis(vrange))

    """ show a truecolor with water mask """
    truecolor = get_rgb(hg,(640, 550, 460))
    m_water = water_mask(hg, thresh=0)
    tc_water = truecolor
    tc_water[np.where(np.logical_not(m_water))] = 0
    print(enh.array_stat(tc_water))
    gt.quick_render(enh.linear_gamma_stretch(tc_water))

    ## Visible to NIR ratio doesn't distinguish any clouds
    visnir = np.squeeze(hg.data(wl=990)/hg.data(wl=500))
    visnir[np.logical_not(m_water)] = np.amin(visnir)
    gt.quick_render(enh.linear_gamma_stretch(visnir))

    '''
    gran_id = 17
    for i in range(len(hg.coords("wl").coords)):
        tmp_wl = hg.coords("wl").coords[i]
        wlstr = str(tmp_wl)[:3]
        tmp_path = Path(f"buffer/desis_17_all/desis_{gran_id}_{wlstr}.png")
        gp.generate_raw_image(enh.linear_gamma_stretch(
            np.squeeze(hg.data(wl=tmp_wl))), tmp_path)
    '''

    """ Plot model  """
    model_wl, sfcref = get_model_ref(Path("data/lut_aero_ocean_sflux.pkl"))
    plot_stats(hg, m_water, sfcref, model_wl*1000)
    exit(0)

    hist_bands = [450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950]
    water_px = [np.squeeze(hg.data(wl=b)[np.where(m_water)])
                for b in hist_bands]
    plot_hists(
            arrays=water_px,
            labels=[f"{b}$ nm$" for b in hist_bands]
            )
