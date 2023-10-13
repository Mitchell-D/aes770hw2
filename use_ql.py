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
from krttdkit.products import FeatureGrid

from HyperGrid import HyperGrid
from CoordSystem import CoordSystem, IntAxis, CoordAxis

if __name__=="__main__":
    ql_path = Path("data/DESIS-HSI-L2A-DT0865788448_017-20230607T153935-QL2.pkl")
    hg_path = Path("data/DESIS-HSI-L2A-DT0865788448_017-20230607T153935.zip")

    hrange=(400,1000)
    vrange=(400,1000)

    '''
    hg = HyperGrid.from_store(hg_path)
    hg = hg.subgrid(x=IntAxis(hrange), y=IntAxis(vrange))
    rgb = [np.squeeze(hg.data(wl=wl)) for wl in (640, 550, 460)]
    gt.quick_render(enh.linear_gamma_stretch(np.dstack(rgb)))
    '''

    fg = FeatureGrid.from_pkl(ql_path)
    fg = fg.subgrid(vrange=vrange, hrange=hrange)
    print(fg.labels)

    aod = fg.data("aod")
    for l in fg.labels:
        print(l, enh.array_stat(fg.data(l)))
    gt.quick_render(fg.data("norm1 aod"))
