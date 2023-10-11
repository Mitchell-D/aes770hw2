
import zarr
import numpy as np
from pathlib import Path
import shlex
from subprocess import Popen, PIPE

from krttdkit.products import FeatureGrid
from krttdkit.visualize import guitools as gt
from krttdkit.visualize import geoplot as gp
from krttdkit.operate import enhance as enh
from krttdkit.operate import Recipe

if __name__=="__main__":
    aero_lut_path = Path("data/aerosol_lut_backup.zip")
    L = zarr.group(store=aero_lut_path)
    print(np.array(L))
