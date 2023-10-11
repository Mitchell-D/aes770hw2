""" """
#from sbdart_info import default_params
from pathlib import Path
import numpy as np
import pickle as pkl
import zarr

from krttdkit.acquire.sbdart import dispatch_sbdart, parse_iout
import krttdkit.visualize.guitools as gt
import krttdkit.visualize.geoplot as gp

from HyperGrid import HyperGrid

aero_types = {1:"Rural", 2:"Urban", 3:"Oceanic", 4:"Tropo"}

if __name__=="__main__":
    tmp_dir = Path("buffer/sbdart")
    pkl_path = Path("buffer/lut.pkl")
    lut_file_ocean = Path("data/lut_aero_ocean_srad.zip")
    flux_file_ocean = Path("data/lut_aero_ocean_sflux.pkl")
    lut_file_noref = Path("data/lut_aero_noref_srad.zip")
    flux_file_noref = Path("data/lut_aero_noref_sflux.pkl")

    sza = 28.92
    saa = 329.53%180
    vza = 10.50
    vaa = 305.89
    raa = abs(vaa-saa)
    plot_spec={
        "xlabel":"Wavelength (nm)",
        "ylabel":"Spectral Radiance ($Wm^{-2}sr{-1}\mu m^{-1}$)",
        }

    '''
    """ Plot each aerosol type and AOD wrt wl for one surface type """
    # (sza, tbaer, iaer, uzen, phi, wl)
    hg = HyperGrid.from_store(lut_file_noref)
    #aero_type_img = "figures/aero_{atype}_ocean.png"
    aero_type_img = "figures/aero_{atype}_noref.png"
    for atype in aero_types.keys():
        sub_data = np.squeeze(hg.data(sza=sza, saa=saa, vza=vza,
                                      uzen=vza, phi=raa, iaer=atype))
        plot_spec["title"] = "SBDART $\\alpha_{sfc}=0$ TOA Aerosol " + \
                "Radiance ({aero_types[atype].lower()})"
        gp.plot_lines(
                domain=hg.coords("wl").coords,
                ylines=[sub_data[i] for i in range(sub_data.shape[0])],
                labels=[f"$\\tau$={hg.coords('tbaer').coords[i]}"
                        for i in range(sub_data.shape[0])],
                show=False,
                plot_spec=plot_spec,
                image_path=Path(aero_type_img.format(
                    atype=aero_types[atype].lower())),
                )
    '''

    """ Plot the difference between oceanic and no-reflectance radiance """
    ocean = HyperGrid.from_store(lut_file_ocean)
    noref = HyperGrid.from_store(lut_file_noref)
    ocean_anom_img = "figures/aero_{atype}_ocean-anom.png"
    for atype in aero_types.keys():
        plot_spec["title"] = "Ocean Reflectance Contribution with " + \
                f"{aero_types[atype]} Aerosol Loading)"
        tmp_oc = np.squeeze(ocean.data(
            sza=sza, saa=saa, vza=vza, uzen=vza, phi=raa, iaer=atype))
        tmp_nr = np.squeeze(noref.data(
            sza=sza, saa=saa, vza=vza, uzen=vza, phi=raa, iaer=atype))
        anom = tmp_oc-tmp_nr
        gp.plot_lines(
                domain=ocean.coords("wl").coords,
                ylines=[anom[i] for i in range(anom.shape[0])],
                labels=[f"$\\tau$={ocean.coords('tbaer').coords[i]}"
                        for i in range(anom.shape[0])],
                show=True,
                plot_spec=plot_spec,
                image_path=Path(ocean_anom_img.format(
                    atype=aero_types[atype].lower())),
                )



    #flux_labels, fluxes = pkl.load(flux_file.open("rb"))

    ## offsets in boxed data from coordinate values??
    #geom_box = hg.around(sza=sza, phi=saa, uzen=vza, as_slice=True)
    #geom_point = hg.data(sza=sza, phi=saa, uzen=vza)
    #point_data = np.squeeze(geom_point)
    #box_data = np.squeeze(hg._data[*geom_box])

