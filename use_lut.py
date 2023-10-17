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

    #flux_file = Path("data/lut_aero_veg_sflux.pkl")
    #srad_file = Path("data/lut_aero_veg_srad.zip")
    flux_file = Path("data/lut_aero_veg-noatmo_sflux.pkl")
    srad_file = Path("data/lut_aero_veg-noatmo_srad.zip")

    sza = 22.47
    saa = 12.78
    vza = 5.32
    vaa = 0
    raa = abs(vaa-saa)
    plot_spec={
        "xlabel":"Wavelength (um)",
        "ylabel":"Spectral Radiance ($Wm^{-2}sr{-1}\mu m^{-1}$)",
        }

    """ Get the downward radiance """
    #flabels, fluxes = pkl.load(flux_file_ocean.open("rb"))
    flabels, fluxes = pkl.load(flux_file.open("rb"))
    # Just use the first one since sfc reflectance should be the same.
    fluxes = fluxes[0]
    sfcup, sfcdn = fluxes[-2:]

    #'''
    """ Plot each aerosol type and AOD wrt wl for one surface type """
    # (sza, tbaer, iaer, uzen, phi, wl)
    hg = HyperGrid.from_store(srad_file)
    #aero_type_img = "figures/model/aero_{atype}_ocean.png"
    aero_type_img = "figures/model/aero_veg_{atype}.png"
    for atype in aero_types.keys():
        sub_data = np.squeeze(hg.data(sza=sza, saa=saa, vza=vza,
                                      uzen=vza, phi=raa, iaer=atype))
        sub_data /= sfcdn
        plot_spec["title"] = "SBDART TOA Reflectance over "+\
                f" vegetation with {aero_types[atype].lower()} Aerosols"
        plot_spec["ylabel"] = "Reflectance Factor"
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
    #'''

    '''
    """ Plot the difference between ocean and no-albedo reflectance """
    ocean = HyperGrid.from_store(lut_file_ocean)
    noref = HyperGrid.from_store(lut_file_noref)
    ocean_anom_img = "figures/model/aero_{atype}_ocean-anom-ref.png"
    for atype in aero_types.keys():
        plot_spec["ylabel"] = "Reflectance Factor"
        plot_spec["title"] = "Ocean Reflectance Contribution with " + \
                f"{aero_types[atype]} Aerosol Loading"
        tmp_oc = np.squeeze(ocean.data(
            sza=sza, saa=saa, vza=vza, uzen=vza, phi=raa, iaer=atype))
        tmp_nr = np.squeeze(noref.data(
            sza=sza, saa=saa, vza=vza, uzen=vza, phi=raa, iaer=atype))
        ## Convert to reflectance by dividing out sfc downward radiance
        anom = (tmp_oc-tmp_nr)/sfcdn
        gp.plot_lines(
                domain=ocean.coords("wl").coords,
                ylines=[anom[i] for i in range(anom.shape[0])],
                labels=[f"$\\tau$={ocean.coords('tbaer').coords[i]}"
                        for i in range(anom.shape[0])],
                show=False,
                plot_spec=plot_spec,
                image_path=Path(ocean_anom_img.format(
                    atype=aero_types[atype].lower())),
                )
    '''

    #'''
    """
    Plot the difference between aerosol and no-aerosol reflectance
    """
    hg = HyperGrid.from_store(srad_file)
    aero_anomaly = "figures/model/aero_{atype}_veg-anom.png"
    for atype in aero_types.keys():
        plot_spec["ylabel"] = "Reflectance Factor"
        plot_spec["title"] = "Aerosol Reflectance Contribution with " + \
                f"{aero_types[atype]} Aerosol Loading"
        plot_spec["yrange"] = (-.04,.04)
        tmp_oc = np.squeeze(hg.data(
            sza=sza, saa=saa, vza=vza, uzen=vza, phi=raa, iaer=atype))
        tmp_na = np.squeeze(hg.data(
            sza=sza, saa=saa, vza=vza, uzen=vza, phi=raa, iaer=atype,
            tbaer=0))
        ## Convert to reflectance by dividing out sfc downward radiance
        anom = (tmp_oc-tmp_na)/sfcdn
        gp.plot_lines(
                domain=hg.coords("wl").coords,
                ylines=[anom[i] for i in range(anom.shape[0])],
                labels=[f"$\\tau$={hg.coords('tbaer').coords[i]}"
                        for i in range(anom.shape[0])],
                show=False,
                plot_spec=plot_spec,
                image_path=Path(aero_anomaly.format(
                    atype=aero_types[atype].lower())),
                )
    #'''

    #'''
    """ Plot model reflectance wrt wavelength """
    flabels, fluxes = pkl.load(flux_file.open("rb"))
    # Just use the first one since sfc reflectance should be the same.
    fluxes = fluxes[0]
    sfcup, sfcdn = fluxes[-2:]
    wl = fluxes[0]
    sfcref = (sfcup/sfcdn)
    gp.plot_lines(
            domain=wl,
            ylines=[sfcref],
            labels=["Vegetation Surface Reflectance"],
            show=True,
            )
    #'''

    #flux_labels, fluxes = pkl.load(flux_file.open("rb"))

    ## offsets in boxed data from coordinate values??
    #geom_box = hg.around(sza=sza, phi=saa, uzen=vza, as_slice=True)
    #geom_point = hg.data(sza=sza, phi=saa, uzen=vza)
    #point_data = np.squeeze(geom_point)
    #box_data = np.squeeze(hg._data[*geom_box])

