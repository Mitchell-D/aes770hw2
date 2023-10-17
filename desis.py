import json
import xmltodict
from pathlib import Path
from collections import namedtuple
from datetime import datetime
from itertools import chain
import rasterio as rio
import zarr
from multiprocessing import Pool

from krttdkit.visualize import guitools as gt
from krttdkit.operate import enhance as enh
from krttdkit.products import FeatureGrid
"""
DESIS granules are uniquely identitfied by a "data take" ID, a tile ID, and
a capture time. A granule corresponds to multiple files, generally including:
- HISTORY.xml
- METADATA.xml
- QL_IMAGE.tif
- QL_QUALITY-2.tif
- QL_QUALITY.tif
- SPECTRAL_IMAGE.hdr
- SPECTRAL_IMAGE.tif
"""
DESIS_Granule = namedtuple("DESIS_File", "take tile time substr")

def parse_granule(desis_file:Path):
    desis_str = "DESIS-HSI-L2A"
    assert desis_str in desis_file.name
    gran_fields = desis_file.name.split("-")[:5]
    _,_,_,take_tile,time = gran_fields
    substr = f"{desis_str}-{take_tile}-{time}"
    take, tile = map(int,take_tile[2:].split("_"))
    time = datetime.strptime(time,"%Y%m%dT%H%M%S")
    return DESIS_Granule(take, tile, time, substr)


def xml_to_dict(xml_file:Path):
    return xmltodict.parse(str(xml_file.open("r").read()))

def desis_dir_dict(desis_dir:Path):
    """
    returns a dictionary mapping a DESIS_File namedtuple to a list of Path
    objects corresponding to all valid data files associated with the
    unique DESIS_File image. This includes metadata, quality flag, and spectral
    data files.

    :@param desis_dir: directory path containing DESIS files
    :@return: dict mapping DESIS_File namedtuples to a list of corresponding
        files.
    """
    desis_str = "DESIS-HSI-L2A"
    files = [f for f in desis_dir.iterdir() if desis_str in f.name]
    #unique = set(tuple(f.name.split("-")[:5]) for f in files)
    granules = list(set(parse_granule(f) for f in files))
    return {g:[f for f in files if g.substr in f.name] for g in granules}
    '''
    granules = []
    for _,_,_,take_tile,time in unique:
        substr = f"{desis_str}-{take_tile}-{time}"
        take, tile = map(int,take_tile[2:].split("_"))
        time = datetime.strptime(time,"%Y%m%dT%H%M%S")
        granules.append(DESIS_Granule(take, tile, time, substr))
    return {g:[f for f in files if g.substr in f.name] for g in granules}
    '''

def get_desis_meta_info(metadata_xml:Path, get_rsrs:bool=False):
    """
    Parses a DESIS granule metadata XML into a meta-dict and list of info
    dicts suited for initializing a FeatureGrid with the DESIS data.

    The returned "meta" dict is serializable and contains information about the
    entire granule, and each dict in the "info" list contains information
    about the corresponding band.

    If get_rsrs is set to True, the spectral response of each band wrt
    wavelengths in nm is included in the band's info dict.

    :@param metadata_xml: DESIS granule XML labeled with "METADATA".
    :@param get_rsrs: If True, returns spectral response of each band along
        with a wavelength domain in nm.
    :@return: 2-tuple like (meta, info) such that "meta" is a dict containing
        information about the entire granule like the bounding box,
        acquisition time, etc, and "info" is a list of dicts corresponding
        to information about each band, including RSRs if get_rsrs is True.
    """
    xml_data = xml_to_dict(metadata_xml)
    o3val = int(xml_data["hsi_doc"]["processing"]["ozoneValue"])

    base = xml_data["hsi_doc"]["base"]
    specific = xml_data["hsi_doc"]["specific"]

    """ store the polygon as a dict of 2-tuple (lat,lon) points """
    polygon = list(
            tuple(point.items()) for point in
            base["spatialCoverage"]["boundingPolygon"]["point"])
    polygon = {label[1]:(lat[1],lon[1]) for label,lat,lon in polygon}
    # Convert the time range strings to datetime
    time_range = tuple(map(
        lambda t:datetime.strptime(t,"%Y-%m-%dT%H:%M:%S.%fZ").strftime("%s"),
        (base["temporalCoverage"]["startTime"],
         base["temporalCoverage"]["endTime"])
        ))

    scene = {
            "mean_elev":float(
                specific["terrain"]["meanGroundElevation"]),
            "mean_slope":float(
                specific["terrain"]["meanSlope"]),
            "mean_H2Ovap":float(
                specific["waterVapour"]["meanWaterVapour"]),
            "mean_aod":float(
                specific["meanAerosolOpticalThickness"]),
            "pct_haze":float(
                specific["haze"]["percentageHaze"]),
            "pct_cloud":float(
                specific["clouds"]["percentageClouds"]),
            "pct_cloudshadow":float(
                specific["cloudShadow"]["percentageCloudShadow"]),
            "pct_toposhadow":float(
                specific["topoShadow"]["percentageTopoShadow"]),
            }

    meta = {
            "polygon":polygon,
            "sza_saa":(float(specific["sunZenithAngle"]),
                       float(specific["sunAzimuthAngle"])),
            "via_vaa":(float(specific["sceneIncidenceAngle"]),
                       float(specific["sceneAzimuthAngle"])),
            "time_range":time_range,
            "sensor_altitude":float(base["altitudeCoverage"]),
            "scene":scene
            }

    # Collect band specific info into a list of dicts
    xml_bands, info = zip(*[
            (b, {
                "band":int(b["bandNumber"]),
                "wl":float(b["wavelengthCenterOfBand"]),
                "bandwidth":float(b["wavelengthWidthOfBand"]),
                "gain":float(b["gainOfBand"]),
                "offset":float(b["offsetOfBand"]),
                "dead_count":float(b["deadPixels"]),
                "sus_count":float(b["suspiciousPixel"]),
                })
            for b in [b for b in specific["bandCharacterisation"]["band"]]
            ])
    """
    If requested, parse the relative spectral response into each band's info
    """
    if get_rsrs:
        for i in range(len(info)):
            info[i]["rsr"] = (
                    tuple(map(float,xml_bands[i]["wavelengths"].split(","))),
                    tuple(map(float,xml_bands[i]["response"].split(","))))
    return meta, info

def get_desis_spectral(spectral_tif:Path, zarr_file:Path,
                       bands:list, meta:dict, info:list):
    """
    Extracts hyperspectral band data from a DESIS tif labeled "SPECTRAL_IMAGE"

    :@param spectral_tif: Path to the DESIS tif datafile.
    :@param bands: list of desired bands, by number and in order.
    :@param:
    """
    assert len(bands) and len(info)
    assert all(type(b) is int and 1<=b<=235 for b in bands)
    assert not zarr_file.exists() and zarr_file.parent.is_dir()
    band_idxs = [b-1 for b in bands]
    with rio.open(spectral_tif) as tif:
        ## Read the first band only in order to get the shap
        idx_0 = band_idxs[0]
        X_0 = tif.read(bands[0])*info[idx_0]["gain"]
        grid_shape = X_0.shape
        assert not zarr_file.exists()
        store = zarr.ZipStore(zarr_file, mode="w")
        ## Coords for each spatial dim, band, and 1 feature (srad)
        Z = zarr.create(shape=(*grid_shape,len(bands), 1), dtype="f16",
                        chunks=(*grid_shape,1,1), store=store)
        Z.oindex[:,:,0,0] = X_0
        new_info = [info[idx_0]]

        ## Read the /est of the bands
        for i in range(1, len(band_idxs)):
            idx = band_idxs[i]
            print(f"Loading band {idx+1}")
            Z.oindex[:,:,i,0] = tif.read(idx)*info[idx]["gain"]
            new_info.append(info[idx])
        ## Wavelength information goes in meta dict since it's a coordinate
        meta.update({"wl":info})
        Z.attrs["meta"] = meta
        Z.attrs["info"] = {}
        Z.attrs["flabels"] = ("srad",)
        Z.attrs["clabels"] = ["y", "x", "wl"]
        ## No lat/lon so spatial coordinates are just indeces
        Z.attrs["coord_arrays"] = (
                tuple(range(grid_shape[0])),
                tuple(range(grid_shape[1])),
                tuple(meta["wl"][i]["wl"] for i in range(len(meta["wl"]))),
                )
        store.close()

def get_desis_granule(files:list, zarr_file:Path, bands:list):
    meta, info = get_desis_meta_info(
            next(f for f in files if "METADATA" in f.name),
            get_rsrs=True,
            )
    get_desis_spectral(
            spectral_tif=next(
                f for f in files
                if "SPECTRAL_IMAGE" in f.name and f.suffix==".tif"),
            zarr_file=zarr_file,
            bands=bands,
            #bands=(28, 59, 104),
            meta=meta,
            info=info,
            )

def _mp_get_desis_granule(args):
    """
    Wrapper method for passing a kwarg dict to get_desis_granule
    within a multiprocessing context
    """
    return get_desis_granule(**args)

def get_desis_quality(quality_tif:Path, data_dir:Path=None):
    labels = [
            "shadow", "land_clear", "snow", "land_haze", "water_haze",
            "land_cloud", "water_cloud", "water_clear", "aod", "pwv"
            ]
    gran_fields = quality_tif.name.split("-")[:5]
    Q = []
    with rio.open(quality_tif) as tif:
        #Q = tif.read(i)
        Q = [tif.read(i) for i in range(1,11)]
    fg = FeatureGrid(labels=labels, data=Q)
    if not data_dir is None:
        fg.to_pkl(data_dir.joinpath("-".join(gran_fields)+"-QL2.pkl"))
    return fg

    for X in range(Q):
        tmp = enh.linear_gamma_stretch(X[i])
        print(enh.array_stat(X[i]))
        gt.quick_render(tmp)
    print(X.shape)



if __name__=="__main__":
    out_dir = Path("data")
    desis_root = Path("data/desis/")
    #desis_dirs = [p for p in desis_root.iterdir() if p.is_dir()]
    #desis_dirs = [desis_root.joinpath("smoke")]
    desis_dirs = [desis_root.joinpath("amazon")]
    #desis_dirs = [desis_root.joinpath("ocean")]
    #desis_dirs = [desis_root.joinpath("smoke_new")]
    desis_grans = list(chain(*[desis_dir_dict(d).items() for d in desis_dirs]))
    all_files = list(chain(*[p for _,p in desis_grans]))

    #'''
    """ Make a FeatureGrid for quality data """
    ql_files = [p for p in all_files if "QUALITY-2" in p.name]
    for ql_path in ql_files:
        get_desis_quality(ql_path,out_dir)
    #'''

    #'''
    """
    Make a list of arguments and save granules as Zarr directories
    containing attributes corresponding to the information needed to
    initialize a FeatureGrid object.
    """
    workers = 10
    bands = list(range(1,236))
    args = [{"files":files,
             "zarr_file":out_dir.joinpath(granule.substr+".zip"),
             "bands":bands,
            } for granule,files in desis_grans]
    with Pool(workers) as p:
        print(p.map(_mp_get_desis_granule, args))
    #'''

