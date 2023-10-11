""" """
#from sbdart_info import default_params
from pathlib import Path
import numpy as np
import pickle as pkl
import zarr
#from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool

from krttdkit.acquire.sbdart import dispatch_sbdart, parse_iout
import krttdkit.visualize.guitools as gt

def _mp_sbdart(args):
    """
    Run SBDART given a temporary directory, and a dict of SBDART style
    arguments that defines an output type (key "iout").

    Return the output from krttdkit.acquire.sbdart.parse_iout given the args.

    This method always removes its temporary directory, even if it errors out.

    :@param args: tuple like (sbdart_args:dict, tmp_dir:Path) identical to
        the parameters of krttdkit.acquire.sbdart.dispatch_sbdart. It is
        enforced that tmp_dir doesn't exist, and that a specific output style
        "iout" is provided.
    :@param return: 2-tuple like (args, output) where args is identical to the
        provided arguments (for asynchronous processing posterity), and output
        is from krttdkit.acquire.sbdart.parse_iout if the model was effectively
        executed. The format of the output depends on the iout specified in
        the sbdart_args dictionary.
    """
    sbdart_args, tmp_dir = args
    assert not tmp_dir.exists()
    assert tmp_dir.parent.is_dir()
    assert "iout" in sbdart_args.keys()
    try:
        return (args, parse_iout(
            sb_out=dispatch_sbdart(sbdart_args, tmp_dir),
            iout_id=sbdart_args["iout"], print_stdout=False))
    except Exception as E:
        raise E
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


def srad_over_fields(
        fields:str, coords:list, tmp_dir_parent:Path, sbdart_args:dict={},
        zarr_file:Path=None, sflux_file:Path=None, description="",
        workers:int=3):
    """
    Method to calculate spectral radiance (and layerwise spectral fluxes)
    by iterating over one or more SBDART parameters, in addition to the
    three default coordinates.

    This method is heavily multiprocessed, and can handle lookup tables that
    are larger than the available memory! If you provide a ".zip" zarr_file
    path, the compressed array will be updated on-disc as the processes
    return.

    The lookup table will start with dimensions corresponding to the order of
    SBDART parameter names in the fields list. Each dimension will have an
    element for each coordinate associated with it.

    sbdart args is a dictionary of arguments to pass to sbdart, which determine
    radiative characteristics and aspects of the returned lookup table size.

    The default coordinates are always the last 3 in the returned lookup table,
    and follow the order...
     1: viewing zenith angle (uzen, nzen)
     2: relative azimuth angle (phi, nphi)
     3: wavelength (wlinf, wlsup, wlinc)

    :@param fields: List of string names of SBDART parameters corresponding to
        the coordinate values in the same-index element of the "coords" list.
    :@param coords: list of lists enumerating argument values for each field.
        The length of each coordinate list is identical to the size of the
        corresponding dimension in the returned lookup table.
    :@param tmp_dir_parent: Directory where hashed temporary directories can
        be written to during SBDART dispatch.
    :@param sbdart_args: Dictionary mapping SBDART argument strings to user-
        -selected values. See deafults in krttdkit.acquire.sbdart docs.
    :@param zarr_file: Path to a zip file to write the zarr archive to. If the
        lookup table would be too big to keep in memory, provide a zarr file
        and it will be offloaded to the disc as each increment is calculated!
    :@param sflux_file: pkl path where fluxes can be written with their labels.
    :@param description: String description of the LUT run to include in the
        returned (stored) zarr grid attributes.
    :@param workers: Number of parallel processes to dispatch while calculating
        lookup tables.
    """
    assert len(fields)==len(coords)
    dims = tuple(len(f) for f in fields)
    # Get a list of all argument combinations in argument space, ie all
    # combinations of (i, j, k, l, ...) indeces of each 'coords' list, in order
    arg_space = np.full([len(coords[i]) for i in range(len(fields))], True)
    coord_idxs = list(zip(*np.where(arg_space)))
    # Make a SBDART argument dictionary with each combination
    fields_coords = [
            {fields[i]:coords[i][aidx[i]] for i in range(len(aidx))}
            for aidx in coord_idxs
            ]
    no_repeat_rint = np.random.default_rng().choice(
            int(1e6), size=len(fields_coords), replace=False)
    tmp_dirs = [tmp_dir_parent.joinpath(f"tmp_{hash(str(abs(x)))}")
                for x in no_repeat_rint]
    # Add default args. At least as of Python 3.10.9, the rightmost
    # elements in a dict expansion get priority.
    full_args = (({**sbdart_args, "iout":5, **fields_coords[i]},
                  tmp_dirs[i])
            for i in range(len(fields_coords)))

    # Use a concurrent process pool to execute each
    #with ProcessPoolExecutor(max_workers=workers) as pool:
    #    lut_dicts = (pool.submit(_mp_sbdart, a) for a in args)
    sbdart_coords = None
    srad = None
    final_shape = None
    sflux_labels = None
    sfluxes = []
    # Create the zarr array, using the storage path if provided.
    store = zarr.ZipStore(zarr_file.as_posix(),mode="w") if zarr_file else None
    try:
        with Pool(workers) as pool:
            for sb_in,sb_out in pool.imap(_mp_sbdart, full_args):
                arg_combo,_ = sb_in
                print(f"Appending "+" ".join(
                    [f"{f}:{arg_combo[f]}" for f in fields]))
                # Collect spectral radiances and spectral fluxes
                tmp_srad = sb_out["srad"]
                sfluxes.append(sb_out["sflux"])
                if srad is None:
                    # Get the labels and coordinate arrays calculated by SBDART
                    labels_coords = [(k,sb_out[k]) for k in
                                     ("uzen", "phi", "wl")]
                    sflux_labels = sb_out["sflux_labels"]
                    ## Determine the ultimate lookup table shape. Ultimately,
                    ## there is an extra hanging dimension for features.
                    final_shape = tuple([
                            *arg_space.shape,
                            *[len(coords) for label, coords in labels_coords],
                            1,
                            ])
                    # Chunk each output array separately
                    chunk_shape = tuple([
                            *[1 for i in range(len(arg_space.shape))],
                            *final_shape[:len(labels_coords)]
                            ])
                    srad = zarr.creation.create(
                            shape=final_shape,
                            chunks=chunk_shape,
                            store=store,
                            mode="w",
                            dtype="f16",
                            )
                    # Add all the general information to the attributes
                    sb_labels, sb_coords = map(list, zip(*labels_coords))
                    srad.attrs.update({
                          "flabels":["srad"],
                          "clabels":fields + sb_labels,
                          "coord_arrays":list(map(list, coords + sb_coords)),
                          "meta":{
                              "sbdart_args":dict(sbdart_args),
                              "desc":description,
                              },
                          })
                # Update the array and save it to storage
                srad.oindex[*coord_idxs.pop(0)] = np.expand_dims(tmp_srad,-1)
                if not store is None:
                    store.flush()
        sfluxes_all = (sflux_labels, sfluxes)
    except Exception as E:
        raise E
    finally:
        if not zarr_file is None:
            store.close()
    if not sflux_file is None:
        pkl.dump(sfluxes_all, sflux_file.open("wb"))
    return srad, sfluxes_all

if __name__=="__main__":
    tmp_dir = Path("buffer/sbdart")
    #zarr_file = Path("data/aerosol_lut.zip")
    #flux_file = Path("data/aerosol_sfluxes.pkl")
    zarr_file = Path("buffer/tmp_lut.zip")
    flux_file = Path("buffer/tmp_flux.pkl")
    description = "Spectral response of several boundary layer AODs in DESIS' wave range; no atmospheric scattering (since L2 data); fixed solar zenith of 23.719deg; rural aerosol types; default aerosol profile"

    """
    """
    sbdart_args = {
            "idatm":2, # Mid-latitude summer
            "pbar":0, # no atmospheric scattering or absorption
            #"isalb":7, # Ocean water
            "isalb":0, # User specified albedo
            #"albcon":0, # No reflection
            "nphi":8,
            "phi":"0,180",
            "nzen":20,
            "uzen":"0,85",
            "wlinf":.4,
            "wlsup":1.,
            "wlinc":0.005,
            #"wlinc":0.1,
            "iaer":1,
            }
    """
    Iterate over solar zenith angle, boundary layer aerosol optical depth,
    and aerosol types
    """
    srad, sfluxes = srad_over_fields(
            zarr_file=zarr_file,
            sflux_file=flux_file,
            #fields=["sza", "tbaer", "iaer"],
            fields=["sza", "tbaer", "albcon"],
            coords=[list(range(0, 70, 5)),
                  [0, 0.05, 0.01, 0.25, 0.5, 0.75, 1, 2, 5, 10],
                  [0, .2,.4,.6,.8,1]],
            #coords = [[0], [0, .005, .05, .5], [1, 3]],
            workers=10,
            sbdart_args=sbdart_args,
            tmp_dir_parent=tmp_dir,
            description=description,
            )
    print(srad.shape)
