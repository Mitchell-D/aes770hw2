"""
Script for generating lokup tables at a given wavelength.
The returned tables are shaped like: (sza, tau, cre, phi, uzen)
for each solar zenith angle, optical depth, cloud effective radius, relative
azimuth, and viewing zenith angle in the specified ranges.
"""
#from sbdart_info import default_params
from pathlib import Path
import numpy as np
import pickle as pkl
import zarr
#from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool

from krttdkit.acquire.sbdart import dispatch_sbdart, parse_iout
import krttdkit.visualize.guitools as gt

"""
For each wavelength, optical depth, and effective radius in the provided
lists, this method provides a 2d array of spectral radiances expected for
the viewing geometries specified by parameters (nphi, phi, nzen, uzen)

spectral radiances: (vza, raa, wl, aod)

:@param wl: Wavelength of the lookup table in um
:@param szas: List of solar zenith angles in degrees
:@param taus: List of optical depths
:@param cres: List of cloud effective radii
:@param sbdart_args: dict of SBDART arguments (If wlinf, wlsup, wlinc,
    tcloud, sza, zcloud, or nre are in the provided dict, they will be
    overwritten.)
"""
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
        iout:int, fields:str, coords:list, sbdart_args:dict, tmp_dir_parent:Path,
        description="", workers:int=3, zarr_file:Path=None,
        ):
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
    full_args = (({**sbdart_args, "iout":iout, **fields_coords[i]},
                  tmp_dirs[i])
            for i in range(len(fields_coords)))

    # Use a concurrent process pool to execute each
    #with ProcessPoolExecutor(max_workers=workers) as pool:
    #    lut_dicts = (pool.submit(_mp_sbdart, a) for a in args)
    sbdart_coords = None
    srad = None
    final_shape = None
    sflux_labels = None
    fluxes = []
    with Pool(workers) as pool:
        for sb_in,sb_out in pool.imap(_mp_sbdart, full_args):
            arg_combo,_ = sb_in
            print(f"Appending "+" ".join(
                [f"{f}:{arg_combo[f]}" for f in fields]))
            # Collect spectral radiances and spectral fluxes
            tmp_srad = sb_out["srad"]
            fluxes.append(sb_out["sflux"])
            if sbdart_coords is None:
                # Get the labels and coordinate arrays calculated by SBDART
                labels_coords = [(k,sb_out[k]) for k in ("uzen", "phi", "wl")]
                sflux_labels = sb_out["sflux_labels"]
                # Determine the ultimate lookup table shape
                final_shape = tuple([
                        *arg_space.shape,
                        *[len(coords) for label, coords in labels_coords],
                        ])
                # Chunk each output array separately
                chunk_shape = tuple([
                        *[1 for i in range(len(arg_space.shape))],
                        *final_shape[:len(labels_coords)]
                        ])
                # Create the zarr array, using the storage path if provided.
                store = zarr.ZipStore(zarr_file, mode="w")
                srad = zarr.create(
                        shape=final_shape,
                        chunks=chunk_shape,
                        dtype="f16",
                        store=store,
                        )
            srad.oindex[*coord_idxs.pop(0)] = tmp_srad
    sb_labels, sb_coords = map(list, zip(*labels_coords))
    srad.attrs = {
          "srad_coords":coords + sb_coords,
          "coord_labels":fields + sb_labels,
          "sfluxes":fluxes,
          "sflux_labels":sflux_labels,
          "args":sbdart_args,
          "desc":description,
          }
    return srad



def srad_over_field(
        field_label:str, args:list, sbdart_args:dict, tmp_dir:Path, sza:float,
        description="", print_stdout=False):
    """
    Dispatches SBDART to solve for spectral radiance and spectral fluxes
    at a series of boundary layer aerosol optical depths ("tbaer").
    """
    luts = []
    for a in args:
        sbdart_args[field_label] = a
        sb_out=dispatch_sbdart(sbdart_args, tmp_dir)
        tmp_lut = parse_iout(iout_id=5, sb_out=sb_out, print_stdout=False)
        luts.append((a, tmp_lut))

    coords, luts = zip(*luts)
    lut = np.stack([L["srad"] for L in luts], axis=-1)
    # spectral radiances: (vza, raa, wl, coord)
    axes = [(l,luts[0].get(l)) for l in ("uzen", "phi", "wl")]
    axes += (field_label, coords)
    clabels,coords = zip(*axes)
    out = {
            "srad":lut,
            "srad_coords":coords,
            "coord_labels":clabels,
            "sfluxes":[L["sflux"] for L in luts],
            "sflux_labels":[luts[0]["sflux_labels"]],
            "args":sbdart_args,
            "desc":description,
            }

if __name__=="__main__":
    tmp_dir = Path("buffer/sbdart")
    pkl_path = Path("buffer/lut.pkl")
    zarr_file = Path("data/aerosol_lut.zip")
    description = "Spectral response of several boundary layer AODs in DESIS' wave range; no atmospheric scattering (since L2 data); fixed solar zenith of 23.719deg; rural aerosol types; default aerosol profile"

    """
    The below dictionaries are sufficient for generating a lookup with shape:
    (sza, tau, cre, uzen, phi)
    """
    sbdart_args = {
            "iout":5,
            "idatm":2, # Mid-latitude summer
            "pbar":0, # no atmospheric scattering or absorption
            "isalb":7, # Ocean water
            #"sza":23.719,
            #"imoma":3, # henyey-greenstein if default 3

            #"iaer":1, # rural aerosol
            #"iaer":2, # urban aerosol
            #"iaer":3, # oceanic aerosol
            #"iaer":4, # tropo aerosol

            "nphi":8,
            "phi":"0,180",
            "nzen":20,
            "uzen":"0,85",
            "wlinf":.4,
            "wlsup":1.,
            "wlinc":0.002551,
            #"btemp":292,
            #"zcloud":2
            }
    print(srad_over_fields(
            zarr_file=zarr_file,
            fields=["sza", "tbaer", "iaer"],
            coords=[list(range(0, 70, 5)),
                  [i*0.025 for i in range(10)],
                  list(range(1,5))],
            iout=5,
            workers=10,
            sbdart_args=sbdart_args,
            tmp_dir_parent=tmp_dir,
            description=description,
            ).shape)

    #'''
