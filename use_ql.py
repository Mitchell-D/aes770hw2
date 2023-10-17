import pickle as pkl
import zarr
import numpy as np
from pathlib import Path
from typing import Callable
from matplotlib import pyplot as plt

from krttdkit.operate import enhance as enh
from krttdkit.operate import recipe_book as rb
from krttdkit.operate.recipe_book import histgauss, gaussnorm
from krttdkit.visualize import guitools as gt
from krttdkit.visualize import geoplot as gp
from krttdkit.visualize import TextFormat as TF
from krttdkit.products import FeatureGrid

from HyperGrid import HyperGrid
from CoordSystem import CoordSystem, IntAxis, CoordAxis

_WL_RED = 660
#_WL_NIR = 990
#_WL_NIR = 850
_WL_NIR = 860
_WL_DEEPRED = 700
_WL_GREEN = 550

def _ddv_step(sub_tile:HyperGrid, L:HyperGrid, red_ub, cand_aod):
    """
    Return a boolean mask with True over dense dark vegetation

    :@param sub_tile: HyperGrid, preferably subsetted around the angles
    :@param L: HyperGrid containing aerosol lookup table
    :@param red_ub: Upper bound for red reflectance of DDV
    :@param cand_aod: A-priori candidate AOD (just for selecting DDV)
    """
    ## Ratio of vegetation index
    m_rvi = np.squeeze(
            sub_tile.data(wl=_WL_NIR) / sub_tile.data(wl=_WL_RED)
            ) >= 3
    ## Near-infrared upper and lower bounds
    m_nir_ub = np.squeeze(sub_tile.data(wl=_WL_NIR)) <= .25
    m_nir_lb = 0.1 <= np.squeeze(sub_tile.data(wl=_WL_NIR))
    ## Variable bound on red
    m_red = np.squeeze(sub_tile.data(wl=_WL_RED)) <= red_ub

    print(np.count_nonzero(m_rvi), np.count_nonzero(m_nir_ub),
          np.count_nonzero(m_nir_lb), np.count_nonzero(m_red))
    ## Combination of all masks is returned
    mask = np.logical_and(np.logical_and(np.logical_and(
        m_rvi, m_nir_lb), m_nir_ub), m_red)
    return mask

def _narrow(sub_tile:HyperGrid, L:HyperGrid, cur_mask, cand_aod):
    """
    Change the red reflectance threshold so that the percentage of DDV
    pixels is as close as possible to 5% of the total tile size.

    :@param sub_tile: HyperGrid, preferably subsetted around the angles
    :@param L: HyperGrid containing aerosol lookup table
    :@param cur_mask: Mask corresponding to all pixels that fulfill the
        base conditions in _ddv_step
    """
    print(f"Trying AOD={cand_aod}")
    ## Calculate the percentage of the mask that is DDV pixels
    ddv_pct = lambda m: np.count_nonzero(m)/sub_tile.size
    tmp_pct = ddv_pct(cur_mask)
    if tmp_pct <= .45:
        if tmp_pct <= .22:
            return cur_mask
        cur_mask = _ddv_step(sub_tile, L, red_ub=.03, cand_aod=cand_aod)
        tmp_pct = ddv_pct(cur_mask)
        if tmp_pct >= .05:
            return cur_mask
        return _ddv_step(sub_tile, L, red_ub=.035, cand_aod=cand_aod)
    else:
        cur_mask = _ddv_step(sub_tile, L, red_ub=.03, cand_aod=cand_aod)
        tmp_pct = ddv_pct(cur_mask)
        if tmp_pct <= .22:
            if tmp_pct >= .05:
                return cur_mask
            return _ddv_step(sub_tile, L, red_ub=.035, cand_aod=cand_aod)
        cur_mask = _ddv_step(sub_tile, L, red_ub=.025, cand_aod=cand_aod)
        tmp_pct = ddv_pct(cur_mask)
        if tmp_pct >= .05:
            return cur_mask
        return _ddv_step(sub_tile, L, red_ub=.03, cand_aod=cand_aod)

def _mask_ddv(sub_tile:HyperGrid, L:HyperGrid):
    """
    Use the methodology (Richter, 2006) to select dense dark vegetation,
    returning a boolean mask the same size as the provided tile.
    """
    ## Calculate the percentage of the mask that is DDV pixels
    ddv_pct = lambda m: np.count_nonzero(m)/sub_tile.size
    ## Get boolean masks for moderate and high visibility
    modvis = _ddv_step(sub_tile, L, red_ub=.04, cand_aod=.25)
    hivis = _ddv_step(sub_tile, L, red_ub=.04, cand_aod=.1)
    is_modvis = np.count_nonzero(modvis)>np.count_nonzero(hivis)
    #print(ddv_pct(modvis))
    if ddv_pct(modvis)<.05 and  ddv_pct(hivis)<.05:
        return np.full_like(hivis, False)
    ## Get boolean masks for moderate and high visibility
    cand_mask = _narrow(
            sub_tile, L, [hivis,modvis][is_modvis], [.1,.25][is_modvis])
    if ddv_pct(cand_mask) >= .05:
        return cand_mask
    lovis = _ddv_step(sub_tile, L, red_ub=.04, cand_aod=.8)
    cand_mask = _narrow(sub_tile, L, lovis, .8)
    if ddv_pct(lovis) >= .05:
        return lovis
    return np.full_like(lovis, False)

def mask_ddv(tile:HyperGrid, aero_lut:HyperGrid, atype:int=1):
    ## Extract the viewing angles
    sza,saa = map(float, tile.meta("sza_saa"))
    vza,vaa = map(float, tile.meta("via_vaa"))
    raa = abs(vaa-saa)
    ## Restrict the LUT to values surrounding the viewing angles
    aero_lut = aero_lut.around(sza=sza, phi=raa, uzen=vza, as_hypergrid=True)
    ## Restrict the LUT to use one aerosol type
    aero_lut = aero_lut.subgrid(iaer=atype)
    #for c,x in zip(aero_lut._clabels, aero_lut._coords):
    #    print(c,x)
    #Y = np.full(tile.shape[:2], 1 if np.any(mask) else 0)
    return _mask_ddv(tile, aero_lut)

def mask_ddv_simple(RED:HyperGrid, NIR:HyperGrid, GREEN:HyperGrid, margin=.2):
    #for aod in aero_lut.coords("tbaer").coords:
    ## Ratio of vegetation index
    red = np.squeeze(RED.data())
    nir = np.squeeze(NIR.data())
    green = np.squeeze(GREEN.data())

    m_rvi = np.squeeze(nir/red) >= 4
    ## Near-infrared upper and lower bounds
    #m_nir_ub = np.squeeze(nir) <= .28
    m_nir_ub = np.squeeze(nir) <= .28
    m_nir_lb = 0.1 <= np.squeeze(nir)
    ## Variable bound on red
    m_red = red <= .04
    m_ndvi = (nir-red)/(nir+red) > .6
    m_red_lb = red > .0
    if np.any((green-nir)/(green+nir) > -.6):
        return np.full_like(m_rvi, False)

    '''
    m_rvi = np.squeeze(nir/red) >= 4
    ## Near-infrared upper and lower bounds
    m_nir_ub = np.squeeze(nir) <= .3
    m_nir_lb = 0.1 <= np.squeeze(nir)
    ## Variable bound on red
    m_red = np.squeeze(red) <= .08
    m_ndvi = np.squeeze((nir-red)/(nir+red)) > .6
    '''

    ## Combination of all masks is returned
    #mask = np.logical_and(np.logical_and(np.logical_and(
    #    m_rvi, m_nir_lb), m_nir_ub), m_red)
    mask = np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(m_rvi, m_red), m_nir_lb), m_nir_ub), m_ndvi), m_red_lb)
    ideal_num = int(margin*mask.size)
    if np.count_nonzero(mask) > ideal_num:
        newmask = np.full_like(m_rvi, False)
        idxs = np.where(mask)
        refs = np.squeeze(RED.data()[mask])
        newcoords = [p[1:] for p in sorted(zip(
            refs, *idxs), key=lambda t:t[0])[:ideal_num]]
        for c in newcoords:
            newmask[*c] = True
        return newmask
    return mask


def show_all_quality(data_dir:Path, field="aod", hrange=None, vrange=None,
                     show=True, fig_dir:Path=None, substr="", mask=None):
    """
    quick-renders all AOD fields for a directory with DESIS quality
    FeatureGrid pkls, notated by substring "QL2.pkl"
    """
    ql_files = [p for p in data_dir.iterdir() if "QL2.pkl" in p.name]
    for f in [q for q in ql_files if substr in q.name]:
        tmp_fg = FeatureGrid.from_pkl(f)
        if not vrange is None:
            tmp_fg = tmp_fg.subgrid(vrange=vrange)
        if not hrange is None:
            tmp_fg = tmp_fg.subgrid(hrange=hrange)
        print()
        print(f"\n{f.name}")
        print(enh.array_stat(tmp_fg.data(field)))
        print(enh.array_stat(tmp_fg.data(field)*.006))
        #print(tmp_fg.data(field).dtype)
        #tmp_mask = tmp_fg.data(field) == np.amax(tmp_fg.data(field))
        #X = tmp_fg.data(field, mask=tmp_mask,
        #                mask_value=np.amax(tmp_fg.data(field)))
        X = tmp_fg.data(field)
        rgb = gt.scal_to_rgb(X)
        #rgb[tmp_mask] = 0
        if show:
            gt.quick_render(rgb)
        if fig_dir:
            fname = "-".join(f.name.split("-")[:5])+f"_{field}.png"
            gp.generate_raw_image(rgb, fig_dir.joinpath(fname))

def gen_hg_tiles(hg:HyperGrid, tile:dict):
    """
    Returns a generator that tiles over a HyperGrid's shape with a kernel
    mapping string coordinate labels to a rooted IntAxis range

    :@return: generator for combos and tuple showing final shape
    """
    '''
    assert all(k for k in tile.keys() in hg.clabels)
    t_labels,t_axes = list(zip(*tile.items()))
    t_shape = np.array([s.size for s in t_axes])
    tile_slices = []
    for cl in hg.clabels:
        if cl not in t_labels:
            tile_slices.
    '''

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

def retrieve_aod(RED, NIR, ddv, srad, flux, iaer=1, DEEPRED=None):
    if not np.any(ddv):
        return np.squeeze(np.full(RED.shape, np.nan, dtype=float))
    ## Insolation amounts per optical depth
    red_inso = np.squeeze(flux.data("botdir", wl=_WL_RED/1000, iaer=iaer))[0]
    nir_inso = np.squeeze(flux.data("botdir", wl=_WL_NIR/1000, iaer=iaer))[0]
    deepred_inso = np.squeeze(flux.data(
        "botdir", wl=_WL_DEEPRED/1000, iaer=iaer))[0]
    red_srad = srad.subgrid(wl=_WL_RED/1000, iaer=iaer)
    nir_srad = srad.subgrid(wl=_WL_NIR/1000, iaer=iaer)

    #nir_ref = np.average(NIR.data()[ddv])
    #red_ref = np.average(RED.data()[ddv])
    #red_ref = nir_ref*.095
    if DEEPRED is None:
        nir_ref = NIR.data()[ddv]
        nir_ref = np.average(nir_ref) - np.std(nir_ref)
        red_ref = nir_ref*.1
    else:
        '''
        red = srad.data(wl=_WL_RED/1000,iaer=1,tbaer=0)/red_inso
        deepred = srad.data(wl=_WL_DEEPRED/1000,iaer=1)/deepred_inso
        print(np.average(deepred/red, axis=(0,2,3,4)),
              np.std(deepred/red, axis=(0,2,3,4)))
        print(np.average(deepred/red))
        '''
        nir_ref = DEEPRED.data()[ddv]
        nir_ref = np.average(nir_ref)
        red_ref = .2584*nir_ref


    #obs_diff = nir_ref*.1-nir_ref
    #model_diff = red_srad.data()/red_inso-nir_srad.data()/nir_inso
    #print(obs_diff, np.squeeze(model_diff))
    #print(model_diff.shape)
    #obs_ratio = red_ref/nir_ref
    #model_ratio = (red_srad.data()/red_inso)/(nir_srad.data()/nir_inso)
    #print(obs_ratio, np.squeeze(model_ratio))
    #print(model_ratio)

    #obs_red_rad = red_ref*red_inso
    #obs_nir_rad = red_ref*red_inso

    ## interpolate between angles
    #sza1,sza2 = tuple()
    #print(red_srad.clabels)
    #print(red_srad.shape)
    ## Extract viewing angles

    #'''
    sza,saa = map(float, hg.meta("sza_saa"))
    uzen,vaa = map(float, hg.meta("via_vaa"))
    phi = abs(vaa-saa)
    sza0,sza1 = tuple(red_srad.coords("sza").coords)
    uzen0,uzen1 = tuple(red_srad.coords("uzen").coords)
    phi0,phi1 = tuple(red_srad.coords("phi").coords)
    drad_dsza = (red_srad.data(sza=sza1, uzen=uzen0, phi=phi0) - \
            red_srad.data(sza=sza0, uzen=uzen0, phi=phi0)) / (sza1-sza0)
    dsza = sza-sza0
    drad_duzen = (red_srad.data(sza=sza0, uzen=uzen1, phi=phi0) - \
            red_srad.data(sza=sza0, uzen=uzen0, phi=phi0)) / (uzen1-uzen0)
    duzen = uzen-uzen0
    drad_dphi = (red_srad.data(sza=sza0, uzen=uzen0, phi=phi1) - \
            red_srad.data(sza=sza0, uzen=uzen0, phi=phi0)) / (phi1-phi0)
    dphi = phi-phi0
    #'''
    ## Account for incremental changes due to viewing geometry
    ref_step = (drad_dsza*dsza+drad_duzen*duzen+drad_dphi*dphi)
    ref0 = red_srad.data(sza=sza0,phi=phi0,uzen=uzen0)
    ref_per_tau = np.squeeze(ref0 + ref_step) / red_inso

    ## Reflectances averaged over viewing angles
    ref_per_tau_n = np.squeeze(np.average(red_srad.data(), axis=(0,3,4)))
    ref_per_tau_n /= red_inso

    #print(ref_per_tau, ref_per_tau_n)

    ## naive method
    #tau = srad.coords("tbaer").coords[np.argmin(np.abs(ref_per_tau-red_ref))]

    if red_ref >= np.amax(ref_per_tau):
        return np.squeeze(np.full(RED.shape, np.nan, dtype=float))
    if red_ref < np.amin(ref_per_tau):
        return np.squeeze(np.full(RED.shape, np.nan, dtype=float))
    #print(red_ref, ref_per_tau)
    tau_idx_0, tau_idx_1, _ = CoordAxis(ref_per_tau).around(red_ref).as_tuple
    tau0 = red_srad.coords("tbaer").coords[tau_idx_0]
    tau1 = red_srad.coords("tbaer").coords[tau_idx_1]
    dtau_dref = (tau1-tau0) / (ref_per_tau[tau_idx_1]-ref_per_tau[tau_idx_0])
    dref = red_ref - ref_per_tau[tau_idx_0]
    tau = tau0 + dtau_dref * dref

    #red_srad = red_srad.subgrid(tbaer=tau)

    #print(np.argmin(ref_per_tau-nir_ref*.1))
    #print(ref_per_tau-red_ref)
    #print(red_ref, ref_per_tau)
    #print(f"TAU: {tau}\n")
    return np.squeeze(np.full(RED.shape, tau, dtype=float))

def validation_curve(A, B, fig_path=None, nbin=128, show=False, plot_spec={}):
    X = enh.linear_gamma_stretch(np.stack((np.ravel(A),np.ravel(B)), axis=1))
    X = np.rint(X*(nbin-1)).astype(int)
    rmse = np.sqrt(np.sum((X[:,0]-X[:,1])**2)/X[:,0].size)
    #print(f"RMSE: {rmse}")
    V = np.zeros((nbin,nbin))
    for i in range(X.shape[0]):
        V[*X[i]] += 1
    gp.plot_heatmap(V, fig_path=fig_path, plot_spec=plot_spec,
                    show_ticks=False, show=show, plot_diagonal=True)

def plot_hists(arrays:list, labels:list=None, nbins=256):
    plt.clf()
    hists,coords = zip(*[enh.get_nd_hist([a], bin_counts=nbins)
                         for a in arrays])
    if labels is None:
        labels = [None for i in range(len(coords))]
    for i in range(len(coords)):
        plt.plot(np.squeeze(coords[i]), hists[i], label=labels[i])
    plt.legend()
    plt.title("DESIS Reflectance Histograms Over Dense Dark Vegetation")
    plt.ylabel("Count")
    plt.xlabel("Reflectance")
    plt.show()

if __name__=="__main__":
    data_dir = Path("data")
    ## hg_path = Path("data/.zip")
    ## ql_path = Path("data/-QL2.pkl")
    ## san fran
    ## ocean
    #hg_path = Path("data/DESIS-HSI-L2A-DT0314290188_005-20190503T164053.zip")
    #ql_path = Path("data/DESIS-HSI-L2A-DT0314290188_005-20190503T164053-QL2.pkl")
    ## smoke
    #hg_path = Path("data/DESIS-HSI-L2A-DT0865788448_014-20230607T153935.zip")
    #ql_path = Path("data/DESIS-HSI-L2A-DT0865788448_014-20230607T153935-QL2.pkl")
    #hg_path = Path("data/DESIS-HSI-L2A-DT0865808324_039-20230607T220625.zip")
    #ql_path = Path("data/DESIS-HSI-L2A-DT0865808324_039-20230607T220625-QL2.pkl")
    #hg_path = Path("data/DESIS-HSI-L2A-DT0865808324_036-20230607T220625.zip")
    #ql_path = Path("data/DESIS-HSI-L2A-DT0865808324_036-20230607T220625-QL2.pkl")
    #hg_path = Path("data/DESIS-HSI-L2A-DT0865808324_037-20230607T220625.zip")
    #ql_path = Path("data/DESIS-HSI-L2A-DT0865808324_037-20230607T220625-QL2.pkl")
    ## amazon
    #hg_path = Path("data/DESIS-HSI-L2A-DT0726511704_011-20220526T170133.zip")
    #ql_path = Path("data/DESIS-HSI-L2A-DT0726511704_011-20220526T170133-QL2.pkl")

    hg_path = Path("data/DESIS-HSI-L2A-DT0903201448_001-20200903T144653.zip")
    ql_path = Path("data/DESIS-HSI-L2A-DT0903201448_001-20200903T144653-QL2.pkl")
    #hg_path = Path("data/DESIS-HSI-L2A-DT0903201448_002-20200903T144653.zip")
    #ql_path = Path("data/DESIS-HSI-L2A-DT0903201448_002-20200903T144653-QL2.pkl")
    #hg_path = Path("data/DESIS-HSI-L2A-DT0903201448_003-20200903T144653.zip")
    #ql_path = Path("data/DESIS-HSI-L2A-DT0903201448_003-20200903T144653-QL2.pkl")
    #hg_path = Path("data/DESIS-HSI-L2A-DT0903201448_004-20200903T144653.zip")
    #ql_path = Path("data/DESIS-HSI-L2A-DT0903201448_004-20200903T144653-QL2.pkl")
    #hg_path = Path("data/DESIS-HSI-L2A-DT0903201448_005-20200903T144653.zip")
    #ql_path = Path("data/DESIS-HSI-L2A-DT0903201448_005-20200903T144653-QL2.pkl")
    fig_dir = Path("figures")

    srad_path = Path("data/lut_aero_veg-noatmo_srad.zip")
    flux_path = Path("data/lut_aero_veg-noatmo_sflux.pkl")

    vrange=(400,1000)
    hrange=(400,1000)
    #hrange=(400,1000)
    #vrange=(300,900)
    #hrange=None
    #vrange=None

    srad = HyperGrid.from_store(srad_path)
    flux_labels, fluxes = pkl.load(flux_path.open("rb"))
    #sfcdn = fluxes[0][-1]
    fluxes = np.dstack(fluxes)
    #print(flux_labels)
    #print(fluxes[-1,50])
    ## (wl, sza, tbaer, iaer, flux)
    #fluxes = fluxes.reshape(8, 121, 14, 12, 4)
    #fluxes = fluxes.transpose((1,2,3,4,0))
    fluxes = fluxes.reshape(8, 121, 4, 12, 14)
    fluxes = fluxes.transpose((1,4,3,2,0))
    flux_clabels = ("wl", "sza", "tbaer", "iaer")
    flux_coords = [srad.coords(f).coords for f in flux_clabels]
    flux = HyperGrid(
            data=fluxes,
            flabels=flux_labels,
            clabels=flux_clabels,
            coord_arrays=flux_coords,
            )
    ##  Load and subset the DESIS data and quality flags
    hg = HyperGrid.from_store(hg_path)
    hg = hg.subgrid(x=IntAxis(hrange), y=IntAxis(vrange))
    fg = FeatureGrid.from_pkl(ql_path)
    fg = fg.subgrid(vrange=vrange, hrange=hrange)

    ## Get a mask for out-of-bounds values
    fg.add_data("oob", np.squeeze(hg.data(wl=500)==np.amin(hg.data(wl=500))))

    '''
    ## Show all quality masks and their stats within constraints
    model_fig_dir = Path("figures/qlmask")
    show_all_quality(data_dir, field="aod", hrange=hrange, vrange=vrange,
            substr="20200903",show=False,
            fig_dir=model_fig_dir,mask=fg.data("oob"))
    '''


    #'''
    ## Generate an image of the AOD mask from the quality file
    val_aod = fg.data("aod")
    val_aod[fg.data("oob")] = np.nanmin(val_aod)
    val_aod[np.where(np.isnan(fg.data("oob")))] = np.nanmin(val_aod)
    #gt.quick_render(gt.scal_to_rgb(val_aod))
    gp.generate_raw_image(
            enh.norm_to_uint(gt.scal_to_rgb(val_aod), 256, np.uint8),
            fig_dir.joinpath(Path(hg_path.name.replace(".zip","-val.png"))))
    #'''

    """
    Iterate over a kernel tiled over the image, running the retrieval algorithm
    on each npx x npx set of pixels
    """
    ## Extract the viewing angles
    sza,saa = map(float, hg.meta("sza_saa"))
    vza,vaa = map(float, hg.meta("via_vaa"))
    raa = abs(vaa-saa)
    print(f"sza: {sza}, raa: {raa}, vza: {vza}")
    ## Restrict the LUT to values surrounding the viewing angles
    srad = srad.around(sza=sza, phi=raa%180, uzen=vza, as_hypergrid=True)
    ## Restrict the LUT to use one aerosol type
    flux = flux.subgrid(sza=sza, as_hypergrid=True)

    ## side length of tile
    npx = 10
    kernel = {"x":IntAxis((0,npx)), "y":IntAxis((0,npx))}

    '''
    #Y = np.zeros((hg.shape[0]//npx, hg.shape[1]//npx))
    TMP = np.full((npx*hg.shape[0]//npx, npx*hg.shape[1]//npx), 0, dtype=float)
    for combo,tmp_hg in gen_hg_tiles(hg, kernel):
        slices = [a.as_slice for a in combo]
        red = np.squeeze(tmp_hg.subgrid(wl=_WL_RED).data())
        red[np.where(np.isnan(red))] = 0
        red[np.where(red<-1)] = 0
        TMP[*slices[::-1]] = red
        #print(TMP[*slices[::-1]], slices)
    #print(enh.array_stat(TMP))
    #exit(0)
    '''

    """ Do the retrieval """

    MDDV = np.full(hg.shape[:2], False)
    AOD = np.zeros(hg.shape[:2])
    for combo,tmp_hg in gen_hg_tiles(hg, kernel):
        RED = tmp_hg.subgrid(wl=_WL_RED)
        NIR = tmp_hg.subgrid(wl=_WL_NIR)
        GREEN = tmp_hg.subgrid(wl=_WL_GREEN)
        DEEPRED = tmp_hg.subgrid(wl=_WL_DEEPRED)
        slices = [a.as_slice for a in combo]
        tile_mddv = mask_ddv_simple(RED, NIR, GREEN, margin=.05)
        #tile_mddv = mask_ddv(tmp_hg, srad, 1)
        MDDV[*slices[::-1]]  = tile_mddv
        AOD[*slices[::-1]] = retrieve_aod(RED, NIR, tile_mddv, srad, flux,
                                          iaer=1,)#DEEPRED=DEEPRED)

    #gt.quick_render(AOD)
    valid_aod = np.logical_not(np.isnan(AOD))
    AOD[np.logical_not(valid_aod)] = 0
    print(TF.BLUE(f"Average AOD:    {np.average(AOD[valid_aod])}"))
    print(enh.array_stat(AOD[valid_aod]))
    print(TF.BLUE(f"Validation AOD: {hg.meta('scene')['mean_aod']}"))

    #'''
    ## Use the DESIS data to generate a truecolor with DDV pixels masked
    truecolor = np.dstack([np.squeeze(hg.data(wl=wl))
                           for wl in (640, 550, 460)])
    truecolor[fg.data("oob")] = 0
    truecolor = enh.linear_gamma_stretch(gaussnorm(truecolor))
    gp.generate_raw_image(
            enh.norm_to_uint(truecolor,256,np.uint8),
            fig_dir.joinpath(hg_path.name.replace(".zip","-tc.png")))
    rgb_ddv = np.copy(truecolor)
    rgb_ddv[MDDV] = np.array([1, 0, 0])
    #gt.quick_render(rgb_ddv)
    gp.generate_raw_image(
        enh.norm_to_uint(rgb_ddv,256,np.uint8),
        fig_dir.joinpath(hg_path.name.replace(".zip","-ddv.png")))
    ## Generate a truecolor with the scaled AOD values.
    tmp_aod = gt.scal_to_rgb(AOD)
    truecolor[valid_aod] = tmp_aod[valid_aod]
    #gt.quick_render(truecolor)
    gp.generate_raw_image(
        enh.norm_to_uint(np.clip(gaussnorm(truecolor),-1,1), 256, np.uint8),
        fig_dir.joinpath(Path(hg_path.name.replace(".zip", "-aod.png"))))
    #'''

    """
    Plot the spectral response of the DDV pixels
    """
    ddv = np.squeeze(hg.data()[MDDV])
    gp.stats_1d(
            {"DDV":{"means":np.average(ddv,axis=0),
                    "stdevs":np.std(ddv,axis=0)}},
            band_labels=hg.coords("wl").coords,
            show=False,
            plot_spec={
                "xlabel":"Wavelength (nm)",
                "ylabel":"Reflectance",
                "title":"Spectral Response of Dense Dark Vegetated Pixels",
                "title_size":26,
                "label_size":20,
                "legend_font_size":20,
                "yrange":(0,.5),
                },
            fig_path=fig_dir.joinpath(
                Path(hg_path.name.replace(".zip", "-ddvspec.png"))),
            )

    """ Plot histograms of DDV curves """
    hist_bands = [450, 500, 550, 600, 660, 700, 750, 800, 850, 900, 950]
    ddv_spectra = [np.squeeze(hg.data(wl=b))[MDDV] for b in hist_bands]
    plot_hists(
            arrays=ddv_spectra,
            labels=[f"{b} nm" for b in hist_bands]
            )

    """ Plot validation curves """
    print(enh.array_stat(fg.data("aod")[valid_aod]/100))
    print(enh.array_stat(AOD[valid_aod]))
    validation_curve(AOD[valid_aod], fg.data("aod")[valid_aod]/100,
                     fig_path=None,
                     nbin=256,
                     show=False,
                     plot_spec={
                         "imshow_norm":"log"
                         }
                     )
