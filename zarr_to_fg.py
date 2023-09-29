
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

def animate_pngs(label:str, frames_dir, out_path:Path, framerate:int=15):
    """
    Given a directory with png files following a {label}_{order_str}.png
    format with unique identifying string labels, and globbable order strings,
    use ffmpeg to generate an animation (ie mp4, gif) from all matching
    PNGs.

    :@param label: String label matching the first underscore-separated field
        of all png files to incorporate into the animation
    :@param frames_dir: Directory containing images to animate
    :@param out_path: Full path to a video file output.
    :@param framerate: Animation framerate

    :@return: ffmpeg subprocess communication object
    """
    ffmpeg_cmd = f"ffmpeg -hide_banner -loglevel error -f image2 "+ \
            f"-r {framerate} -pattern_type glob -i " + \
            f"'{frames_dir.as_posix()}/{label}_*.png' -vb 140M {out_path}"
    proc = Popen(shlex.split(ffmpeg_cmd), stdout=PIPE)
    return proc.communicate()

def zarr_to_fg(Z:zarr.core.Array, bands:list=None,
               vrange:tuple=None, hrange:tuple=None):
    """
    Return the Zarr array as a FeatureGrid, probably with a subset of the
    current shape.
    """
    zarr_info, meta = Z.attrs["info"], Z.attrs["meta"]
    zarr_labels = [str(inf["band"]) for inf in zarr_info]
    bands = zarr_labels if bands is None else bands
    idxs = [zarr_labels.index(str(b)) for b in bands]
    return FeatureGrid(
            labels=[f"b{int(b):03}" for b in bands],
            data=[np.array(Z.oindex[slice(*vrange),slice(*hrange),i]) for i in idxs],
            info=[zarr_info[idx] for idx in idxs],
            meta=dict(meta)
            )

def do_3d_fourier(Z:zarr.core.Array):
    F = np.fft.fftn(np.asarray(Z))
    print(F.shape)

def get_zarr_generator(Z, hrange=(None,None), vrange=(None,None), bands=None):
    """
    """
    old_info = Z.attrs["info"]
    old_bands = [old_info[j]["band"] for j in range(len(old_info))]
    return ((old_bands[idx],
             Z.oindex[slice(*vrange), slice(*hrange), old_bands[idx]],
             old_info[idx])
            for idx in (old_bands.index(b)
                        for b in (old_bands if bands is None else
                                  (b for b in bands if b in old_bands))))

def make_video(array_generator, frames_dir:Path, video_path:Path,
               label="tmp", padding=3, framerate=15, keep_frames:bool=False):
    i = 0
    paths = []
    for X in array_generator:
        tmp_path = frames_dir.joinpath(f"{label}_{i:0{padding}}.png")
        gp.generate_raw_image(X, tmp_path)
        paths.append(tmp_path)
        i += 1
    animate_pngs(label=label, frames_dir=frames_dir,
                 out_path=video_path, framerate=framerate)
    if not keep_frames:
        for p in paths:
            p.unlink()
    return video_path
def cycle(F:FeatureGrid, array_generator, dont_drop):
    """
    Drop the first-loaded band outside the window and add the next band
    from the Z based generator.
    """
    if dont_drop:
        return F
    return F.drop_data(F.labels[0]).add_data(*next(array_generator))

if __name__=="__main__":
    zarr_path = Path("data/DESIS-HSI-L2A-DT0865788448_014-20230607T153935.zarr")
    #zarr_path = Path("data/DESIS-HSI-L2A-DT0865788448_015-20230607T153935.zarr")
    #zarr_path = Path("data/DESIS-HSI-L2A-DT0865788448_016-20230607T153935.zarr") # nice
    #zarr_path = Path("data/DESIS-HSI-L2A-DT0865788448_017-20230607T153935.zarr")
    offset = (400,400)
    size = (600,600)
    window = 5

    vrange = (offset[0],offset[0]+size[0])
    hrange = (offset[1],offset[1]+size[1])
    Z = zarr.open(zarr_path)
    zarr_gen = get_zarr_generator(
            Z=Z, hrange=hrange, vrange=vrange, bands=None)
    fg = FeatureGrid()
    # Load the initial window's worth of data into a FeatureGrid
    for i in range(window):
        fg.add_data(*next(zarr_gen))
    data_series = (
            fg.add_data(*next(zarr_gen)).drop_data(fg.labels[0]).data(
                f"colorize logfft2d {fg.labels[-1]}")
            for i in range(Z.shape[-1]-window-1))
    video = make_video(
            array_generator=data_series,
            frames_dir=Path("buffer"),
            video_path=Path("./video.mp4"),
            label="tmp",
            padding=3,
            framerate=15,
            keep_frames=False,
            )
    #do_3d_fourier(Z)
    exit(0)
    #offset = (500,300)
    #size = (1080//3, 1920//3)
    fg = zarr_to_fg(
            Z=Z,
            #bands=list(range(1, 236, 3)),
            vrange=(offset[0],offset[0]+size[0]),
            hrange=(offset[1],offset[1]+size[1]),
            )
    image_dir = Path("figures/animation")
    border_mask = fg.data(fg.labels[0])<-1
    fg.add_recipe(
            "truecolor",
            Recipe(
                args=[f"norm256 histgauss b{b:03}" for b in (104, 59, 28)],
                func=lambda r,g,b:np.dstack([r,g,b]),
                name="truecolor",
                desc="Gaussian histogram-matched truecolor RGB"
                ))
    #tc = np.dstack([enh.norm_to_uint(fg.data(f"b{b:03}"),256,np.uint8)
    #                for b in (104,59,28)])
    #tc[border_mask] = np.average(tc[np.where(np.logical_not(border_mask))])
    gt.quick_render(fg.data("truecolor"))
    #'''
    width = 40
    #for i in range(back, len(fg.labels)-forward):
    for i in range(width, len(fg.labels)-width):
        '''
        R = fg.data(f"norm256 histgauss b{i-width+1:03}")
        G = fg.data(f"norm256 histgauss b{i:03}")
        B = fg.data(f"norm256 histgauss b{i+width+1:03}")
        RGB = np.dstack([R,G,B])
        print(RGB.shape)
        #gt.quick_render(RGB)
        '''
        #'''
        diff = fg.data(f"b{i+1+width:03}") - \
                fg.data(f"b{i+1-width:03}")
        R = enh.norm_to_uint(np.abs(diff), 256, np.uint8)
        B = np.copy(R)
        R[diff<=0] = 0
        B[diff>=0] = 0
        G = np.zeros_like(R)
        RGB = np.dstack((R,G,B))
        RGB = gt.label_at_index(
                RGB, (550,300),
                f"{fg.info(f'b{i+1:03}')['wl']}nm, w{width}nm")
        gp.generate_raw_image(
                RGB, image_dir.joinpath(f"desis-diff_w{width}_{i+1:03}.png"))
    '''
    for i in range(len(fg.labels)):
        idxs = (
                (width*2+i)%len(fg.labels),
                (width+i)%len(fg.labels),
                i%len(fg.labels),
                )
        bands = [f"{idx+1:03}" for idx in idxs]
        image_fname = f"desis_{'-'.join(bands)}.png"
        rgb = np.dstack([fg.data(f"norm256 b{b}") for b in bands])
        rgb[border_mask] = 0
        gp.generate_raw_image(
                RGB=rgb, image_path=image_dir.joinpath(Path(image_fname)))
    '''
    #fg.get_pixels("colorize 13", show=True)
    #'''
