
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

def text_on_mp4(in_path:Path, out_path, text:str, color="white",
                size=24, offset:tuple=(0,0), box=1, wbox=2, obox=.5,
                font="/usr/share/fonts/TTF/FiraCodeNerdFontMono-Regular.ttf"):
    """
    Write static text on an MP4 video with an optional background box.

    :@param offset:2-tuple of values in [0,1] representing (respectively) the
        vertical and horizontal percentage offset of the top left corner of the
        text in terms of the dimensions of the full image.
    """
    filter_dict = {
            "drawtext=fontfile":font,
            "text":text,
            "fontcolor":color,
            "fontsize":size,
            "box":box,
            "boxcolor":f"{color}@{obox}",
            "boxborderw":wbox,
            "x":f"{offset[1]}*w",
            "y":f"{offset[0]}*h",
            }
    filter_str = ":".join("=".join(map(str,t)) for t in filter_dict.items())
    ffmpeg_cmd = f"ffmpeg -i {in_path.as_posix()} -vf \"{filter_str}\" " + \
            f"-qscale 0 -codec:a copy {out_path.as_posix()}"
    proc = Popen(shlex.split(ffmpeg_cmd), stdout=PIPE)
    return proc.communicate()

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

def get_zarr_generator(Z, hrange=(None,None), vrange=(None,None), bands=None):
    """
    Returns a generator returning a subset
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
        try:
            tmp_path = frames_dir.joinpath(f"{label}_{i:0{padding}}.png")
            gp.generate_raw_image(X, tmp_path)
            paths.append(tmp_path)
            i += 1
        except Exception as e:
            print(e)
            break
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
    #zarr_path = Path("data/DESIS-HSI-L2A-DT0865788448_014-20230607T153935.zarr")
    zarr_path = Path("data/DESIS-HSI-L2A-DT0865788448_015-20230607T153935.zarr")
    #zarr_path = Path("data/DESIS-HSI-L2A-DT0865788448_016-20230607T153935.zarr") # nice
    #zarr_path = Path("data/DESIS-HSI-L2A-DT0865788448_017-20230607T153935.zarr")
    offset = (400,400)
    size = (600,600)
    window = 30
    video_path = Path("./desis_015_w30.mp4")

    vrange = (offset[0],offset[0]+size[0])
    hrange = (offset[1],offset[1]+size[1])
    Z = zarr.open(zarr_path)
    zarr_gen = get_zarr_generator(
            Z=Z, hrange=hrange, vrange=vrange, bands=None)
    fg = FeatureGrid()
    # Load the initial window's worth of data into a FeatureGrid
    for i in range(window):
        fg.add_data(*next(zarr_gen))

    iter_range = list(range(Z.shape[-1]-window-1))
    fft_series = (
            np.roll(fg.add_data(*next(zarr_gen)).data(
                f"norm256 colorize logfft2d gaussnorm {fg.labels[-(window//2)]}"
                ), shift=[idx//2 for idx in fg.shape[:2]], axis=(0,1))
            for i in iter_range)
    vis_series = (
            np.dstack([
                fg.add_data(*next(zarr_gen)).drop_data(fg.labels[0]).data(
                    f"norm256 gaussnorm {fg.labels[j]}"
                    ) for j in (0, window//2, window-1)])
                for i in iter_range)
    data_series = (np.concatenate((next(vis_series), next(fft_series)), axis=0)
                   for i in iter_range)

    video = make_video(
            array_generator=data_series,
            frames_dir=Path("buffer"),
            video_path=video_path,
            label="tmp",
            padding=3,
            framerate=15,
            keep_frames=True,
            )
