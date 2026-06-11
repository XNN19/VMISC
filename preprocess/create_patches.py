import argparse
import glob
import os
import time
from pathlib import Path

import numpy as np
import openslide
import pandas as pd

try:
    from .wsi_core.WholeSlideImage import WholeSlideImage
    from .wsi_core.batch_process_utils import initialize_df
    from .wsi_core.wsi_utils import StitchCoords
except ImportError:
    from wsi_core.WholeSlideImage import WholeSlideImage
    from wsi_core.batch_process_utils import initialize_df
    from wsi_core.wsi_utils import StitchCoords


DEFAULT_SEG = {"seg_level": -1, "sthresh": 8, "mthresh": 7, "close": 4, "use_otsu": False, "keep_ids": "none", "exclude_ids": "none"}
DEFAULT_FILTER = {"a_t": 100, "a_h": 16, "max_n_holes": 8}
DEFAULT_VIS = {"vis_level": -1, "line_thickness": 250}
DEFAULT_PATCH = {"use_padding": True, "contour_fn": "four_pt"}


def load_params(preset):
    params = {
        "seg_params": DEFAULT_SEG.copy(),
        "filter_params": DEFAULT_FILTER.copy(),
        "vis_params": DEFAULT_VIS.copy(),
        "patch_params": DEFAULT_PATCH.copy(),
    }
    if not preset:
        return params

    row = pd.read_csv(Path(__file__).with_name("presets") / preset).iloc[0]
    for group in params.values():
        for key in group:
            if key in row:
                group[key] = row[key]
    return params


def split_ids(value):
    value = str(value)
    return [] if value == "none" or not value else np.array(value.split(",")).astype(int)


def choose_level(wsi_object, requested_level):
    if requested_level >= 0 or len(wsi_object.level_dim) == 1:
        return max(requested_level, 0)
    return wsi_object.getOpenSlide().get_best_level_for_downsample(64)


def slide_path(source, slide, pattern):
    if not pattern:
        return os.path.join(source, slide)
    matches = glob.glob(os.path.join(source, pattern, slide))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one match for {slide}, found {len(matches)}: {matches}")
    return matches[0]


def infer_patch_size(wsi_object, fallback, physical_size):
    if fallback:
        return fallback
    mpp = wsi_object.wsi.properties[openslide.PROPERTY_NAME_MPP_X]
    return int(float(physical_size) / float(mpp))


def timed(fn, *args, **kwargs):
    start = time.time()
    result = fn(*args, **kwargs)
    return result, time.time() - start


def patch_slide(wsi_object, patch_dir, patch_level, patch_size, patch_params):
    params = dict(patch_params, patch_level=patch_level, patch_size=patch_size, step_size=patch_size, save_path=patch_dir)
    return wsi_object.process_contours(**params)


def stitch_slide(patch_path, wsi_object, output_path):
    heatmap, tissue_ratio = StitchCoords(patch_path, wsi_object, downscale=64, bg_color=(0, 0, 0), alpha=-1, draw_grid=True)
    heatmap.save(output_path)
    pd.DataFrame(tissue_ratio, columns=["coords", "tissue_ratio"]).to_csv(
        output_path.replace(".jpg", "_tissueratio.csv"), index=False
    )


def build_process_frame(source, process_list, skip_list, params):
    slides = sorted(slide for slide in os.listdir(source) if os.path.isfile(os.path.join(source, slide)) and not slide.endswith(".png"))
    df = pd.read_csv(process_list) if process_list else slides
    df = initialize_df(df, params["seg_params"], params["filter_params"], params["vis_params"], params["patch_params"])
    if skip_list:
        df = df[~df["slide_id"].isin(pd.read_csv(skip_list)["slide_id"])]
    df["objective_power"] = 0
    return df


def process_slides(args):
    output = Path(args.save_dir)
    patch_dir = output / "patches"
    mask_dir = output / "masks"
    stitch_dir = output / "stitches"
    for directory in (output, patch_dir, mask_dir, stitch_dir):
        directory.mkdir(parents=True, exist_ok=True)

    params = load_params(args.preset)
    df = build_process_frame(args.source, args.process_list, args.skip_list, params)
    process_stack = df[df["process"] == 1]
    skip_ids = []
    timings = {"seg": [], "patch": [], "stitch": []}

    for n, idx in enumerate(process_stack.index, start=1):
        df.to_csv(output / "process_list_autogen.csv", index=False)
        slide = process_stack.loc[idx, "slide_id"]
        slide_id, slide_ext = os.path.splitext(slide)
        print(f"\nprogress: {n}/{len(process_stack)}\nprocessing {slide}")

        if slide_ext.lower() == ".sdpc":
            continue
        if args.auto_skip and (patch_dir / f"{slide_id}.h5").is_file():
            df.loc[idx, "status"] = "already_exist"
            continue

        try:
            wsi_object = WholeSlideImage(slide_path(args.source, slide, args.source_glob))
        except Exception as exc:
            print(f"Error: {exc}")
            skip_ids.append(slide_id)
            df = df[df["slide_id"] != slide_id]
            continue

        seg_params = {key: df.loc[idx, key] for key in params["seg_params"]}
        filter_params = {key: df.loc[idx, key] for key in params["filter_params"]}
        vis_params = {key: df.loc[idx, key] for key in params["vis_params"]}
        patch_params = {key: df.loc[idx, key] for key in params["patch_params"]}

        vis_params["vis_level"] = choose_level(wsi_object, vis_params["vis_level"])
        seg_params["seg_level"] = choose_level(wsi_object, seg_params["seg_level"])
        seg_params["keep_ids"] = split_ids(seg_params["keep_ids"])
        seg_params["exclude_ids"] = split_ids(seg_params["exclude_ids"])
        df.loc[idx, ["vis_level", "seg_level"]] = [vis_params["vis_level"], seg_params["seg_level"]]

        width, height = wsi_object.level_dim[seg_params["seg_level"]]
        if width * height > args.max_seg_pixels:
            print(f"level_dim {width} x {height} is too large for segmentation")
            df.loc[idx, "status"] = "failed_seg"
            continue

        seg_time = patch_time = stitch_time = -1
        if args.seg:
            _, seg_time = timed(wsi_object.segmentTissue, **seg_params, filter_params=filter_params)
        if args.save_mask:
            wsi_object.visWSI(**vis_params).save(mask_dir / f"{slide_id}.jpg")
        if args.patch:
            patch_size = infer_patch_size(wsi_object, args.patch_size, args.physical_size)
            _, patch_time = timed(patch_slide, wsi_object, patch_dir, args.patch_level, patch_size, patch_params)
        if args.stitch:
            patch_path = patch_dir / f"{slide_id}.h5"
            if patch_path.is_file():
                _, stitch_time = timed(stitch_slide, str(patch_path), wsi_object, str(stitch_dir / f"{slide_id}.jpg"))

        df.loc[idx, "status"] = "processed"
        for name, value in (("seg", seg_time), ("patch", patch_time), ("stitch", stitch_time)):
            timings[name].append(value)
            print(f"{name} took {value} seconds")

    df.to_csv(output / "process_list_autogen.csv", index=False)
    pd.DataFrame(skip_ids, columns=["slide_id"]).to_csv(output / "skip_list_autogen.csv", index=False)
    for name, values in timings.items():
        if values:
            print(f"average {name} time in s per slide: {np.mean(values)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Segment, patch, and stitch WSI files.")
    parser.add_argument("--source", required=True, help="Folder with raw WSI files.")
    parser.add_argument("--save_dir", required=True, help="Output directory.")
    parser.add_argument("--preset", default=None, help="Preset CSV under preprocess/presets.")
    parser.add_argument("--process_list", default=None, help="Optional process-list CSV.")
    parser.add_argument("--skip_list", default=None, help="Optional skip-list CSV with slide_id.")
    parser.add_argument("--source_glob", default="", help="Optional subdirectory glob, for example 'changhai*'.")
    parser.add_argument("--patch_level", type=int, default=0)
    parser.add_argument("--patch_size", type=int, default=0, help="Fixed pixel patch size. Defaults to physical-size / MPP.")
    parser.add_argument("--physical_size", type=float, default=128 * 0.25, help="Physical patch size in microns when inferring patch size.")
    parser.add_argument("--max_seg_pixels", type=float, default=1e12)
    parser.add_argument("--seg", action="store_true")
    parser.add_argument("--patch", action="store_true")
    parser.add_argument("--stitch", action="store_true")
    parser.add_argument("--save_mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--auto_skip", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    process_slides(parse_args())
