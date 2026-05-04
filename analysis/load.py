from enum import IntEnum

import os
import tomllib
import re
from shapely.geometry import Polygon, LineString

import numpy as np
import pandas as pd 
import shapely.geometry.polygon as shp

from .standard_variables import strings_ref
#strings_ref = ("_X", "_Y", "_A", "_B", "_Theta", "_R", "_M", "_Class", "_beam")

class ObjectType(IntEnum):
    ELLIPSE = 0
    YSO = 1
    POLYGON = -1

def read_toml(file):
    with open(file, "rb") as f:
        cfg = tomllib.load(f)
    return cfg

def _load_ellipses(df, cfg):
    df[strings_ref[0]] = df.pop(cfg["columns_names_in_catalog"]["Xposition"]) * cfg["unit_conversion"]["Xposition_to_deg"]
    df[strings_ref[1]] = df.pop(cfg["columns_names_in_catalog"]["Yposition"]) * cfg["unit_conversion"]["Yposition_to_deg"]
    df[strings_ref[2]] = df.pop(cfg["columns_names_in_catalog"]["SemiMajorAxis"]) * cfg["unit_conversion"]["SemiMajorAxis_to_deg"]
    df[strings_ref[3]] = df.pop(cfg["columns_names_in_catalog"]["SemiMinorAxis"]) * cfg["unit_conversion"]["SemiMinorAxis_to_deg"]
    df[strings_ref[4]] = df.pop(cfg["columns_names_in_catalog"]["PosAngle"]) * cfg["unit_conversion"]["PosAngle_to_rad"] 
    df[strings_ref[4]] += cfg["unit_conversion"]["PosAngle_offset"]

    df[strings_ref[2]] *= cfg["unit_conversion"]["n_sigma"]
    df[strings_ref[3]] *= cfg["unit_conversion"]["n_sigma"]

    # size in arcsec
    df[strings_ref[5]] = np.sqrt( df[strings_ref[2]] * df[strings_ref[3]] ) * 3600.0

    if cfg["columns_names_in_catalog"]["Mass"] in df.columns:
        df[strings_ref[6]] = df.pop(cfg["columns_names_in_catalog"]["Mass"]) * cfg["unit_conversion"]["Mass_to_Msun"] 

    df[strings_ref[8]] = cfg["observation"]["beam"]

def _load_yso(df, cfg):
    df[strings_ref[0]] = df.pop(cfg["columns_names_in_catalog"]["Xposition"]) * cfg["unit_conversion"]["Xposition_to_deg"]
    df[strings_ref[1]] = df.pop(cfg["columns_names_in_catalog"]["Yposition"]) * cfg["unit_conversion"]["Yposition_to_deg"]
    
    data_frame[[strings_ref[2], strings_ref[3]]] = cfg["observation"]["beam"] / 3600.0
    data_frame[strings_ref[4]] = 0.0
    
    df[strings_ref[5]] = np.sqrt( df[strings_ref[2]] * df[strings_ref[3]] ) * 3600.0

    if cfg["columns_names_in_catalog"]["Mass"] in df.columns:
        df[strings_ref[6]] = df.pop(cfg["columns_names_in_catalog"]["Mass"]) * cfg["unit_conversion"]["Mass_to_Msun"]

    if cfg["columns_names_in_catalog"]["Class"] in df.columns:
        df[strings_ref[7]] = df.pop(cfg["columns_names_in_catalog"]["Class"]) * cfg["unit_conversion"]["Mass_to_Msun"]

    df[strings_ref[8]] = cfg["observation"]["beam"]

def _load_polygons(df, cfg):
    df[strings_ref[0]] = df.pop(cfg["columns_names_in_catalog"]["Xposition"]) * cfg["unit_conversion"]["Xposition_to_deg"]
    df[strings_ref[1]] = df.pop(cfg["columns_names_in_catalog"]["Yposition"]) * cfg["unit_conversion"]["Yposition_to_deg"]

    df[strings_ref[8]] = cfg["observation"]["beam"]
    
def load_data(file, reader_to_df):
    cfg = read_toml(file)

    path = cfg["files"]["catalog"]
    df = reader_to_df(path)

    if cfg["objects"]["type"] == ObjectType.ELLIPSE:
        _load_ellipses(df, cfg)

    elif metadata["object_type"] == ObjectType.YSO:
        _load_yso(df, cfg)

    elif metadata["object_type"] == ObjectType.POLYGON:
        _load_polygons(df, cfg)

    else:
        return
        
    metadata = {
        "df": df,
        "comment": cfg["objects"]["comment"],
        "name":cfg["dataset"]["name"]
    }
    metadata.update(cfg["files"])
    metadata.update(cfg["observation"])
    return metadata

def read_ds9_polygons(path):
    path = Path(path)
    polygons = []

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        # ignore les lignes de frame, ex: image / fk5 / physical
        if line.lower() in {"image", "physical", "fk5", "icrs", "galactic", "ecliptic"}:
            continue

        if line.lower().startswith("polygon("):
            inside = line[line.find("(") + 1: line.rfind(")")]
            coords = [float(x) for x in re.split(r"\s*,\s*", inside)]

            if len(coords) < 6 or len(coords) % 2 != 0:
                raise ValueError(f"Polygone invalide: {line}")

            xy = list(zip(coords[::2], coords[1::2]))
            polygons.append(Polygon(xy))

    return polygons