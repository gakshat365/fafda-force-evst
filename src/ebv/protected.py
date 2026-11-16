"""Protected areas (national parks, wildlife sanctuaries, tiger and bird reserves) from OpenStreetMap."""
import json
import re
import time

import numpy as np
import pandas as pd
import requests
from shapely.geometry import LineString, Polygon, box
from shapely.ops import polygonize, unary_union
from shapely.strtree import STRtree

from . import config

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
STRICT_CLASSES = {"1", "1a", "1b", "2", "3", "4", "5"}     # IUCN-style protect_class values kept
NAME_PATTERN = re.compile(r"national park|(wild ?life|birds?) sanct\w*|\bwls\b|tiger reserve|"
                          r"conservation reserve", re.I)
EXCLUDE_PATTERN = re.compile(r"eco[- ]?sensitive|\besz\b|buffer|corridor|proposed|zone of influence", re.I)


def fetch_osm(bounds: tuple, path) -> dict:
    """Download (once) protected-area ways and relations with full geometry."""
    if path.exists():
        return json.loads(path.read_text())
    west, south, east, north = bounds
    bbox = f"{south},{west},{north},{east}"
    query = (f"[out:json][timeout:600];("
             f'relation["boundary"="protected_area"]({bbox});way["boundary"="protected_area"]({bbox});'
             f'relation["leisure"="nature_reserve"]({bbox});way["leisure"="nature_reserve"]({bbox});'
             f'relation["boundary"="national_park"]({bbox});way["boundary"="national_park"]({bbox});'
             f");out geom;")
    for attempt in range(5):                     # the public server is often briefly overloaded
        response = requests.post(OVERPASS_URL, data={"data": query}, timeout=900,
                                 headers={"User-Agent": "ebv-south-india-student-project/1.0"})
        if response.ok:
            path.write_text(response.text)
            return response.json()
        time.sleep(60 * (attempt + 1))
    response.raise_for_status()


def is_protected_area(tags: dict) -> bool:
    """National parks, sanctuaries and reserves; not eco-sensitive buffers or corridors."""
    name = tags.get("name", "") + " " + tags.get("protection_title", "")
    if EXCLUDE_PATTERN.search(name):
        return False
    return tags.get("protect_class") in STRICT_CLASSES or bool(NAME_PATTERN.search(name))


def _polygon(element: dict):
    """Polygon of a closed way, or of a multipolygon relation (outer minus inner rings)."""
    if element["type"] == "way":
        coords = [(p["lon"], p["lat"]) for p in element.get("geometry", [])]
        return Polygon(coords) if len(coords) >= 4 and coords[0] == coords[-1] else None
    rings = {"outer": [], "inner": []}
    for member in element.get("members", []):
        if member.get("type") == "way" and member.get("geometry"):
            role = "inner" if member.get("role") == "inner" else "outer"
            rings[role].append(LineString([(p["lon"], p["lat"]) for p in member["geometry"]]))
    outer = unary_union(list(polygonize(rings["outer"])))
    inner = unary_union(list(polygonize(rings["inner"])))
    shape = outer.difference(inner) if not inner.is_empty else outer
    return None if shape.is_empty else shape


def protected_areas(osm: dict) -> pd.DataFrame:
    """One row per protected area: name, OSM class and valid polygon geometry."""
    rows = []
    for element in osm["elements"]:
        tags = element.get("tags", {})
        if not is_protected_area(tags):
            continue
        geometry = _polygon(element)
        if geometry is not None:
            rows.append({"name": tags.get("name"), "protect_class": tags.get("protect_class"),
                         "geometry": geometry.buffer(0)})
    return pd.DataFrame(rows)


def protected_fraction(cells: pd.DataFrame, geometries) -> np.ndarray:
    """Share of each grid cell's area inside any protected area (overlaps counted once)."""
    tree = STRtree(list(geometries))
    out = np.zeros(len(cells))
    for i, cell in enumerate(cells.itertuples()):
        square = box(cell.lon0, cell.lat0, cell.lon0 + config.CELL_DEG, cell.lat0 + config.CELL_DEG)
        hits = [tree.geometries[j] for j in tree.query(square, predicate="intersects")]
        if hits:
            out[i] = unary_union(hits).intersection(square).area / square.area
    return out
