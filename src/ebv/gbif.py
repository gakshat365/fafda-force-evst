"""A small, polite client for the public GBIF occurrence and species APIs.

GBIF throttles clients that send many parallel requests, so every request goes
through one session, at most one at a time, with a minimum gap between calls and
exponential back-off when GBIF asks us to slow down.
"""
import time

import requests

from . import config

THROTTLE_MESSAGE = "Too many API requests"


class GbifClient:
    def __init__(self, interval_s: float = config.REQUEST_INTERVAL_S, max_retries: int = 8):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "ebv-south-india-student-project/1.0"
        self.interval_s = interval_s
        self.max_retries = max_retries
        self._last_call = 0.0

    def _get(self, path: str, **params) -> dict:
        for attempt in range(self.max_retries):
            wait = self.interval_s - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            try:
                response = self.session.get(f"{config.GBIF_API}/{path}", params=params, timeout=90)
                data = response.json() if response.content else {}
                throttled = response.status_code == 429 or THROTTLE_MESSAGE in str(data)[:200]
                if response.ok and isinstance(data, dict) and not throttled:
                    return data
                if response.status_code == 404:
                    return {}
            except (requests.RequestException, ValueError):
                pass
            time.sleep(min(15 * 2 ** attempt, 600))
        raise RuntimeError(f"GBIF request failed after {self.max_retries} attempts: {path} {params}")

    @staticmethod
    def _bird_filters() -> dict:
        return dict(classKey=config.AVES_CLASS_KEY, basisOfRecord="HUMAN_OBSERVATION",
                    year=f"{config.YEARS[0]},{config.YEARS[1]}", hasCoordinate="true",
                    hasGeospatialIssue="false", occurrenceStatus="PRESENT")

    def species_counts(self, west: float, south: float, east: float, north: float) -> dict:
        """Total bird records and records per species key inside a box.

        Box edges are half-open (east and north excluded) so neighbouring cells never
        share a record.
        """
        eps = 1e-6
        data = self._get("occurrence/search", limit=0, facet="speciesKey", facetLimit=3000,
                         decimalLatitude=f"{south:.6f},{north - eps:.6f}",
                         decimalLongitude=f"{west:.6f},{east - eps:.6f}", **self._bird_filters())
        facets = data.get("facets") or [{"counts": []}]
        return {"n_records": data["count"],
                "species": {c["name"]: c["count"] for c in facets[0]["counts"]}}

    def species_info(self, key: str) -> dict:
        """Name, family, order and IUCN Red List category ("NE" when never assessed)."""
        taxon = self._get(f"species/{key}")
        iucn = self._get(f"species/{key}/iucnRedListCategory")
        return dict(species_key=key, scientific_name=taxon.get("canonicalName"),
                    english_name=taxon.get("vernacularName"), family=taxon.get("family"),
                    order=taxon.get("order"), iucn=iucn.get("code") or "NE")
