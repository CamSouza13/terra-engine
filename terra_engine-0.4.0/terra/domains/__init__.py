"""Domain registry. Every domain is a config on the one engine."""
from . import aquaculture, soil, bioremediation, blss

DOMAINS = {
    "aquaculture": aquaculture,
    "soil": soil,
    "bioremediation": bioremediation,
    "blss": blss,
}

__all__ = ["DOMAINS", "aquaculture", "soil", "bioremediation", "blss"]
