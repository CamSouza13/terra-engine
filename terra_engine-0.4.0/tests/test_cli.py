"""CLI tests (numpy-only subcommands)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from terra.cli import main


def test_domains():
    assert main(["domains"]) == 0


def test_validate_each_domain():
    for d in ("aquaculture", "soil", "bioremediation", "blss"):
        assert main(["validate", "--domain", d]) == 0


def test_demo():
    assert main(["demo", "--domain", "aquaculture"]) == 0


if __name__ == "__main__":
    n = 0
    for fn in (test_domains, test_validate_each_domain, test_demo):
        fn()
        print(f"PASS  {fn.__name__}")
        n += 1
    print(f"\n{n}/{n} passed")
