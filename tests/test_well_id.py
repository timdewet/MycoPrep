"""Plate well-ID parsing and output-filename construction.

Covers ``mycoprep.core.split_czi_plate``: ``normalize_well_id``,
``well_sort_key``, and ``build_output_filename``. Wrong sort orders or
mis-normalised well IDs silently mis-route data, so these are worth
pinning even though the logic is small.
"""

from __future__ import annotations

import pytest

from mycoprep.core.split_czi_plate import (
    build_output_filename,
    normalize_well_id,
    well_sort_key,
)


class TestNormalizeWellId:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("A1", "A1"),
            ("A01", "A1"),
            ("A001", "A1"),
            ("a1", "A1"),
            ("a01", "A1"),
            (" A01 ", "A1"),
            ("H12", "H12"),
            ("H012", "H12"),
            ("B10", "B10"),
        ],
    )
    def test_standard_well_ids(self, raw, expected):
        assert normalize_well_id(raw) == expected

    def test_unrecognised_well_passes_through_uppercased(self):
        # The regex only matches LETTER + digits, so anything else falls through.
        assert normalize_well_id("foo") == "FOO"
        assert normalize_well_id("a1-1") == "A1-1"


class TestWellSortKey:
    def test_numeric_part_is_compared_as_integer(self):
        wells = ["A10", "A2", "A1"]
        assert sorted(wells, key=well_sort_key) == ["A1", "A2", "A10"]

    def test_letter_part_is_primary_key(self):
        wells = ["B1", "A12", "A1"]
        assert sorted(wells, key=well_sort_key) == ["A1", "A12", "B1"]

    def test_full_plate_row_order(self):
        # 96-well plate: rows A..H, cols 1..12.
        wells = [f"{r}{c}" for r in "BAH" for c in (1, 10, 2)]
        ordered = sorted(wells, key=well_sort_key)
        assert ordered == ["A1", "A2", "A10", "B1", "B2", "B10", "H1", "H2", "H10"]

    def test_malformed_well_falls_back_without_crashing(self):
        # Unknown shape → (raw, 0); sorting still works.
        key = well_sort_key("not-a-well")
        assert key == ("not-a-well", 0)


class TestBuildOutputFilename:
    def test_basic_three_fields(self):
        assert (
            build_output_filename("WT", "GFP", "DMSO")
            == "WT__GFP__DMSO.tif"
        )

    def test_spaces_replaced_with_underscores(self):
        assert (
            build_output_filename("strain A", "RFP reporter", "drug X")
            == "strain_A__RFP_reporter__drug_X.tif"
        )

    def test_replica_appended_with_r_prefix(self):
        assert (
            build_output_filename("WT", "GFP", "DMSO", replica="2")
            == "WT__GFP__DMSO__R2.tif"
        )

    def test_empty_replica_omitted(self):
        assert (
            build_output_filename("WT", "GFP", "DMSO", replica="")
            == "WT__GFP__DMSO.tif"
        )

    def test_whitespace_only_replica_omitted(self):
        assert (
            build_output_filename("WT", "GFP", "DMSO", replica="   ")
            == "WT__GFP__DMSO.tif"
        )
