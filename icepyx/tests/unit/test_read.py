from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

import icepyx.core.read as read


# note isdir will issue a TypeError if a tuple is passed
def test_parse_source_bad_input_type():
    ermesg = (
        "data_source should be a list of files, a directory, the path to a file, "
        "or a glob string."
    )
    with pytest.raises(TypeError, match=ermesg):
        read._parse_source(150)
        read._parse_source({"myfiles": "./my_valid_path/file.h5"})


def test_parse_source_pathlib_input(tmp_path):
    f = tmp_path / "fake.h5"
    f.touch()
    filelist = read._parse_source(f)
    assert len(filelist) == 1
    assert Path(filelist[0]) == f


def test_parse_source_list_of_wrong_types_raises():
    with pytest.raises(TypeError, match="must be a str or pathlib.Path"):
        read._parse_source([1, 2, 3])


def test_make_np_datetime_multielement_z_suffix():
    ds = xr.Dataset(
        {
            "time": (
                "time_idx",
                [
                    b"2019-01-11T05:26:31.323722Z",
                    b"2019-01-12T05:26:31.323722Z",
                ],
            )
        },
        coords={"time_idx": [0, 1]},
    )
    out = read._make_np_datetime(ds, "time")
    assert out["time"].dtype == np.dtype("datetime64[ns]")
    assert out["time"].size == 2


def test_build_single_file_dataset_visits_all_nested_groups(tmp_path):
    """Two siblings under "gt1l/land_ice_segments" must both reach
    `_combine_nested_vars`. Pre-fix, the inner loop mutated
    `wanted_groups_list` while iterating, so the second sibling was
    skipped and instead landed in the outer loop's `_add_vars_to_ds`
    (top-level path), producing wrong coordinate alignment.
    """
    f = tmp_path / "processed_ATL06_20190221_08410203_007_01.h5"
    f.touch()

    with (
        patch.object(read.is2ref, "extract_product", return_value="ATL06"),
        patch.object(read.is2ref, "extract_version", return_value="007"),
    ):
        r = read.Read([str(f)])

    groups_list = [
        "orbit_info/sc_orient",
        "orbit_info/cycle_number",
        "orbit_info/rgt",
        "ancillary_data/atlas_sdp_gps_epoch",
        "ancillary_data/data_start_utc",
        "ancillary_data/data_end_utc",
        "gt1l/land_ice_segments/h_li",
        "gt1l/land_ice_segments/fit_statistics/dh_fit_dx",
        "gt1l/land_ice_segments/ground_track/x_atc",
    ]

    nested_calls = []
    top_level_calls = []

    def fake_read_single_grp(self, file, grp_path):
        return xr.Dataset()

    def fake_add_vars_to_ds(is2ds, ds, grp_path, *a, **kw):
        top_level_calls.append(grp_path)
        return is2ds, ds

    def fake_combine_nested_vars(is2ds, sub_ds, grp_path, wanted_dict):
        nested_calls.append(grp_path)
        return is2ds

    with (
        patch.object(read.Read, "_read_single_grp", fake_read_single_grp),
        patch.object(read.Read, "_add_vars_to_ds", fake_add_vars_to_ds),
        patch.object(read.Read, "_combine_nested_vars", fake_combine_nested_vars),
    ):
        r._build_single_file_dataset(str(f), groups_list)

    assert "gt1l/land_ice_segments/fit_statistics" in nested_calls
    assert "gt1l/land_ice_segments/ground_track" in nested_calls
    # Neither nested child should have leaked into the top-level path.
    assert "gt1l/land_ice_segments/fit_statistics" not in top_level_calls
    assert "gt1l/land_ice_segments/ground_track" not in top_level_calls


def test_parse_source_no_files():
    ermesg = (
        "No files found matching the specified `data_source`. Check your glob "
        "string or file list."
    )
    with pytest.raises(KeyError, match=ermesg):
        read._parse_source("./icepyx/bogus_glob")


@pytest.mark.parametrize(
    "source, expect",
    [
        (  # check list input
            [
                "./icepyx/core/is2ref.py",
                "./icepyx/tests/unit/test_is2class_query.py",
            ],
            sorted(
                [
                    "./icepyx/core/is2ref.py",
                    "./icepyx/tests/unit/test_is2class_query.py",
                ]
            ),
        ),
        (  # check dir input
            "./examples",
            [
                "./examples/README.md",
            ],
        ),
        (  # check filename string with glob pattern input
            "./icepyx/**/*is2*.py",
            sorted(
                [
                    "./icepyx/core/is2ref.py",
                    "./icepyx/tests/unit/test_is2class_query.py",
                    "./icepyx/tests/unit/test_is2ref.py",
                ]
            ),
        ),
        (  # check filename string without glob pattern input
            "./icepyx/core/is2ref.py",
            [
                "./icepyx/core/is2ref.py",
            ],
        ),
        (  # check s3 filename string
            (
                "s3://nsidc-cumulus-prod-protected/ATLAS/"
                "ATL03/006/2019/11/30/ATL03_20191130221008_09930503_006_01.h5"
            ),
            [
                (
                    "s3://nsidc-cumulus-prod-protected/ATLAS/"
                    "ATL03/006/2019/11/30/ATL03_20191130221008_09930503_006_01.h5"
                ),
            ],
        ),
        (
            "./icepyx/core/is2*.py",
            ["./icepyx/core/is2ref.py"],
        ),
    ],
)
def test_parse_source(source, expect):
    filelist = read._parse_source(source, glob_kwargs={"recursive": True})
    assert (sorted(filelist)) == expect


@pytest.mark.parametrize(
    "grp_path, exp_track_str, exp_spot_dim_name, exp_spot_var_name",
    [
        ("gt1l", "gt1l", "spot", "gt"),
        ("gt3r", "gt3r", "spot", "gt"),
        ("profile_2", "profile_2", "profile", "prof"),
        ("pt1", "pt1", "pair_track", "pt"),
    ],
)
def test_get_track_type_str(
    grp_path, exp_track_str, exp_spot_dim_name, exp_spot_var_name
):
    obs_track_str, obs_spot_dim_name, obs_spot_var_name = read._get_track_type_str(
        grp_path
    )
    assert (obs_track_str, obs_spot_dim_name, obs_spot_var_name) == (
        exp_track_str,
        exp_spot_dim_name,
        exp_spot_var_name,
    )
