from unittest.mock import patch

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


def _make_read_with_fake_files(tmp_path, n=3):
    """Build a Read against `n` empty files, bypassing all HDF5 access."""
    files = []
    for i in range(n):
        f = tmp_path / f"processed_ATL06_2019022{i}_0841020{i}_007_01.h5"
        f.touch()
        files.append(str(f))

    class _FakeVars:
        wanted = {"h_li": ["gt1l/land_ice_segments/h_li"]}

        def append(self, *a, **kw):
            pass

    with (
        patch.object(read.is2ref, "extract_product", return_value="ATL06"),
        patch.object(read.is2ref, "extract_version", return_value="007"),
    ):
        r = read.Read(files)
    r._read_vars = _FakeVars()
    return r, files


_COORDLESS_MERGE_WARN = "Your inputs could not be automatically merged"


@pytest.mark.parametrize("max_workers", [1, 2, 4])
def test_load_max_workers_preserves_order_and_count(tmp_path, max_workers):
    # Sentinels with no coords always fall through to the list-return path.
    r, files = _make_read_with_fake_files(tmp_path, n=4)
    sentinels = {f: xr.Dataset(attrs={"src": f}) for f in files}

    def fake_build(self, file, groups_list):
        return sentinels[file]

    with (
        patch.object(read.Read, "_build_single_file_dataset", fake_build),
        pytest.warns(UserWarning, match=_COORDLESS_MERGE_WARN),
    ):
        result = r.load(max_workers=max_workers)
    assert isinstance(result, list)
    assert [d.attrs["src"] for d in result] == files


def test_load_max_workers_caps_to_filelist_length(tmp_path):
    r, files = _make_read_with_fake_files(tmp_path, n=2)
    sentinels = {f: xr.Dataset(attrs={"src": f}) for f in files}

    def fake_build(self, file, groups_list):
        return sentinels[file]

    with (
        patch.object(read.Read, "_build_single_file_dataset", fake_build),
        pytest.warns(UserWarning, match=_COORDLESS_MERGE_WARN),
    ):
        result = r.load(max_workers=64)
    assert [d.attrs["src"] for d in result] == files


def test_load_max_workers_one_skips_threadpool(tmp_path):
    r, files = _make_read_with_fake_files(tmp_path, n=3)
    sentinels = {f: xr.Dataset(attrs={"src": f}) for f in files}

    def fake_build(self, file, groups_list):
        return sentinels[file]

    with (
        patch.object(read.Read, "_build_single_file_dataset", fake_build),
        patch.object(read, "ThreadPoolExecutor") as fake_pool,
        pytest.warns(UserWarning, match=_COORDLESS_MERGE_WARN),
    ):
        r.load(max_workers=1)
    fake_pool.assert_not_called()


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
