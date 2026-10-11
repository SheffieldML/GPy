"""Targeted coverage for open()/tarfile.open() paths ahead of SIM115.

These helpers historically had no active tests. Fixtures use temp directories so
CI does not need network downloads of the real datasets.
"""

from __future__ import annotations

import tarfile

import numpy as np
import pytest

from GPy.util import datasets, mocap, netpbmfile


def test_netpbm_roundtrip_via_filename(tmp_path):
    """NetpbmFile(filename) opens with open() and must read back saved pixels."""
    path = tmp_path / "tiny.pgm"
    image = np.array([[0, 1], [254, 255]], dtype=np.uint8)
    netpbmfile.imsave(str(path), image)
    loaded = netpbmfile.imread(str(path))
    np.testing.assert_array_equal(loaded, image)


def test_netpbmfile_keeps_handle_for_asarray(tmp_path):
    """Stored file handle (_fh) must remain usable after construction."""
    path = tmp_path / "tiny.pgm"
    image = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    netpbmfile.imsave(str(path), image)
    pam = netpbmfile.NetpbmFile(str(path))
    try:
        assert pam._fh is not None
        np.testing.assert_array_equal(pam.asarray(), image)
    finally:
        pam.close()


def test_mocap_read_connections(tmp_path):
    path = tmp_path / "connections.txt"
    path.write_text("head,neck\nneck,torso\n")
    names = np.array(["head", "neck", "torso"])
    connect = mocap.read_connections(str(path), names)
    assert connect.shape == (3, 3)
    assert connect[0, 1] and connect[1, 0]
    assert connect[1, 2] and connect[2, 1]
    assert not connect[0, 2]


def test_mocap_parse_text(tmp_path):
    # Header layout expected by parse_text: [frame, time, x0,y0,z0, x1,y1,z1, ...]
    path = tmp_path / "osu.txt"
    path.write_text(
        "Frame Time LHip_X LHip_Y LHip_Z RHip_X RHip_Y RHip_Z\n"
        "1 0.0 1.0 2.0 3.0 4.0 5.0 6.0\n"
        "2 0.1 1.1 2.1 3.1 4.1 5.1 6.1\n"
    )
    points, point_names, times = mocap.parse_text(str(path))
    assert list(point_names) == ["LHip", "RHip"]
    np.testing.assert_allclose(times, [0.0, 0.1])
    assert len(points) == 3
    np.testing.assert_allclose(points[0][:, 0], [1.0, 1.1])


def test_datasets_oil_reads_tab_files(tmp_path, monkeypatch):
    root = tmp_path / "three_phase_oil_flow"
    root.mkdir()
    rng = np.random.default_rng(0)

    def write_tab(name, arr):
        # Match np.fromfile(..., sep="\t") consumption used by oil().
        (root / name).write_text("\t".join(f"{v:g}" for v in arr.ravel()) + "\n")

    write_tab("DataTrn.txt", rng.normal(size=(4, 12)))
    write_tab("DataTst.txt", rng.normal(size=(3, 12)))
    write_tab("DataVdn.txt", rng.normal(size=(2, 12)))
    write_tab("DataTrnLbls.txt", rng.integers(0, 2, size=(4, 3)))
    write_tab("DataTstLbls.txt", rng.integers(0, 2, size=(3, 3)))
    write_tab("DataVdnLbls.txt", rng.integers(0, 2, size=(2, 3)))

    monkeypatch.setattr(datasets, "data_path", str(tmp_path))
    monkeypatch.setattr(datasets, "data_available", lambda name=None: True)

    data = datasets.oil()
    assert data["X"].shape == (4, 12)
    assert data["Xtest"].shape == (3, 12)
    assert data["Xvalid"].shape == (2, 12)
    assert data["Y"].shape == (4, 3)
    assert data["Ytest"].shape == (3, 3)
    assert data["Yvalid"].shape == (2, 3)


def _write_tiny_tar(archive_path, member_name, content: bytes, gz=False):
    mode = "w:gz" if gz else "w"
    with tarfile.open(archive_path, mode) as tar:
        import io

        info = tarfile.TarInfo(name=member_name)
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))


def test_datasets_creep_extracts_tar(tmp_path, monkeypatch, capsys):
    ds_name = "creep_rupture"
    root = tmp_path / ds_name
    root.mkdir()
    # 31 columns: feature 0, target 1, features 2..30 (2 rows so loadtxt stays 2d)
    row = " ".join(str(float(i)) for i in range(31))
    content = (row + "\n" + row + "\n").encode("ascii")
    _write_tiny_tar(root / "creeprupt.tar", "taka", content)

    monkeypatch.setattr(datasets, "data_path", str(tmp_path))
    # Force the extract branch (tar open), then load the extracted file.
    monkeypatch.setattr(datasets, "data_available", lambda name=None: False)
    monkeypatch.setattr(datasets, "download_data", lambda name=None: None)

    data = datasets.creep_data()
    assert data["X"].shape[1] == 30
    assert data["y"].shape[1] == 1
    assert (tmp_path / ds_name / "taka").exists()


def test_datasets_download_rogers_extracts_tar(tmp_path, monkeypatch):
    ds_name = "rogers_girolami_data"
    root = tmp_path / ds_name
    root.mkdir()
    _write_tiny_tar(root / "firstcoursemldata.tar.gz", "data/marker.txt", b"ok\n", gz=True)

    monkeypatch.setattr(datasets, "data_path", str(tmp_path))
    monkeypatch.setattr(datasets, "data_available", lambda name=None: False)
    monkeypatch.setattr(datasets, "download_data", lambda name=None: None)

    datasets.download_rogers_girolami_data()
    assert (tmp_path / ds_name / "data" / "marker.txt").read_text() == "ok\n"


def test_datasets_pumadyn_extracts_then_loads(tmp_path, monkeypatch):
    ds_name = "pumadyn-32nm"
    root = tmp_path / ds_name
    root.mkdir()
    # After extract, pumadyn loads pumadyn-32nm/Dataset.data.gz
    inner = "pumadyn-32nm/Dataset.data.gz"
    # gzip-compressed numeric table: 10 rows x 34 cols (32 inputs + unused + target)
    import gzip
    import io

    table = np.arange(10 * 34, dtype=float).reshape(10, 34)
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="w") as gz:
        np.savetxt(gz, table)
    _write_tiny_tar(root / "pumadyn-32nm.tar.gz", inner, buf.getvalue(), gz=True)

    monkeypatch.setattr(datasets, "data_path", str(tmp_path))
    monkeypatch.setattr(datasets, "data_available", lambda name=None: False)
    monkeypatch.setattr(datasets, "download_data", lambda name=None: None)
    monkeypatch.setattr(np.random, "permutation", lambda n: np.arange(n))

    data = datasets.pumadyn(seed=0)
    assert data["X"].shape[1] == 32
    assert data["Y"].ndim == 2


def test_datasets_cifar10_patches_extracts(tmp_path, monkeypatch):
    import pickle

    ds_name = "cifar-10"
    root = tmp_path / ds_name
    root.mkdir()
    batch = {
        "data": np.zeros((2, 3 * 32 * 32), dtype=np.uint8),
    }
    batch_bytes = pickle.dumps(batch, protocol=2)
    _write_tiny_tar(root / "cifar-10-python.tar.gz", "cifar-10-batches-py/data_batch_1", batch_bytes, gz=True)

    monkeypatch.setattr(datasets, "data_path", str(tmp_path))
    monkeypatch.setattr(datasets, "data_available", lambda name=None: False)
    monkeypatch.setattr(datasets, "download_data", lambda name=None: None)

    data = datasets.cifar10_patches()
    assert "Y" in data
    assert data["Y"].shape[1] == 5 * 5 * 3


@pytest.mark.skipif(not datasets.gpxpy_available, reason="gpxpy not installed")
def test_datasets_epomeo_gpx_parses_temp_files(tmp_path, monkeypatch):
    ds_name = "epomeo_gpx"
    gpx_dir = tmp_path / ds_name
    gpx_dir.mkdir()
    # Minimal GPX track used by epomeo_gpx()
    gpx_xml = """<?xml version="1.0"?>
<gpx version="1.1" creator="test">
  <trk><name>t</name><trkseg>
    <trkpt lat="40.0" lon="14.0"><ele>100</ele><time>2013-08-21T10:00:00Z</time></trkpt>
    <trkpt lat="40.1" lon="14.1"><ele>110</ele><time>2013-08-21T10:00:10Z</time></trkpt>
  </trkseg></trk>
</gpx>
"""
    files = ["endomondo_1", "endomondo_2", "garmin_watch_via_endomondo", "viewranger_phone", "viewranger_tablet"]
    for name in files:
        (gpx_dir / f"{name}.gpx").write_text(gpx_xml)

    monkeypatch.setattr(datasets, "data_path", str(tmp_path))
    monkeypatch.setattr(datasets, "data_available", lambda name=None: True)

    data = datasets.epomeo_gpx(sample_every=1)
    assert len(data["X"]) == len(files)
    assert data["X"][0].shape[1] == 4
