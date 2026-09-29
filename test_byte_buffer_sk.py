"""Standalone test for shapekey deltas import (txt + binary paths).

Currently 3D Migoto/XXMI emits the shapekey offset/count in `SKOffsets.csv`.
The intended format embeds those `offset,count` pairs directly in the deltas
txt header as `sk offsets:` / `sk counts:` lines (plus a total `sk count:`),
which `MigotoFormat` parses via its existing `sk_offsets`/`sk_counts` fields.
Like every other header entry (`index count`, `first index`, `vb0 stride`,
etc.) the keys are space-separated; the underscore forms are also accepted.
This test builds that intended format from the CSV so the library code reads
it as designed.

Run with: python test_byte_buffer_sk.py
"""

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy

from migoto.data.byte_buffer import MigotoFormat, NumpyBuffer, Semantic

DATA_DIR = Path(
    r"C:\Users\Leandro\AppData\Roaming\XXMI Launcher\guisk\_Extracted\Ramielle"
)
VB_PATH = DATA_DIR / "RamielleEyebrowsA-vb0=da475732.txt"
DELTAS_TXT = DATA_DIR / "RamielleEyebrowsSKDeltas.txt"
DELTAS_BUF = DATA_DIR / "RamielleEyebrowsSKDeltas.buf"
OFFSETS_CSV = DATA_DIR / "RamielleEyebrowsSKOffsets.csv"


def add_sk_header(deltas_text: str) -> str:
    """Injects `sk count:`/`sk offsets:`/`sk counts:` header lines (sourced from
    SKOffsets.csv) into the deltas txt to reproduce the intended XXMI Deltas.txt
    format. Keys are space-separated like the other header entries."""
    rows = [
        line.strip()
        for line in OFFSETS_CSV.read_text().splitlines()
        if line.strip() and not line.startswith("offset")
    ]
    offsets = ",".join(row.split(",")[0] for row in rows)
    counts = ",".join(row.split(",")[1] for row in rows)
    return deltas_text.replace(
        "topology: trianglelist",
        f"topology: trianglelist\nsk count: {len(rows)}\nsk offsets: {offsets}\nsk counts: {counts}",
        1,
    )


def sk_field_names(count: int) -> list[str]:
    return ["SHAPEKEY"] + [f"SHAPEKEY{i}" for i in range(1, count)]


def main() -> None:
    vb_text = VB_PATH.read_text()
    deltas_text = add_sk_header(DELTAS_TXT.read_text())
    deltas_fmt = MigotoFormat.from_txt_file(io.StringIO(deltas_text))

    assert deltas_fmt.sk_count == 11
    assert deltas_fmt.sk_offsets == [0, 42, 84, 126, 168, 210, 294, 336, 378, 420, 462]
    assert deltas_fmt.sk_counts == [42] * 5 + [84] + [42] * 5
    sk_names = sk_field_names(len(deltas_fmt.sk_counts))
    print("ok: sk count/sk offsets/sk counts parsed from deltas header")

    # Merged format (VB + deltas) must keep the base VB elements and append ShapeKeys
    with open(VB_PATH) as vb_file, io.StringIO(deltas_text) as deltas_file:
        fmt = MigotoFormat.from_files(vb_file, None, deltas_file)
    assert fmt.vb_layout is not None
    layout_names = [s.get_name() for s in fmt.vb_layout.semantics]
    assert layout_names == [
        "POSITION",
        "NORMAL",
        "TANGENT",
        "BLENDINDICES",
        "COLOR",
        "TEXCOORD.xy",
        "TEXCOORD1.xy",
        "TEXCOORD2.xy",
        "TEXCOORD3.xy",
    ] + sk_names, layout_names
    assert fmt.vb_layout.stride == 92 + 12 * len(sk_names)
    print("ok: merged layout keeps VB elements and appends", len(sk_names), "shapekeys")

    # Text import: VB + deltas scatter matches the binary reference
    buffer = NumpyBuffer(fmt.vb_layout, size=fmt.vertex_count)
    buffer.import_txt_data(vb_text, deltas_data=deltas_text)
    assert buffer.data is not None and len(buffer.data) == fmt.vertex_count == 84

    sk_dtype = numpy.dtype(
        [
            ("VINDEX", numpy.uint32),
            ("POSITION", numpy.float32, 3),
            ("NORMAL", numpy.float32, 3),
            ("TANGENT", numpy.float32, 3),
        ]
    )
    sk_pool = numpy.frombuffer(DELTAS_BUF.read_bytes(), dtype=sk_dtype)
    for i, field in enumerate(sk_names):
        expected = numpy.zeros((84, 3), dtype=numpy.float32)
        chunk = sk_pool[deltas_fmt.sk_offsets[i] : deltas_fmt.sk_offsets[i] + deltas_fmt.sk_counts[i]]
        expected[chunk["VINDEX"]] = chunk["POSITION"]
        got = buffer.get_field(field)
        assert got is not None, field
        assert numpy.allclose(got, expected, atol=1e-6), field
    print("ok: txt import shapekeys match binary reference")

    # Pure VB import (no deltas) zero-fills shapekeys
    vb_only = NumpyBuffer(fmt.vb_layout, size=fmt.vertex_count)
    vb_only.import_txt_data(vb_text)
    assert numpy.all(vb_only.get_field(Semantic.ShapeKey) == 0)
    print("ok: pure VB import zero-fills shapekeys")

    # Binary path: expand_sk_bytes + NumpyMesh.from_bytes match the txt result
    with open(VB_PATH) as vb_file:
        vb_only_fmt = MigotoFormat.from_files(vb_file, None, None)
    vb_only_buffer = NumpyBuffer(vb_only_fmt.vb_layout, size=fmt.vertex_count)
    vb_only_buffer.import_txt_data(vb_text)
    expanded = NumpyBuffer.expand_sk_bytes(
        fmt.vb_layout,
        fmt.sk_counts,
        fmt.sk_offsets,
        vb_only_buffer.get_bytes(),
        DELTAS_BUF.read_bytes(),
    )
    bin_buffer = NumpyBuffer(fmt.vb_layout)
    bin_buffer.import_raw_data(expanded)
    for field in sk_names:
        assert numpy.allclose(bin_buffer.get_field(field), buffer.get_field(field), atol=1e-6), field

    from migoto.data.numpy_mesh import NumpyMesh

    mesh = NumpyMesh.from_bytes(
        fmt, vb_bytes=vb_only_buffer.get_bytes(), deltas_bytes=DELTAS_BUF.read_bytes()
    )
    assert mesh.vertex_buffer is not None
    for field in sk_names:
        assert numpy.allclose(
            mesh.vertex_buffer.get_field(field), buffer.get_field(field), atol=1e-6
        ), field
    print("ok: expand_sk_bytes / NumpyMesh.from_bytes match txt path")


if __name__ == "__main__":
    main()