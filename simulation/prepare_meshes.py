"""Prepare oversized binary STL files for MuJoCo without dropping triangles."""

from __future__ import annotations

import argparse
from pathlib import Path
import struct

from simulation.model_builder import HAND_VARIANTS


STL_HEADER_BYTES = 80
TRIANGLE_BYTES = 50
MUJOCO_FACE_LIMIT = 200_000
PART_FACE_LIMIT = 180_000


def split_oversized_binary_stl(path: Path) -> list[Path]:
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"Not a binary STL: {path}")
    face_count = struct.unpack_from("<I", data, STL_HEADER_BYTES)[0]
    expected_size = STL_HEADER_BYTES + 4 + face_count * TRIANGLE_BYTES
    if expected_size != len(data):
        raise ValueError(f"Only binary STL is supported: {path}")
    if face_count <= MUJOCO_FACE_LIMIT:
        return [path]

    triangle_data = data[84:]
    outputs: list[Path] = []
    for part_index, start_face in enumerate(range(0, face_count, PART_FACE_LIMIT), start=1):
        part_faces = min(PART_FACE_LIMIT, face_count - start_face)
        start_byte = start_face * TRIANGLE_BYTES
        end_byte = start_byte + part_faces * TRIANGLE_BYTES
        output = path.with_name(f"{path.stem}_part{part_index}{path.suffix}")
        header = data[:STL_HEADER_BYTES]
        output.write_bytes(header + struct.pack("<I", part_faces) + triangle_data[start_byte:end_byte])
        outputs.append(output)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", choices=sorted(HAND_VARIANTS), default="left")
    args = parser.parse_args()
    directory = HAND_VARIANTS[args.hand].meshes
    for mesh in sorted(directory.glob("*.STL")):
        if "_part" in mesh.stem:
            continue
        outputs = split_oversized_binary_stl(mesh)
        if outputs != [mesh]:
            print(f"Split {mesh.name} into {', '.join(item.name for item in outputs)}")


if __name__ == "__main__":
    main()
