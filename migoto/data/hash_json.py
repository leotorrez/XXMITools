import json
from dataclasses import dataclass, field
from pathlib import Path
from bpy.types import Mesh, Object


@dataclass
class SubObj:
    collection_name: str
    depth: int
    name: str
    obj: Object
    mesh: Mesh
    vertex_count: int = 0
    index_count: int = 0
    index_offset: int = 0


@dataclass
class TextureData:
    name: str
    extension: str
    hash: str
    fullname: str
    path: Path


@dataclass
class Part:
    name: str
    fullname: str
    objects: list[SubObj]
    textures: list[TextureData]
    first_index: int
    index_count: int = 0
    first_vertex: int = 0
    vertex_count: int = 0

    def __hash__(self):
        return hash(
            self.fullname
            + "".join(tex.hash for tex in self.textures)
            + str(self.first_index)
        )

    def get_texture_by_name(self, name: str) -> TextureData | None:
        return next((t for t in self.textures if t.name.lower() == "diffuse"), None)


@dataclass
class Component:
    name: str
    fullname: str
    parts: list[Part]
    root_vs: str
    draw_vb: str
    position_vb: str
    blend_vb: str
    texcoord_vb: str
    ib: str
    vertex_count: int = 0
    strides: dict[str, int] = field(default_factory=dict)


class HashJsonData:
    def __init__(self, path: Path):
        name = path.parent.stem
        self.path: Path = path
        with open(path, "r") as f:
            data = json.load(f)
        self.components = self.parse_components(data, name)
        self.find_missing_textures()

    def find_missing_textures(self):
        texture_map: dict[str, Path] = {}
        missing_textures: list[TextureData] = []
        for comp in self.components:
            for part in comp.parts:
                for tex in part.textures:
                    if tex.path.exists():
                        texture_map[tex.hash] = tex.path
                    else:
                        missing_textures.append(tex)
        if missing_textures:
            print("Missing texture files:")
            for tex in missing_textures:
                if tex.hash in texture_map:
                    tex.path = texture_map[tex.hash]
                    print(
                        f"{tex.name}{tex.extension} (hash: {tex.hash}) - found at {tex.path}"
                    )
                else:
                    print(
                        f"{tex.name}{tex.extension} (hash: {tex.hash}) at {tex.path} - file not found"
                    )

    def parse_components(self, data: list[dict], name: str) -> list[Component]:
        comps: list[Component] = []
        for comp in data:
            parts = self.parse_parts(comp, name)
            comps.append(
                Component(
                    name=comp["component_name"],
                    fullname=name + comp["component_name"],
                    parts=parts,
                    root_vs=comp.get("root_vs", ""),
                    draw_vb=comp.get("draw_vb", ""),
                    position_vb=comp.get("position_vb", ""),
                    blend_vb=comp.get("blend_vb", ""),
                    texcoord_vb=comp.get("texcoord_vb", ""),
                    ib=comp.get("ib", ""),
                )
            )
        return comps

    def parse_parts(self, comp: dict, name: str) -> list[Part]:
        parts = []
        for tex_hashes, _, obj_class in zip(
            comp["texture_hashes"],
            comp["object_indexes"],
            comp["object_classifications"],
        ):
            part_fullname = name + comp["component_name"] + obj_class
            textures = self.parse_textures(tex_hashes, part_fullname)
            ib_path = self.path.parent.glob(part_fullname + "-ib" + "*.txt")
            ib_path = next(ib_path, None)
            index_count, first_index = self.parse_ib_metadata(ib_path)
            parts.append(
                Part(
                    name=obj_class,
                    fullname=part_fullname,
                    textures=textures,
                    first_index=first_index,
                    index_count=index_count,
                    objects=[],
                )
            )
        return parts

    def parse_ib_metadata(self, ib_path: Path | None) -> tuple[int, int]:
        if ib_path is None or not ib_path.exists():
            return 0, 0
        index_count = 0
        first_index = 0
        with open(ib_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                if index_count != 0 and first_index != 0:
                    break
                if line.startswith("index count:"):
                    index_count = int(line.split(":")[1].strip())
                elif line.startswith("first index:"):
                    first_index = int(line.split(":")[1].strip())
        return index_count, first_index

    def parse_textures(
        self, tex_hashes: list[list], part_fullname: str
    ) -> list[TextureData]:
        textures = []
        for name, ext, hash in tex_hashes:
            path = self.path.parent / (part_fullname + name + ext)
            textures.append(
                TextureData(
                    name=name,
                    fullname=part_fullname + name,
                    extension=ext,
                    hash=hash,
                    path=path,
                )
            )
        return textures

    def get_part_by_fullname(self, fullname: str) -> Part:
        for comp in self.components:
            for part in comp.parts:
                if part.fullname == fullname:
                    return part
        raise ValueError(f"Part with fullname {fullname} not found")

    def get_component_by_fullname(self, fullname: str) -> Component:
        for comp in self.components:
            if comp.fullname == fullname:
                return comp
        raise ValueError(f"Component with fullname {fullname} not found")
