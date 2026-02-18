import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TextureData:
    name: str
    extension: str
    hash: str


@dataclass
class Part:
    fullname: str
    textures: list[TextureData]
    first_index: int

    def __hash__(self):
        return hash(
            self.fullname
            + "".join(tex.hash for tex in self.textures)
            + str(self.first_index)
        )


@dataclass
class Component:
    fullname: str
    parts: list[Part]
    root_vs: str
    draw_vb: str
    position_vb: str
    blend_vb: str
    texcoord_vb: str
    ib: str


class HashJsonData:
    def __init__(self, path: Path):
        name = path.parent.stem
        with open(path, "r") as f:
            data = json.load(f)
        self.components = self.parse_components(data, name)

    def parse_components(self, data: list[dict], name: str) -> list[Component]:
        comps: list[Component] = []
        for comp in data:
            parts = self.parse_parts(comp, name)
            comps.append(
                Component(
                    fullname=name + comp["component_name"],
                    parts=parts,
                    root_vs=comp["root_vs"],
                    draw_vb=comp["draw_vb"],
                    position_vb=comp["position_vb"],
                    blend_vb=comp["blend_vb"],
                    texcoord_vb=comp["texcoord_vb"],
                    ib=comp["ib"],
                )
            )
        return comps

    def parse_parts(self, comp: dict, name: str) -> list[Part]:
        parts = []
        for tex_hashes, obj_index, obj_class in zip(
            comp["texture_hashes"],
            comp["object_indexes"],
            comp["object_classifications"],
        ):
            textures = [
                TextureData(
                    name=tex_type,
                    extension=tex_ext,
                    hash=tex_hash,
                )
                for tex_type, tex_ext, tex_hash in tex_hashes
            ]
            parts.append(
                Part(
                    fullname=name + comp["component_name"] + obj_class,
                    textures=textures,
                    first_index=obj_index,
                )
            )
        return parts

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
