import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TextureData:
    name: str
    fullname: str
    extension: str
    hash: str
    path: Path


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
            part_fullname = name + comp["component_name"] + obj_class
            textures = self.parse_textures(tex_hashes, part_fullname)
            parts.append(
                Part(
                    fullname=part_fullname,
                    textures=textures,
                    first_index=obj_index,
                )
            )
        return parts

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
