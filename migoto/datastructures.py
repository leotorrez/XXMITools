import collections
import io
import itertools
import re
import struct
import textwrap
from enum import Enum
import numpy
from mathutils import Matrix

IOOBJOrientationHelper = type("DummyIOOBJOrientationHelper", (object,), {})
vertex_color_layer_channels = 4


# FIXME: hardcoded values in a very weird way cause blender EnumProperties are odd-
class GameEnum(str, Enum):
    HonkaiImpact3rd = "Honkai Impact 3rd"
    GenshinImpact = "Genshin Impact"
    HonkaiStarRail = "Honkai Star Rail"
    ZenlessZoneZero = "Zenless Zone Zero"
    HonkaiImpactPart2 = "Honkai Impact 3rd Part 2"


game_enum = [None]+[
    (
        game.name,
        game.value,
        game.value,
    )
    for game in GameEnum
]

supported_topologies = ("trianglelist", "pointlist", "trianglestrip")

ImportPaths = collections.namedtuple(
    "ImportPaths", ("vb_paths", "ib_paths", "use_bin", "pose_path")
)


def keys_to_ints(d):
    return {k.isdecimal() and int(k) or k: v for k, v in d.items()}


def keys_to_strings(d):
    return {str(k): v for k, v in d.items()}


class Fatal(Exception):
    pass


# TODO: Support more DXGI formats:
f32_pattern = re.compile(r"""(?:DXGI_FORMAT_)?(?:[RGBAD]32)+_FLOAT""")
f16_pattern = re.compile(r"""(?:DXGI_FORMAT_)?(?:[RGBAD]16)+_FLOAT""")
u32_pattern = re.compile(r"""(?:DXGI_FORMAT_)?(?:[RGBAD]32)+_UINT""")
u16_pattern = re.compile(r"""(?:DXGI_FORMAT_)?(?:[RGBAD]16)+_UINT""")
u8_pattern = re.compile(r"""(?:DXGI_FORMAT_)?(?:[RGBAD]8)+_UINT""")
s32_pattern = re.compile(r"""(?:DXGI_FORMAT_)?(?:[RGBAD]32)+_SINT""")
s16_pattern = re.compile(r"""(?:DXGI_FORMAT_)?(?:[RGBAD]16)+_SINT""")
s8_pattern = re.compile(r"""(?:DXGI_FORMAT_)?(?:[RGBAD]8)+_SINT""")
unorm16_pattern = re.compile(r"""(?:DXGI_FORMAT_)?(?:[RGBAD]16)+_UNORM""")
unorm8_pattern = re.compile(r"""(?:DXGI_FORMAT_)?(?:[RGBAD]8)+_UNORM""")
snorm16_pattern = re.compile(r"""(?:DXGI_FORMAT_)?(?:[RGBAD]16)+_SNORM""")
snorm8_pattern = re.compile(r"""(?:DXGI_FORMAT_)?(?:[RGBAD]8)+_SNORM""")

misc_float_pattern = re.compile(
    r"""(?:DXGI_FORMAT_)?(?:[RGBAD][0-9]+)+_(?:FLOAT|UNORM|SNORM)"""
)
misc_int_pattern = re.compile(r"""(?:DXGI_FORMAT_)?(?:[RGBAD][0-9]+)+_[SU]INT""")


def EncoderDecoder(fmt):
    if f32_pattern.match(fmt):
        return (
            lambda data: b"".join(struct.pack("<f", x) for x in data),
            lambda data: numpy.frombuffer(data, numpy.float32).tolist(),
        )
    if f16_pattern.match(fmt):
        return (
            lambda data: numpy.fromiter(data, numpy.float16).tobytes(),
            lambda data: numpy.frombuffer(data, numpy.float16).tolist(),
        )
    if u32_pattern.match(fmt):
        return (
            lambda data: numpy.fromiter(data, numpy.uint32).tobytes(),
            lambda data: numpy.frombuffer(data, numpy.uint32).tolist(),
        )
    if u16_pattern.match(fmt):
        return (
            lambda data: numpy.fromiter(data, numpy.uint16).tobytes(),
            lambda data: numpy.frombuffer(data, numpy.uint16).tolist(),
        )
    if u8_pattern.match(fmt):
        return (
            lambda data: numpy.fromiter(data, numpy.uint8).tobytes(),
            lambda data: numpy.frombuffer(data, numpy.uint8).tolist(),
        )
    if s32_pattern.match(fmt):
        return (
            lambda data: numpy.fromiter(data, numpy.int32).tobytes(),
            lambda data: numpy.frombuffer(data, numpy.int32).tolist(),
        )
    if s16_pattern.match(fmt):
        return (
            lambda data: numpy.fromiter(data, numpy.int16).tobytes(),
            lambda data: numpy.frombuffer(data, numpy.int16).tolist(),
        )
    if s8_pattern.match(fmt):
        return (
            lambda data: numpy.fromiter(data, numpy.int8).tobytes(),
            lambda data: numpy.frombuffer(data, numpy.int8).tolist(),
        )

    if unorm16_pattern.match(fmt):
        return (
            lambda data: numpy.around((numpy.fromiter(data, numpy.float32) * 65535.0))
            .astype(numpy.uint16)
            .tobytes(),
            lambda data: (numpy.frombuffer(data, numpy.uint16) / 65535.0).tolist(),
        )
    if unorm8_pattern.match(fmt):
        return (
            lambda data: numpy.around((numpy.fromiter(data, numpy.float32) * 255.0))
            .astype(numpy.uint8)
            .tobytes(),
            lambda data: (numpy.frombuffer(data, numpy.uint8) / 255.0).tolist(),
        )
    if snorm16_pattern.match(fmt):
        return (
            lambda data: numpy.around((numpy.fromiter(data, numpy.float32) * 32767.0))
            .astype(numpy.int16)
            .tobytes(),
            lambda data: (numpy.frombuffer(data, numpy.int16) / 32767.0).tolist(),
        )
    if snorm8_pattern.match(fmt):
        return (
            lambda data: numpy.around((numpy.fromiter(data, numpy.float32) * 127.0))
            .astype(numpy.int8)
            .tobytes(),
            lambda data: (numpy.frombuffer(data, numpy.int8) / 127.0).tolist(),
        )

    raise Fatal("File uses an unsupported DXGI Format: %s" % fmt)


components_pattern = re.compile(r"""(?<![0-9])[0-9]+(?![0-9])""")


def format_components(fmt):
    return len(components_pattern.findall(fmt))


def format_size(fmt):
    matches = components_pattern.findall(fmt)
    return sum(map(int, matches)) // 8


class InputLayoutElement(object):
    def __init__(self, arg):
        self.RemappedSemanticName = None
        self.RemappedSemanticIndex = None
        if isinstance(arg, io.IOBase):
            self.from_file(arg)
        else:
            self.from_dict(arg)

        self.encoder, self.decoder = EncoderDecoder(self.Format)

    def from_file(self, f):
        self.SemanticName = self.next_validate(f, "SemanticName")
        self.SemanticIndex = int(self.next_validate(f, "SemanticIndex"))
        (self.RemappedSemanticName, line) = self.next_optional(
            f, "RemappedSemanticName"
        )
        if line is None:
            self.RemappedSemanticIndex = int(
                self.next_validate(f, "RemappedSemanticIndex")
            )
        self.Format = self.next_validate(f, "Format", line)
        self.InputSlot = int(self.next_validate(f, "InputSlot"))
        self.AlignedByteOffset = self.next_validate(f, "AlignedByteOffset")
        if self.AlignedByteOffset == "append":
            raise Fatal(
                'Input layouts using "AlignedByteOffset=append" are not yet supported'
            )
        self.AlignedByteOffset = int(self.AlignedByteOffset)
        self.InputSlotClass = self.next_validate(f, "InputSlotClass")
        self.InstanceDataStepRate = int(self.next_validate(f, "InstanceDataStepRate"))
        self.format_len = format_components(self.Format)

    def to_dict(self):
        d = {}
        d["SemanticName"] = self.SemanticName
        d["SemanticIndex"] = self.SemanticIndex
        if self.RemappedSemanticName is not None:
            d["RemappedSemanticName"] = self.RemappedSemanticName
            d["RemappedSemanticIndex"] = self.RemappedSemanticIndex
        d["Format"] = self.Format
        d["InputSlot"] = self.InputSlot
        d["AlignedByteOffset"] = self.AlignedByteOffset
        d["InputSlotClass"] = self.InputSlotClass
        d["InstanceDataStepRate"] = self.InstanceDataStepRate
        return d

    def to_string(self, indent=2):
        ret = textwrap.dedent("""
            SemanticName: %s
            SemanticIndex: %i
        """).lstrip() % (
            self.SemanticName,
            self.SemanticIndex,
        )
        if self.RemappedSemanticName is not None:
            ret += textwrap.dedent("""
                RemappedSemanticName: %s
                RemappedSemanticIndex: %i
            """).lstrip() % (
                self.RemappedSemanticName,
                self.RemappedSemanticIndex,
            )
        ret += textwrap.dedent("""
            Format: %s
            InputSlot: %i
            AlignedByteOffset: %i
            InputSlotClass: %s
            InstanceDataStepRate: %i
        """).lstrip() % (
            self.Format,
            self.InputSlot,
            self.AlignedByteOffset,
            self.InputSlotClass,
            self.InstanceDataStepRate,
        )
        return textwrap.indent(ret, " " * indent)

    def from_dict(self, d):
        self.SemanticName = d["SemanticName"]
        self.SemanticIndex = d["SemanticIndex"]
        try:
            self.RemappedSemanticName = d["RemappedSemanticName"]
            self.RemappedSemanticIndex = d["RemappedSemanticIndex"]
        except KeyError:
            pass
        self.Format = d["Format"]
        self.InputSlot = d["InputSlot"]
        self.AlignedByteOffset = d["AlignedByteOffset"]
        self.InputSlotClass = d["InputSlotClass"]
        self.InstanceDataStepRate = d["InstanceDataStepRate"]
        self.format_len = format_components(self.Format)

    @staticmethod
    def next_validate(f, field, line=None):
        if line is None:
            line = next(f).strip()
        assert line.startswith(field + ": ")
        return line[len(field) + 2 :]

    @staticmethod
    def next_optional(f, field, line=None):
        if line is None:
            line = next(f).strip()
        if line.startswith(field + ": "):
            return (line[len(field) + 2 :], None)
        return (None, line)

    @property
    def name(self):
        if self.SemanticIndex:
            return "%s%i" % (self.SemanticName, self.SemanticIndex)
        return self.SemanticName

    @property
    def remapped_name(self):
        if self.RemappedSemanticName is None:
            return self.name
        if self.RemappedSemanticIndex:
            return "%s%i" % (self.RemappedSemanticName, self.RemappedSemanticIndex)
        return self.RemappedSemanticName

    def pad(self, data, val):
        padding = self.format_len - len(data)
        assert padding >= 0
        data.extend([val] * padding)
        return data

    def clip(self, data):
        return data[: format_components(self.Format)]

    def size(self):
        return format_size(self.Format)

    def is_float(self):
        return misc_float_pattern.match(self.Format)

    def is_int(self):
        return misc_int_pattern.match(self.Format)

    def encode(self, data):
        # print(self.Format, data)
        return self.encoder(data)

    def decode(self, data):
        return self.decoder(data)

    def __eq__(self, other):
        return (
            self.SemanticName == other.SemanticName
            and self.SemanticIndex == other.SemanticIndex
            and self.Format == other.Format
            and self.InputSlot == other.InputSlot
            and self.AlignedByteOffset == other.AlignedByteOffset
            and self.InputSlotClass == other.InputSlotClass
            and self.InstanceDataStepRate == other.InstanceDataStepRate
        )


class InputLayout(object):
    def __init__(self, custom_prop=[]):
        self.semantic_translations_cache = None
        self.elems = collections.OrderedDict()
        for item in custom_prop:
            elem = InputLayoutElement(item)
            self.elems[elem.name] = elem

    def serialise(self):
        return [x.to_dict() for x in self.elems.values()]

    def to_string(self):
        ret = ""
        for i, elem in enumerate(self.elems.values()):
            ret += "element[%i]:\n" % i
            ret += elem.to_string()
        return ret

    def parse_element(self, f):
        elem = InputLayoutElement(f)
        self.elems[elem.name] = elem

    def __iter__(self):
        return iter(self.elems.values())

    def __getitem__(self, semantic):
        return self.elems[semantic]

    def untranslate_semantic(
        self, translated_semantic_name, translated_semantic_index=0
    ):
        semantic_translations = self.get_semantic_remap()
        reverse_semantic_translations = {v: k for k, v in semantic_translations.items()}
        semantic = reverse_semantic_translations[
            (translated_semantic_name, translated_semantic_index)
        ]
        return self[semantic]

    def encode(self, vertex, vbuf_idx, stride):
        buf = bytearray(stride)

        for semantic, data in vertex.items():
            if semantic.startswith("~"):
                continue
            elem = self.elems[semantic]
            if vbuf_idx.isnumeric() and elem.InputSlot != int(vbuf_idx):
                # Belongs to a different vertex buffer
                continue
            data = elem.encode(data)
            buf[elem.AlignedByteOffset : elem.AlignedByteOffset + len(data)] = data

        assert len(buf) == stride
        return buf

    def decode(self, buf, vbuf_idx):
        vertex = {}
        for elem in self.elems.values():
            if elem.InputSlot != vbuf_idx:
                # Belongs to a different vertex buffer
                continue
            data = buf[elem.AlignedByteOffset : elem.AlignedByteOffset + elem.size()]
            vertex[elem.name] = elem.decode(data)
        return vertex

    def __eq__(self, other):
        return self.elems == other.elems

    def apply_semantic_remap(self, operator):
        semantic_translations = {}
        semantic_highest_indices = {}

        for elem in self.elems.values():
            semantic_highest_indices[elem.SemanticName.upper()] = max(
                semantic_highest_indices.get(elem.SemanticName.upper(), 0),
                elem.SemanticIndex,
            )

        def find_free_elem_index(semantic):
            idx = semantic_highest_indices.get(semantic, -1) + 1
            semantic_highest_indices[semantic] = idx
            return idx

        for remap in operator.properties.semantic_remap:
            if remap.semantic_to == "None":
                continue
            if remap.semantic_from in semantic_translations:
                operator.report(
                    {"ERROR"},
                    "semantic remap for {} specified multiple times, only the first will be used".format(
                        remap.semantic_from
                    ),
                )
                continue
            if remap.semantic_from not in self.elems:
                operator.report(
                    {"WARNING"},
                    'semantic "{}" not found in imported file, double check your semantic remaps'.format(
                        remap.semantic_from
                    ),
                )
                continue

            remapped_semantic_idx = find_free_elem_index(remap.semantic_to)

            operator.report(
                {"INFO"},
                "Remapping semantic {} -> {}{}".format(
                    remap.semantic_from, remap.semantic_to, remapped_semantic_idx or ""
                ),
            )

            self.elems[remap.semantic_from].RemappedSemanticName = remap.semantic_to
            self.elems[
                remap.semantic_from
            ].RemappedSemanticIndex = remapped_semantic_idx
            semantic_translations[remap.semantic_from] = (
                remap.semantic_to,
                remapped_semantic_idx,
            )

        self.semantic_translations_cache = semantic_translations
        return semantic_translations

    def get_semantic_remap(self):
        if self.semantic_translations_cache:
            return self.semantic_translations_cache
        semantic_translations = {}
        for elem in self.elems.values():
            if elem.RemappedSemanticName is not None:
                semantic_translations[elem.name] = (
                    elem.RemappedSemanticName,
                    elem.RemappedSemanticIndex,
                )
        self.semantic_translations_cache = semantic_translations
        return semantic_translations


class HashableVertex(dict):
    def __hash__(self):
        # Convert keys and values into immutable types that can be hashed
        immutable = tuple((k, tuple(v)) for k, v in sorted(self.items()))
        return hash(immutable)


class IndividualVertexBuffer(object):
    """
    One individual vertex buffer. Multiple vertex buffers may contain
    individual semantics which when combined together make up a vertex buffer
    group.
    """

    vb_elem_pattern = re.compile(
        r"""vb\d+\[\d*\]\+\d+ (?P<semantic>[^:]+): (?P<data>.*)$"""
    )

    def __init__(self, idx, f=None, layout=None, load_vertices=True):
        self.vertices = []
        self.layout = layout and layout or InputLayout()
        self.first = 0
        self.vertex_count = 0
        self.offset = 0
        self.topology = "trianglelist"
        self.stride = 0
        self.idx = idx

        if f is not None:
            self.parse_vb_txt(f, load_vertices)

    def parse_vb_txt(self, f, load_vertices):
        split_vb_stride = "vb%i stride:" % self.idx
        for line in map(str.strip, f):
            # print(line)
            if line.startswith("byte offset:"):
                self.offset = int(line[13:])
            if line.startswith("first vertex:"):
                self.first = int(line[14:])
            if line.startswith("vertex count:"):
                self.vertex_count = int(line[14:])
            if line.startswith("stride:"):
                self.stride = int(line[7:])
            if line.startswith(split_vb_stride):
                self.stride = int(line[len(split_vb_stride) :])
            if line.startswith("element["):
                self.layout.parse_element(f)
            if line.startswith("topology:"):
                self.topology = line[10:]
                if self.topology not in supported_topologies:
                    raise Fatal('"%s" is not yet supported' % line)
            if line.startswith("vertex-data:"):
                if not load_vertices:
                    return
                self.parse_vertex_data(f)
        # If the buffer is only per-instance elements there won't be any
        # vertices. If the buffer has any per-vertex elements than we should
        # have the number of vertices declared in the header.
        if self.vertices:
            assert len(self.vertices) == self.vertex_count

    def parse_vb_bin(self, f, use_drawcall_range=False):
        f.seek(self.offset)
        if use_drawcall_range:
            f.seek(self.first * self.stride, 1)
        else:
            self.first = 0
        for i in itertools.count():
            if use_drawcall_range and i == self.vertex_count:
                break
            vertex = f.read(self.stride)
            if not vertex:
                break
            self.vertices.append(self.layout.decode(vertex, self.idx))
        # We intentionally disregard the vertex count when loading from a
        # binary file, as we assume frame analysis might have only dumped a
        # partial buffer to the .txt files (e.g. if this was from a dump where
        # the draw call index count was overridden it may be cut short, or
        # where the .txt files contain only sub-meshes from each draw call and
        # we are loading the .buf file because it contains the entire mesh):
        self.vertex_count = len(self.vertices)

    def append(self, vertex):
        self.vertices.append(vertex)
        self.vertex_count += 1

    def parse_vertex_data(self, f):
        vertex = {}
        for line in map(str.strip, f):
            # print(line)
            if line.startswith("instance-data:"):
                break

            match = self.vb_elem_pattern.match(line)
            if match:
                vertex[match.group("semantic")] = self.parse_vertex_element(match)
            elif line == "" and vertex:
                self.vertices.append(vertex)
                vertex = {}
        if vertex:
            self.vertices.append(vertex)

    @staticmethod
    def ms_float(val):
        x = val.split(".#")
        s = float(x[0])
        if len(x) == 1:
            return s
        if x[1].startswith("INF"):
            return s * numpy.inf  # Will preserve sign
        # TODO: Differentiate between SNAN / QNAN / IND
        if s == -1:  # Multiplying -1 * nan doesn't preserve sign
            return -numpy.nan  # so must use unary - operator
        return numpy.nan

    def parse_vertex_element(self, match):
        fields = match.group("data").split(",")

        if self.layout[match.group("semantic")].Format.endswith("INT"):
            return tuple(map(int, fields))

        return tuple(map(self.ms_float, fields))


class VertexBufferGroup(object):
    """
    All the per-vertex data, which may be loaded/saved from potentially
    multiple individual vertex buffers with different semantics in each.
    """

    vb_idx_pattern = re.compile(r"""[-\.]vb([0-9]+)""")

    # Python gotcha - do not set layout=InputLayout() in the default function
    # parameters, as they would all share the *same* InputLayout since the
    # default values are only evaluated once on file load
    def __init__(self, files=None, layout=None, load_vertices=True, topology=None):
        self.vertices = []
        self.layout = layout and layout or InputLayout()
        self.first = 0
        self.vertex_count = 0
        self.topology = topology or "trianglelist"
        self.vbs = []
        self.slots = {}

        if files is not None:
            self.parse_vb_txt(files, load_vertices)

    def parse_vb_txt(self, files, load_vertices):
        for f in files:
            match = self.vb_idx_pattern.search(f)
            if match is None:
                raise Fatal("Cannot determine vertex buffer index from filename %s" % f)
            idx = int(match.group(1))
            vb = IndividualVertexBuffer(idx, open(f, "r"), self.layout, load_vertices)
            if vb.vertices:
                self.vbs.append(vb)
                self.slots[idx] = vb

        self.flag_invalid_semantics()

        # Non buffer specific info:
        self.first = self.vbs[0].first
        self.vertex_count = self.vbs[0].vertex_count
        self.topology = self.vbs[0].topology

        if load_vertices:
            self.merge_vbs(self.vbs)
            assert len(self.vertices) == self.vertex_count

    def parse_vb_bin(self, files, use_drawcall_range=False):
        for bin_f, fmt_f in files:
            match = self.vb_idx_pattern.search(bin_f)
            if match is not None:
                idx = int(match.group(1))
            else:
                print(
                    "Cannot determine vertex buffer index from filename %s, assuming 0 for backwards compatibility"
                    % bin_f
                )
                idx = 0
            vb = IndividualVertexBuffer(idx, open(fmt_f, "r"), self.layout, False)
            vb.parse_vb_bin(open(bin_f, "rb"), use_drawcall_range)
            if vb.vertices:
                self.vbs.append(vb)
                self.slots[idx] = vb

        self.flag_invalid_semantics()

        # Non buffer specific info:
        self.first = self.vbs[0].first
        self.vertex_count = self.vbs[0].vertex_count
        self.topology = self.vbs[0].topology

        self.merge_vbs(self.vbs)
        assert len(self.vertices) == self.vertex_count

    def append(self, vertex):
        self.vertices.append(vertex)
        self.vertex_count += 1

    def remap_blendindices(self, obj, mapping):
        def lookup_vgmap(x):
            vgname = obj.vertex_groups[x].name
            return mapping.get(vgname, mapping.get(x, x))

        for vertex in self.vertices:
            for semantic in list(vertex):
                if semantic.startswith("BLENDINDICES"):
                    vertex["~" + semantic] = vertex[semantic]
                    vertex[semantic] = tuple(lookup_vgmap(x) for x in vertex[semantic])

    def revert_blendindices_remap(self):
        # Significantly faster than doing a deep copy
        for vertex in self.vertices:
            for semantic in list(vertex):
                if semantic.startswith("BLENDINDICES"):
                    vertex[semantic] = vertex["~" + semantic]
                    del vertex["~" + semantic]

    def disable_blendweights(self):
        for vertex in self.vertices:
            for semantic in list(vertex):
                if semantic.startswith("BLENDINDICES"):
                    vertex[semantic] = (0, 0, 0, 0)

    def write(self, output_prefix, strides, operator=None):
        for vbuf_idx, stride in strides.items():
            with open(output_prefix + vbuf_idx, "wb") as output:
                for vertex in self.vertices:
                    output.write(self.layout.encode(vertex, vbuf_idx, stride))

                msg = "Wrote %i vertices to %s" % (len(self), output.name)
                if operator:
                    operator.report({"INFO"}, msg)
                else:
                    print(msg)

    def __len__(self):
        return len(self.vertices)

    def merge_vbs(self, vbs):
        self.vertices = self.vbs[0].vertices
        del self.vbs[0].vertices
        assert len(self.vertices) == self.vertex_count
        for vb in self.vbs[1:]:
            assert len(vb.vertices) == self.vertex_count
            [self.vertices[i].update(vertex) for i, vertex in enumerate(vb.vertices)]
            del vb.vertices

    def merge(self, other):
        if self.layout != other.layout:
            raise Fatal(
                "Vertex buffers have different input layouts - ensure you are only trying to merge the same vertex buffer split across multiple draw calls"
            )
        if self.first != other.first:
            # FIXME: Future 3DMigoto might automatically set first from the
            # index buffer and chop off unreferenced vertices to save space
            raise Fatal(
                "Cannot merge multiple vertex buffers - please check for updates of the 3DMigoto import script, or import each buffer separately"
            )
        self.vertices.extend(other.vertices[self.vertex_count :])
        self.vertex_count = max(self.vertex_count, other.vertex_count)
        assert len(self.vertices) == self.vertex_count

    def wipe_semantic_for_testing(self, semantic, val=0):
        print("WARNING: WIPING %s FOR TESTING PURPOSES!!!" % semantic)
        semantic, _, components = semantic.partition(".")
        if components:
            components = [{"x": 0, "y": 1, "z": 2, "w": 3}[c] for c in components]
        else:
            components = range(4)
        for vertex in self.vertices:
            for s in list(vertex):
                if s == semantic:
                    v = list(vertex[semantic])
                    for component in components:
                        if component < len(v):
                            v[component] = val
                    vertex[semantic] = v

    def flag_invalid_semantics(self):
        # This refactors some of the logic that used to be in import_vertices()
        # and get_valid_semantics() - Any semantics that re-use the same offset
        # of an earlier semantic is considered invalid and will be ignored when
        # importing the vertices. These are usually a quirk of how certain
        # engines handle unused semantics and at best will be repeating data we
        # already imported in another semantic and at worst may be
        # misinterpreting the data as a completely different type.
        #
        # Is is theoretically possible for the earlier semantic to be the
        # invalid one - if we ever encounter that we might want to allow the
        # user to choose which of the semantics sharing the same offset should
        # be considerd the valid one.
        #
        # This also makes sure the corresponding vertex buffer is present and
        # can fit the semantic.
        seen_offsets = set()
        for elem in self.layout:
            if elem.InputSlotClass != "per-vertex":
                # Instance data isn't invalid, we just don't import it yet
                continue
            if (elem.InputSlot, elem.AlignedByteOffset) in seen_offsets:
                # Setting two flags to avoid changing behaviour in the refactor
                # - might be able to simplify this to one flag, but want to
                # test semantics that [partially] overflow the stride first,
                # and make sure that export flow (stride won't be set) works.
                elem.reused_offset = True
                elem.invalid_semantic = True
                continue
            seen_offsets.add((elem.InputSlot, elem.AlignedByteOffset))
            elem.reused_offset = False

            try:
                stride = self.slots[elem.InputSlot].stride
            except KeyError:
                # UE4 claiming it uses vertex buffers that it doesn't bind.
                elem.invalid_semantic = True
                continue

            if elem.AlignedByteOffset + format_size(elem.Format) > stride:
                elem.invalid_semantic = True
                continue

            elem.invalid_semantic = False

    def get_valid_semantics(self):
        self.flag_invalid_semantics()
        return set(
            [
                elem.name
                for elem in self.layout
                if elem.InputSlotClass == "per-vertex" and not elem.invalid_semantic
            ]
        )


class IndexBuffer(object):
    def __init__(self, *args, load_indices=True):
        self.faces = []
        self.first = 0
        self.index_count = 0
        self.format = "DXGI_FORMAT_UNKNOWN"
        self.offset = 0
        self.topology = "trianglelist"
        self.used_in_drawcall = None

        if isinstance(args[0], io.IOBase):
            assert len(args) == 1
            self.parse_ib_txt(args[0], load_indices)
        else:
            (self.format,) = args

        self.encoder, self.decoder = EncoderDecoder(self.format)

    def append(self, face):
        self.faces.append(face)
        self.index_count += len(face)

    def parse_ib_txt(self, f, load_indices):
        for line in map(str.strip, f):
            if line.startswith("byte offset:"):
                self.offset = int(line[13:])
                # If we see this line we are looking at a 3DMigoto frame
                # analysis dump, not a .fmt file exported by this script.
                # If it was an indexed draw call it will be followed by "first
                # index" and "index count", while if it was not an indexed draw
                # call they will be absent. So by the end of parsing:
                # used_in_drawcall = None signifies loading a .fmt file from a previous export
                # used_in_drawcall = False signifies draw call did not use the bound IB
                # used_in_drawcall = True signifies an indexed draw call
                self.used_in_drawcall = False
            if line.startswith("first index:"):
                self.first = int(line[13:])
                self.used_in_drawcall = True
            elif line.startswith("index count:"):
                self.index_count = int(line[13:])
                self.used_in_drawcall = True
            elif line.startswith("topology:"):
                self.topology = line[10:]
                if self.topology not in supported_topologies:
                    raise Fatal('"%s" is not yet supported' % line)
            elif line.startswith("format:"):
                self.format = line[8:]
            elif line == "":
                if not load_indices:
                    return
                self.parse_index_data(f)
        if self.used_in_drawcall is not False:
            assert (
                len(self.faces) * self.indices_per_face + self.extra_indices
                == self.index_count
            )

    def parse_ib_bin(self, f, use_drawcall_range=False):
        f.seek(self.offset)
        stride = format_size(self.format)
        if use_drawcall_range:
            f.seek(self.first * stride, 1)
        else:
            self.first = 0

        face = []
        for i in itertools.count():
            if use_drawcall_range and i == self.index_count:
                break
            index = f.read(stride)
            if not index:
                break
            face.append(*self.decoder(index))
            if len(face) == self.indices_per_face:
                self.faces.append(tuple(face))
                face = []
        assert len(face) == 0, "Index buffer has incomplete face at end of file"
        self.expand_strips()

        if use_drawcall_range:
            assert (
                len(self.faces) * self.indices_per_face + self.extra_indices
                == self.index_count
            )
        else:
            # We intentionally disregard the index count when loading from a
            # binary file, as we assume frame analysis might have only dumped a
            # partial buffer to the .txt files (e.g. if this was from a dump where
            # the draw call index count was overridden it may be cut short, or
            # where the .txt files contain only sub-meshes from each draw call and
            # we are loading the .buf file because it contains the entire mesh):
            self.index_count = (
                len(self.faces) * self.indices_per_face + self.extra_indices
            )

    def parse_index_data(self, f):
        for line in map(str.strip, f):
            face = tuple(map(int, line.split()))
            assert len(face) == self.indices_per_face
            self.faces.append(face)
        self.expand_strips()

    def expand_strips(self):
        if self.topology == "trianglestrip":
            # Every 2nd face has the vertices out of order to keep all faces in the same orientation:
            # https://learn.microsoft.com/en-us/windows/win32/direct3d9/triangle-strips
            self.faces = [
                (
                    self.faces[i - 2][0],
                    self.faces[i % 2 and i or i - 1][0],
                    self.faces[i % 2 and i - 1 or i][0],
                )
                for i in range(2, len(self.faces))
            ]
        elif self.topology == "linestrip":
            raise Fatal("linestrip topology conversion is untested")
            self.faces = [
                (self.faces[i - 1][0], self.faces[i][0])
                for i in range(1, len(self.faces))
            ]

    def merge(self, other):
        if self.format != other.format:
            raise Fatal(
                "Index buffers have different formats - ensure you are only trying to merge the same index buffer split across multiple draw calls"
            )
        self.first = min(self.first, other.first)
        self.index_count += other.index_count
        self.faces.extend(other.faces)

    def write(self, output, operator=None):
        for face in self.faces:
            output.write(self.encoder(face))

        msg = "Wrote %i indices to %s" % (len(self), output.name)
        if operator:
            operator.report({"INFO"}, msg)
        else:
            print(msg)

    @property
    def indices_per_face(self):
        return {
            "trianglelist": 3,
            "pointlist": 1,
            "trianglestrip": 1,  # + self.extra_indices for 1st tri
            "linelist": 2,
            "linestrip": 1,  # + self.extra_indices for 1st line
        }[self.topology]

    @property
    def extra_indices(self):
        if len(self.faces) >= 1:
            if self.topology == "trianglestrip":
                return 2
            if self.topology == "linestrip":
                return 1
        return 0

    def __len__(self):
        return len(self.faces) * self.indices_per_face + self.extra_indices


class ConstantBuffer(object):
    def __init__(self, f, start_idx, end_idx):
        self.entries = []
        entry = []
        i = 0
        for line in map(str.strip, f):
            if line.startswith("buf") or line.startswith("cb"):
                entry.append(float(line.split()[1]))
                if len(entry) == 4:
                    if i >= start_idx:
                        self.entries.append(entry)
                    else:
                        print("Skipping", entry)
                    entry = []
                    i += 1
                    if end_idx and i > end_idx:
                        break
        assert entry == []

    def as_3x4_matrices(self):
        return [Matrix(self.entries[i : i + 3]) for i in range(0, len(self.entries), 3)]


class FALogFile(object):
    """
    Class that is able to parse frame analysis log files, query bound resource
    state at the time of a given draw call, and search for resource usage.

    TODO: Support hold frame analysis log files that include multiple frames
    TODO: Track bound shaders
    TODO: Merge deferred context log files into main log file
    TODO: Track CopyResource / other ways resources can be updated
    """

    ResourceUse = collections.namedtuple(
        "ResourceUse", ["draw_call", "slot_type", "slot"]
    )

    class SparseSlots(dict):
        """
        Allows the resources bound in each slot to be stored + queried by draw
        call. There can be gaps with draw calls that don't change any of the
        given slot type, in which case it will return the slots in the most
        recent draw call that did change that slot type.

        Requesting a draw call higher than any seen so far will return a *copy*
        of the most recent slots, intended for modification during parsing.
        """

        def __init__(self):
            dict.__init__(self, {0: {}})
            self.last_draw_call = 0

        def prev_draw_call(self, draw_call):
            return max([i for i in self.keys() if i < draw_call])

        # def next_draw_call(self, draw_call):
        #    return min([ i for i in self.keys() if i > draw_call ])
        def subsequent_draw_calls(self, draw_call):
            return [i for i in sorted(self.keys()) if i >= draw_call]

        def __getitem__(self, draw_call):
            if draw_call > self.last_draw_call:
                dict.__setitem__(
                    self, draw_call, dict.__getitem__(self, self.last_draw_call).copy()
                )
                self.last_draw_call = draw_call
            elif draw_call not in self.keys():
                return dict.__getitem__(self, self.prev_draw_call(draw_call))
            return dict.__getitem__(self, draw_call)

    class FALogParser(object):
        """
        Base class implementing some common parsing functions
        """

        pattern = None

        def parse(self, line, q, state):
            match = self.pattern.match(line)
            if match:
                remain = line[match.end() :]
                self.matched(match, remain, q, state)
            return match

        def matched(self, match, remain, q, state):
            raise NotImplementedError()

    class FALogParserDrawcall(FALogParser):
        """
        Parses a typical line in a frame analysis log file that begins with a
        draw call number. Additional parsers can be registered with this one to
        parse the remainder of such lines.
        """

        pattern = re.compile(r"""^(?P<drawcall>\d+) """)
        next_parsers_classes = []

        @classmethod
        def register(cls, parser):
            cls.next_parsers_classes.append(parser)

        def __init__(self, state):
            self.next_parsers = []
            for parser in self.next_parsers_classes:
                self.next_parsers.append(parser(state))

        def matched(self, match, remain, q, state):
            drawcall = int(match.group("drawcall"))
            state.draw_call = drawcall
            for parser in self.next_parsers:
                parser.parse(remain, q, state)

    class FALogParserBindResources(FALogParser):
        """
        Base class for any parsers that bind resources (and optionally views)
        to the pipeline. Will consume all following lines matching the resource
        pattern and update the log file state and resource lookup index for the
        current draw call.
        """

        resource_pattern = re.compile(
            r"""^\s+(?P<slot>[0-9D]+): (?:view=(?P<view>0x[0-9A-F]+) )?resource=(?P<address>0x[0-9A-F]+) hash=(?P<hash>[0-9a-f]+)$""",
            re.MULTILINE,
        )
        FALogResourceBinding = collections.namedtuple(
            "FALogResourceBinding",
            ["slot", "view_address", "resource_address", "resource_hash"],
        )
        slot_prefix = None
        bind_clears_all_slots = False

        def __init__(self, state):
            if self.slot_prefix is None:
                raise NotImplementedError()
            self.sparse_slots = FALogFile.SparseSlots()
            state.slot_class[self.slot_prefix] = self.sparse_slots

        def matched(self, api_match, remain, q, state):
            if self.bind_clears_all_slots:
                self.sparse_slots[state.draw_call].clear()
            else:
                start_slot = self.start_slot(api_match)
                for i in range(self.num_bindings(api_match)):
                    self.sparse_slots[state.draw_call].pop(start_slot + i, None)
            bindings = self.sparse_slots[state.draw_call]
            while self.resource_pattern.match(q[0]):
                # FIXME: Inefficiently calling match twice. I hate that Python
                # lacks a do/while and every workaround is ugly in some way.
                resource_match = self.resource_pattern.match(q.popleft())
                slot = resource_match.group("slot")
                if slot.isnumeric():
                    slot = int(slot)
                view = resource_match.group("view")
                if view:
                    view = int(view, 16)
                address = int(resource_match.group("address"), 16)
                resource_hash = int(resource_match.group("hash"), 16)
                bindings[slot] = self.FALogResourceBinding(
                    slot, view, address, resource_hash
                )
                state.resource_index[address].add(
                    FALogFile.ResourceUse(state.draw_call, self.slot_prefix, slot)
                )
            # print(sorted(bindings.items()))

        def start_slot(self, match):
            return int(match.group("StartSlot"))

        def num_bindings(self, match):
            return int(match.group("NumBindings"))

    class FALogParserSOSetTargets(FALogParserBindResources):
        pattern = re.compile(r"""SOSetTargets\(.*\)$""")
        slot_prefix = "so"
        bind_clears_all_slots = True

    FALogParserDrawcall.register(FALogParserSOSetTargets)

    class FALogParserIASetVertexBuffers(FALogParserBindResources):
        pattern = re.compile(
            r"""IASetVertexBuffers\(StartSlot:(?P<StartSlot>[0-9]+), NumBuffers:(?P<NumBindings>[0-9]+),.*\)$"""
        )
        slot_prefix = "vb"

    FALogParserDrawcall.register(FALogParserIASetVertexBuffers)

    # At the moment we don't need to track other pipeline slots, so to keep
    # things faster and use less memory we don't bother with slots we don't
    # need to know about. but if we wanted to the above makes it fairly trivial
    # to add additional slot classes, e.g. to track bound texture slots (SRVs)
    # for all shader types uncomment the following:
    # class FALogParserVSSetShaderResources(FALogParserBindResources):
    #    pattern = re.compile(r'''VSSetShaderResources\(StartSlot:(?P<StartSlot>[0-9]+), NumViews:(?P<NumBindings>[0-9]+),.*\)$''')
    #    slot_prefix = 'vs-t'
    # class FALogParserDSSetShaderResources(FALogParserBindResources):
    #    pattern = re.compile(r'''DSSetShaderResources\(StartSlot:(?P<StartSlot>[0-9]+), NumViews:(?P<NumBindings>[0-9]+),.*\)$''')
    #    slot_prefix = 'ds-t'
    # class FALogParserHSSetShaderResources(FALogParserBindResources):
    #    pattern = re.compile(r'''HSSetShaderResources\(StartSlot:(?P<StartSlot>[0-9]+), NumViews:(?P<NumBindings>[0-9]+),.*\)$''')
    #    slot_prefix = 'hs-t'
    # class FALogParserGSSetShaderResources(FALogParserBindResources):
    #    pattern = re.compile(r'''GSSetShaderResources\(StartSlot:(?P<StartSlot>[0-9]+), NumViews:(?P<NumBindings>[0-9]+),.*\)$''')
    #    slot_prefix = 'gs-t'
    # class FALogParserPSSetShaderResources(FALogParserBindResources):
    #    pattern = re.compile(r'''PSSetShaderResources\(StartSlot:(?P<StartSlot>[0-9]+), NumViews:(?P<NumBindings>[0-9]+),.*\)$''')
    #    slot_prefix = 'ps-t'
    # class FALogParserCSSetShaderResources(FALogParserBindResources):
    #    pattern = re.compile(r'''CSSetShaderResources\(StartSlot:(?P<StartSlot>[0-9]+), NumViews:(?P<NumBindings>[0-9]+),.*\)$''')
    #    slot_prefix = 'cs-t'
    # FALogParserDrawcall.register(FALogParserVSSetShaderResources)
    # FALogParserDrawcall.register(FALogParserDSSetShaderResources)
    # FALogParserDrawcall.register(FALogParserHSSetShaderResources)
    # FALogParserDrawcall.register(FALogParserGSSetShaderResources)
    # FALogParserDrawcall.register(FALogParserPSSetShaderResources)
    # FALogParserDrawcall.register(FALogParserCSSetShaderResources)

    # Uncomment these to track bound constant buffers:
    # class FALogParserVSSetConstantBuffers(FALogParserBindResources):
    #    pattern = re.compile(r'''VSSetConstantBuffers\(StartSlot:(?P<StartSlot>[0-9]+), NumBuffers:(?P<NumBindings>[0-9]+),.*\)$''')
    #    slot_prefix = 'vs-cb'
    # class FALogParserDSSetConstantBuffers(FALogParserBindResources):
    #    pattern = re.compile(r'''DSSetConstantBuffers\(StartSlot:(?P<StartSlot>[0-9]+), NumBuffers:(?P<NumBindings>[0-9]+),.*\)$''')
    #    slot_prefix = 'ds-cb'
    # class FALogParserHSSetConstantBuffers(FALogParserBindResources):
    #    pattern = re.compile(r'''HSSetConstantBuffers\(StartSlot:(?P<StartSlot>[0-9]+), NumBuffers:(?P<NumBindings>[0-9]+),.*\)$''')
    #    slot_prefix = 'hs-cb'
    # class FALogParserGSSetConstantBuffers(FALogParserBindResources):
    #    pattern = re.compile(r'''GSSetConstantBuffers\(StartSlot:(?P<StartSlot>[0-9]+), NumBuffers:(?P<NumBindings>[0-9]+),.*\)$''')
    #    slot_prefix = 'gs-cb'
    # class FALogParserPSSetConstantBuffers(FALogParserBindResources):
    #    pattern = re.compile(r'''PSSetConstantBuffers\(StartSlot:(?P<StartSlot>[0-9]+), NumBuffers:(?P<NumBindings>[0-9]+),.*\)$''')
    #    slot_prefix = 'ps-cb'
    # class FALogParserCSSetConstantBuffers(FALogParserBindResources):
    #    pattern = re.compile(r'''CSSetConstantBuffers\(StartSlot:(?P<StartSlot>[0-9]+), NumBuffers:(?P<NumBindings>[0-9]+),.*\)$''')
    #    slot_prefix = 'cs-cb'
    # FALogParserDrawcall.register(FALogParserVSSetConstantBuffers)
    # FALogParserDrawcall.register(FALogParserDSSetConstantBuffers)
    # FALogParserDrawcall.register(FALogParserHSSetConstantBuffers)
    # FALogParserDrawcall.register(FALogParserGSSetConstantBuffers)
    # FALogParserDrawcall.register(FALogParserPSSetConstantBuffers)
    # FALogParserDrawcall.register(FALogParserCSSetConstantBuffers)

    # Uncomment to tracks render targets (note that this doesn't yet take into
    # account games using OMSetRenderTargetsAndUnorderedAccessViews)
    # class FALogParserOMSetRenderTargets(FALogParserBindResources):
    #    pattern = re.compile(r'''OMSetRenderTargets\(NumViews:(?P<NumBindings>[0-9]+),.*\)$''')
    #    slot_prefix = 'o'
    #    bind_clears_all_slots = True
    # FALogParserDrawcall.register(FALogParserOMSetRenderTargets)

    def __init__(self, f):
        self.draw_call = None
        self.slot_class = {}
        self.resource_index = collections.defaultdict(set)
        draw_call_parser = self.FALogParserDrawcall(self)
        # Using a deque for a concise way to use a pop iterator and be able to
        # peek/consume the following line. Maybe overkill, but shorter code
        q = collections.deque(f)
        q.append(None)
        for line in iter(q.popleft, None):
            # print(line)
            if not draw_call_parser.parse(line, q, self):
                # print(line)
                pass

    def find_resource_uses(self, resource_address, slot_class=None):
        """
        Find draw calls + slots where this resource is used.
        """
        # return [ x for x in sorted(self.resource_index[resource_address]) if x.slot_type == slot_class ]
        ret = set()
        for bound in sorted(self.resource_index[resource_address]):
            if slot_class is not None and bound.slot_type != slot_class:
                continue
            # Resource was bound in this draw call, but could potentially have
            # been left bound in subsequent draw calls that we also want to
            # return, so return a range of draw calls if appropriate:
            sparse_slots = self.slot_class[bound.slot_type]
            for sparse_draw_call in sparse_slots.subsequent_draw_calls(bound.draw_call):
                if (
                    bound.slot not in sparse_slots[sparse_draw_call]
                    or sparse_slots[sparse_draw_call][bound.slot].resource_address
                    != resource_address
                ):
                    # print('x', sparse_draw_call, sparse_slots[sparse_draw_call][bound.slot])
                    for draw_call in range(bound.draw_call, sparse_draw_call):
                        ret.add(
                            FALogFile.ResourceUse(
                                draw_call, bound.slot_type, bound.slot
                            )
                        )
                    break
                # print('y', sparse_draw_call, sparse_slots[sparse_draw_call][bound.slot])
            else:
                # I love Python's for/else clause. Means we didn't hit the
                # above break so the resource was still bound at end of frame
                for draw_call in range(bound.draw_call, self.draw_call):
                    ret.add(
                        FALogFile.ResourceUse(draw_call, bound.slot_type, bound.slot)
                    )
        return ret


VBSOMapEntry = collections.namedtuple("VBSOMapEntry", ["draw_call", "slot"])
