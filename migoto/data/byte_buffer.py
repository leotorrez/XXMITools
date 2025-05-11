import copy
import textwrap
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Union


import numpy
from numpy.typing import DTypeLike, NDArray

from .dxgi_format import DXGIFormat


class Semantic(str, Enum):
    VertexId = "VERTEXID"
    Index = "INDEX"
    Tangent = "TANGENT"
    BitangentSign = "BITANGENTSIGN"
    Normal = "NORMAL"
    TexCoord = "TEXCOORD"
    Color = "COLOR"
    Position = "POSITION"
    Blendindices = "BLENDINDICES"
    Blendweight = "BLENDWEIGHT"
    ShapeKey = "SHAPEKEY"
    Binormal = "BINORMAL"
    RawData = "RAWDATA"

    def __str__(self) -> str:
        return f"{self.value}"

    def __repr__(self) -> str:
        return f"{self.value}"


@dataclass
class AbstractSemantic:
    enum: Semantic
    index: int = 0

    def __init__(self, semantic, semantic_index=0) -> None:
        self.enum = semantic
        self.index = semantic_index

    def __hash__(self) -> int:
        return hash((self.enum, self.index))

    def __str__(self) -> str:
        return f"{self.enum}_{self.index}"

    def __repr__(self) -> str:
        return f"{self.enum}_{self.index}"

    def get_name(self) -> str:
        name = self.enum.value
        if self.index > 0:
            name += str(self.index)
        if self.enum == Semantic.TexCoord:
            name += ".xy"
        return name


@dataclass
class BufferSemantic:
    abstract: AbstractSemantic
    format: DXGIFormat
    stride: int = 0
    offset: int = 0
    input_slot: int = 0
    data_step_rate: int = 0
    extract_format: Optional[DXGIFormat] = None
    name: Optional[str] = None
    remapped_abstract: Optional[AbstractSemantic] = None

    def __post_init__(self) -> None:
        # Calculate byte stride
        if self.stride == 0:
            self.stride = self.format.byte_width

    def __hash__(self) -> int:
        return hash((self.abstract, self.format.format, self.stride, self.offset))

    def __repr__(self) -> str:
        return f"{self.abstract} ({self.format.format} stride={self.stride} offset={self.offset})"

    def to_string(self, indent=2) -> str:
        return textwrap.indent(
            textwrap.dedent(f"""
            SemanticName: {self.abstract.enum}
            SemanticIndex: {self.abstract.index}
            Format: {self.format.format}
            InputSlot: {self.input_slot}
            AlignedByteOffset: {self.offset}
            InputSlotClass: per-vertex
            InstanceDataStepRate: {self.data_step_rate}
        """).lstrip(),
            " " * indent,
        )

    def get_format(self) -> str:
        return self.format.get_format()

    def get_name(self) -> str:
        return self.name if self.name else self.abstract.get_name()

    def get_num_values(self) -> int:
        return self.format.get_num_values(self.stride)

    def get_numpy_type(self) -> DTypeLike:
        return self.format.get_numpy_type(self.stride)


@dataclass
class BufferLayout:
    semantics: list[BufferSemantic]
    stride: int = 0
    force_stride: bool = False


    def __post_init__(self) -> None:
        # Autofill byte Stride and Offsets
        if self.stride == 0:
            # Calculate byte stride
            for element in self.semantics:
                self.stride += element.stride
            # Calculate byte offsets
            offset = 0
            for element in self.semantics:
                element.offset = offset
                offset += element.stride
        # Autofill Semantic Index
        groups = {}
        for semantic in self.semantics:
            if semantic not in groups:
                groups[semantic] = 0
                continue
            if semantic.abstract.index == 0:
                groups[semantic] += 1
                semantic.abstract.index = groups[semantic]

    def get_element(self, abstract: AbstractSemantic) -> Optional[BufferSemantic]:
        """Returns the first element with the same semantic name and index"""
        for element in self.semantics:
            if abstract == element.abstract:
                return element
        return None

    def add_element(self, semantic: BufferSemantic) -> None:
        """Adds a new element to the layout"""
        if self.get_element(semantic.abstract) is not None:
            return
        semantic = copy.deepcopy(semantic)
        semantic.offset = self.stride
        self.semantics.append(semantic)
        self.stride += semantic.stride

    def merge(self, layout) -> None:
        for semantic in layout.semantics:
            if not self.get_element(semantic):
                self.add_element(semantic)

    def to_string(self) -> str:
        ret = ""
        for i, semantic in enumerate(self.semantics):
            ret += "element[%i]:\n" % i
            ret += semantic.to_string()
        return ret

    def get_numpy_type(self) -> DTypeLike:
        dtype = numpy.dtype([])
        for semantic in self.semantics:
            dtype = numpy.dtype(
                dtype.descr
                + [(semantic.abstract.get_name(), (semantic.get_numpy_type()))]
            )
        return dtype


class NumpyBuffer:
    layout: BufferLayout
    data: NDArray

    def __init__(
        self, layout: BufferLayout, data: Optional[NDArray] = None, size=0
    ) -> None:
        self.set_layout(layout)
        self.set_data(data, size)

    def set_layout(self, layout: BufferLayout) -> None:
        self.layout = layout

    def set_data(self, data: Optional[NDArray], size=0) -> None:
        if data is not None:
            self.data = data
        elif size > 0:
            self.data = numpy.zeros(size, dtype=self.layout.get_numpy_type())

    def set_field(self, field: str, data: NDArray) -> None:
        try:
            self.data[field] = data
        except ValueError as e:
            raise ValueError(
                f"Failed to set field {field}: {e}"
            )

    def get_data(self, indices: Optional[NDArray] = None) -> NDArray:
        if indices is None:
            return self.data
        else:
            return self.data[indices]

    def get_field(self, field: str) -> NDArray:
        return self.data[field]

    def remove_duplicates(self, keep_order=True) -> None:
        if keep_order:
            _, unique_index = numpy.unique(self.data, return_index=True)
            self.data = self.data[numpy.sort(unique_index)]
        else:
            self.data = numpy.unique(self.data)

    def import_semantic_data(
        self,
        data: NDArray,
        semantic: Union[BufferSemantic, int],
        semantic_converters: Optional[list[Callable]] = None,
        format_converters: Optional[list[Callable]] = None,
    ) -> None:
        if isinstance(semantic, int):
            semantic = self.layout.semantics[semantic]
        current_semantic = self.layout.get_element(semantic.abstract)
        if current_semantic is None:
            raise ValueError(
                f"NumpyBuffer is missing {semantic.abstract} semantic data!"
            )
        if semantic_converters is not None:
            for data_converter in semantic_converters:
                data = data_converter(data)
        if current_semantic.format != semantic.format:
            data = current_semantic.format.type_encoder(data)
        if format_converters is not None:
            for data_converter in format_converters:
                data = data_converter(data)
        self.set_field(current_semantic.get_name(), data)

    def import_data(
        self,
        data: "NumpyBuffer",
        semantic_converters: dict[AbstractSemantic, list[Callable]],
        format_converters: dict[AbstractSemantic, list[Callable]],
    ) -> None:
        for buffer_semantic in self.layout.semantics:
            data_semantic = data.layout.get_element(buffer_semantic.abstract)

            if data_semantic is None:
                continue
            field_data = data.get_field(buffer_semantic.get_name())

            self.import_semantic_data(
                field_data,
                data_semantic,
                semantic_converters.get(buffer_semantic.abstract, []),
                format_converters.get(buffer_semantic.abstract, []),
            )

    def import_raw_data(self, data: NDArray) -> None:
        self.data = numpy.frombuffer(data, dtype=self.layout.get_numpy_type())

    def get_bytes(self) -> bytes:
        return self.data.tobytes()

    def __len__(self) -> int:
        return len(self.data)

    def to_file(self, file: Path) -> None:
        """Writes the buffer to a file in the specified format"""
        with open(file, "wb", encoding="utf-8") as f:
            f.write(self.get_bytes())


class MigotoFmt:
    vb_strides: list[int]
    vb_layout: BufferLayout
    ib_layout: BufferLayout

    def __init__(
        self,
        vb_layout: BufferLayout,
        vb_strides: list[int],
        ib_layout: BufferLayout,
    ) -> None:
        self.vb_layout = vb_layout
        self.vb_strides = vb_strides
        self.ib_layout = ib_layout

    @classmethod
    def from_files(cls, fmt_files: list) -> "MigotoFmt":
        """Takes a list of fmt files and parses them into a list of BufferLayouts"""
        vb_strides: list[int] = []
        ib_layout: BufferLayout = BufferLayout([])
        vb_semantics: list[BufferSemantic] = []
        element: Optional[dict] = None
        converters: dict[str, Callable] = {
            "SemanticName": lambda value: Semantic(value),
            "SemanticIndex": lambda value: int(value),
            "Format": lambda value: DXGIFormat(value),
            "AlignedByteOffset": lambda value: int(value),
            "InputSlot": lambda value: int(value),
            "InputSlotClass": lambda value: int(value),
            "InstanceDataStepRate": lambda value: str(value),
        }

        for idx, f in enumerate(fmt_files):
            for line in map(str.strip, f):
                data = line.split(":")
                if len(data) != 2:
                    continue
                data = list(map(str.strip, data))
                key, value = data[0], data[1]

                split_vb_stride = "vb%i stride:" % idx
                if line.startswith(split_vb_stride):
                    vb_strides[idx] = int(line[len(split_vb_stride) :])
                elif line.startswith("stride"):
                    vb_strides[idx] = int(value)
                elif line.startswith("format") and element is None:
                    ib_format = DXGIFormat(value)
                    if ib_format == DXGIFormat.R16_UINT:
                        ib_format = DXGIFormat.R32_UINT
                    ib_layout = BufferLayout(
                        [BufferSemantic(AbstractSemantic(Semantic.Index), ib_format)]
                    )
                elif line.startswith("element["):
                    if element is not None:
                        if len(element) == 4:
                            vb_semantics.append(
                                BufferSemantic(
                                    AbstractSemantic(
                                        Semantic(element["SemanticName"]),
                                        element["SemanticIndex"],
                                    ),
                                    element["Format"],
                                )
                            )
                        elif len(element) > 0:
                            raise ValueError(
                                f"malformed buffer element format: {element}"
                            )
                    element = {}
                else:
                    assert element is not None, f"missing element definition for {key}"
                    for search_key, converter in converters.items():
                        if key.startswith(search_key):
                            element[search_key] = converter(value)
                            break
            assert element is not None, "missing element definition for last element"
            if len(element) == 7:
                vb_semantics.append(
                    BufferSemantic(
                        AbstractSemantic(
                            Semantic(element["SemanticName"]), element["SemanticIndex"]
                        ),
                        element["Format"],
                    )
                )

            vb_layout = BufferLayout(vb_semantics)
            if vb_strides[idx] != vb_layout.stride:
                raise ValueError(
                    f"vb{idx} buffer layout format stride mismatch: {vb_strides[idx]} != {vb_layout.stride}"
                )
        return cls(vb_layout, vb_strides, ib_layout)
