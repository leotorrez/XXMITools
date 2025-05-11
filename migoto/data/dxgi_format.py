import numpy
from enum import Enum
from typing import Callable

from numpy.typing import DTypeLike


class Topology(str, Enum):
    # https://learn.microsoft.com/en-us/windows/win32/direct3d11/d3d11-primitive-topology
    UNSOPORTED = "unsoported"
    TRIANGLELIST = "trianglelist"
    POINTLIST = "pointlist"
    TRIANGLESTRIP = "trianglestrip"

    @staticmethod
    def get_from_string(topology: str) -> "Topology":
        if topology == "trianglelist":
            return Topology.TRIANGLELIST
        elif topology == "pointlist":
            return Topology.POINTLIST
        elif topology == "trianglestrip":
            return Topology.TRIANGLESTRIP
        else:
            return Topology.UNSOPORTED


class DXGIType(Enum):
    # dxgi_type.value = (numpy_type, list_encoder, list_decoder, type_encoder, type_decoder)
    FLOAT32 = (numpy.float32, None, None, None, None)
    FLOAT16 = (numpy.float16, None, None, None, None)
    UINT32 = (numpy.uint32, None, None, None, None)
    UINT16 = (numpy.uint16, None, None, None, None)
    UINT8 = (numpy.uint8, None, None, None, None)
    SINT32 = (numpy.int32, None, None, None, None)
    SINT16 = (numpy.int16, None, None, None, None)
    SINT8 = (numpy.int8, None, None, None, None)
    UNORM16 = (
        numpy.uint16,
        lambda data: numpy.fromiter(data, numpy.float32),
        None,
        lambda data: numpy.around(data * 65535.0).astype(numpy.uint16),
        lambda data: data / 65535.0,
    )
    UNORM8 = (
        numpy.uint8,
        lambda data: numpy.fromiter(data, numpy.float32),
        None,
        lambda data: numpy.around(data * 255.0).astype(numpy.uint8),
        lambda data: data / 255.0,
    )
    SNORM16 = (
        numpy.int16,
        lambda data: numpy.fromiter(data, numpy.float32),
        None,
        lambda data: numpy.around(data * 32767.0).astype(numpy.int16),
        lambda data: data / 32767.0,
    )
    SNORM8 = (
        numpy.int8,
        lambda data: numpy.fromiter(data, numpy.float32),
        None,
        lambda data: numpy.around(data * 127.0).astype(numpy.int8),
        lambda data: data / 127.0,
    )


class DXGIFormat(Enum):
    @classmethod
    def from_type(cls, dxgi_type: DXGIType, dimensions) -> "DXGIFormat":
        for member in cls:
            if member.dxgi_type == dxgi_type and member.num_values == dimensions:
                return member
        raise ValueError(
            f"DXGIFormat not found for {dxgi_type} and {dimensions} dimensions!"
        )

    @classmethod
    def _missing_(cls, value: str):
        if value.startswith("DXGI_FORMAT_"):
            value = value[12:]
        for member in cls:
            if member.value == value:
                return member
        return None

    def __new__(cls, fmt, dxgi_type):
        (numpy_type, list_encoder, list_decoder, type_encoder, type_decoder) = (
            dxgi_type.value
        )
        obj = object.__new__(cls)
        obj._value_ = fmt
        obj.format = fmt
        obj.byte_width = 0
        obj.num_values = 0
        obj.value_bit_width = 0
        obj.value_byte_width = 0
        obj.dxgi_type = dxgi_type
        obj.numpy_base_type = numpy_type
        obj.type_encoder = type_encoder
        obj.type_decoder = type_decoder

        if list_encoder is None:
            obj.encoder = lambda data: numpy.fromiter(data, obj.numpy_base_type)
        else:
            obj.encoder = list_encoder

        if list_decoder is None:
            obj.decoder = lambda data: numpy.frombuffer(data, obj.numpy_base_type)
        else:
            obj.decoder = list_decoder

        if type_encoder is not None:
            obj.encoder = lambda data: type_encoder(obj.encoder(data))  # type: ignore
        else:
            # Special encoder is not defined, lets use basic type conversion
            # We shouldn't do it earlier, as list encoder already does it via fromiter
            obj.type_encoder = lambda data: data.astype(obj.numpy_base_type)

        if type_decoder is not None:
            obj.decoder = lambda data: type_decoder(obj.decoder(data))  # type: ignore

        for value_bit_width, value_byte_width in {"32": 4, "16": 2, "8": 1}.items():
            if value_bit_width in obj.dxgi_type.name:
                obj.num_values = obj.format.count(value_bit_width)
                obj.byte_width = obj.num_values * value_byte_width
                obj.value_bit_width = int(value_bit_width)
                obj.value_byte_width = value_byte_width
                break

        if obj.byte_width <= 0:
            raise ValueError(f"Invalid byte width {obj.byte_width} for {obj.format}!")

        return obj

    def __init__(self, fmt, dxgi_type):
        # This is called only once, when the class is created
        # We can use it to set up the instance variables
        self.format: str
        self.dxgi_type: DXGIType
        self.byte_width: int
        self.num_values: int
        self.value_bit_width: int
        self.value_byte_width: int
        self.numpy_base_type: DTypeLike
        self.encoder: Callable
        self.decoder: Callable
        self.type_encoder: Callable
        self.type_decoder: Callable

    def get_format(self) -> str:
        return "DXGI_FORMAT_" + self.format

    def get_num_values(self, data_stride=0) -> int:
        if data_stride > 0:
            # Caller specified data_stride, number of values may differ from the base dtype
            return int(data_stride / self.value_byte_width)
        else:
            return self.num_values

    def get_numpy_type(self, data_stride=0) -> DTypeLike:
        num_values = self.get_num_values(data_stride)
        # Tuple format of (type, 1) is deprecated, so we have to take special care
        if num_values == 1:
            return self.numpy_base_type
        else:
            return (self.numpy_base_type, num_values)

    # Float 32
    R32G32B32A32_FLOAT = "R32G32B32A32_FLOAT", DXGIType.FLOAT32
    R32G32B32_FLOAT = "R32G32B32_FLOAT", DXGIType.FLOAT32
    R32G32_FLOAT = "R32G32_FLOAT", DXGIType.FLOAT32
    R32_FLOAT = "R32_FLOAT", DXGIType.FLOAT32
    # Float 16
    R16G16B16A16_FLOAT = "R16G16B16A16_FLOAT", DXGIType.FLOAT16
    R16G16B16_FLOAT = "R16G16B16_FLOAT", DXGIType.FLOAT16
    R16G16_FLOAT = "R16G16_FLOAT", DXGIType.FLOAT16
    R16_FLOAT = "R16_FLOAT", DXGIType.FLOAT16
    # UINT 32
    R32G32B32A32_UINT = "R32G32B32A32_UINT", DXGIType.UINT32
    R32G32B32_UINT = "R32G32B32_UINT", DXGIType.UINT32
    R32G32_UINT = "R32G32_UINT", DXGIType.UINT32
    R32_UINT = "R32_UINT", DXGIType.UINT32
    # UINT 16
    R16G16B16A16_UINT = "R16G16B16A16_UINT", DXGIType.UINT16
    R16G16B16_UINT = "R16G16B16_UINT", DXGIType.UINT16
    R16G16_UINT = "R16G16_UINT", DXGIType.UINT16
    R16_UINT = "R16_UINT", DXGIType.UINT16
    # UINT 8
    R8G8B8A8_UINT = "R8G8B8A8_UINT", DXGIType.UINT8
    R8G8B8_UINT = "R8G8B8_UINT", DXGIType.UINT8
    R8G8_UINT = "R8G8_UINT", DXGIType.UINT8
    R8_UINT = "R8_UINT", DXGIType.UINT8
    # SINT 32
    R32G32B32A32_SINT = "R32G32B32A32_SINT", DXGIType.SINT32
    R32G32B32_SINT = "R32G32B32_SINT", DXGIType.SINT32
    R32G32_SINT = "R32G32_SINT", DXGIType.SINT32
    R32_SINT = "R32_SINT", DXGIType.SINT32
    # SINT 16
    R16G16B16A16_SINT = "R16G16B16A16_SINT", DXGIType.SINT16
    R16G16B16_SINT = "R16G16B16_SINT", DXGIType.SINT16
    R16G16_SINT = "R16G16_SINT", DXGIType.SINT16
    R16_SINT = "R16_SINT", DXGIType.SINT16
    # SINT 8
    R8G8B8A8_SINT = "R8G8B8A8_SINT", DXGIType.SINT8
    R8G8B8_SINT = "R8G8B8_SINT", DXGIType.SINT8
    R8G8_SINT = "R8G8_SINT", DXGIType.SINT8
    R8_SINT = "R8_SINT", DXGIType.SINT8
    # UNORM 16
    R16G16B16A16_UNORM = "R16G16B16A16_UNORM", DXGIType.UNORM16
    R16G16B16_UNORM = "R16G16B16_UNORM", DXGIType.UNORM16
    R16G16_UNORM = "R16G16_UNORM", DXGIType.UNORM16
    R16_UNORM = "R16_UNORM", DXGIType.UNORM16
    # UNORM 8
    R8G8B8A8_UNORM = "R8G8B8A8_UNORM", DXGIType.UNORM8
    R8G8B8_UNORM = "R8G8B8_UNORM", DXGIType.UNORM8
    R8G8_UNORM = "R8G8_UNORM", DXGIType.UNORM8
    R8_UNORM = "R8_UNORM", DXGIType.UNORM8
    # SNORM 16
    R16G16B16A16_SNORM = "R16G16B16A16_SNORM", DXGIType.SNORM16
    R16G16B16_SNORM = "R16G16B16_SNORM", DXGIType.SNORM16
    R16G16_SNORM = "R16G16_SNORM", DXGIType.SNORM16
    R16_SNORM = "R16_SNORM", DXGIType.SNORM16
    # SNORM 8
    R8G8B8A8_SNORM = "R8G8B8A8_SNORM", DXGIType.SNORM8
    R8G8B8_SNORM = "R8G8B8_SNORM", DXGIType.SNORM8
    R8G8_SNORM = "R8G8_SNORM", DXGIType.SNORM8
    R8_SNORM = "R8_SNORM", DXGIType.SNORM8
