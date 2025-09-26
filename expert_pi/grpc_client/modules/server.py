# GENERATED FILE from pySTEM 1.16.1,build 4717,rev b31140633df20050ad44d1030889522d4f5f8256

from .. import instrument_pb2
from .. import serializer
from ._common import *
from .. import channel as _channel

_get_version = None
_get_state = None
_enable_live_streaming = None


def connect():
    global _get_version
    global _get_state
    global _enable_live_streaming

    _get_version = _channel.channel.unary_unary(
        '/pystem.api.grpc.Server/GetVersion',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.GetVersionMessage).DictFromString,
    )

    _get_state = _channel.channel.unary_unary(
        '/pystem.api.grpc.Server/GetState',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.GetStateRequest).DictFromString,
    )

    _enable_live_streaming = _channel.channel.unary_unary(
        '/pystem.api.grpc.Server/EnableLiveStreaming',
        request_serializer=instrument_pb2.BoolMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.BoolMessage).DictFromString,
    )


def get_version() -> str:
    return _get_version(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_version_future() -> str:
    get_version.last_future = _get_version.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_version.last_future

get_version.future=_get_version_future

def get_state() -> {"version": str,
                    "uptime": float,
                    "cpu_load": float,
                    "memory_percent": float,
                    "memory_consumed": float,
                    "memory_total": float}:
    return _get_state(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_state_future() -> {"version": str,
                    "uptime": float,
                    "cpu_load": float,
                    "memory_percent": float,
                    "memory_consumed": float,
                    "memory_total": float}:
    get_state.last_future = _get_state.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_state.last_future

get_state.future=_get_state_future

def enable_live_streaming(value: bool) -> bool:
    return _enable_live_streaming(serializer.serialize({"value":value}, instrument_pb2.BoolMessage.DESCRIPTOR))


def _enable_live_streaming_future(value: bool) -> bool:
    enable_live_streaming.last_future = _enable_live_streaming.future(serializer.serialize({"value":value}, instrument_pb2.BoolMessage.DESCRIPTOR))
    return enable_live_streaming.last_future

enable_live_streaming.future=_enable_live_streaming_future
