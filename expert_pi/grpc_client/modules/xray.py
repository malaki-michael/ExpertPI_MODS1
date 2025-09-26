# GENERATED FILE from pySTEM 1.16.1,build 4717,rev b31140633df20050ad44d1030889522d4f5f8256

from .. import instrument_pb2
from .. import serializer
from ._common import *
from .. import channel as _channel

_get_xray_seconds_per_event = None
_get_xray_filter_type = None
_set_xray_filter_type = None


def connect():
    global _get_xray_seconds_per_event
    global _get_xray_filter_type
    global _set_xray_filter_type

    _get_xray_seconds_per_event = _channel.channel.unary_unary(
        '/pystem.api.grpc.Xray/GetXraySecondsPerEvent',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _get_xray_filter_type = _channel.channel.unary_unary(
        '/pystem.api.grpc.Xray/GetXrayFilterType',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.GetXrayFilterTypeMessage).DictFromString,
    )

    _set_xray_filter_type = _channel.channel.unary_unary(
        '/pystem.api.grpc.Xray/SetXrayFilterType',
        request_serializer=instrument_pb2.GetXrayFilterTypeMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.GetXrayFilterTypeMessage).DictFromString,
    )


def get_xray_seconds_per_event() -> float:
    """the minimal time of the event - this is used for the dead time correction"""
    return _get_xray_seconds_per_event(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_xray_seconds_per_event_future() -> float:
    get_xray_seconds_per_event.last_future = _get_xray_seconds_per_event.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_xray_seconds_per_event.last_future

get_xray_seconds_per_event.future=_get_xray_seconds_per_event_future

def get_xray_filter_type() -> XrayFilterType:
    return _get_xray_filter_type(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_xray_filter_type_future() -> XrayFilterType:
    get_xray_filter_type.last_future = _get_xray_filter_type.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_xray_filter_type.last_future

get_xray_filter_type.future=_get_xray_filter_type_future

def set_xray_filter_type(value: XrayFilterType) -> XrayFilterType:
    return _set_xray_filter_type(serializer.serialize({"value":value}, instrument_pb2.GetXrayFilterTypeMessage.DESCRIPTOR))


def _set_xray_filter_type_future(value: XrayFilterType) -> XrayFilterType:
    set_xray_filter_type.last_future = _set_xray_filter_type.future(serializer.serialize({"value":value}, instrument_pb2.GetXrayFilterTypeMessage.DESCRIPTOR))
    return set_xray_filter_type.last_future

set_xray_filter_type.future=_set_xray_filter_type_future
