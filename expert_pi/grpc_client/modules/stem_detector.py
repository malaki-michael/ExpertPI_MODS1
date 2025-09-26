# GENERATED FILE from pySTEM 1.16.1,build 4717,rev b31140633df20050ad44d1030889522d4f5f8256

from .. import instrument_pb2
from .. import serializer
from ._common import *
from .. import channel as _channel

_get_is_inserted = None
_set_is_inserted = None
_get_gain = None
_set_gain = None
_get_gain_range = None
_get_offset = None
_set_offset = None
_get_offset_range = None
_get_scale = None
_set_scale = None
_get_scale_range = None


def connect():
    global _get_is_inserted
    global _set_is_inserted
    global _get_gain
    global _set_gain
    global _get_gain_range
    global _get_offset
    global _set_offset
    global _get_offset_range
    global _get_scale
    global _set_scale
    global _get_scale_range

    _get_is_inserted = _channel.channel.unary_unary(
        '/pystem.api.grpc.StemDetector/GetIsInserted',
        request_serializer=instrument_pb2.GetIsInsertedMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.BoolMessage).DictFromString,
    )

    _set_is_inserted = _channel.channel.unary_unary(
        '/pystem.api.grpc.StemDetector/SetIsInserted',
        request_serializer=instrument_pb2.SetIsInsertedMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.BoolMessage).DictFromString,
    )

    _get_gain = _channel.channel.unary_unary(
        '/pystem.api.grpc.StemDetector/GetGain',
        request_serializer=instrument_pb2.GetIsInsertedMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _set_gain = _channel.channel.unary_unary(
        '/pystem.api.grpc.StemDetector/SetGain',
        request_serializer=instrument_pb2.SetGainMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _get_gain_range = _channel.channel.unary_unary(
        '/pystem.api.grpc.StemDetector/GetGainRange',
        request_serializer=instrument_pb2.GetIsInsertedMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatRangeMessage).DictFromString,
    )

    _get_offset = _channel.channel.unary_unary(
        '/pystem.api.grpc.StemDetector/GetOffset',
        request_serializer=instrument_pb2.GetIsInsertedMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _set_offset = _channel.channel.unary_unary(
        '/pystem.api.grpc.StemDetector/SetOffset',
        request_serializer=instrument_pb2.SetGainMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _get_offset_range = _channel.channel.unary_unary(
        '/pystem.api.grpc.StemDetector/GetOffsetRange',
        request_serializer=instrument_pb2.GetIsInsertedMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatRangeMessage).DictFromString,
    )

    _get_scale = _channel.channel.unary_unary(
        '/pystem.api.grpc.StemDetector/GetScale',
        request_serializer=instrument_pb2.GetIsInsertedMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _set_scale = _channel.channel.unary_unary(
        '/pystem.api.grpc.StemDetector/SetScale',
        request_serializer=instrument_pb2.SetGainMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _get_scale_range = _channel.channel.unary_unary(
        '/pystem.api.grpc.StemDetector/GetScaleRange',
        request_serializer=instrument_pb2.GetIsInsertedMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatRangeMessage).DictFromString,
    )


def get_is_inserted(detector: DetectorType) -> bool:
    return _get_is_inserted(serializer.serialize({"detector":detector}, instrument_pb2.GetIsInsertedMessage.DESCRIPTOR))


def _get_is_inserted_future(detector: DetectorType) -> bool:
    get_is_inserted.last_future = _get_is_inserted.future(serializer.serialize({"detector":detector}, instrument_pb2.GetIsInsertedMessage.DESCRIPTOR))
    return get_is_inserted.last_future

get_is_inserted.future=_get_is_inserted_future

def set_is_inserted(detector: DetectorType, value: bool) -> bool:
    """will produce vibrations of the column due to moving detector for several seconds"""
    return _set_is_inserted(serializer.serialize({"detector":detector, "value":value}, instrument_pb2.SetIsInsertedMessage.DESCRIPTOR))


def _set_is_inserted_future(detector: DetectorType, value: bool) -> bool:
    set_is_inserted.last_future = _set_is_inserted.future(serializer.serialize({"detector":detector, "value":value}, instrument_pb2.SetIsInsertedMessage.DESCRIPTOR))
    return set_is_inserted.last_future

set_is_inserted.future=_set_is_inserted_future

def get_gain(detector: DetectorType) -> float:
    """Sets the gain of the specified detector, in dimensionless units.  The signal intensity read out from the detector
        will be given by Intensity = Gain * SignalCurrent / BeamCurrent / Scale * MAXIMUM_INTENSITY
        where MAXIMUM_INTENSITY depends on the number of bits read out (255 for 8 bits and 65535 for 16 bits)"""
    return _get_gain(serializer.serialize({"detector":detector}, instrument_pb2.GetIsInsertedMessage.DESCRIPTOR))


def _get_gain_future(detector: DetectorType) -> float:
    get_gain.last_future = _get_gain.future(serializer.serialize({"detector":detector}, instrument_pb2.GetIsInsertedMessage.DESCRIPTOR))
    return get_gain.last_future

get_gain.future=_get_gain_future

def set_gain(detector: DetectorType, value: float) -> float:
    return _set_gain(serializer.serialize({"detector":detector, "value":value}, instrument_pb2.SetGainMessage.DESCRIPTOR))


def _set_gain_future(detector: DetectorType, value: float) -> float:
    set_gain.last_future = _set_gain.future(serializer.serialize({"detector":detector, "value":value}, instrument_pb2.SetGainMessage.DESCRIPTOR))
    return set_gain.last_future

set_gain.future=_set_gain_future

def get_gain_range(detector: DetectorType) -> {"start": float, "end": float}:
    return _get_gain_range(serializer.serialize({"detector":detector}, instrument_pb2.GetIsInsertedMessage.DESCRIPTOR))


def _get_gain_range_future(detector: DetectorType) -> {"start": float, "end": float}:
    get_gain_range.last_future = _get_gain_range.future(serializer.serialize({"detector":detector}, instrument_pb2.GetIsInsertedMessage.DESCRIPTOR))
    return get_gain_range.last_future

get_gain_range.future=_get_gain_range_future

def get_offset(detector: DetectorType) -> float:
    return _get_offset(serializer.serialize({"detector":detector}, instrument_pb2.GetIsInsertedMessage.DESCRIPTOR))


def _get_offset_future(detector: DetectorType) -> float:
    get_offset.last_future = _get_offset.future(serializer.serialize({"detector":detector}, instrument_pb2.GetIsInsertedMessage.DESCRIPTOR))
    return get_offset.last_future

get_offset.future=_get_offset_future

def set_offset(detector: DetectorType, value: float) -> float:
    return _set_offset(serializer.serialize({"detector":detector, "value":value}, instrument_pb2.SetGainMessage.DESCRIPTOR))


def _set_offset_future(detector: DetectorType, value: float) -> float:
    set_offset.last_future = _set_offset.future(serializer.serialize({"detector":detector, "value":value}, instrument_pb2.SetGainMessage.DESCRIPTOR))
    return set_offset.last_future

set_offset.future=_set_offset_future

def get_offset_range(detector: DetectorType) -> {"start": float, "end": float}:
    return _get_offset_range(serializer.serialize({"detector":detector}, instrument_pb2.GetIsInsertedMessage.DESCRIPTOR))


def _get_offset_range_future(detector: DetectorType) -> {"start": float, "end": float}:
    get_offset_range.last_future = _get_offset_range.future(serializer.serialize({"detector":detector}, instrument_pb2.GetIsInsertedMessage.DESCRIPTOR))
    return get_offset_range.last_future

get_offset_range.future=_get_offset_range_future

def get_scale(detector: DetectorType) -> float:
    return _get_scale(serializer.serialize({"detector":detector}, instrument_pb2.GetIsInsertedMessage.DESCRIPTOR))


def _get_scale_future(detector: DetectorType) -> float:
    get_scale.last_future = _get_scale.future(serializer.serialize({"detector":detector}, instrument_pb2.GetIsInsertedMessage.DESCRIPTOR))
    return get_scale.last_future

get_scale.future=_get_scale_future

def set_scale(detector: DetectorType, value: float) -> float:
    """set the scale to the portion of the beam current on the stem detector"""
    return _set_scale(serializer.serialize({"detector":detector, "value":value}, instrument_pb2.SetGainMessage.DESCRIPTOR))


def _set_scale_future(detector: DetectorType, value: float) -> float:
    set_scale.last_future = _set_scale.future(serializer.serialize({"detector":detector, "value":value}, instrument_pb2.SetGainMessage.DESCRIPTOR))
    return set_scale.last_future

set_scale.future=_set_scale_future

def get_scale_range(detector: DetectorType) -> {"start": float, "end": float}:
    return _get_scale_range(serializer.serialize({"detector":detector}, instrument_pb2.GetIsInsertedMessage.DESCRIPTOR))


def _get_scale_range_future(detector: DetectorType) -> {"start": float, "end": float}:
    get_scale_range.last_future = _get_scale_range.future(serializer.serialize({"detector":detector}, instrument_pb2.GetIsInsertedMessage.DESCRIPTOR))
    return get_scale_range.last_future

get_scale_range.future=_get_scale_range_future
