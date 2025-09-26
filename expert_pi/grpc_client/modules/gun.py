# GENERATED FILE from pySTEM 1.16.1,build 4717,rev b31140633df20050ad44d1030889522d4f5f8256

from .. import instrument_pb2
from .. import serializer
from ._common import *
from .. import channel as _channel

_get_high_voltage = None
_get_emission_current = None
_get_extractor_voltage = None
_set_extractor_voltage = None
_get_energy_width = None
_get_brightness = None


def connect():
    global _get_high_voltage
    global _get_emission_current
    global _get_extractor_voltage
    global _set_extractor_voltage
    global _get_energy_width
    global _get_brightness

    _get_high_voltage = _channel.channel.unary_unary(
        '/pystem.api.grpc.Gun/GetHighVoltage',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _get_emission_current = _channel.channel.unary_unary(
        '/pystem.api.grpc.Gun/GetEmissionCurrent',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _get_extractor_voltage = _channel.channel.unary_unary(
        '/pystem.api.grpc.Gun/GetExtractorVoltage',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _set_extractor_voltage = _channel.channel.unary_unary(
        '/pystem.api.grpc.Gun/SetExtractorVoltage',
        request_serializer=instrument_pb2.FloatMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _get_energy_width = _channel.channel.unary_unary(
        '/pystem.api.grpc.Gun/GetEnergyWidth',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _get_brightness = _channel.channel.unary_unary(
        '/pystem.api.grpc.Gun/GetBrightness',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )


def get_high_voltage() -> float:
    return _get_high_voltage(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_high_voltage_future() -> float:
    get_high_voltage.last_future = _get_high_voltage.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_high_voltage.last_future

get_high_voltage.future=_get_high_voltage_future

def get_emission_current() -> float:
    return _get_emission_current(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_emission_current_future() -> float:
    get_emission_current.last_future = _get_emission_current.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_emission_current.last_future

get_emission_current.future=_get_emission_current_future

def get_extractor_voltage() -> float:
    return _get_extractor_voltage(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_extractor_voltage_future() -> float:
    get_extractor_voltage.last_future = _get_extractor_voltage.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_extractor_voltage.last_future

get_extractor_voltage.future=_get_extractor_voltage_future

def set_extractor_voltage(value: float) -> float:
    """not implemented yet"""
    return _set_extractor_voltage(serializer.serialize({"value":value}, instrument_pb2.FloatMessage.DESCRIPTOR))


def _set_extractor_voltage_future(value: float) -> float:
    set_extractor_voltage.last_future = _set_extractor_voltage.future(serializer.serialize({"value":value}, instrument_pb2.FloatMessage.DESCRIPTOR))
    return set_extractor_voltage.last_future

set_extractor_voltage.future=_set_extractor_voltage_future

def get_energy_width() -> float:
    return _get_energy_width(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_energy_width_future() -> float:
    get_energy_width.last_future = _get_energy_width.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_energy_width.last_future

get_energy_width.future=_get_energy_width_future

def get_brightness() -> float:
    """get brightness at actual energy units: m**(-2)*rad**(-2)"""
    return _get_brightness(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_brightness_future() -> float:
    get_brightness.last_future = _get_brightness.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_brightness.last_future

get_brightness.future=_get_brightness_future
