# GENERATED FILE from pySTEM 1.16.1,build 4717,rev b31140633df20050ad44d1030889522d4f5f8256

from .. import instrument_pb2
from .. import serializer
from ._common import *
from .. import channel as _channel

_get_state = None
_set_state = None
_get_optical_mode = None
_set_optical_mode = None
_get_energy = None
_set_energy = None
_get_energy_range = None


def connect():
    global _get_state
    global _set_state
    global _get_optical_mode
    global _set_optical_mode
    global _get_energy
    global _set_energy
    global _get_energy_range

    _get_state = _channel.channel.unary_unary(
        '/pystem.api.grpc.Microscope/GetState',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.GetStateMessage).DictFromString,
    )

    _set_state = _channel.channel.unary_unary(
        '/pystem.api.grpc.Microscope/SetState',
        request_serializer=instrument_pb2.GetStateMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.GetStateMessage).DictFromString,
    )

    _get_optical_mode = _channel.channel.unary_unary(
        '/pystem.api.grpc.Microscope/GetOpticalMode',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.GetOpticalModeMessage).DictFromString,
    )

    _set_optical_mode = _channel.channel.unary_unary(
        '/pystem.api.grpc.Microscope/SetOpticalMode',
        request_serializer=instrument_pb2.GetOpticalModeMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.GetOpticalModeMessage).DictFromString,
    )

    _get_energy = _channel.channel.unary_unary(
        '/pystem.api.grpc.Microscope/GetEnergy',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _set_energy = _channel.channel.unary_unary(
        '/pystem.api.grpc.Microscope/SetEnergy',
        request_serializer=instrument_pb2.FloatMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _get_energy_range = _channel.channel.unary_unary(
        '/pystem.api.grpc.Microscope/GetEnergyRange',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.GetEnergyRangeMessage).DictFromString,
    )


def get_state() -> MicroscopeState:
    return _get_state(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_state_future() -> MicroscopeState:
    get_state.last_future = _get_state.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_state.last_future

get_state.future=_get_state_future

def set_state(value: MicroscopeState) -> MicroscopeState:
    """only Standby or Ready states can be set
        going to Standby state will close the valves of the column and stop the stage feedback loop.
        in Ready state there is a limit of 30 minutes after which the microscope will be automatically turned to Standby state again. Any new acquisition will reset this limit
        If the microscope is in Acquiring state, you need to stop the acquisition first"""
    return _set_state(serializer.serialize({"value":value}, instrument_pb2.GetStateMessage.DESCRIPTOR))


def _set_state_future(value: MicroscopeState) -> MicroscopeState:
    set_state.last_future = _set_state.future(serializer.serialize({"value":value}, instrument_pb2.GetStateMessage.DESCRIPTOR))
    return set_state.last_future

set_state.future=_set_state_future

def get_optical_mode() -> OpticalMode:
    return _get_optical_mode(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_optical_mode_future() -> OpticalMode:
    get_optical_mode.last_future = _get_optical_mode.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_optical_mode.last_future

get_optical_mode.future=_get_optical_mode_future

def set_optical_mode(value: OpticalMode) -> OpticalMode:
    """Low magnification mode is done by moving the stage closer to the polepiece (about 1.3mm), it is meant just for the general navigation. Diffraction pattern and tilting will be very limited"""
    return _set_optical_mode(serializer.serialize({"value":value}, instrument_pb2.GetOpticalModeMessage.DESCRIPTOR))


def _set_optical_mode_future(value: OpticalMode) -> OpticalMode:
    set_optical_mode.last_future = _set_optical_mode.future(serializer.serialize({"value":value}, instrument_pb2.GetOpticalModeMessage.DESCRIPTOR))
    return set_optical_mode.last_future

set_optical_mode.future=_set_optical_mode_future

def get_energy() -> float:
    return _get_energy(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_energy_future() -> float:
    get_energy.last_future = _get_energy.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_energy.last_future

get_energy.future=_get_energy_future

def set_energy(value: float) -> float:
    """use with care"""
    return _set_energy(serializer.serialize({"value":value}, instrument_pb2.FloatMessage.DESCRIPTOR))


def _set_energy_future(value: float) -> float:
    set_energy.last_future = _set_energy.future(serializer.serialize({"value":value}, instrument_pb2.FloatMessage.DESCRIPTOR))
    return set_energy.last_future

set_energy.future=_set_energy_future

def get_energy_range() -> list[float]:
    return _get_energy_range(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_energy_range_future() -> list[float]:
    get_energy_range.last_future = _get_energy_range.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_energy_range.last_future

get_energy_range.future=_get_energy_range_future
