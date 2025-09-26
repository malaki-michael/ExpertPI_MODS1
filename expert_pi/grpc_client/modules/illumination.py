# GENERATED FILE from pySTEM 1.16.1,build 4717,rev b31140633df20050ad44d1030889522d4f5f8256

from .. import instrument_pb2
from .. import serializer
from ._common import *
from .. import channel as _channel

_get_spot_size = None
_get_maximized_current = None
_get_convergence_half_angle = None
_get_beam_diameter = None
_get_current = None
_get_illumination_mode = None
_set_illumination_values = None
_set_parallel_illumination_values = None
_get_condenser_defocus = None
_set_condenser_defocus = None
_get_condenser_defocus_range = None
_get_objective_defocus = None
_set_objective_defocus = None
_get_objective_defocus_range = None
_get_stigmator = None
_set_stigmator = None
_get_stigmator_range = None
_get_tilt = None
_set_tilt = None
_get_tilt_range = None
_get_shift = None
_set_shift = None
_get_shift_range = None
_get_aperture_size = None
_get_aperture_position = None
_set_aperture_position = None
_get_aperture_position_range = None


def connect():
    global _get_spot_size
    global _get_maximized_current
    global _get_convergence_half_angle
    global _get_beam_diameter
    global _get_current
    global _get_illumination_mode
    global _set_illumination_values
    global _set_parallel_illumination_values
    global _get_condenser_defocus
    global _set_condenser_defocus
    global _get_condenser_defocus_range
    global _get_objective_defocus
    global _set_objective_defocus
    global _get_objective_defocus_range
    global _get_stigmator
    global _set_stigmator
    global _get_stigmator_range
    global _get_tilt
    global _set_tilt
    global _get_tilt_range
    global _get_shift
    global _set_shift
    global _get_shift_range
    global _get_aperture_size
    global _get_aperture_position
    global _set_aperture_position
    global _get_aperture_position_range

    _get_spot_size = _channel.channel.unary_unary(
        '/pystem.api.grpc.Illumination/GetSpotSize',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.GetSpotSizeMessage).DictFromString,
    )

    _get_maximized_current = _channel.channel.unary_unary(
        '/pystem.api.grpc.Illumination/GetMaximizedCurrent',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.GetMaximizedCurrentMessage).DictFromString,
    )

    _get_convergence_half_angle = _channel.channel.unary_unary(
        '/pystem.api.grpc.Illumination/GetConvergenceHalfAngle',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _get_beam_diameter = _channel.channel.unary_unary(
        '/pystem.api.grpc.Illumination/GetBeamDiameter',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _get_current = _channel.channel.unary_unary(
        '/pystem.api.grpc.Illumination/GetCurrent',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _get_illumination_mode = _channel.channel.unary_unary(
        '/pystem.api.grpc.Illumination/GetIlluminationMode',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.GetIlluminationModeMessage).DictFromString,
    )

    _set_illumination_values = _channel.channel.unary_unary(
        '/pystem.api.grpc.Illumination/SetIlluminationValues',
        request_serializer=instrument_pb2.SetIlluminationValuesMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.GetMaximizedCurrentMessage).DictFromString,
    )

    _set_parallel_illumination_values = _channel.channel.unary_unary(
        '/pystem.api.grpc.Illumination/SetParallelIlluminationValues',
        request_serializer=instrument_pb2.SetParallelIlluminationValuesMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.SetParallelIlluminationValuesRequest).DictFromString,
    )

    _get_condenser_defocus = _channel.channel.unary_unary(
        '/pystem.api.grpc.Illumination/GetCondenserDefocus',
        request_serializer=instrument_pb2.GetCondenserDefocusMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _set_condenser_defocus = _channel.channel.unary_unary(
        '/pystem.api.grpc.Illumination/SetCondenserDefocus',
        request_serializer=instrument_pb2.SetCondenserDefocusMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _get_condenser_defocus_range = _channel.channel.unary_unary(
        '/pystem.api.grpc.Illumination/GetCondenserDefocusRange',
        request_serializer=instrument_pb2.GetCondenserDefocusMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatRangeMessage).DictFromString,
    )

    _get_objective_defocus = _channel.channel.unary_unary(
        '/pystem.api.grpc.Illumination/GetObjectiveDefocus',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _set_objective_defocus = _channel.channel.unary_unary(
        '/pystem.api.grpc.Illumination/SetObjectiveDefocus',
        request_serializer=instrument_pb2.FloatMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _get_objective_defocus_range = _channel.channel.unary_unary(
        '/pystem.api.grpc.Illumination/GetObjectiveDefocusRange',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatRangeMessage).DictFromString,
    )

    _get_stigmator = _channel.channel.unary_unary(
        '/pystem.api.grpc.Illumination/GetStigmator',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.PositionMessage).DictFromString,
    )

    _set_stigmator = _channel.channel.unary_unary(
        '/pystem.api.grpc.Illumination/SetStigmator',
        request_serializer=instrument_pb2.PositionMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.PositionMessage).DictFromString,
    )

    _get_stigmator_range = _channel.channel.unary_unary(
        '/pystem.api.grpc.Illumination/GetStigmatorRange',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.GetStigmatorRangeMessage).DictFromString,
    )

    _get_tilt = _channel.channel.unary_unary(
        '/pystem.api.grpc.Illumination/GetTilt',
        request_serializer=instrument_pb2.GetTiltMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.PositionMessage).DictFromString,
    )

    _set_tilt = _channel.channel.unary_unary(
        '/pystem.api.grpc.Illumination/SetTilt',
        request_serializer=instrument_pb2.SetTiltMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.PositionMessage).DictFromString,
    )

    _get_tilt_range = _channel.channel.unary_unary(
        '/pystem.api.grpc.Illumination/GetTiltRange',
        request_serializer=instrument_pb2.GetTiltMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.GetStigmatorRangeMessage).DictFromString,
    )

    _get_shift = _channel.channel.unary_unary(
        '/pystem.api.grpc.Illumination/GetShift',
        request_serializer=instrument_pb2.GetTiltMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.PositionMessage).DictFromString,
    )

    _set_shift = _channel.channel.unary_unary(
        '/pystem.api.grpc.Illumination/SetShift',
        request_serializer=instrument_pb2.SetTiltMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.PositionMessage).DictFromString,
    )

    _get_shift_range = _channel.channel.unary_unary(
        '/pystem.api.grpc.Illumination/GetShiftRange',
        request_serializer=instrument_pb2.GetTiltMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.GetStigmatorRangeMessage).DictFromString,
    )

    _get_aperture_size = _channel.channel.unary_unary(
        '/pystem.api.grpc.Illumination/GetApertureSize',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _get_aperture_position = _channel.channel.unary_unary(
        '/pystem.api.grpc.Illumination/GetAperturePosition',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.PositionMessage).DictFromString,
    )

    _set_aperture_position = _channel.channel.unary_unary(
        '/pystem.api.grpc.Illumination/SetAperturePosition',
        request_serializer=instrument_pb2.PositionMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.PositionMessage).DictFromString,
    )

    _get_aperture_position_range = _channel.channel.unary_unary(
        '/pystem.api.grpc.Illumination/GetAperturePositionRange',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.GetStigmatorRangeMessage).DictFromString,
    )


def get_spot_size() -> {"current": float, "angle": float, "d50": float}:
    """best theoretical achievable diameter of STEM spot size in which is 50% of beam intensity"""
    return _get_spot_size(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_spot_size_future() -> {"current": float, "angle": float, "d50": float}:
    get_spot_size.last_future = _get_spot_size.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_spot_size.last_future

get_spot_size.future=_get_spot_size_future

def get_maximized_current() -> {"current": float, "angle": float}:
    """not implemented yet"""
    return _get_maximized_current(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_maximized_current_future() -> {"current": float, "angle": float}:
    get_maximized_current.last_future = _get_maximized_current.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_maximized_current.last_future

get_maximized_current.future=_get_maximized_current_future

def get_convergence_half_angle() -> float:
    """converge illumination - half angle on sample
        parallel mode - theoretical incidence angle"""
    return _get_convergence_half_angle(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_convergence_half_angle_future() -> float:
    get_convergence_half_angle.last_future = _get_convergence_half_angle.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_convergence_half_angle.last_future

get_convergence_half_angle.future=_get_convergence_half_angle_future

def get_beam_diameter() -> float:
    """converge illumination - theoretical d50
        parallel mode - diameter of the parallel disc"""
    return _get_beam_diameter(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_beam_diameter_future() -> float:
    get_beam_diameter.last_future = _get_beam_diameter.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_beam_diameter.last_future

get_beam_diameter.future=_get_beam_diameter_future

def get_current() -> float:
    """current on sample"""
    return _get_current(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_current_future() -> float:
    get_current.last_future = _get_current.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_current.last_future

get_current.future=_get_current_future

def get_illumination_mode() -> IlluminationMode:
    return _get_illumination_mode(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_illumination_mode_future() -> IlluminationMode:
    get_illumination_mode.last_future = _get_illumination_mode.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_illumination_mode.last_future

get_illumination_mode.future=_get_illumination_mode_future

def set_illumination_values(current: float, angle: float, keep_adjustments: bool) -> {"current": float, "angle": float}:
    """set stem illumination optics, use keep_adjustments option if you want preserve previously set adjustments on top of the preset"""
    return _set_illumination_values(serializer.serialize({"current":current, "angle":angle, "keep_adjustments":keep_adjustments}, instrument_pb2.SetIlluminationValuesMessage.DESCRIPTOR))


def _set_illumination_values_future(current: float, angle: float, keep_adjustments: bool) -> {"current": float, "angle": float}:
    set_illumination_values.last_future = _set_illumination_values.future(serializer.serialize({"current":current, "angle":angle, "keep_adjustments":keep_adjustments}, instrument_pb2.SetIlluminationValuesMessage.DESCRIPTOR))
    return set_illumination_values.last_future

set_illumination_values.future=_set_illumination_values_future

def set_parallel_illumination_values(current: float, radius: float, keep_adjustments: bool) -> {"current": float, "radius": float}:
    """set parallel beam n sample,
        use keep_adjustments option if you want preserve previously set adjustments on top of the preset"""
    return _set_parallel_illumination_values(serializer.serialize({"current":current, "radius":radius, "keep_adjustments":keep_adjustments}, instrument_pb2.SetParallelIlluminationValuesMessage.DESCRIPTOR))


def _set_parallel_illumination_values_future(current: float, radius: float, keep_adjustments: bool) -> {"current": float, "radius": float}:
    set_parallel_illumination_values.last_future = _set_parallel_illumination_values.future(serializer.serialize({"current":current, "radius":radius, "keep_adjustments":keep_adjustments}, instrument_pb2.SetParallelIlluminationValuesMessage.DESCRIPTOR))
    return set_parallel_illumination_values.last_future

set_parallel_illumination_values.future=_set_parallel_illumination_values_future

def get_condenser_defocus(type: CondenserFocusType) -> float:
    return _get_condenser_defocus(serializer.serialize({"type":type}, instrument_pb2.GetCondenserDefocusMessage.DESCRIPTOR))


def _get_condenser_defocus_future(type: CondenserFocusType) -> float:
    get_condenser_defocus.last_future = _get_condenser_defocus.future(serializer.serialize({"type":type}, instrument_pb2.GetCondenserDefocusMessage.DESCRIPTOR))
    return get_condenser_defocus.last_future

get_condenser_defocus.future=_get_condenser_defocus_future

def set_condenser_defocus(value: float, type: CondenserFocusType) -> float:
    """C3 defocus will not keep the fixed illumination angle
        FixedAngle defocus uses C2 and C3 combination, thus it might have worse stabilization time.
        Setting on of the type will reset the previous type to zero"""
    return _set_condenser_defocus(serializer.serialize({"value":value, "type":type}, instrument_pb2.SetCondenserDefocusMessage.DESCRIPTOR))


def _set_condenser_defocus_future(value: float, type: CondenserFocusType) -> float:
    set_condenser_defocus.last_future = _set_condenser_defocus.future(serializer.serialize({"value":value, "type":type}, instrument_pb2.SetCondenserDefocusMessage.DESCRIPTOR))
    return set_condenser_defocus.last_future

set_condenser_defocus.future=_set_condenser_defocus_future

def get_condenser_defocus_range(type: CondenserFocusType) -> {"start": float, "end": float}:
    return _get_condenser_defocus_range(serializer.serialize({"type":type}, instrument_pb2.GetCondenserDefocusMessage.DESCRIPTOR))


def _get_condenser_defocus_range_future(type: CondenserFocusType) -> {"start": float, "end": float}:
    get_condenser_defocus_range.last_future = _get_condenser_defocus_range.future(serializer.serialize({"type":type}, instrument_pb2.GetCondenserDefocusMessage.DESCRIPTOR))
    return get_condenser_defocus_range.last_future

get_condenser_defocus_range.future=_get_condenser_defocus_range_future

def get_objective_defocus() -> float:
    return _get_objective_defocus(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_objective_defocus_future() -> float:
    get_objective_defocus.last_future = _get_objective_defocus.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_objective_defocus.last_future

get_objective_defocus.future=_get_objective_defocus_future

def set_objective_defocus(value: float) -> float:
    """objective defocus will slightly modify the stigmation"""
    return _set_objective_defocus(serializer.serialize({"value":value}, instrument_pb2.FloatMessage.DESCRIPTOR))


def _set_objective_defocus_future(value: float) -> float:
    set_objective_defocus.last_future = _set_objective_defocus.future(serializer.serialize({"value":value}, instrument_pb2.FloatMessage.DESCRIPTOR))
    return set_objective_defocus.last_future

set_objective_defocus.future=_set_objective_defocus_future

def get_objective_defocus_range() -> {"start": float, "end": float}:
    return _get_objective_defocus_range(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_objective_defocus_range_future() -> {"start": float, "end": float}:
    get_objective_defocus_range.last_future = _get_objective_defocus_range.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_objective_defocus_range.last_future

get_objective_defocus_range.future=_get_objective_defocus_range_future

def get_stigmator() -> {"x": float, "y": float}:
    """get normal and skew axis  of stigmators in amperes"""
    return _get_stigmator(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_stigmator_future() -> {"x": float, "y": float}:
    get_stigmator.last_future = _get_stigmator.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_stigmator.last_future

get_stigmator.future=_get_stigmator_future

def set_stigmator(x: float, y: float) -> {"x": float, "y": float}:
    """set normal and skew axis  of stigmators in amperes"""
    return _set_stigmator(serializer.serialize({"x":x, "y":y}, instrument_pb2.PositionMessage.DESCRIPTOR))


def _set_stigmator_future(x: float, y: float) -> {"x": float, "y": float}:
    set_stigmator.last_future = _set_stigmator.future(serializer.serialize({"x":x, "y":y}, instrument_pb2.PositionMessage.DESCRIPTOR))
    return set_stigmator.last_future

set_stigmator.future=_set_stigmator_future

def get_stigmator_range() -> list[{"x": float, "y": float}]:
    return _get_stigmator_range(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_stigmator_range_future() -> list[{"x": float, "y": float}]:
    get_stigmator_range.last_future = _get_stigmator_range.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_stigmator_range.last_future

get_stigmator_range.future=_get_stigmator_range_future

def get_tilt(type: DeflectorType) -> {"x": float, "y": float}:
    return _get_tilt(serializer.serialize({"type":type}, instrument_pb2.GetTiltMessage.DESCRIPTOR))


def _get_tilt_future(type: DeflectorType) -> {"x": float, "y": float}:
    get_tilt.last_future = _get_tilt.future(serializer.serialize({"type":type}, instrument_pb2.GetTiltMessage.DESCRIPTOR))
    return get_tilt.last_future

get_tilt.future=_get_tilt_future

def set_tilt(position: {"x": float, "y": float}, type: DeflectorType) -> {"x": float, "y": float}:
    """use primary the precession type of the tilt - it is calibrated with respect to the precession"""
    return _set_tilt(serializer.serialize({"position":position, "type":type}, instrument_pb2.SetTiltMessage.DESCRIPTOR))


def _set_tilt_future(position: {"x": float, "y": float}, type: DeflectorType) -> {"x": float, "y": float}:
    set_tilt.last_future = _set_tilt.future(serializer.serialize({"position":position, "type":type}, instrument_pb2.SetTiltMessage.DESCRIPTOR))
    return set_tilt.last_future

set_tilt.future=_set_tilt_future

def get_tilt_range(type: DeflectorType) -> list[{"x": float, "y": float}]:
    return _get_tilt_range(serializer.serialize({"type":type}, instrument_pb2.GetTiltMessage.DESCRIPTOR))


def _get_tilt_range_future(type: DeflectorType) -> list[{"x": float, "y": float}]:
    get_tilt_range.last_future = _get_tilt_range.future(serializer.serialize({"type":type}, instrument_pb2.GetTiltMessage.DESCRIPTOR))
    return get_tilt_range.last_future

get_tilt_range.future=_get_tilt_range_future

def get_shift(type: DeflectorType) -> {"x": float, "y": float}:
    return _get_shift(serializer.serialize({"type":type}, instrument_pb2.GetTiltMessage.DESCRIPTOR))


def _get_shift_future(type: DeflectorType) -> {"x": float, "y": float}:
    get_shift.last_future = _get_shift.future(serializer.serialize({"type":type}, instrument_pb2.GetTiltMessage.DESCRIPTOR))
    return get_shift.last_future

get_shift.future=_get_shift_future

def set_shift(position: {"x": float, "y": float}, type: DeflectorType) -> {"x": float, "y": float}:
    """use primary the scanning type of the shift,
        higher range can be achieved by precession dynamic type, but it cannot be combined together with precession"""
    return _set_shift(serializer.serialize({"position":position, "type":type}, instrument_pb2.SetTiltMessage.DESCRIPTOR))


def _set_shift_future(position: {"x": float, "y": float}, type: DeflectorType) -> {"x": float, "y": float}:
    set_shift.last_future = _set_shift.future(serializer.serialize({"position":position, "type":type}, instrument_pb2.SetTiltMessage.DESCRIPTOR))
    return set_shift.last_future

set_shift.future=_set_shift_future

def get_shift_range(type: DeflectorType) -> list[{"x": float, "y": float}]:
    return _get_shift_range(serializer.serialize({"type":type}, instrument_pb2.GetTiltMessage.DESCRIPTOR))


def _get_shift_range_future(type: DeflectorType) -> list[{"x": float, "y": float}]:
    get_shift_range.last_future = _get_shift_range.future(serializer.serialize({"type":type}, instrument_pb2.GetTiltMessage.DESCRIPTOR))
    return get_shift_range.last_future

get_shift_range.future=_get_shift_range_future

def get_aperture_size() -> float:
    return _get_aperture_size(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_aperture_size_future() -> float:
    get_aperture_size.last_future = _get_aperture_size.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_aperture_size.last_future

get_aperture_size.future=_get_aperture_size_future

def get_aperture_position() -> {"x": float, "y": float}:
    return _get_aperture_position(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_aperture_position_future() -> {"x": float, "y": float}:
    get_aperture_position.last_future = _get_aperture_position.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_aperture_position.last_future

get_aperture_position.future=_get_aperture_position_future

def set_aperture_position(x: float, y: float) -> {"x": float, "y": float}:
    """this is the electronic shift by the double deflector before and after the aperture"""
    return _set_aperture_position(serializer.serialize({"x":x, "y":y}, instrument_pb2.PositionMessage.DESCRIPTOR))


def _set_aperture_position_future(x: float, y: float) -> {"x": float, "y": float}:
    set_aperture_position.last_future = _set_aperture_position.future(serializer.serialize({"x":x, "y":y}, instrument_pb2.PositionMessage.DESCRIPTOR))
    return set_aperture_position.last_future

set_aperture_position.future=_set_aperture_position_future

def get_aperture_position_range() -> list[{"x": float, "y": float}]:
    return _get_aperture_position_range(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_aperture_position_range_future() -> list[{"x": float, "y": float}]:
    get_aperture_position_range.last_future = _get_aperture_position_range.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_aperture_position_range.last_future

get_aperture_position_range.future=_get_aperture_position_range_future
