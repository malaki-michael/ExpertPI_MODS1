# GENERATED FILE from pySTEM 1.16.1,build 4717,rev b31140633df20050ad44d1030889522d4f5f8256

from .. import instrument_pb2
from .. import serializer
from ._common import *
from .. import channel as _channel

_get_tilt = None
_set_tilt = None
_get_tilt_range = None
_get_shift = None
_set_shift = None
_get_shift_range = None
_get_camera_shift = None
_set_camera_shift = None
_get_camera_shift_range = None
_get_is_off_axis_stem_enabled = None
_set_is_off_axis_stem_enabled = None
_get_max_detector_angles = None
_get_max_camera_angle = None
_set_max_camera_angle = None
_get_max_camera_angle_range = None
_get_camera_to_stage_rotation = None
_get_projector_defocus = None
_set_projector_defocus = None
_get_projector_defocus_range = None


def connect():
    global _get_tilt
    global _set_tilt
    global _get_tilt_range
    global _get_shift
    global _set_shift
    global _get_shift_range
    global _get_camera_shift
    global _set_camera_shift
    global _get_camera_shift_range
    global _get_is_off_axis_stem_enabled
    global _set_is_off_axis_stem_enabled
    global _get_max_detector_angles
    global _get_max_camera_angle
    global _set_max_camera_angle
    global _get_max_camera_angle_range
    global _get_camera_to_stage_rotation
    global _get_projector_defocus
    global _set_projector_defocus
    global _get_projector_defocus_range

    _get_tilt = _channel.channel.unary_unary(
        '/pystem.api.grpc.Projection/GetTilt',
        request_serializer=instrument_pb2.GetTiltMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.PositionMessage).DictFromString,
    )

    _set_tilt = _channel.channel.unary_unary(
        '/pystem.api.grpc.Projection/SetTilt',
        request_serializer=instrument_pb2.SetTiltMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.PositionMessage).DictFromString,
    )

    _get_tilt_range = _channel.channel.unary_unary(
        '/pystem.api.grpc.Projection/GetTiltRange',
        request_serializer=instrument_pb2.GetTiltMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.GetStigmatorRangeMessage).DictFromString,
    )

    _get_shift = _channel.channel.unary_unary(
        '/pystem.api.grpc.Projection/GetShift',
        request_serializer=instrument_pb2.GetTiltMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.PositionMessage).DictFromString,
    )

    _set_shift = _channel.channel.unary_unary(
        '/pystem.api.grpc.Projection/SetShift',
        request_serializer=instrument_pb2.SetTiltMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.PositionMessage).DictFromString,
    )

    _get_shift_range = _channel.channel.unary_unary(
        '/pystem.api.grpc.Projection/GetShiftRange',
        request_serializer=instrument_pb2.GetTiltMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.GetStigmatorRangeMessage).DictFromString,
    )

    _get_camera_shift = _channel.channel.unary_unary(
        '/pystem.api.grpc.Projection/GetCameraShift',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.PositionMessage).DictFromString,
    )

    _set_camera_shift = _channel.channel.unary_unary(
        '/pystem.api.grpc.Projection/SetCameraShift',
        request_serializer=instrument_pb2.PositionMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.PositionMessage).DictFromString,
    )

    _get_camera_shift_range = _channel.channel.unary_unary(
        '/pystem.api.grpc.Projection/GetCameraShiftRange',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.GetStigmatorRangeMessage).DictFromString,
    )

    _get_is_off_axis_stem_enabled = _channel.channel.unary_unary(
        '/pystem.api.grpc.Projection/GetIsOffAxisStemEnabled',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.BoolMessage).DictFromString,
    )

    _set_is_off_axis_stem_enabled = _channel.channel.unary_unary(
        '/pystem.api.grpc.Projection/SetIsOffAxisStemEnabled',
        request_serializer=instrument_pb2.BoolMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.BoolMessage).DictFromString,
    )

    _get_max_detector_angles = _channel.channel.unary_unary(
        '/pystem.api.grpc.Projection/GetMaxDetectorAngles',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.GetMaxDetectorAnglesMessage).DictFromString,
    )

    _get_max_camera_angle = _channel.channel.unary_unary(
        '/pystem.api.grpc.Projection/GetMaxCameraAngle',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _set_max_camera_angle = _channel.channel.unary_unary(
        '/pystem.api.grpc.Projection/SetMaxCameraAngle',
        request_serializer=instrument_pb2.SetMaxCameraAngleMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _get_max_camera_angle_range = _channel.channel.unary_unary(
        '/pystem.api.grpc.Projection/GetMaxCameraAngleRange',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatRangeMessage).DictFromString,
    )

    _get_camera_to_stage_rotation = _channel.channel.unary_unary(
        '/pystem.api.grpc.Projection/GetCameraToStageRotation',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _get_projector_defocus = _channel.channel.unary_unary(
        '/pystem.api.grpc.Projection/GetProjectorDefocus',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _set_projector_defocus = _channel.channel.unary_unary(
        '/pystem.api.grpc.Projection/SetProjectorDefocus',
        request_serializer=instrument_pb2.FloatMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _get_projector_defocus_range = _channel.channel.unary_unary(
        '/pystem.api.grpc.Projection/GetProjectorDefocusRange',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatRangeMessage).DictFromString,
    )


def get_tilt(type: DeflectorType) -> {"x": float, "y": float}:
    return _get_tilt(serializer.serialize({"type":type}, instrument_pb2.GetTiltMessage.DESCRIPTOR))


def _get_tilt_future(type: DeflectorType) -> {"x": float, "y": float}:
    get_tilt.last_future = _get_tilt.future(serializer.serialize({"type":type}, instrument_pb2.GetTiltMessage.DESCRIPTOR))
    return get_tilt.last_future

get_tilt.future=_get_tilt_future

def set_tilt(position: {"x": float, "y": float}, type: DeflectorType) -> {"x": float, "y": float}:
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

def get_camera_shift() -> {"x": float, "y": float}:
    """in pixels"""
    return _get_camera_shift(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_camera_shift_future() -> {"x": float, "y": float}:
    get_camera_shift.last_future = _get_camera_shift.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_camera_shift.last_future

get_camera_shift.future=_get_camera_shift_future

def set_camera_shift(x: float, y: float) -> {"x": float, "y": float}:
    """camera shift coils are located after the projection lenses, thus they will be very precised to place the beam to required place on camera,
        this value is used also in off_axis detector to selection of the chosen signal"""
    return _set_camera_shift(serializer.serialize({"x":x, "y":y}, instrument_pb2.PositionMessage.DESCRIPTOR))


def _set_camera_shift_future(x: float, y: float) -> {"x": float, "y": float}:
    set_camera_shift.last_future = _set_camera_shift.future(serializer.serialize({"x":x, "y":y}, instrument_pb2.PositionMessage.DESCRIPTOR))
    return set_camera_shift.last_future

set_camera_shift.future=_set_camera_shift_future

def get_camera_shift_range() -> list[{"x": float, "y": float}]:
    return _get_camera_shift_range(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_camera_shift_range_future() -> list[{"x": float, "y": float}]:
    get_camera_shift_range.last_future = _get_camera_shift_range.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_camera_shift_range.last_future

get_camera_shift_range.future=_get_camera_shift_range_future

def get_is_off_axis_stem_enabled() -> bool:
    return _get_is_off_axis_stem_enabled(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_is_off_axis_stem_enabled_future() -> bool:
    get_is_off_axis_stem_enabled.last_future = _get_is_off_axis_stem_enabled.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_is_off_axis_stem_enabled.last_future

get_is_off_axis_stem_enabled.future=_get_is_off_axis_stem_enabled_future

def set_is_off_axis_stem_enabled(value: bool) -> bool:
    """when BF detector is retracted it can be used for quick overview images,
        the switch from camera to the detector is in order of hundreds of ms, done just by deflectors,
        acquisition will be stopped, camera shift reseted"""
    return _set_is_off_axis_stem_enabled(serializer.serialize({"value":value}, instrument_pb2.BoolMessage.DESCRIPTOR))


def _set_is_off_axis_stem_enabled_future(value: bool) -> bool:
    set_is_off_axis_stem_enabled.last_future = _set_is_off_axis_stem_enabled.future(serializer.serialize({"value":value}, instrument_pb2.BoolMessage.DESCRIPTOR))
    return set_is_off_axis_stem_enabled.last_future

set_is_off_axis_stem_enabled.future=_set_is_off_axis_stem_enabled_future

def get_max_detector_angles() -> {"haadf": {"start": float, "end": float},
                                  "bf": {"start": float, "end": float},
                                  "camera": {"start": float, "end": float}}:
    """returns detector half angle ranges based on the optics tracing, this method is slow around 200ms"""
    return _get_max_detector_angles(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_max_detector_angles_future() -> {"haadf": {"start": float, "end": float},
                                  "bf": {"start": float, "end": float},
                                  "camera": {"start": float, "end": float}}:
    get_max_detector_angles.last_future = _get_max_detector_angles.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_max_detector_angles.last_future

get_max_detector_angles.future=_get_max_detector_angles_future

def get_max_camera_angle() -> float:
    return _get_max_camera_angle(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_max_camera_angle_future() -> float:
    get_max_camera_angle.last_future = _get_max_camera_angle.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_max_camera_angle.last_future

get_max_camera_angle.future=_get_max_camera_angle_future

def set_max_camera_angle(detector: DetectorType, value: float, keep_adjustments: bool) -> float:
    """calibrated half angle on camera"""
    return _set_max_camera_angle(serializer.serialize({"detector":detector, "value":value, "keep_adjustments":keep_adjustments}, instrument_pb2.SetMaxCameraAngleMessage.DESCRIPTOR))


def _set_max_camera_angle_future(detector: DetectorType, value: float, keep_adjustments: bool) -> float:
    set_max_camera_angle.last_future = _set_max_camera_angle.future(serializer.serialize({"detector":detector, "value":value, "keep_adjustments":keep_adjustments}, instrument_pb2.SetMaxCameraAngleMessage.DESCRIPTOR))
    return set_max_camera_angle.last_future

set_max_camera_angle.future=_set_max_camera_angle_future

def get_max_camera_angle_range() -> {"start": float, "end": float}:
    return _get_max_camera_angle_range(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_max_camera_angle_range_future() -> {"start": float, "end": float}:
    get_max_camera_angle_range.last_future = _get_max_camera_angle_range.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_max_camera_angle_range.last_future

get_max_camera_angle_range.future=_get_max_camera_angle_range_future

def get_camera_to_stage_rotation() -> float:
    return _get_camera_to_stage_rotation(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_camera_to_stage_rotation_future() -> float:
    get_camera_to_stage_rotation.last_future = _get_camera_to_stage_rotation.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_camera_to_stage_rotation.last_future

get_camera_to_stage_rotation.future=_get_camera_to_stage_rotation_future

def get_projector_defocus() -> float:
    """in At of P1"""
    return _get_projector_defocus(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_projector_defocus_future() -> float:
    get_projector_defocus.last_future = _get_projector_defocus.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_projector_defocus.last_future

get_projector_defocus.future=_get_projector_defocus_future

def set_projector_defocus(value: float) -> float:
    """in At of P1"""
    return _set_projector_defocus(serializer.serialize({"value":value}, instrument_pb2.FloatMessage.DESCRIPTOR))


def _set_projector_defocus_future(value: float) -> float:
    set_projector_defocus.last_future = _set_projector_defocus.future(serializer.serialize({"value":value}, instrument_pb2.FloatMessage.DESCRIPTOR))
    return set_projector_defocus.last_future

set_projector_defocus.future=_set_projector_defocus_future

def get_projector_defocus_range() -> {"start": float, "end": float}:
    return _get_projector_defocus_range(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_projector_defocus_range_future() -> {"start": float, "end": float}:
    get_projector_defocus_range.last_future = _get_projector_defocus_range.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_projector_defocus_range.last_future

get_projector_defocus_range.future=_get_projector_defocus_range_future
