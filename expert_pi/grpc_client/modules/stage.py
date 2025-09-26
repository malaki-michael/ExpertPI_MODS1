# GENERATED FILE from pySTEM 1.16.1,build 4717,rev b31140633df20050ad44d1030889522d4f5f8256

from .. import instrument_pb2
from .. import serializer
from ._common import *
from .. import channel as _channel

_get_calibration = None
_set_calibration = None
_get_x_y = None
_set_x_y = None
_get_x_y_range = None
_get_z = None
_set_z = None
_get_z_range = None
_get_alpha = None
_set_alpha = None
_get_alpha_range = None
_is_beta_enabled = None
_get_beta = None
_set_beta = None
_get_beta_range = None
_recalculate_ranges = None
_get_gamma = None
_get_tilt = None
_set_tilt = None
_get_tilt_range = None
_get_holder_info = None
_stop = None
_get_optics_to_sample_projection = None
_set_x_y_z = None
_set_x_y_z_a_b = None
_read_x_y_z_a_b = None


def connect():
    global _get_calibration
    global _set_calibration
    global _get_x_y
    global _set_x_y
    global _get_x_y_range
    global _get_z
    global _set_z
    global _get_z_range
    global _get_alpha
    global _set_alpha
    global _get_alpha_range
    global _is_beta_enabled
    global _get_beta
    global _set_beta
    global _get_beta_range
    global _recalculate_ranges
    global _get_gamma
    global _get_tilt
    global _set_tilt
    global _get_tilt_range
    global _get_holder_info
    global _stop
    global _get_optics_to_sample_projection
    global _set_x_y_z
    global _set_x_y_z_a_b
    global _read_x_y_z_a_b

    _get_calibration = _channel.channel.unary_unary(
        '/pystem.api.grpc.Stage/GetCalibration',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.GetCalibrationMessage).DictFromString,
    )

    _set_calibration = _channel.channel.unary_unary(
        '/pystem.api.grpc.Stage/SetCalibration',
        request_serializer=instrument_pb2.GetCalibrationMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.GetCalibrationMessage).DictFromString,
    )

    _get_x_y = _channel.channel.unary_unary(
        '/pystem.api.grpc.Stage/GetXY',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.PositionMessage).DictFromString,
    )

    _set_x_y = _channel.channel.unary_unary(
        '/pystem.api.grpc.Stage/SetXY',
        request_serializer=instrument_pb2.SetXYMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.PositionMessage).DictFromString,
    )

    _get_x_y_range = _channel.channel.unary_unary(
        '/pystem.api.grpc.Stage/GetXYRange',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.GetStigmatorRangeMessage).DictFromString,
    )

    _get_z = _channel.channel.unary_unary(
        '/pystem.api.grpc.Stage/GetZ',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _set_z = _channel.channel.unary_unary(
        '/pystem.api.grpc.Stage/SetZ',
        request_serializer=instrument_pb2.SetZMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _get_z_range = _channel.channel.unary_unary(
        '/pystem.api.grpc.Stage/GetZRange',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatRangeMessage).DictFromString,
    )

    _get_alpha = _channel.channel.unary_unary(
        '/pystem.api.grpc.Stage/GetAlpha',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _set_alpha = _channel.channel.unary_unary(
        '/pystem.api.grpc.Stage/SetAlpha',
        request_serializer=instrument_pb2.SetAlphaMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _get_alpha_range = _channel.channel.unary_unary(
        '/pystem.api.grpc.Stage/GetAlphaRange',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatRangeMessage).DictFromString,
    )

    _is_beta_enabled = _channel.channel.unary_unary(
        '/pystem.api.grpc.Stage/IsBetaEnabled',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.BoolMessage).DictFromString,
    )

    _get_beta = _channel.channel.unary_unary(
        '/pystem.api.grpc.Stage/GetBeta',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _set_beta = _channel.channel.unary_unary(
        '/pystem.api.grpc.Stage/SetBeta',
        request_serializer=instrument_pb2.SetBetaMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _get_beta_range = _channel.channel.unary_unary(
        '/pystem.api.grpc.Stage/GetBetaRange',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatRangeMessage).DictFromString,
    )

    _recalculate_ranges = _channel.channel.unary_unary(
        '/pystem.api.grpc.Stage/RecalculateRanges',
        request_serializer=instrument_pb2.RecalculateRangesMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _get_gamma = _channel.channel.unary_unary(
        '/pystem.api.grpc.Stage/GetGamma',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _get_tilt = _channel.channel.unary_unary(
        '/pystem.api.grpc.Stage/GetTilt',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.StageTiltMessage).DictFromString,
    )

    _set_tilt = _channel.channel.unary_unary(
        '/pystem.api.grpc.Stage/SetTilt',
        request_serializer=instrument_pb2.SetTiltRequest.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.StageTiltMessage).DictFromString,
    )

    _get_tilt_range = _channel.channel.unary_unary(
        '/pystem.api.grpc.Stage/GetTiltRange',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.GetTiltRangeMessage).DictFromString,
    )

    _get_holder_info = _channel.channel.unary_unary(
        '/pystem.api.grpc.Stage/GetHolderInfo',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.GetHolderInfoMessage).DictFromString,
    )

    _stop = _channel.channel.unary_unary(
        '/pystem.api.grpc.Stage/Stop',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.EmptyMessage).DictFromString,
    )

    _get_optics_to_sample_projection = _channel.channel.unary_unary(
        '/pystem.api.grpc.Stage/GetOpticsToSampleProjection',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.GetEnergyRangeMessage).DictFromString,
    )

    _set_x_y_z = _channel.channel.unary_unary(
        '/pystem.api.grpc.Stage/SetXYZ',
        request_serializer=instrument_pb2.SetXYZMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.SetXYZMessage).DictFromString,
    )

    _set_x_y_z_a_b = _channel.channel.unary_unary(
        '/pystem.api.grpc.Stage/SetXYZAB',
        request_serializer=instrument_pb2.SetXYZABMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.GetCalibrationMessage).DictFromString,
    )

    _read_x_y_z_a_b = _channel.channel.unary_unary(
        '/pystem.api.grpc.Stage/ReadXYZAB',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.GetCalibrationMessage).DictFromString,
    )


def get_calibration() -> {"x": float,
                          "y": float,
                          "z": float,
                          "alpha": float,
                          "beta": float,
                          "gamma": float}:
    return _get_calibration(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_calibration_future() -> {"x": float,
                          "y": float,
                          "z": float,
                          "alpha": float,
                          "beta": float,
                          "gamma": float}:
    get_calibration.last_future = _get_calibration.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_calibration.last_future

get_calibration.future=_get_calibration_future

def set_calibration(x: float, y: float, z: float,
                    alpha: float, beta: float, gamma: float) \
        -> {"x": float,
            "y": float,
            "z": float,
            "alpha": float,
            "beta": float,
            "gamma": float}:
    """calibration of the holder to the currently loaded sample, x,y,z shifts and alpha,beta ganna intrinsic euler angles"""
    return _set_calibration(serializer.serialize({"x":x, "y":y, "z":z, "alpha":alpha, "beta":beta, "gamma":gamma}, instrument_pb2.GetCalibrationMessage.DESCRIPTOR))


def _set_calibration_future(x: float, y: float, z: float,
                    alpha: float, beta: float, gamma: float) \
        -> {"x": float,
            "y": float,
            "z": float,
            "alpha": float,
            "beta": float,
            "gamma": float}:
    set_calibration.last_future = _set_calibration.future(serializer.serialize({"x":x, "y":y, "z":z, "alpha":alpha, "beta":beta, "gamma":gamma}, instrument_pb2.GetCalibrationMessage.DESCRIPTOR))
    return set_calibration.last_future

set_calibration.future=_set_calibration_future

def get_x_y() -> {"x": float, "y": float}:
    return _get_x_y(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_x_y_future() -> {"x": float, "y": float}:
    get_x_y.last_future = _get_x_y.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_x_y.last_future

get_x_y.future=_get_x_y_future

def set_x_y(x: float, y: float, nowait: bool = False, fixed_alpha: bool = False, fixed_beta: bool = False) -> {"x": float, "y": float}:
    """x,y coordinates are with respect to the sample, plane, note that it will not correspond to scanning xy positions at non-zero tilts"""
    return _set_x_y(serializer.serialize({"x":x, "y":y, "nowait":nowait, "fixed_alpha":fixed_alpha, "fixed_beta":fixed_beta}, instrument_pb2.SetXYMessage.DESCRIPTOR))


def _set_x_y_future(x: float, y: float, nowait: bool = False, fixed_alpha: bool = False, fixed_beta: bool = False) -> {"x": float, "y": float}:
    set_x_y.last_future = _set_x_y.future(serializer.serialize({"x":x, "y":y, "nowait":nowait, "fixed_alpha":fixed_alpha, "fixed_beta":fixed_beta}, instrument_pb2.SetXYMessage.DESCRIPTOR))
    return set_x_y.last_future

set_x_y.future=_set_x_y_future

def get_x_y_range() -> list[{"x": float, "y": float}]:
    """approximated polygon of the allowed ranges, this will be different for each alpha, beta rotations and z position"""
    return _get_x_y_range(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_x_y_range_future() -> list[{"x": float, "y": float}]:
    get_x_y_range.last_future = _get_x_y_range.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_x_y_range.last_future

get_x_y_range.future=_get_x_y_range_future

def get_z() -> float:
    return _get_z(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_z_future() -> float:
    get_z.last_future = _get_z.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_z.last_future

get_z.future=_get_z_future

def set_z(value: float, nowait: bool = False, fixed_alpha: bool = False, fixed_beta: bool = False) -> float:
    """z coordinates is always in the direction of the beam - even at tilted sample"""
    return _set_z(serializer.serialize({"value":value, "nowait":nowait, "fixed_alpha":fixed_alpha, "fixed_beta":fixed_beta}, instrument_pb2.SetZMessage.DESCRIPTOR))


def _set_z_future(value: float, nowait: bool = False, fixed_alpha: bool = False, fixed_beta: bool = False) -> float:
    set_z.last_future = _set_z.future(serializer.serialize({"value":value, "nowait":nowait, "fixed_alpha":fixed_alpha, "fixed_beta":fixed_beta}, instrument_pb2.SetZMessage.DESCRIPTOR))
    return set_z.last_future

set_z.future=_set_z_future

def get_z_range() -> {"start": float, "end": float}:
    return _get_z_range(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_z_range_future() -> {"start": float, "end": float}:
    get_z_range.last_future = _get_z_range.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_z_range.last_future

get_z_range.future=_get_z_range_future

def get_alpha() -> float:
    return _get_alpha(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_alpha_future() -> float:
    get_alpha.last_future = _get_alpha.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_alpha.last_future

get_alpha.future=_get_alpha_future

def set_alpha(value: float, nowait: bool = False, fixed_beta: bool = False) -> float:
    """tilting alpha with respect to the selected x,y,z position on the sample"""
    return _set_alpha(serializer.serialize({"value":value, "nowait":nowait, "fixed_beta":fixed_beta}, instrument_pb2.SetAlphaMessage.DESCRIPTOR))


def _set_alpha_future(value: float, nowait: bool = False, fixed_beta: bool = False) -> float:
    set_alpha.last_future = _set_alpha.future(serializer.serialize({"value":value, "nowait":nowait, "fixed_beta":fixed_beta}, instrument_pb2.SetAlphaMessage.DESCRIPTOR))
    return set_alpha.last_future

set_alpha.future=_set_alpha_future

def get_alpha_range() -> {"start": float, "end": float}:
    return _get_alpha_range(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_alpha_range_future() -> {"start": float, "end": float}:
    get_alpha_range.last_future = _get_alpha_range.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_alpha_range.last_future

get_alpha_range.future=_get_alpha_range_future

def is_beta_enabled() -> bool:
    """check if the current holder supports beta rotation"""
    return _is_beta_enabled(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _is_beta_enabled_future() -> bool:
    is_beta_enabled.last_future = _is_beta_enabled.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return is_beta_enabled.last_future

is_beta_enabled.future=_is_beta_enabled_future

def get_beta() -> float:
    return _get_beta(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_beta_future() -> float:
    get_beta.last_future = _get_beta.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_beta.last_future

get_beta.future=_get_beta_future

def set_beta(value: float, nowait: bool = False, fixed_alpha: bool = False) -> float:
    """tilting beta with respect to the selected x,y,z position on the sample"""
    return _set_beta(serializer.serialize({"value":value, "nowait":nowait, "fixed_alpha":fixed_alpha}, instrument_pb2.SetBetaMessage.DESCRIPTOR))


def _set_beta_future(value: float, nowait: bool = False, fixed_alpha: bool = False) -> float:
    set_beta.last_future = _set_beta.future(serializer.serialize({"value":value, "nowait":nowait, "fixed_alpha":fixed_alpha}, instrument_pb2.SetBetaMessage.DESCRIPTOR))
    return set_beta.last_future

set_beta.future=_set_beta_future

def get_beta_range() -> {"start": float, "end": float}:
    return _get_beta_range(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_beta_range_future() -> {"start": float, "end": float}:
    get_beta_range.last_future = _get_beta_range.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_beta_range.last_future

get_beta_range.future=_get_beta_range_future

def recalculate_ranges(max_time: float) -> float:
    return _recalculate_ranges(serializer.serialize({"max_time":max_time}, instrument_pb2.RecalculateRangesMessage.DESCRIPTOR))


def _recalculate_ranges_future(max_time: float) -> float:
    recalculate_ranges.last_future = _recalculate_ranges.future(serializer.serialize({"max_time":max_time}, instrument_pb2.RecalculateRangesMessage.DESCRIPTOR))
    return recalculate_ranges.last_future

recalculate_ranges.future=_recalculate_ranges_future

def get_gamma() -> float:
    """this is residual gamma rotation due to stage mechanics, the scan rotation is actually applied on top of this rotation"""
    return _get_gamma(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_gamma_future() -> float:
    get_gamma.last_future = _get_gamma.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_gamma.last_future

get_gamma.future=_get_gamma_future

def get_tilt() -> {"alpha": float, "beta": float}:
    return _get_tilt(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_tilt_future() -> {"alpha": float, "beta": float}:
    get_tilt.last_future = _get_tilt.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_tilt.last_future

get_tilt.future=_get_tilt_future

def set_tilt(alpha: float, beta: float, nowait: bool = False) -> {"alpha": float, "beta": float}:
    """set alpha,beta simultaneously"""
    return _set_tilt(serializer.serialize({"alpha":alpha, "beta":beta, "nowait":nowait}, instrument_pb2.SetTiltRequest.DESCRIPTOR))


def _set_tilt_future(alpha: float, beta: float, nowait: bool = False) -> {"alpha": float, "beta": float}:
    set_tilt.last_future = _set_tilt.future(serializer.serialize({"alpha":alpha, "beta":beta, "nowait":nowait}, instrument_pb2.SetTiltRequest.DESCRIPTOR))
    return set_tilt.last_future

set_tilt.future=_set_tilt_future

def get_tilt_range() -> list[{"alpha": float, "beta": float}]:
    """approximated polygon of the allowed ranges, this will be different for each xyz position"""
    return _get_tilt_range(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_tilt_range_future() -> list[{"alpha": float, "beta": float}]:
    get_tilt_range.last_future = _get_tilt_range.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_tilt_range.last_future

get_tilt_range.future=_get_tilt_range_future

def get_holder_info() -> {"loaded": uint64, "holder_name": str, "holder_type": HolderType}:
    return _get_holder_info(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_holder_info_future() -> {"loaded": uint64, "holder_name": str, "holder_type": HolderType}:
    get_holder_info.last_future = _get_holder_info.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_holder_info.last_future

get_holder_info.future=_get_holder_info_future

def stop():
    return _stop(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _stop_future():
    stop.last_future = _stop.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return stop.last_future

stop.future=_stop_future

def get_optics_to_sample_projection() -> list[float]:
    """(3x3) flattened affine transformation \\(P\_{os}\\) - radians and meters,
        to get the position on the sample plane simply calculate: $${P\_{os}\cdot\left(\begin{matrix}x\\\y\\\1\end{matrix}\right)}$$
        where \\(x,y\\) are coordinates on the scanned image (assuming the center has coordinates (0,0))"""
    return _get_optics_to_sample_projection(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_optics_to_sample_projection_future() -> list[float]:
    get_optics_to_sample_projection.last_future = _get_optics_to_sample_projection.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_optics_to_sample_projection.last_future

get_optics_to_sample_projection.future=_get_optics_to_sample_projection_future

def set_x_y_z(x: float, y: float, z: float) -> {"x": float, "y": float, "z": float}:
    """fast setting of xyz"""
    return _set_x_y_z(serializer.serialize({"x":x, "y":y, "z":z}, instrument_pb2.SetXYZMessage.DESCRIPTOR))


def _set_x_y_z_future(x: float, y: float, z: float) -> {"x": float, "y": float, "z": float}:
    set_x_y_z.last_future = _set_x_y_z.future(serializer.serialize({"x":x, "y":y, "z":z}, instrument_pb2.SetXYZMessage.DESCRIPTOR))
    return set_x_y_z.last_future

set_x_y_z.future=_set_x_y_z_future

def set_x_y_z_a_b(x: float, y: float, z: float, alpha: float, beta: float) -> {"x": float, "y": float, "z": float, "alpha": float, "beta": float, "gamma": float}:
    """fast setting of xyz,alpha beta, if single tilt holder beta is ignored"""
    return _set_x_y_z_a_b(serializer.serialize({"x":x, "y":y, "z":z, "alpha":alpha, "beta":beta}, instrument_pb2.SetXYZABMessage.DESCRIPTOR))


def _set_x_y_z_a_b_future(x: float, y: float, z: float, alpha: float, beta: float) -> {"x": float, "y": float, "z": float, "alpha": float, "beta": float, "gamma": float}:
    set_x_y_z_a_b.last_future = _set_x_y_z_a_b.future(serializer.serialize({"x":x, "y":y, "z":z, "alpha":alpha, "beta":beta}, instrument_pb2.SetXYZABMessage.DESCRIPTOR))
    return set_x_y_z_a_b.last_future

set_x_y_z_a_b.future=_set_x_y_z_a_b_future

def read_x_y_z_a_b() -> {"x": float, "y": float, "z": float, "alpha": float, "beta": float, "gamma": float}:
    """read actual stage position by encoders"""
    return _read_x_y_z_a_b(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _read_x_y_z_a_b_future() -> {"x": float, "y": float, "z": float, "alpha": float, "beta": float, "gamma": float}:
    read_x_y_z_a_b.last_future = _read_x_y_z_a_b.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return read_x_y_z_a_b.last_future

read_x_y_z_a_b.future=_read_x_y_z_a_b_future
