# GENERATED FILE from pySTEM 1.16.1,build 4717,rev b31140633df20050ad44d1030889522d4f5f8256

from .. import instrument_pb2
from .. import serializer
from ._common import *
from .. import channel as _channel

_get_field_width = None
_set_field_width = None
_get_field_width_range = None
_get_fov_ranges_precession_scan = None
_set_zoom_range_index = None
_get_zoom_range_index = None
_get_applied_zoom_range_indices = None
_get_rotation = None
_get_shear = None
_set_shear = None
_get_aspect_y = None
_set_aspect_y = None
_set_rotation = None
_get_maximum_scan_field_number_of_pixels = None
_get_precession_frequency = None
_set_precession_frequency = None
_get_aligned_precession_frequencies = None
_get_precession_angle = None
_set_precession_angle = None
_get_precession_angle_range = None
_get_precession_height = None
_set_precession_height = None
_get_precession_height_range = None
_get_precession_height_correction = None
_set_precession_height_correction = None
_get_deprecession_height = None
_set_deprecession_height = None
_get_deprecession_height_range = None
_get_deprecession_tilt_correction = None
_set_deprecession_tilt_correction = None
_start_rectangle_scan = None
_start_tilt_series = None
_stop_scanning = None
_get_drift_rate = None
_set_drift_rate = None
_get_scale_factor = None
_set_scale_factor = None
_start_camera = None
_stop_camera = None
_set_camera_roi = None
_get_camera_roi = None
_get_camera_max_fps = None
_camera_is_acquiring = None
_get_camera_bith_depth = None
_get_distortion_coefficients = None
_set_distortion_coefficients = None
_set_blanker_mode = None
_get_blanker_mode = None
_set_image_compression = None
_get_image_compression = None
_get_dead_pixels = None


def connect():
    global _get_field_width
    global _set_field_width
    global _get_field_width_range
    global _get_fov_ranges_precession_scan
    global _set_zoom_range_index
    global _get_zoom_range_index
    global _get_applied_zoom_range_indices
    global _get_rotation
    global _get_shear
    global _set_shear
    global _get_aspect_y
    global _set_aspect_y
    global _set_rotation
    global _get_maximum_scan_field_number_of_pixels
    global _get_precession_frequency
    global _set_precession_frequency
    global _get_aligned_precession_frequencies
    global _get_precession_angle
    global _set_precession_angle
    global _get_precession_angle_range
    global _get_precession_height
    global _set_precession_height
    global _get_precession_height_range
    global _get_precession_height_correction
    global _set_precession_height_correction
    global _get_deprecession_height
    global _set_deprecession_height
    global _get_deprecession_height_range
    global _get_deprecession_tilt_correction
    global _set_deprecession_tilt_correction
    global _start_rectangle_scan
    global _start_tilt_series
    global _stop_scanning
    global _get_drift_rate
    global _set_drift_rate
    global _get_scale_factor
    global _set_scale_factor
    global _start_camera
    global _stop_camera
    global _set_camera_roi
    global _get_camera_roi
    global _get_camera_max_fps
    global _camera_is_acquiring
    global _get_camera_bith_depth
    global _get_distortion_coefficients
    global _set_distortion_coefficients
    global _set_blanker_mode
    global _get_blanker_mode
    global _set_image_compression
    global _get_image_compression
    global _get_dead_pixels

    _get_field_width = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/GetFieldWidth',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _set_field_width = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/SetFieldWidth',
        request_serializer=instrument_pb2.FloatMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _get_field_width_range = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/GetFieldWidthRange',
        request_serializer=instrument_pb2.GetFieldWidthRangeMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatRangeMessage).DictFromString,
    )

    _get_fov_ranges_precession_scan = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/GetFovRangesPrecessionScan',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.GetEnergyRangeMessage).DictFromString,
    )

    _set_zoom_range_index = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/SetZoomRangeIndex',
        request_serializer=instrument_pb2.SetZoomRangeIndexMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.SetZoomRangeIndexRequest).DictFromString,
    )

    _get_zoom_range_index = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/GetZoomRangeIndex',
        request_serializer=instrument_pb2.GetZoomRangeIndexMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.SetZoomRangeIndexRequest).DictFromString,
    )

    _get_applied_zoom_range_indices = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/GetAppliedZoomRangeIndices',
        request_serializer=instrument_pb2.GetZoomRangeIndexMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.GetAppliedZoomRangeIndicesMessage).DictFromString,
    )

    _get_rotation = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/GetRotation',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _get_shear = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/GetShear',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _set_shear = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/SetShear',
        request_serializer=instrument_pb2.FloatMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _get_aspect_y = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/GetAspectY',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _set_aspect_y = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/SetAspectY',
        request_serializer=instrument_pb2.FloatMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _set_rotation = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/SetRotation',
        request_serializer=instrument_pb2.FloatMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _get_maximum_scan_field_number_of_pixels = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/GetMaximumScanFieldNumberOfPixels',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.UIntMessage).DictFromString,
    )

    _get_precession_frequency = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/GetPrecessionFrequency',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _set_precession_frequency = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/SetPrecessionFrequency',
        request_serializer=instrument_pb2.FloatMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.EmptyMessage).DictFromString,
    )

    _get_aligned_precession_frequencies = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/GetAlignedPrecessionFrequencies',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.GetEnergyRangeMessage).DictFromString,
    )

    _get_precession_angle = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/GetPrecessionAngle',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _set_precession_angle = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/SetPrecessionAngle',
        request_serializer=instrument_pb2.SetPrecessionAngleMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _get_precession_angle_range = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/GetPrecessionAngleRange',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatRangeMessage).DictFromString,
    )

    _get_precession_height = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/GetPrecessionHeight',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.PositionMessage).DictFromString,
    )

    _set_precession_height = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/SetPrecessionHeight',
        request_serializer=instrument_pb2.PositionMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.PositionMessage).DictFromString,
    )

    _get_precession_height_range = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/GetPrecessionHeightRange',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.GetStigmatorRangeMessage).DictFromString,
    )

    _get_precession_height_correction = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/GetPrecessionHeightCorrection',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.GetEnergyRangeMessage).DictFromString,
    )

    _set_precession_height_correction = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/SetPrecessionHeightCorrection',
        request_serializer=instrument_pb2.SetPrecessionHeightCorrectionMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.GetEnergyRangeMessage).DictFromString,
    )

    _get_deprecession_height = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/GetDeprecessionHeight',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.PositionMessage).DictFromString,
    )

    _set_deprecession_height = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/SetDeprecessionHeight',
        request_serializer=instrument_pb2.PositionMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.PositionMessage).DictFromString,
    )

    _get_deprecession_height_range = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/GetDeprecessionHeightRange',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.GetStigmatorRangeMessage).DictFromString,
    )

    _get_deprecession_tilt_correction = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/GetDeprecessionTiltCorrection',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.GetEnergyRangeMessage).DictFromString,
    )

    _set_deprecession_tilt_correction = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/SetDeprecessionTiltCorrection',
        request_serializer=instrument_pb2.SetPrecessionHeightCorrectionMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.GetEnergyRangeMessage).DictFromString,
    )

    _start_rectangle_scan = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/StartRectangleScan',
        request_serializer=instrument_pb2.StartRectangleScanMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.StartRectangleScanRequest).DictFromString,
    )

    _start_tilt_series = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/StartTiltSeries',
        request_serializer=instrument_pb2.StartTiltSeriesMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.StartRectangleScanRequest).DictFromString,
    )

    _stop_scanning = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/StopScanning',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.EmptyMessage).DictFromString,
    )

    _get_drift_rate = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/GetDriftRate',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.PositionMessage).DictFromString,
    )

    _set_drift_rate = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/SetDriftRate',
        request_serializer=instrument_pb2.PositionMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.PositionMessage).DictFromString,
    )

    _get_scale_factor = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/GetScaleFactor',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _set_scale_factor = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/SetScaleFactor',
        request_serializer=instrument_pb2.FloatMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _start_camera = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/StartCamera',
        request_serializer=instrument_pb2.StartCameraMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.EmptyMessage).DictFromString,
    )

    _stop_camera = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/StopCamera',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.EmptyMessage).DictFromString,
    )

    _set_camera_roi = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/SetCameraRoi',
        request_serializer=instrument_pb2.SetCameraRoiMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.SetCameraRoiMessage).DictFromString,
    )

    _get_camera_roi = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/GetCameraRoi',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.SetCameraRoiMessage).DictFromString,
    )

    _get_camera_max_fps = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/GetCameraMaxFps',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.FloatMessage).DictFromString,
    )

    _camera_is_acquiring = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/CameraIsAcquiring',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.BoolMessage).DictFromString,
    )

    _get_camera_bith_depth = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/GetCameraBithDepth',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.UIntMessage).DictFromString,
    )

    _get_distortion_coefficients = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/GetDistortionCoefficients',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.GetDistortionCoefficientsMessage).DictFromString,
    )

    _set_distortion_coefficients = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/SetDistortionCoefficients',
        request_serializer=instrument_pb2.GetDistortionCoefficientsMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.GetDistortionCoefficientsMessage).DictFromString,
    )

    _set_blanker_mode = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/SetBlankerMode',
        request_serializer=instrument_pb2.SetBlankerModeMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.SetBlankerModeMessage).DictFromString,
    )

    _get_blanker_mode = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/GetBlankerMode',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.SetBlankerModeMessage).DictFromString,
    )

    _set_image_compression = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/SetImageCompression',
        request_serializer=instrument_pb2.SetImageCompressionMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.SetImageCompressionMessage).DictFromString,
    )

    _get_image_compression = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/GetImageCompression',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.SetImageCompressionMessage).DictFromString,
    )

    _get_dead_pixels = _channel.channel.unary_unary(
        '/pystem.api.grpc.Scanning/GetDeadPixels',
        request_serializer=instrument_pb2.EmptyMessage.SerializeToString,
        response_deserializer=serializer.append_DictFromString(instrument_pb2.GetDeadPixelsMessage).DictFromString,
    )


def get_field_width() -> float:
    return _get_field_width(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_field_width_future() -> float:
    get_field_width.last_future = _get_field_width.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_field_width.last_future

get_field_width.future=_get_field_width_future

def set_field_width(value: float) -> float:
    return _set_field_width(serializer.serialize({"value":value}, instrument_pb2.FloatMessage.DESCRIPTOR))


def _set_field_width_future(value: float) -> float:
    set_field_width.last_future = _set_field_width.future(serializer.serialize({"value":value}, instrument_pb2.FloatMessage.DESCRIPTOR))
    return set_field_width.last_future

set_field_width.future=_set_field_width_future

def get_field_width_range(pixel_time: float, scan_field_number_of_pixels: int) -> {"start": float, "end": float}:
    """There is a overhead which is used for compensation of the dynamical effect, for slower acquisition the available fov is larger than for faster acquisitions.
        If the scan hexade is set (not auto -1), it will be used in the calculation, otherwise maximum range is used."""
    return _get_field_width_range(serializer.serialize({"pixel_time":pixel_time, "scan_field_number_of_pixels":scan_field_number_of_pixels}, instrument_pb2.GetFieldWidthRangeMessage.DESCRIPTOR))


def _get_field_width_range_future(pixel_time: float, scan_field_number_of_pixels: int) -> {"start": float, "end": float}:
    get_field_width_range.last_future = _get_field_width_range.future(serializer.serialize({"pixel_time":pixel_time, "scan_field_number_of_pixels":scan_field_number_of_pixels}, instrument_pb2.GetFieldWidthRangeMessage.DESCRIPTOR))
    return get_field_width_range.last_future

get_field_width_range.future=_get_field_width_range_future

def get_fov_ranges_precession_scan() -> list[float]:
    """Returns a list of maximal table-fov (m) of precession scans for each hexade (index=hexade).
        These values determine scan-hexade by set fov during precessioon-scans if the scan-hexade is not forced by SetZoomRangeIndex."""
    return _get_fov_ranges_precession_scan(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_fov_ranges_precession_scan_future() -> list[float]:
    get_fov_ranges_precession_scan.last_future = _get_fov_ranges_precession_scan.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_fov_ranges_precession_scan.last_future

get_fov_ranges_precession_scan.future=_get_fov_ranges_precession_scan_future

def set_zoom_range_index(value: int64, type_scan: ScanSystemType) -> int64:
    """Hexades 0-5, -1 means automatic changing (and 'unfreezing').
        For the Scanning system it also 'freezes' the hexade. For the Precession system it can still be overwritten in pystem, if the max amplifier currents aren't be enough."""
    return _set_zoom_range_index(serializer.serialize({"value":value, "type_scan":type_scan}, instrument_pb2.SetZoomRangeIndexMessage.DESCRIPTOR))


def _set_zoom_range_index_future(value: int64, type_scan: ScanSystemType) -> int64:
    set_zoom_range_index.last_future = _set_zoom_range_index.future(serializer.serialize({"value":value, "type_scan":type_scan}, instrument_pb2.SetZoomRangeIndexMessage.DESCRIPTOR))
    return set_zoom_range_index.last_future

set_zoom_range_index.future=_set_zoom_range_index_future

def get_zoom_range_index(type_scan: ScanSystemType) -> int64:
    return _get_zoom_range_index(serializer.serialize({"type_scan":type_scan}, instrument_pb2.GetZoomRangeIndexMessage.DESCRIPTOR))


def _get_zoom_range_index_future(type_scan: ScanSystemType) -> int64:
    get_zoom_range_index.last_future = _get_zoom_range_index.future(serializer.serialize({"type_scan":type_scan}, instrument_pb2.GetZoomRangeIndexMessage.DESCRIPTOR))
    return get_zoom_range_index.last_future

get_zoom_range_index.future=_get_zoom_range_index_future

def get_applied_zoom_range_indices(type_scan: ScanSystemType) -> list[int]:
    """Returns the last actually set hexades on the specified scanning system in the order [O1DS, O1Dl, O2Dl, O2DS] - for debugging and testing."""
    return _get_applied_zoom_range_indices(serializer.serialize({"type_scan":type_scan}, instrument_pb2.GetZoomRangeIndexMessage.DESCRIPTOR))


def _get_applied_zoom_range_indices_future(type_scan: ScanSystemType) -> list[int]:
    get_applied_zoom_range_indices.last_future = _get_applied_zoom_range_indices.future(serializer.serialize({"type_scan":type_scan}, instrument_pb2.GetZoomRangeIndexMessage.DESCRIPTOR))
    return get_applied_zoom_range_indices.last_future

get_applied_zoom_range_indices.future=_get_applied_zoom_range_indices_future

def get_rotation() -> float:
    return _get_rotation(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_rotation_future() -> float:
    get_rotation.last_future = _get_rotation.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_rotation.last_future

get_rotation.future=_get_rotation_future

def get_shear() -> float:
    return _get_shear(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_shear_future() -> float:
    get_shear.last_future = _get_shear.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_shear.last_future

get_shear.future=_get_shear_future

def set_shear(value: float) -> float:
    return _set_shear(serializer.serialize({"value":value}, instrument_pb2.FloatMessage.DESCRIPTOR))


def _set_shear_future(value: float) -> float:
    set_shear.last_future = _set_shear.future(serializer.serialize({"value":value}, instrument_pb2.FloatMessage.DESCRIPTOR))
    return set_shear.last_future

set_shear.future=_set_shear_future

def get_aspect_y() -> float:
    return _get_aspect_y(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_aspect_y_future() -> float:
    get_aspect_y.last_future = _get_aspect_y.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_aspect_y.last_future

get_aspect_y.future=_get_aspect_y_future

def set_aspect_y(value: float) -> float:
    return _set_aspect_y(serializer.serialize({"value":value}, instrument_pb2.FloatMessage.DESCRIPTOR))


def _set_aspect_y_future(value: float) -> float:
    set_aspect_y.last_future = _set_aspect_y.future(serializer.serialize({"value":value}, instrument_pb2.FloatMessage.DESCRIPTOR))
    return set_aspect_y.last_future

set_aspect_y.future=_set_aspect_y_future

def set_rotation(value: float) -> float:
    return _set_rotation(serializer.serialize({"value":value}, instrument_pb2.FloatMessage.DESCRIPTOR))


def _set_rotation_future(value: float) -> float:
    set_rotation.last_future = _set_rotation.future(serializer.serialize({"value":value}, instrument_pb2.FloatMessage.DESCRIPTOR))
    return set_rotation.last_future

set_rotation.future=_set_rotation_future

def get_maximum_scan_field_number_of_pixels() -> int:
    return _get_maximum_scan_field_number_of_pixels(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_maximum_scan_field_number_of_pixels_future() -> int:
    get_maximum_scan_field_number_of_pixels.last_future = _get_maximum_scan_field_number_of_pixels.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_maximum_scan_field_number_of_pixels.last_future

get_maximum_scan_field_number_of_pixels.future=_get_maximum_scan_field_number_of_pixels_future

def get_precession_frequency() -> float:
    return _get_precession_frequency(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_precession_frequency_future() -> float:
    get_precession_frequency.last_future = _get_precession_frequency.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_precession_frequency.last_future

get_precession_frequency.future=_get_precession_frequency_future

def set_precession_frequency(value: float):
    return _set_precession_frequency(serializer.serialize({"value":value}, instrument_pb2.FloatMessage.DESCRIPTOR))


def _set_precession_frequency_future(value: float):
    set_precession_frequency.last_future = _set_precession_frequency.future(serializer.serialize({"value":value}, instrument_pb2.FloatMessage.DESCRIPTOR))
    return set_precession_frequency.last_future

set_precession_frequency.future=_set_precession_frequency_future

def get_aligned_precession_frequencies() -> list[float]:
    return _get_aligned_precession_frequencies(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_aligned_precession_frequencies_future() -> list[float]:
    get_aligned_precession_frequencies.last_future = _get_aligned_precession_frequencies.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_aligned_precession_frequencies.last_future

get_aligned_precession_frequencies.future=_get_aligned_precession_frequencies_future

def get_precession_angle() -> float:
    return _get_precession_angle(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_precession_angle_future() -> float:
    get_precession_angle.last_future = _get_precession_angle.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_precession_angle.last_future

get_precession_angle.future=_get_precession_angle_future

def set_precession_angle(value: float, skip_deprecession: bool = False) -> float:
    return _set_precession_angle(serializer.serialize({"value":value, "skip_deprecession":skip_deprecession}, instrument_pb2.SetPrecessionAngleMessage.DESCRIPTOR))


def _set_precession_angle_future(value: float, skip_deprecession: bool = False) -> float:
    set_precession_angle.last_future = _set_precession_angle.future(serializer.serialize({"value":value, "skip_deprecession":skip_deprecession}, instrument_pb2.SetPrecessionAngleMessage.DESCRIPTOR))
    return set_precession_angle.last_future

set_precession_angle.future=_set_precession_angle_future

def get_precession_angle_range() -> {"start": float, "end": float}:
    """Uses the set precession frequency with the corresponding precession matrices."""
    return _get_precession_angle_range(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_precession_angle_range_future() -> {"start": float, "end": float}:
    get_precession_angle_range.last_future = _get_precession_angle_range.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_precession_angle_range.last_future

get_precession_angle_range.future=_get_precession_angle_range_future

def get_precession_height() -> {"x": float, "y": float}:
    """depreceated use precession_height_correction"""
    return _get_precession_height(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_precession_height_future() -> {"x": float, "y": float}:
    get_precession_height.last_future = _get_precession_height.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_precession_height.last_future

get_precession_height.future=_get_precession_height_future

def set_precession_height(x: float, y: float) -> {"x": float, "y": float}:
    """depreceated use precession_height_correction"""
    return _set_precession_height(serializer.serialize({"x":x, "y":y}, instrument_pb2.PositionMessage.DESCRIPTOR))


def _set_precession_height_future(x: float, y: float) -> {"x": float, "y": float}:
    set_precession_height.last_future = _set_precession_height.future(serializer.serialize({"x":x, "y":y}, instrument_pb2.PositionMessage.DESCRIPTOR))
    return set_precession_height.last_future

set_precession_height.future=_set_precession_height_future

def get_precession_height_range() -> list[{"x": float, "y": float}]:
    """depreceated use precession_height_correction"""
    return _get_precession_height_range(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_precession_height_range_future() -> list[{"x": float, "y": float}]:
    get_precession_height_range.last_future = _get_precession_height_range.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_precession_height_range.last_future

get_precession_height_range.future=_get_precession_height_range_future

def get_precession_height_correction() -> list[float]:
    """2x2 flattened matrix in meters"""
    return _get_precession_height_correction(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_precession_height_correction_future() -> list[float]:
    get_precession_height_correction.last_future = _get_precession_height_correction.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_precession_height_correction.last_future

get_precession_height_correction.future=_get_precession_height_correction_future

def set_precession_height_correction(matrix: list[float]) -> list[float]:
    """use this values for tunning the precession on the sample 2x2 flattened matrix in meters"""
    return _set_precession_height_correction(serializer.serialize({"matrix":matrix}, instrument_pb2.SetPrecessionHeightCorrectionMessage.DESCRIPTOR))


def _set_precession_height_correction_future(matrix: list[float]) -> list[float]:
    set_precession_height_correction.last_future = _set_precession_height_correction.future(serializer.serialize({"matrix":matrix}, instrument_pb2.SetPrecessionHeightCorrectionMessage.DESCRIPTOR))
    return set_precession_height_correction.last_future

set_precession_height_correction.future=_set_precession_height_correction_future

def get_deprecession_height() -> {"x": float, "y": float}:
    """depreciated use deprecession_tilt_correction instead"""
    return _get_deprecession_height(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_deprecession_height_future() -> {"x": float, "y": float}:
    get_deprecession_height.last_future = _get_deprecession_height.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_deprecession_height.last_future

get_deprecession_height.future=_get_deprecession_height_future

def set_deprecession_height(x: float, y: float) -> {"x": float, "y": float}:
    """depreciated use deprecession_tilt_correction instead"""
    return _set_deprecession_height(serializer.serialize({"x":x, "y":y}, instrument_pb2.PositionMessage.DESCRIPTOR))


def _set_deprecession_height_future(x: float, y: float) -> {"x": float, "y": float}:
    set_deprecession_height.last_future = _set_deprecession_height.future(serializer.serialize({"x":x, "y":y}, instrument_pb2.PositionMessage.DESCRIPTOR))
    return set_deprecession_height.last_future

set_deprecession_height.future=_set_deprecession_height_future

def get_deprecession_height_range() -> list[{"x": float, "y": float}]:
    """depreciated use deprecession_tilt_correction instead"""
    return _get_deprecession_height_range(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_deprecession_height_range_future() -> list[{"x": float, "y": float}]:
    get_deprecession_height_range.last_future = _get_deprecession_height_range.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_deprecession_height_range.last_future

get_deprecession_height_range.future=_get_deprecession_height_range_future

def get_deprecession_tilt_correction() -> list[float]:
    """2x2 flattened matrix in rads"""
    return _get_deprecession_tilt_correction(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_deprecession_tilt_correction_future() -> list[float]:
    get_deprecession_tilt_correction.last_future = _get_deprecession_tilt_correction.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_deprecession_tilt_correction.last_future

get_deprecession_tilt_correction.future=_get_deprecession_tilt_correction_future

def set_deprecession_tilt_correction(matrix: list[float]) -> list[float]:
    """2x2 flattened matrix for tuning diffractions spot while precessing
        in radians!"""
    return _set_deprecession_tilt_correction(serializer.serialize({"matrix":matrix}, instrument_pb2.SetPrecessionHeightCorrectionMessage.DESCRIPTOR))


def _set_deprecession_tilt_correction_future(matrix: list[float]) -> list[float]:
    set_deprecession_tilt_correction.last_future = _set_deprecession_tilt_correction.future(serializer.serialize({"matrix":matrix}, instrument_pb2.SetPrecessionHeightCorrectionMessage.DESCRIPTOR))
    return set_deprecession_tilt_correction.last_future

set_deprecession_tilt_correction.future=_set_deprecession_tilt_correction_future

def start_rectangle_scan(number_of_frames: int, width: int, height: int, left: int, top: int,
                         pixel_time: float, scan_field_number_of_pixels: int,
                         enabled_detectors: list[DetectorType],
                         is_precession_enabled: bool, is_cross_elimination_enabled: bool,
                         camera_exposure: float, is_beam_blanked: bool = False) -> {"scan_id": str}:
    """camera_exposure: for low dose application you can set the camera exposure to its limit approx 1/4000,
        but still limit the dose by setting lower pixel_time. If left default zero the pixel_time will be used as camera_exposure"""
    return _start_rectangle_scan(serializer.serialize({"number_of_frames":number_of_frames, "width":width, "height":height, "left":left, "top":top, "pixel_time":pixel_time, "scan_field_number_of_pixels":scan_field_number_of_pixels, "enabled_detectors":enabled_detectors, "is_precession_enabled":is_precession_enabled, "is_cross_elimination_enabled":is_cross_elimination_enabled, "camera_exposure":camera_exposure, "is_beam_blanked":is_beam_blanked}, instrument_pb2.StartRectangleScanMessage.DESCRIPTOR))


def _start_rectangle_scan_future(number_of_frames: int, width: int, height: int, left: int, top: int,
                         pixel_time: float, scan_field_number_of_pixels: int,
                         enabled_detectors: list[DetectorType],
                         is_precession_enabled: bool, is_cross_elimination_enabled: bool,
                         camera_exposure: float, is_beam_blanked: bool = False) -> {"scan_id": str}:
    start_rectangle_scan.last_future = _start_rectangle_scan.future(serializer.serialize({"number_of_frames":number_of_frames, "width":width, "height":height, "left":left, "top":top, "pixel_time":pixel_time, "scan_field_number_of_pixels":scan_field_number_of_pixels, "enabled_detectors":enabled_detectors, "is_precession_enabled":is_precession_enabled, "is_cross_elimination_enabled":is_cross_elimination_enabled, "camera_exposure":camera_exposure, "is_beam_blanked":is_beam_blanked}, instrument_pb2.StartRectangleScanMessage.DESCRIPTOR))
    return start_rectangle_scan.last_future

start_rectangle_scan.future=_start_rectangle_scan_future

def start_tilt_series(number_of_frames: int, width: int, height: int, left: int, top: int,
                      pixel_time: float, scan_field_number_of_pixels: int,
                      enabled_detectors: list[DetectorType],
                      tilt_factors: list[{"x": float, "y": float}],
                      camera_exposure: float, is_beam_blanked: bool = False,
                      ) -> {"scan_id": str}:
    """tilt_factors are factors of the precession angle (-1 to 1), number of frames will be multipled by the lenght of this list"""
    return _start_tilt_series(serializer.serialize({"number_of_frames":number_of_frames, "width":width, "height":height, "left":left, "top":top, "pixel_time":pixel_time, "scan_field_number_of_pixels":scan_field_number_of_pixels, "enabled_detectors":enabled_detectors, "tilt_factors":tilt_factors, "camera_exposure":camera_exposure, "is_beam_blanked":is_beam_blanked}, instrument_pb2.StartTiltSeriesMessage.DESCRIPTOR))


def _start_tilt_series_future(number_of_frames: int, width: int, height: int, left: int, top: int,
                      pixel_time: float, scan_field_number_of_pixels: int,
                      enabled_detectors: list[DetectorType],
                      tilt_factors: list[{"x": float, "y": float}],
                      camera_exposure: float, is_beam_blanked: bool = False,
                      ) -> {"scan_id": str}:
    start_tilt_series.last_future = _start_tilt_series.future(serializer.serialize({"number_of_frames":number_of_frames, "width":width, "height":height, "left":left, "top":top, "pixel_time":pixel_time, "scan_field_number_of_pixels":scan_field_number_of_pixels, "enabled_detectors":enabled_detectors, "tilt_factors":tilt_factors, "camera_exposure":camera_exposure, "is_beam_blanked":is_beam_blanked}, instrument_pb2.StartTiltSeriesMessage.DESCRIPTOR))
    return start_tilt_series.last_future

start_tilt_series.future=_start_tilt_series_future

def stop_scanning():
    return _stop_scanning(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _stop_scanning_future():
    stop_scanning.last_future = _stop_scanning.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return stop_scanning.last_future

stop_scanning.future=_stop_scanning_future

def get_drift_rate() -> {"x": float, "y": float}:
    return _get_drift_rate(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_drift_rate_future() -> {"x": float, "y": float}:
    get_drift_rate.last_future = _get_drift_rate.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_drift_rate.last_future

get_drift_rate.future=_get_drift_rate_future

def set_drift_rate(x: float, y: float) -> {"x": float, "y": float}:
    """any scanning recipe will be adjusted for the drift rate correction, the scale factor will need to be set prior to the scan, otherwise the DSP outputs will overflow"""
    return _set_drift_rate(serializer.serialize({"x":x, "y":y}, instrument_pb2.PositionMessage.DESCRIPTOR))


def _set_drift_rate_future(x: float, y: float) -> {"x": float, "y": float}:
    set_drift_rate.last_future = _set_drift_rate.future(serializer.serialize({"x":x, "y":y}, instrument_pb2.PositionMessage.DESCRIPTOR))
    return set_drift_rate.last_future

set_drift_rate.future=_set_drift_rate_future

def get_scale_factor() -> float:
    return _get_scale_factor(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_scale_factor_future() -> float:
    get_scale_factor.last_future = _get_scale_factor.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_scale_factor.last_future

get_scale_factor.future=_get_scale_factor_future

def set_scale_factor(value: float) -> float:
    """use this factor to leave room for drift, or if you want to speed up the switchinf between off axis detector and camera acuiqsition"""
    return _set_scale_factor(serializer.serialize({"value":value}, instrument_pb2.FloatMessage.DESCRIPTOR))


def _set_scale_factor_future(value: float) -> float:
    set_scale_factor.last_future = _set_scale_factor.future(serializer.serialize({"value":value}, instrument_pb2.FloatMessage.DESCRIPTOR))
    return set_scale_factor.last_future

set_scale_factor.future=_set_scale_factor_future

def start_camera(exposure: float, fps: float, frame_count: int):
    return _start_camera(serializer.serialize({"exposure":exposure, "fps":fps, "frame_count":frame_count}, instrument_pb2.StartCameraMessage.DESCRIPTOR))


def _start_camera_future(exposure: float, fps: float, frame_count: int):
    start_camera.last_future = _start_camera.future(serializer.serialize({"exposure":exposure, "fps":fps, "frame_count":frame_count}, instrument_pb2.StartCameraMessage.DESCRIPTOR))
    return start_camera.last_future

start_camera.future=_start_camera_future

def stop_camera():
    return _stop_camera(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _stop_camera_future():
    stop_camera.last_future = _stop_camera.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return stop_camera.last_future

stop_camera.future=_stop_camera_future

def set_camera_roi(roi_mode: RoiMode, use16bit: bool = False) -> {"roi_mode": RoiMode, "use16bit": bool}:
    """set roi central rectangle (width will be always 512) for faster acquisition"""
    return _set_camera_roi(serializer.serialize({"roi_mode":roi_mode, "use16bit":use16bit}, instrument_pb2.SetCameraRoiMessage.DESCRIPTOR))


def _set_camera_roi_future(roi_mode: RoiMode, use16bit: bool = False) -> {"roi_mode": RoiMode, "use16bit": bool}:
    set_camera_roi.last_future = _set_camera_roi.future(serializer.serialize({"roi_mode":roi_mode, "use16bit":use16bit}, instrument_pb2.SetCameraRoiMessage.DESCRIPTOR))
    return set_camera_roi.last_future

set_camera_roi.future=_set_camera_roi_future

def get_camera_roi() -> {"roi_mode": RoiMode, "use16bit": bool}:
    return _get_camera_roi(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_camera_roi_future() -> {"roi_mode": RoiMode, "use16bit": bool}:
    get_camera_roi.last_future = _get_camera_roi.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_camera_roi.last_future

get_camera_roi.future=_get_camera_roi_future

def get_camera_max_fps() -> float:
    return _get_camera_max_fps(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_camera_max_fps_future() -> float:
    get_camera_max_fps.last_future = _get_camera_max_fps.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_camera_max_fps.last_future

get_camera_max_fps.future=_get_camera_max_fps_future

def camera_is_acquiring() -> bool:
    return _camera_is_acquiring(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _camera_is_acquiring_future() -> bool:
    camera_is_acquiring.last_future = _camera_is_acquiring.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return camera_is_acquiring.last_future

camera_is_acquiring.future=_camera_is_acquiring_future

def get_camera_bith_depth() -> int:
    return _get_camera_bith_depth(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_camera_bith_depth_future() -> int:
    get_camera_bith_depth.last_future = _get_camera_bith_depth.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_camera_bith_depth.last_future

get_camera_bith_depth.future=_get_camera_bith_depth_future

def get_distortion_coefficients() -> {"scan": correction_structure,
                                      "descan": correction_structure,
                                      "precession": correction_structure,
                                      "deprecession": correction_structure,
                                      }:
    return _get_distortion_coefficients(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_distortion_coefficients_future() -> {"scan": correction_structure,
                                      "descan": correction_structure,
                                      "precession": correction_structure,
                                      "deprecession": correction_structure,
                                      }:
    get_distortion_coefficients.last_future = _get_distortion_coefficients.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_distortion_coefficients.last_future

get_distortion_coefficients.future=_get_distortion_coefficients_future

def set_distortion_coefficients(scan: correction_structure,
                                descan: correction_structure,
                                precession: correction_structure,
                                deprecession: correction_structure
                                ) -> {"scan": correction_structure,
                                      "descan": correction_structure,
                                      "precession": correction_structure,
                                      "deprecession": correction_structure,
                                      }:
    return _set_distortion_coefficients(serializer.serialize({"scan":scan, "descan":descan, "precession":precession, "deprecession":deprecession}, instrument_pb2.GetDistortionCoefficientsMessage.DESCRIPTOR))


def _set_distortion_coefficients_future(scan: correction_structure,
                                descan: correction_structure,
                                precession: correction_structure,
                                deprecession: correction_structure
                                ) -> {"scan": correction_structure,
                                      "descan": correction_structure,
                                      "precession": correction_structure,
                                      "deprecession": correction_structure,
                                      }:
    set_distortion_coefficients.last_future = _set_distortion_coefficients.future(serializer.serialize({"scan":scan, "descan":descan, "precession":precession, "deprecession":deprecession}, instrument_pb2.GetDistortionCoefficientsMessage.DESCRIPTOR))
    return set_distortion_coefficients.last_future

set_distortion_coefficients.future=_set_distortion_coefficients_future

def set_blanker_mode(value: BeamBlankType) -> BeamBlankType:
    return _set_blanker_mode(serializer.serialize({"value":value}, instrument_pb2.SetBlankerModeMessage.DESCRIPTOR))


def _set_blanker_mode_future(value: BeamBlankType) -> BeamBlankType:
    set_blanker_mode.last_future = _set_blanker_mode.future(serializer.serialize({"value":value}, instrument_pb2.SetBlankerModeMessage.DESCRIPTOR))
    return set_blanker_mode.last_future

set_blanker_mode.future=_set_blanker_mode_future

def get_blanker_mode() -> BeamBlankType:
    return _get_blanker_mode(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_blanker_mode_future() -> BeamBlankType:
    get_blanker_mode.last_future = _get_blanker_mode.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_blanker_mode.last_future

get_blanker_mode.future=_get_blanker_mode_future

def set_image_compression(value: Compression) -> Compression:
    return _set_image_compression(serializer.serialize({"value":value}, instrument_pb2.SetImageCompressionMessage.DESCRIPTOR))


def _set_image_compression_future(value: Compression) -> Compression:
    set_image_compression.last_future = _set_image_compression.future(serializer.serialize({"value":value}, instrument_pb2.SetImageCompressionMessage.DESCRIPTOR))
    return set_image_compression.last_future

set_image_compression.future=_set_image_compression_future

def get_image_compression() -> Compression:
    return _get_image_compression(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_image_compression_future() -> Compression:
    get_image_compression.last_future = _get_image_compression.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_image_compression.last_future

get_image_compression.future=_get_image_compression_future

def get_dead_pixels() -> list[{"x": int64, "y": int64}]:
    return _get_dead_pixels(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))


def _get_dead_pixels_future() -> list[{"x": int64, "y": int64}]:
    get_dead_pixels.last_future = _get_dead_pixels.future(serializer.serialize({}, instrument_pb2.EmptyMessage.DESCRIPTOR))
    return get_dead_pixels.last_future

get_dead_pixels.future=_get_dead_pixels_future
