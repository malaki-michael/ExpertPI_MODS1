# GENERATED FILE from pySTEM 1.16.1,build 4717,rev b31140633df20050ad44d1030889522d4f5f8256
from enum import Enum


class IlluminationMode(Enum):
    Convergent = 0
    Parallel = 1


class CondenserFocusType(Enum):
    C3 = 0
    FixedAngle = 1


class DeflectorType(Enum):
    Scan = 0
    Precession = 1
    PrecessionDynamic = 2


class MicroscopeState(Enum):
    Standby = 0
    Ready = 1
    Undefined = 2
    Error = 3
    Acquiring = 4
    SampleLoading = 5


class OpticalMode(Enum):
    Off = 0
    ConvergentBeam = 1
    LowMagnification = 2
    ParallelBeam = 3


class DetectorType(Enum):
    BF = 0
    HAADF = 1
    Camera = 2
    EDX0 = 3
    EDX1 = 4


class ScanSystemType(Enum):
    SCAN = 0
    PRECESSION = 1


class RoiMode(Enum):
    Disabled = 0
    Lines_256 = 1
    Lines_128 = 2


class BeamBlankType(Enum):
    BeamOff = 0
    BeamOn = 1
    BeamAcq = 2


class Compression(Enum):
    NoCompression = 0
    Bslz4 = 1


class HolderType(Enum):
    Unknown = 0
    SingleTilt = 1
    DoubleTilt = 2
    Needle = 3


class XrayFilterType(Enum):
    CountRate = 0
    EnergyResolution = 1
    Mixed = 2

class uint64:
    pass

class int64:
    pass

correction_structure = {"orig_rot": float,
                        "coeffs": list[{
                            "n": int64,
                            "k": int64,
                            "real": float,
                            "imag": float}]
                        }

