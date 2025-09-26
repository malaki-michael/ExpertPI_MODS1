from enum import Enum


class StatesHolder:
    class InteractionMode(Enum):
        survey = 0
        stem_4d = 1
        navigation = 2
        measurements = 3

    class _Acquisition:
        def __init__(self):
            self.max_fov = None
            self.automation_thread = None

    def __init__(self) -> None:
        self.acquisition = self._Acquisition()

        self.image_view_channel = "BF"
        self.diffraction_view_channel = "camera"

        self.live_fft_enabled = False

        self.interaction_mode = StatesHolder.InteractionMode.survey

        self.optical_mode = "Unknown"
        self.adjustments_per_mode = {
            "ParallelBeam": {"illumination": [1, 0.5]},  # nA, um
            "ConvergentBeam": {"illumination": [1, 10]},  # nA, mrad
        }

        self.measurement_type = None

        self.energy_value = 100_000
        self.last_microscope_state = {}
