import numpy as np

allowed_fovs = np.array([])  # um - modified by acquire reference

use_rectangle_selection = False
rectangles = None
total_pixels = None
fovs_total = None
centers = None

requested_fov = 5  # um  - modified by start_tracking

minimum_fov_for_moving = 1  # um

reference_N = 512
tracking_N = 256

tilt_for_z = np.array([20, 0])  # if None z measurement will be skipped

pixel_time = 100  # ns

model = 'TEMRegistrationMedium'
tem_registration_batch = True

# expected_std_image = 0.02  # to calculate std along x/y direction
# correlation_shift_factor = 0.05

fov_factor = np.array([4, 4, 2])

# regulation:
target_xy_offset = np.zeros(2)
target_z_offset = 0

enable_regulation = False

keypoints_matching = "median"  # translation/median/mean
offset_filtering = False
filtering_samples = 10
linear_regression_z = True

P = 0
PI = np.array([0.1, 0.1, 0.1])
PD = 0

P_moving = 0
PI_moving = np.array([0.5, 0.5, 0.1])
PD_moving = 0

# alpha measuring:

stage_shifting_timeout = 0.05  # will auto retrigger the stage motion and input chain

electronic_shift_limit = 1  # um
shift_mode = "combined"

alpha_speed = 10  # deg per second for measurement
alpha_dt = 0.05
alpha_acceleration = 10  # deg *s-2

allowed_error = np.array([0.1, 0.1, 0.2])  # um dynamically set at the beginning of tomography measurement
# allowed_error = np.array([0.05, 0.05, 0.1])  # um  dynamically set at the beginning of tomography measurement
stabilization_time = 0.5
stabilization_alpha_skip = None  # deg or None dynamically set at the beginning of tomography measurement
