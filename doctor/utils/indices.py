# doctor/utils/indices.py
"""
Maps the 24-element float64 array returned by integrate_path_doctor 
to human-readable indices for the diagnostic tensor.
"""

IDX_CAPTURED             = 0
IDX_TERM_REASON          = 1
IDX_STEPS                = 2
IDX_MIN_R                = 3
IDX_MAX_R                = 4
IDX_FINAL_R              = 5
IDX_FINAL_THETA          = 6
IDX_ORBIT_COUNT          = 7
IDX_EQ_CROSSINGS         = 8
IDX_HIT_COUNT            = 9
IDX_E                    = 10
IDX_L                    = 11
IDX_Q                    = 12
IDX_MAX_DH               = 13
IDX_MAX_DE               = 14
IDX_MAX_DL               = 15
IDX_MAX_DQ               = 16
IDX_IMPACT_PARAM         = 17
IDX_CARTER_CONST         = 18
IDX_MIN_POLE_GAP         = 19
IDX_THETA_TURNS          = 20
IDX_STEPS_IN_ERGO        = 21
IDX_ENTERED_ERGO         = 22
IDX_MIN_DELTA            = 23  # The metric that caught the artifact!

NUM_DOCTOR_METRICS       = 24