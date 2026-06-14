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
IDX_MAX_DH               = 13  # Hamiltonian drift (The Physics Validator)
IDX_IMPACT_PARAM         = 14
IDX_CARTER_CONST         = 15
IDX_MIN_POLE_GAP         = 16
IDX_THETA_TURNS          = 17
IDX_STEPS_IN_ERGO        = 18
IDX_ENTERED_ERGO         = 19
IDX_MIN_DELTA            = 20
IDX_HIT_R_1              = 21  # Storing the first hit radius for mapping
IDX_HIT_R_2              = 22
IDX_HIT_R_3              = 23
NUM_DOCTOR_METRICS       = 24