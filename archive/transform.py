import numpy as np

# Constants
h_camera = 340# 185  # mm
alpha_camera = 90 + 20  # degrees

# Translation vector T(C)->(B)
T_CB = np.array([0, 0, -h_camera])

def alpha_to_radians(alpha_degrees):
    return alpha_degrees * (np.pi / 180)

def get_BR_C(alpha_camera):
    alpha_rad = alpha_to_radians(alpha_camera)
    sin_alpha = np.sin(-0.5 * np.pi + alpha_rad)
    cos_alpha = np.cos(-0.5 * np.pi + alpha_rad)
    
    # Transpose of the product of the matrices as calculated above
    BR_C = np.array([
        [0, sin_alpha, cos_alpha],
        [-1, 0, 0],
        [0, cos_alpha, -sin_alpha]
    ])
    return BR_C

# Calculate (B)R(C)
BR_C = get_BR_C(alpha_camera)

def transform_to_body_frame(px_c, py_c, pz_c, BR_C = BR_C, T_CB = T_CB):
    p_c = np.array([px_c, py_c, pz_c])
    p_b = BR_C @ p_c - T_CB
    x_w, y_w, z_w = p_b
    return x_w, (-1)*y_w, z_w

# Sample camera frame coordinates (you can replace these with any values)
# px_c, py_c, pz_c = 1.0, 2.0, 3.0

# # Perform transformation
# p_b = transform_to_body_frame(px_c, py_c, pz_c, BR_C, T_CB)

# p_b