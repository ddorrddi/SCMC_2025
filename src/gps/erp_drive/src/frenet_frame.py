# frenet_utils.py
import numpy as np
from scipy.interpolate import interp1d

def compute_s_and_yaw(ref_path):
    s_list = [0.0]
    yaw_list = []
    for i in range(1, len(ref_path)):
        x0 = ref_path[i-1].pose.position.x
        y0 = ref_path[i-1].pose.position.y
        x1 = ref_path[i].pose.position.x
        y1 = ref_path[i].pose.position.y

        dx = x1 - x0
        dy = y1 - y0
        s_list.append(s_list[-1] + np.hypot(dx, dy))
        yaw_list.append(np.arctan2(dy, dx))
    yaw_list.append(yaw_list[-1])  # 마지막 yaw 복사
    return np.array(s_list), np.array(yaw_list)

def cartesian_to_frenet(x, y, ref_path, s_list):
    min_dist = float('inf')
    min_idx = 0
    for i, pt in enumerate(ref_path):
        dx = x - pt.pose.position.x
        dy = y - pt.pose.position.y
        dist = dx**2 + dy**2
        if dist < min_dist:
            min_dist = dist
            min_idx = i

    ref_x = ref_path[min_idx].pose.position.x
    ref_y = ref_path[min_idx].pose.position.y
    ref_yaw = np.arctan2(
        ref_path[min_idx+1].pose.position.y - ref_path[min_idx].pose.position.y,
        ref_path[min_idx+1].pose.position.x - ref_path[min_idx].pose.position.x
    )

    dx = x - ref_x
    dy = y - ref_y

    s = s_list[min_idx]
    l = dx * -np.sin(ref_yaw) + dy * np.cos(ref_yaw)
    return s, l

def frenet_to_cartesian(s, l, ref_path, s_list, yaw_list):
    fx = interp1d(s_list, [p.pose.position.x for p in ref_path], fill_value="extrapolate")
    fy = interp1d(s_list, [p.pose.position.y for p in ref_path], fill_value="extrapolate")
    fyaw = interp1d(s_list, yaw_list, fill_value="extrapolate")

    x_ref = fx(s)
    y_ref = fy(s)
    yaw = fyaw(s)

    x = x_ref - l * np.sin(yaw)
    y = y_ref + l * np.cos(yaw)
    return x, y, yaw

def generate_quintic_path(s0, l0, s1, l1):
    A = np.array([
        [1, s0, s0**2,   s0**3,     s0**4,      s0**5],
        [0, 1,  2*s0,    3*s0**2,   4*s0**3,    5*s0**4],
        [0, 0,  2,       6*s0,     12*s0**2,   20*s0**3],
        [1, s1, s1**2,   s1**3,     s1**4,      s1**5],
        [0, 1,  2*s1,    3*s1**2,   4*s1**3,    5*s1**4],
        [0, 0,  2,       6*s1,     12*s1**2,   20*s1**3],
    ])
    b = np.array([l0, 0, 0, l1, 0, 0])
    coeffs = np.linalg.solve(A, b)
    return lambda s: np.polyval(coeffs[::-1], s)
# def generate_quintic_path(s0, l0, s1, l1, v0=0.0, v1=0.0, a0=0.0, a1=0.0):
#     A = np.array([
#         [1, s0, s0**2,   s0**3,     s0**4,      s0**5],
#         [0, 1,  2*s0,    3*s0**2,   4*s0**3,    5*s0**4],
#         [0, 0,  2,       6*s0,     12*s0**2,   20*s0**3],
#         [1, s1, s1**2,   s1**3,     s1**4,      s1**5],
#         [0, 1,  2*s1,    3*s1**2,   4*s1**3,    5*s1**4],
#         [0, 0,  2,       6*s1,     12*s1**2,   20*s1**3],
#     ])
#     b = np.array([l0, v0, a0, l1, v1, a1])
#     coeffs = np.linalg.solve(A, b)
#     return lambda s: np.polyval(coeffs[::-1], s)