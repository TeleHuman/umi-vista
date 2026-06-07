# -*- coding: utf-8 -*-
import math
import numpy as np

# Standard convention in this module: quaternions are [w,x,y,z], and pose7 is [x,y,z,qw,qx,qy,qz].
def hat(v):
    x,y,z = v
    return np.array([[0,-z, y],[z, 0,-x],[-y,x,0]], dtype=float)

def rvec_to_R(r):
    r = np.asarray(r, dtype=float)
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3) + hat(r)
    n = r/th
    K = hat(n)
    return np.eye(3) + math.sin(th)*K + (1-math.cos(th))*(K@K)

def R_to_rvec(R):
    R = np.asarray(R, dtype=float)
    c = (np.trace(R)-1.0)/2.0
    c = max(-1.0, min(1.0, c))
    th = math.acos(c)
    if th < 1e-12:
        return np.array([(R[2,1]-R[1,2])/2.0, (R[0,2]-R[2,0])/2.0, (R[1,0]-R[0,1])/2.0])
    return (th/(2*math.sin(th))) * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])

def quat_to_R(qw,qx,qy,qz):
    n = math.sqrt(qx*qx+qy*qy+qz*qz+qw*qw)
    if n == 0:
        return np.eye(3)
    qx,qy,qz, qw = qx/n, qy/n, qz/n, qw/n
    xx,yy,zz = qx*qx, qy*qy, qz*qz
    xy,xz,yz = qx*qy, qx*qz, qy*qz
    wx,wy,wz = qw*qx, qw*qy, qw*qz
    return np.array([
        [1-2*(yy+zz),   2*(xy-wz),    2*(xz+wy)],
        [  2*(xy+wz), 1-2*(xx+zz),    2*(yz-wx)],
        [  2*(xz-wy),   2*(yz+wx),  1-2*(xx+yy)]
    ], dtype=float)

def rpy_to_R(roll, pitch, yaw):
    cr, sr = math.cos(roll),  math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]], dtype=float)
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]], dtype=float)
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]], dtype=float)
    return Rz @ Ry @ Rx

def R_to_rpy(R):
    R = np.asarray(R, dtype=float)
    sy = -R[2,0]
    sy = max(-1.0, min(1.0, sy))
    ry = math.asin(sy)
    cy = math.cos(ry)
    if abs(cy) < 1e-8:
        rx = 0.0
        rz = math.atan2(-R[0,1], R[0,2])
    else:
        rx = math.atan2(R[2,1], R[2,2]);   rz = math.atan2(R[1,0], R[0,0])
    return [rx, ry, rz]

def T_to_pose6(T):
    R = T[:3,:3]; t = T[:3,3]
    rx,ry,rz = R_to_rpy(R)
    return [float(t[0]), float(t[1]), float(t[2]), float(rx), float(ry), float(rz)]

def T_to_rvec(T):
    R = T[:3,:3]; t = T[:3,3]
    rx,ry,rz = R_to_rvec(R)
    return [float(t[0]), float(t[1]), float(t[2]), float(rx), float(ry), float(rz)]

def T_to_pose7(T):
    R = T[:3,:3]; t = T[:3,3]
    r = R_to_rvec(R)
    th = np.linalg.norm(r)
    if th < 1e-12:
        qw = 1.0; qx = 0.0; qy = 0.0; qz = 0.0
    else:
        n = r/th
        qw = math.cos(th/2.0)
        s  = math.sin(th/2.0)
        qx = n[0]*s; qy = n[1]*s; qz = n[2]*s
    return [float(t[0]), float(t[1]), float(t[2]), float(qw), float(qx), float(qy), float(qz)]

def pose7_to_T(pose7):
    x,y,z,qw,qx,qy,qz = pose7
    T = np.eye(4)
    T[:3,:3] = quat_to_R(qw,qx,qy,qz)
    T[:3, 3] = [x,y,z]
    return T

def rvec_to_T(pose6):
    x,y,z, rx,ry,rz = pose6
    T = np.eye(4)
    T[:3,:3] = rvec_to_R([rx,ry,rz])
    T[:3, 3] = [x,y,z]
    return T

def inv_T(T):
    R = T[:3,:3]; p = T[:3,3]
    Ti = np.eye(4)
    Ti[:3,:3] = R.T
    Ti[:3, 3] = - R.T @ p
    return Ti

def pose7_to_rvec(pose7):
    x,y,z,qw,qx,qy,qz = pose7
    R = quat_to_R(qw,qx,qy,qz)
    r = R_to_rvec(R)
    return [float(x), float(y), float(z), float(r[0]), float(r[1]), float(r[2])]

def rvec_to_pose7(pose6):
    x,y,z = pose6[:3]
    r = pose6[3:] #rvec
    th = np.linalg.norm(r)
    if th < 1e-12:
        qw = 1.0; qx = 0.0; qy = 0.0; qz = 0.0
    else:
        n = r/th
        qw = math.cos(th/2.0)
        s  = math.sin(th/2.0)
        qx = n[0]*s; qy = n[1]*s; qz = n[2]*s
    return [float(x), float(y), float(z), float(qw), float(qx), float(qy), float(qz)]

def quat_multiply(q1, q2):
    """
    Quaternion multiplication: q1 * q2 (q2 is base, q1 is the applied rotation).
    Quaternion format: [w, x, y, z].
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return normalize(np.array([w, x, y, z]))

def inv_quat(q):
    """
    Quaternion inverse: q^-1.
    Quaternion format: [w, x, y, z].
    """
    w, x, y, z = q
    norm_sq = w*w + x*x + y*y + z*z
    if norm_sq == 0:
        return np.array([0.0, 0.0, 0.0, 0.0])
    return np.array([w/norm_sq, -x/norm_sq, -y/norm_sq, -z/norm_sq])

def normalize(q):
    """Normalize a quaternion."""
    return q / np.linalg.norm(q)

def quat_to_rpy(q):
    """
    Convert quaternion [w,x,y,z] to ZYX Euler angles (yaw, pitch, roll), in radians.
    Returned order is: roll (x axis), pitch (y axis), yaw (z axis).
    """
    if q is None:
        return 0.0, 0.0, 0.0
    w, x, y, z = [float(v) for v in q]
    # Build rotation matrix from quaternion
    m00 = 1 - 2*(y*y + z*z)
    m10 = 2*(x*y + z*w)
    m20 = 2*(x*z - y*w)
    m21 = 2*(y*z + x*w)
    m22 = 1 - 2*(x*x + y*y)
    # ZYX Euler
    yaw = math.atan2(m10, m00)
    pitch = math.asin(max(-1.0, min(1.0, -m20)))
    roll = math.atan2(m21, m22)
    return roll, pitch, yaw

def rpy_to_quat(roll, pitch, yaw):
    """
    Convert ZYX Euler angles (yaw, pitch, roll) to quaternion [w,x,y,z], in radians.
    Input order: roll (x axis), pitch (y axis), yaw (z axis).
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return normalize(np.array([w, x, y, z]))

def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between unit quaternions q0 and q1.
    Ensures shortest path by flipping q1 if needed. t in [0,1].
    """
    q0 = normalize(q0)
    q1 = normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    DOT_THRESH = 0.9995
    if dot > DOT_THRESH:
        # Nearly linear, fall back to lerp then normalize
        res = q0 + t * (q1 - q0)
        return normalize(res)
    theta0 = math.acos(max(min(dot, 1.0), -1.0))
    theta = theta0 * t
    sin_theta0 = math.sin(theta0)
    sin_theta = math.sin(theta)
    s0 = math.cos(theta) - dot * sin_theta / (sin_theta0 if sin_theta0 != 0.0 else 1.0)
    s1 = sin_theta / (sin_theta0 if sin_theta0 != 0.0 else 1.0)
    return s0 * q0 + s1 * q1

def quat_flip(q_ref: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Flip quaternion q to be closest to reference quaternion q_ref.
    """
    q_ref=normalize(q_ref)
    q=normalize(q)
    if np.dot(q_ref, q) < 0.0:
        # print("[WARN] quat_flip: flipping quaternion to ensure continuity: qref=", q_ref, " q=", q)
        return -q
    return q

def pose7_flip(pose7_ref: np.ndarray, pose7: np.ndarray) -> np.ndarray:
    """
    Flip the quaternion part of pose7 to be closest to reference pose7.
    """
    q_ref = pose7_ref[3:]
    q = pose7[3:]
    q_flipped = quat_flip(q_ref, q)
    return np.concatenate([pose7[:3], q_flipped])

def pose7_to_axisAngle(pose7):
    """Convert pose7 [x,y,z,qw,qx,qy,qz] to axis-angle [ax,ay,az,angle]."""
    x,y,z,qw,qx,qy,qz = pose7
    th = 2.0 * math.acos(max(-1.0, min(1.0, qw)))
    s = math.sqrt(1 - qw*qw)
    if s < 1e-12:
        return np.array([x, y, z, 1.0, 0.0, 0.0, 0.0])
    ax = qx / s
    ay = qy / s
    az = qz / s
    return np.array([x,y,z,ax, ay, az, th])

def axisAngle_to_pose7(axisAngle):
    """Convert axis-angle [ax,ay,az,angle] to pose7 [x,y,z,qw,qx,qy,qz]."""
    x,y,z,ax,ay,az,angle = axisAngle
    s = math.sin(angle / 2.0)
    qw = math.cos(angle / 2.0)
    qx = ax * s
    qy = ay * s
    qz = az * s
    return np.array([x, y, z, qw, qx, qy, qz])

def axisAngle_to_R(axisAngle):
    """Convert axis-angle [ax,ay,az,angle] to rotation matrix R."""
    ax, ay, az, angle = axisAngle
    s = math.sin(angle)
    c = math.cos(angle)
    t = 1 - c
    R = np.array([
        [t*ax*ax + c,     t*ax*ay - s*az, t*ax*az + s*ay],
        [t*ax*ay + s*az,  t*ay*ay + c,    t*ay*az - s*ax],
        [t*ax*az - s*ay,  t*ay*az + s*ax, t*az*az + c   ]
    ], dtype=float)
    return R

def R_to_axisAngle(R):
    """Convert rotation matrix R to axis-angle [ax,ay,az,angle]."""
    R = np.asarray(R, dtype=float)
    c = (np.trace(R)-1.0)/2.0
    c = max(-1.0, min(1.0, c))
    angle = math.acos(c)
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    s = math.sin(angle)
    ax = (R[2,1]-R[1,2]) / (2.0*s)
    ay = (R[0,2]-R[2,0]) / (2.0*s)
    az = (R[1,0]-R[0,1]) / (2.0*s)
    return np.array([ax, ay, az, angle])

def get_nearest_axisAngle(last_axisAngle, cur_axisAngle):
    """Return the axis-angle equivalent of `cur_axisAngle` that is numerically
    closest to `last_axisAngle` to improve temporal continuity for filtering.

    Axis-angle equivalences used:
      (a, θ) ≡ (a, θ + 2πk) and (a, θ) ≡ (-a, -θ).

    We consider two candidates (with angle wrapped near the last angle):
      1) ( a,  wrap_to_near( θ, last_θ))
      2) (-a, wrap_to_near(-θ, last_θ))
    and pick the one closer to the last sample using a simple metric.
    """
    last_axisAngle = np.asarray(last_axisAngle, dtype=float)
    cur_axisAngle  = np.asarray(cur_axisAngle, dtype=float)

    last_axis = np.asarray(last_axisAngle[:3], dtype=float)
    last_ang  = float(last_axisAngle[3])
    axis      = np.asarray(cur_axisAngle[:3], dtype=float)
    ang       = float(cur_axisAngle[3])

    # Normalize current axis (if valid)
    n = float(np.linalg.norm(axis))
    if n > 1e-12:
        axis = axis / n

    def wrap_to_near(a: float, ref: float) -> float:
        two_pi = 2.0 * math.pi
        d = a - ref
        d = (d + math.pi) % (two_pi) - math.pi
        return ref + d

    # Build two equivalent candidates and wrap their angles near last_ang
    ang1  = wrap_to_near(ang, last_ang)
    axis1 = axis.copy()
    ang2  = wrap_to_near(-ang, last_ang)
    axis2 = -axis

    # If angle is near zero, axis is not meaningful → keep last axis to reduce jitter
    EPS_ANG = 1e-9
    if abs(ang1) < EPS_ANG and np.linalg.norm(last_axis) > 1e-12:
        axis1 = last_axis.copy()
    if abs(ang2) < EPS_ANG and np.linalg.norm(last_axis) > 1e-12:
        axis2 = last_axis.copy()

    # Scoring: prioritize angle closeness, then axis similarity
    def score(axv: np.ndarray, angv: float) -> float:
        da  = abs(angv - last_ang)
        dax = float(np.linalg.norm(axv - last_axis))
        return da + 0.5 * dax

    s1 = score(axis1, ang1)
    s2 = score(axis2, ang2)
    if s2 < s1:
        return np.array([axis2[0], axis2[1], axis2[2], ang2], dtype=float)
    else:
        return np.array([axis1[0], axis1[1], axis1[2], ang1], dtype=float)

def quat_xyzw_to_wxyz(q):
    """Convert quaternion from [x,y,z,w] to [w,x,y,z] format."""
    x, y, z, w = q
    return np.array([w, x, y, z])

def pose7_xyzw_to_wxyz(pose7):
    """Convert pose7 [x,y,z,qx,qy,qz,qw] to [x,y,z,qw,qx,qy,qz]."""
    x, y, z, qx, qy, qz, qw = pose7
    return np.array([x, y, z, qw, qx, qy, qz])

def quat_wxyz_to_xyzw(q):
    """Convert quaternion from [w,x,y,z] to [x,y,z,w] format."""
    w, x, y, z = q
    return np.array([x, y, z, w])

def pose7_wxyz_to_xyzw(pose7):
    """Convert pose7 [x,y,z,qw,qx,qy,qz] to [x,y,z,qx,qy,qz,qw]."""
    x, y, z, qw, qx, qy, qz = pose7
    return np.array([x, y, z, qx, qy, qz, qw])