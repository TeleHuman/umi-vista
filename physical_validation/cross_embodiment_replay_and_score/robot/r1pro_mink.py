from pathlib import Path

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter
import math
import time
import threading
import numpy as np
import utils.self_math as smath
import utils.lowpass_filter as flt

import mink
from robot.mink_limits import FixedJointLimit

_HERE = Path(__file__).parent.parent
_XML = _HERE / "model" / "r1pro" / "ground_scene.xml"
_MAPPING = [{'mj_name': 'torso_joint1', 'robot_index': 0, 'scale': 1.0, 'offset': 0.0, 'sign': 1.0},
            {'mj_name': 'torso_joint2', 'robot_index': 1, 'scale': 1.0, 'offset': 0.0, 'sign': 1.0},
            {'mj_name': 'torso_joint3', 'robot_index': 2, 'scale': 1.0, 'offset': 0.0, 'sign': 1.0},
            {'mj_name': 'torso_joint4', 'robot_index': 3, 'scale': 1.0, 'offset': 0.0, 'sign': 1.0},
            {'mj_name': 'left_arm_joint1', 'robot_index': 4, 'scale': 1.0, 'offset': 0.0, 'sign': 1.0},
            {'mj_name': 'left_arm_joint2', 'robot_index': 5, 'scale': 1.0, 'offset': 0.0, 'sign': 1.0},
            {'mj_name': 'left_arm_joint3', 'robot_index': 6, 'scale': 1.0, 'offset': 0.0, 'sign': 1.0},
            {'mj_name': 'left_arm_joint4', 'robot_index': 7, 'scale': 1.0, 'offset': 0.0, 'sign': 1.0},
            {'mj_name': 'left_arm_joint5', 'robot_index': 8, 'scale': 1.0, 'offset': 0.0, 'sign': 1.0},
            {'mj_name': 'left_arm_joint6', 'robot_index': 9, 'scale': 1.0, 'offset': 0.0, 'sign': 1.0},
            {'mj_name': 'left_arm_joint7', 'robot_index': 10, 'scale': 1.0, 'offset': 0.0, 'sign': 1.0},
            {'mj_name': 'right_arm_joint1', 'robot_index': 11, 'scale': 1.0, 'offset': 0.0, 'sign': 1.0},
            {'mj_name': 'right_arm_joint2', 'robot_index': 12, 'scale': 1.0, 'offset': 0.0, 'sign': 1.0},
            {'mj_name': 'right_arm_joint3', 'robot_index': 13, 'scale': 1.0, 'offset': 0.0, 'sign': 1.0},
            {'mj_name': 'right_arm_joint4', 'robot_index': 14, 'scale': 1.0, 'offset': 0.0, 'sign': 1.0},
            {'mj_name': 'right_arm_joint5', 'robot_index': 15, 'scale': 1.0, 'offset': 0.0, 'sign': 1.0},
            {'mj_name': 'right_arm_joint6', 'robot_index': 16, 'scale': 1.0, 'offset': 0.0, 'sign': 1.0},
            {'mj_name': 'right_arm_joint7', 'robot_index': 17, 'scale': 1.0, 'offset': 0.0, 'sign': 1.0},
            {'mj_name': 'left_gripper_finger_joint1', 'robot_index': 18, 'scale': 1.0, 'offset': 0.0, 'sign': 1.0},
            {'mj_name': 'left_gripper_finger_joint2', 'robot_index': 18, 'scale': 1.0, 'offset': 0.0, 'sign': 1.0},
            {'mj_name': 'right_gripper_finger_joint1', 'robot_index': 19, 'scale': 1.0, 'offset': 0.0, 'sign': 1.0},
            {'mj_name': 'right_gripper_finger_joint2', 'robot_index': 19, 'scale': 1.0, 'offset': 0.0, 'sign': 1.0}]
# Fixed joint set: defined by the first N entries of _MAPPING.
_LOCKED_JOINTS_MAPPING_HEAD = [m["mj_name"] for m in _MAPPING[:4]]

class R1ProMink:
    def __init__(self):
        model = mujoco.MjModel.from_xml_path(_XML.as_posix())

        self.configuration = mink.Configuration(model)
        self.hands = ["left_gripper_site", "right_gripper_site"]
        # Initialize tasks (avoid walrus assignment to attributes inside list).
        self.torso_task = mink.FrameTask(
            frame_name="torso_site",
            frame_type="site",
            position_cost=np.array([0.0, 0.0, 0.0], dtype=float),
            orientation_cost=np.array([0.1, 0.1, 0.1], dtype=float),
            lm_damping=1.0,
        )
        self.posture_task = mink.PostureTask(model, cost=1e-1)
        self.com_task = mink.ComTask(cost=10.0)
        self.tasks = [
            self.torso_task,
            self.posture_task,
            self.com_task,
        ]
        self.hand_tasks = []
        for hand in self.hands:
            task = mink.FrameTask(
                frame_name=hand,
                frame_type="site",
                position_cost=5.0,
                orientation_cost=1.0,
                lm_damping=1.0,
            )
            self.hand_tasks.append(task)
        self.tasks.extend(self.hand_tasks)
        collision_pairs = [(["left_hand_collision", "right_hand_collision","left_elbow_collision", "right_elbow_collision"], ["torso_collision"])]
        collision_avoidance_limit = mink.CollisionAvoidanceLimit(
            model=model,
            geom_pairs=collision_pairs,  # type: ignore
            minimum_distance_from_collisions=0.01,
            collision_detection_distance=0.1,
        )
        self.limits = [
            mink.ConfigurationLimit(model),
            collision_avoidance_limit,
        ]
        self.com_mid = model.body("com_target").mocapid[0]
        self.hands_mid = [model.body(f"{hand}_target").mocapid[0] for hand in self.hands]
        self.torso_mid = model.body("torso_site_target").mocapid[0]

        self.model = self.configuration.model
        self.data = self.configuration.data
        # Separate render buffer to avoid viewer touching live data directly
        self._render_data = mujoco.MjData(self.model)
        # Concurrent-access protection: prevent viewer and control loop from modifying/copying mjData at the same time,
        # which can trigger mj_copyDataVisual errors. Use a reentrant lock to avoid deadlocks in nested locked calls.
        self._mj_lock = threading.RLock()
        self.solver = "daqp"
        # Mapping: joint_name -> target_qpos (rad for hinge, m for slide).
        # Upper layer provides one bool; which joints are locked is fixed by _MAPPING[:6].
        self._fixed_joint_targets: Dict[str, float] = {}
        self._lock_mapping_head_enabled: bool = False
        # COM support-triangle constraint parameters
        self.com_polygon_links = ["steer_motor_link1", "steer_motor_link2", "steer_motor_link3"]
        self.com_boundary_margin = 0.15  # m, inward offset margin from triangle edges
        self.enforce_com_polygon = True
        # Cache: whether COM is inside support triangle (updated by latest target computation)
        self._com_inside = False
    
        # Initialize to the home keyframe.
        self.configuration.update_from_keyframe("teleop")
        self.posture_task.set_target_from_configuration(self.configuration)
        self.torso_task.set_target_from_configuration(self.configuration)
        # Initialize mocap bodies at their respective sites.
        for hand in self.hands:
            mink.move_mocap_to_frame(self.model, self.data, f"{hand}_target", hand, "site")
        self.data.mocap_pos[self.com_mid] = self.data.subtree_com[1]
        mink.move_mocap_to_frame(self.model, self.data, "torso_site_target", "torso_site", "site")

        # Record baseline mocap positions (hands + COM) and (quaternions, MuJoCo order [w,x,y,z]).
        self.hands_base = [self.data.mocap_pos[mid].copy() for mid in self.hands_mid]
        self.hands_base_quat = [self.data.mocap_quat[mid].copy() for mid in self.hands_mid]
        self.com_base = self.data.mocap_pos[self.com_mid].copy()
        self.torso_base = mink.SE3.from_mocap_id(self.data, self.torso_mid)
        # Gripper joint name list (for unit conversion mm -> m)
        self.gripper_joint_names = [
            'left_gripper_finger_joint1', 'left_gripper_finger_joint2',
            'right_gripper_finger_joint1', 'right_gripper_finger_joint2'
        ]
        
    def solve_ik(self, tgt_hand_pose7):
        """Solve IK for given hand mocap targets (position + optional orientation).

        Args:
            tgt_hand_pos: list/tuple of two 3D position arrays.
            tgt_hand_quat: optional list of two quaternions [w,x,y,z]; if provided will set mocap orientations.
        Returns:
            numpy.ndarray: Updated joint positions `qpos` after integration (shape: (nq,)).
        """
        with self._mj_lock:
            # Apply fixed joint absolute targets BEFORE solving.
            if self._fixed_joint_targets:
                for jname, qval in self._fixed_joint_targets.items():
                    jid = self.model.joint(jname).id
                    qadr = int(self.model.jnt_qposadr[jid])
                    self.data.qpos[qadr] = float(qval)
                # Refresh kinematics/COM/constraints for the updated qpos.
                # Using mink's update() keeps it consistent and lighter than mj_fwdPosition.
                self.configuration.update()
                
            self.data.mocap_pos[self.hands_mid[0]] = np.array(tgt_hand_pose7[0][0:3])
            self.data.mocap_pos[self.hands_mid[1]] = np.array(tgt_hand_pose7[1][0:3])
            self.data.mocap_quat[self.hands_mid[0]] = np.array(tgt_hand_pose7[0][3:7])
            self.data.mocap_quat[self.hands_mid[1]] = np.array(tgt_hand_pose7[1][3:7])
            # Constrain COM target using support triangle and margin
            if self.enforce_com_polygon:
                com_xy, com_z = self._compute_safe_com_target_xy()
                self.com_task.set_target(np.array([com_xy[0], com_xy[1], com_z], dtype=float))
            else:
                self.com_task.set_target(self.com_base)
            # Compute and set a safe torso target (position x/z and orientation pitch/yaw constrained within bounds)
            try:
                torso_pose7 = self._compute_safe_torso_target()
                # Do not update mocap using torso_pose7 here, otherwise mocap keeps tracking actual pose and set_target becomes ineffective
                # self.data.mocap_pos[self.torso_mid] = np.array(torso_pose7[0:3], dtype=float)
                # self.data.mocap_quat[self.torso_mid] = np.array(torso_pose7[3:7], dtype=float)
                # self.torso_task.set_target(mink.SE3.from_mocap_id(self.data, self.torso_mid))
                self.torso_task.set_target(mink.SE3.from_pose7(np.asarray(torso_pose7, dtype=float)))
            except Exception:
                # If computation fails, keep base target
                self.torso_task.set_target(self.torso_base)
            
            # Dynamically adjust costs of hands, COM, and torso (based on COM inside/outside support triangle)
            self._update_costs_based_on_com_and_torso()
            # print("torso_cost:", self.torso_task.position_cost, self.torso_task.orientation_cost)
            # print("com_cost:", self.com_task.cost)
            # print("hand_costs:", [ht.position_cost for ht in self.hand_tasks], [ht.orientation_cost for ht in self.hand_tasks])

            for i, hand_task in enumerate(self.hand_tasks):
                hand_task.set_target(mink.SE3.from_mocap_id(self.data, self.hands_mid[i]))
            limits = self.limits
            if self._fixed_joint_targets:
                limits = list(self.limits) + [FixedJointLimit(self.model, list(self._fixed_joint_targets.keys()))]
            vel = mink.solve_ik(
                self.configuration, self.tasks, 0.005, self.solver, 1e-1, limits=limits
            )
            self.configuration.integrate_inplace(vel, 0.005)
            _camlight = getattr(mujoco, "mj_camlight", None)
            if _camlight is not None:
                _camlight(self.model, self.data)
            # Collect mapped qpos for return while still holding the lock
            qpos = np.array([float(self.data.qpos[self.model.joint(m['mj_name']).id]) for m in _MAPPING], dtype=float)
            qret = np.concatenate([qpos[:-4],qpos[-3:-2],qpos[-1:]], axis=0)  # exclude duplicated gripper joints
        return qret

    def set_lock_flag(self, enabled: bool):
        """Use a boolean switch to hard-lock the first 6 joints in `_MAPPING` to constants (QP constraints).
        - enabled=True: capture current qpos as constants and enforce Δq=0 for these joints in QP
        - enabled=False: unlock these 6 joints (does not affect other joints manually locked via set_fixed_joint)
        """
        enabled = bool(enabled)
        if enabled == self._lock_mapping_head_enabled:
            return

        with self._mj_lock:
            if enabled:
                for jname in _LOCKED_JOINTS_MAPPING_HEAD:
                    jid = self.model.joint(jname).id
                    qadr = int(self.model.jnt_qposadr[jid])
                    self._fixed_joint_targets[str(jname)] = float(self.data.qpos[qadr])
            else:
                for jname in _LOCKED_JOINTS_MAPPING_HEAD:
                    self._fixed_joint_targets.pop(str(jname), None)

        self._lock_mapping_head_enabled = enabled
        
    def set_com_boundary_margin(self, margin_m: float):
        """Set inward boundary margin of the COM support triangle (meters)."""
        self.com_boundary_margin = max(0.0, float(margin_m))

    def set_com_polygon_links(self, link_names):
        """Set names of the three body links forming the support triangle."""
        if len(link_names) != 3:
            raise ValueError("Expect 3 link names for triangle")
        self.com_polygon_links = list(link_names)

    def _compute_safe_com_target_xy(self):
        """Compute COM target (xy) and z under support-triangle constraints.

        - Extract world coordinates of three chassis steering-motor links and project to plane (x,y).
        - Build a CCW triangle and offset inward along edge normals by `self.com_boundary_margin`.
        - Project current COM onto the shrunken triangle and return projected (xy) with current COM z.
        """
        tri = self._get_triangle_xy_ccw()
        if tri is None:
            # Fallback to base target for degenerate cases
            return self.com_base[:2].copy(), float(self.com_base[2])
        tri_shrunk = self._shrink_triangle_ccw(tri, self.com_boundary_margin)
        # Current COM position
        com_now = self.data.subtree_com[1].copy()
        p_xy = com_now[:2]
        # Project onto triangle and get inside/outside flag
        proj_xy, inside = self._project_point_to_triangle_2d(p_xy, tri_shrunk)
        # Cache inside/outside result for later use and avoid recomputation
        self._com_inside = bool(inside)
        return proj_xy, float(com_now[2])

    def _get_triangle_xy_ccw(self):
        pts = []
        try:
            for name in self.com_polygon_links:
                bid = self.model.body(name).id
                p = self.data.xpos[bid][:2].copy()
                pts.append(p)
        except Exception:
            return None
        pts = np.asarray(pts, dtype=float)
        # Sort in CCW order
        c = np.mean(pts, axis=0)
        ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        order = np.argsort(ang)
        tri = pts[order]
        # Ensure CCW orientation (positive area)
        area2 = (tri[1, 0] - tri[0, 0]) * (tri[2, 1] - tri[0, 1]) - (tri[1, 1] - tri[0, 1]) * (tri[2, 0] - tri[0, 0])
        if area2 < 0:
            tri = tri[[0, 2, 1]]
        return tri

    def _shrink_triangle_ccw(self, tri_ccw: np.ndarray, margin: float) -> np.ndarray:
        """Shrink a CCW triangle inward by margin along three edges; return new vertices (3x2)."""
        margin = float(max(0.0, margin))
        if margin == 0.0:
            return tri_ccw.copy()
        v = tri_ccw
        verts = []
        for i in range(3):
            i2 = (i + 1) % 3
            # Edge vector and unit inward normal (for CCW polygon, inward side is left normal)
            e = v[i2] - v[i]
            elen = np.linalg.norm(e)
            if elen < 1e-9:
                return tri_ccw.copy()
            t = e / elen
            n_in = np.array([-t[1], t[0]])  # Rotate 90 degrees CCW to get left normal (points inside polygon)
            # Line equation: n·x = b, where b = n·v[i] + margin
            b = float(n_in.dot(v[i]) + margin)
            verts.append((n_in, b))
        # Intersect pairs of the three offset lines to obtain shrunken triangle vertices
        shrink_pts = []
        for i in range(3):
            n1, b1 = verts[i]
            n2, b2 = verts[(i + 1) % 3]
            A = np.vstack([n1, n2])
            B = np.array([b1, b2])
            try:
                x = np.linalg.solve(A, B)
            except np.linalg.LinAlgError:
                return tri_ccw.copy()
            shrink_pts.append(x)
        return np.asarray(shrink_pts, dtype=float)

    def _project_point_to_triangle_2d(self, p: np.ndarray, tri: np.ndarray):
        """Project point p to the closest point inside 2D triangle tri (3x2).

        Return (proj_xy, inside); inside=True means point is inside triangle (or on boundary).
        """
        # First try fast inside test via barycentric coordinates
        a, b, c = tri[0], tri[1], tri[2]
        v0 = b - a
        v1 = c - a
        v2 = p - a
        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)
        denom = d00 * d11 - d01 * d01
        if denom > 1e-12:
            v = (d11 * d20 - d01 * d21) / denom
            w = (d00 * d21 - d01 * d20) / denom
            u = 1.0 - v - w
            if (u >= 0.0) and (v >= 0.0) and (w >= 0.0):
                # Already inside triangle
                return p.copy(), True
        # If outside, project to each edge and take the nearest
        def proj_to_segment(pt, s0, s1):
            d = s1 - s0
            t = 0.0
            denom = np.dot(d, d)
            if denom > 1e-12:
                t = np.clip(np.dot(pt - s0, d) / denom, 0.0, 1.0)
            return s0 + t * d
        cand = [proj_to_segment(p, a, b), proj_to_segment(p, b, c), proj_to_segment(p, c, a)]
        dists = [np.dot(p - q, p - q) for q in cand]
        return cand[int(np.argmin(dists))].copy(), False

    def _is_com_inside_support(self) -> bool:
        """Check if current COM XY lies inside the shrunk support triangle.

        Prefer returning cached result from the latest `_compute_safe_com_target_xy` call to avoid repeated computation.
        If cache is unavailable (rare), fall back to geometric computation.
        """
        # Prefer cache; solve_ik updates it every loop via `_compute_safe_com_target_xy`.
        if isinstance(getattr(self, "_com_inside", None), bool):
            return bool(self._com_inside)
        # Fallback geometric computation (uncommon path)
        tri = self._get_triangle_xy_ccw()
        if tri is None:
            return False
        tri_shrunk = self._shrink_triangle_ccw(tri, self.com_boundary_margin)
        com_now = self.data.subtree_com[1].copy()
        p = com_now[:2]
        _, inside = self._project_point_to_triangle_2d(p, tri_shrunk)
        self._com_inside = bool(inside)
        return self._com_inside

    def _compute_safe_torso_target(self):
        """Compute a safe torso_site target pose7 constrained within bounds.

        Bounds:
            x in [-0.1, 0.25]
            z in [0.551, 1.149]
            pitch in [-1.05, 1.05] (ry)
            yaw in [-1.76, 1.76] (rz)

        Returns:
            pose7: [x,y,z,qw,qx,qy,qz] numpy array.
        """
        # Bounds
        x_lo, x_hi = -0.1, 0.25
        z_lo, z_hi = 0.551, 1.149
        pitch_lo, pitch_hi = -1.05, 1.05
        yaw_lo, yaw_hi = -1.76, 1.76
        # Current torso pose
        pos, quat = self.forward_kinematics("torso_site", "site", sync_mocap=False)
        if quat is None:
            quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        roll,pitch,yaw = smath.quat_to_rpy(quat)
        # Clamp within bounds
        x = float(np.clip(pos[0], x_lo, x_hi))
        y = float(pos[1])
        z = float(np.clip(pos[2], z_lo, z_hi))
        pitch_c = float(np.clip(pitch, pitch_lo, pitch_hi))
        yaw_c = float(np.clip(yaw, yaw_lo, yaw_hi))
        # Reconstruct rotation matrix (ZYX) then pose7
        R = smath.rpy_to_R(roll, pitch_c, yaw_c)
        T = np.eye(4, dtype=float)
        T[:3, :3] = R
        T[:3, 3] = np.array([x, y, z], dtype=float)
        pose7 = np.array(smath.T_to_pose7(T), dtype=float)
        return pose7

    def _update_costs_based_on_com_and_torso(self):
        """Adjust costs using both COM support polygon and torso_site boundary proximity.

        - COM outside shrunk triangle: decrease hand costs, increase COM cost.
        - Torso site nearing boundary (x/z position, pitch/yaw): increase torso costs,
          decrease hand costs.
        """
        # Base (inside) and extreme costs
        hand_pos_inside = 200.0
        hand_ori_inside = 200.0
        hand_pos_outside = 20.0
        hand_ori_outside = 20.0
        com_inside = np.array([1.0, 1.0, 1.0], dtype=float)
        com_outside = np.array([100.0, 100.0, 100.0], dtype=float)
        pos_inside = np.array([1.0, 1.0, 1.0], dtype=float)
        pos_outside = np.array([100.0, 100.0, 100.0], dtype=float)
        ori_inside = np.array([1.0, 1.0, 1.0], dtype=float)
        ori_outside = np.array([100.0, 100.0, 100.0], dtype=float)

        # 1) COM proximity to support triangle
        d_out = 0.0
        try:
            tri = self._get_triangle_xy_ccw()
            if tri is not None:
                tri_shrunk = self._shrink_triangle_ccw(tri, self.com_boundary_margin)
                com_now = self.data.subtree_com[1].copy()
                p_xy = com_now[:2]
                proj_xy, inside = self._project_point_to_triangle_2d(p_xy, tri_shrunk)
                d_out = 0.0 if inside else float(np.linalg.norm(p_xy - proj_xy))
                self._com_inside = bool(inside)
            else:
                inside = False
        except Exception:
            inside = False
            d_out = 0.0
        # COM ramp
        d0_com = 0.00
        dmax_com = 0.05
        gamma_com = 3.0
        if d_out <= d0_com:
            s_com = 0.0
        else:
            s_com = min(1.0, ((d_out - d0_com) / max(1e-6, (dmax_com - d0_com))))
            s_com = s_com ** gamma_com

        # 2) Torso site proximity to boundary box
        x_lo, x_hi = -0.1, 0.25
        z_lo, z_hi = 0.551, 1.149
        pitch_lo, pitch_hi = -0.25, 1.05
        yaw_lo, yaw_hi = -1.76, 1.76
        try:
            pos, quat = self.forward_kinematics("torso_site", "site", sync_mocap=False)
        except Exception:
            pos, quat = None, None
        if quat is None:
            quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        _roll, pitch, yaw = smath.quat_to_rpy(quat)
        x = float(pos[0]) if pos is not None else 0.0
        z = float(pos[2]) if pos is not None else 0.0

        def _edge_dist(v, lo, hi):
            if v < lo:
                return v - lo  # negative outside
            if v > hi:
                return hi - v  # negative outside
            return min(v - lo, hi - v)

        dx_edge = _edge_dist(x, x_lo, x_hi)
        dz_edge = _edge_dist(z, z_lo, z_hi)
        dp_edge = _edge_dist(pitch, pitch_lo, pitch_hi)
        dy_edge = _edge_dist(yaw, yaw_lo, yaw_hi)

        def _ramp(dist_inside, start=0.1, gamma=3.0):
            if dist_inside < 0.0:
                return 1.0
            if dist_inside >= start:
                return 0.0
            s = (start - dist_inside) / max(1e-6, start)
            return min(1.0, s ** gamma)

        s_pos_torso = max(_ramp(dx_edge,0.1), _ramp(dz_edge,0.1))
        s_ori_torso = max(_ramp(dp_edge,0.25), _ramp(dy_edge,0.25))
        s_total = max(s_com, s_pos_torso, s_ori_torso)

        # Update hand costs using combined proximity
        hand_pos_cost = (1.0 - s_total) * hand_pos_inside + s_total * hand_pos_outside
        hand_ori_cost = (1.0 - s_total) * hand_ori_inside + s_total * hand_ori_outside
        for ht in self.hand_tasks:
            try:
                ht.position_cost = float(hand_pos_cost)
                ht.orientation_cost = float(hand_ori_cost)
            except Exception:
                pass

        # Update COM cost using com proximity
        com_cost = (1.0 - s_com) * com_inside + s_com * com_outside
        try:
            self.com_task.cost = np.asarray(com_cost, dtype=float)
        except Exception:
            pass

        # Boost torso task costs near boundaries using torso proximity
        try:            
            pos_cost_full = (1.0 - s_pos_torso) * pos_inside + s_pos_torso * pos_outside
            ori_cost_full = (1.0 - s_ori_torso) * ori_inside + s_ori_torso * ori_outside
            # Apply to relevant axes: position x/z, orientation pitch(y), yaw(z)
            new_pos = np.asarray(self.torso_task.position_cost, dtype=float).copy()
            new_pos[0] = float(pos_cost_full[0])  # x
            new_pos[2] = float(pos_cost_full[2])  # z
            new_ori = np.asarray(self.torso_task.orientation_cost, dtype=float).copy()
            new_ori[1] = float(ori_cost_full[1])  # pitch (ry)
            new_ori[2] = float(ori_cost_full[2])  # yaw (rz)
            self.torso_task.position_cost = new_pos
            self.torso_task.orientation_cost = new_ori
        except Exception:
            pass
    
    def forward_kinematics(self, frame_name: str, frame_type: str = "site", sync_mocap: bool = False):
        """Return current world pose of a site/body.

        Args:
            frame_name: name of the site or body in the model.
            frame_type: "site" or "body".

        Returns:
            (pos, quat): tuple of numpy arrays. pos shape (3,), quat shape (4,w,x,y,z).
        """
        # Ensure all MuJoCo access happens under the lock to avoid stack-in-use errors
        with self._mj_lock:
            _fwd_pos = getattr(mujoco, "mj_fwdPosition", None)
            if _fwd_pos is not None:
                _fwd_pos(self.model, self.data)
            if sync_mocap:
                self._sync_mocap_targets_locked()

            if frame_type == "site":
                sid = self.model.site(frame_name).id
                pos = self.data.site_xpos[sid].copy()
                if hasattr(self.data, "site_xquat"):
                    quat = self.data.site_xquat[sid].copy()
                elif hasattr(self.data, "site_xmat"):
                    mat9 = self.data.site_xmat[sid].copy()
                    quat = np.empty(4, dtype=float)
                    try:
                        mujoco.mju_mat2Quat(quat, mat9)
                    except Exception:
                        # Fallback: identity if conversion utility is unavailable
                        quat = None
                else:
                    quat = None
                # print("*********************",pos,quat)
                return pos, quat
            elif frame_type == "body":
                bid = self.model.body(frame_name).id
                pos = self.data.xpos[bid].copy()
                quat = self.data.xquat[bid].copy()
                return pos, quat
            else:
                raise ValueError(f"Unsupported frame_type: {frame_type}. Use 'site' or 'body'.")
        
    def solve_fk(self):
        """Return current 7D poses of both hands as [x,y,z,qw,qx,qy,qz].

        Uses the actual hand sites (not the mocap target bodies) so this reflects
        the robot's current end-effector pose. Safe for seed/home capture.
        """
        poses7 = []
        for hand in self.hands:
            # site frame (not *_target) gives real TCP pose
            pos, quat = self.forward_kinematics(hand, "site", sync_mocap=False)# set_real_qpos has already synchronized this
            if quat is None:
                # Fallback: identity quaternion if missing
                quat = np.array([1.0,0.0,0.0,0.0], dtype=float)
            # MuJoCo site_xquat ordering is [w,x,y,z]; maintain same ordering used elsewhere
            pose7 = np.concatenate([pos.astype(float), quat.astype(float)])
            poses7.append(pose7)
        return poses7

    def set_real_qpos(self, q_real, sync_mocap: bool = False):
        """Set MuJoCo qpos from real robot encoders using a mapping.

        Args:
            q_real: sequence or numpy array of real robot joint positions (encoder readings).
            mapping: list of dicts, each with keys:
                - 'mj_name': MuJoCo joint name
                - 'robot_index': index in q_real
                - 'scale': unit scale (e.g., deg->rad). Default 1.0
                - 'offset': additive offset. Default 0.0
                - 'sign': +1 or -1 for direction. Default +1

        Writes mapped values into self.data.qpos in MuJoCo joint order, then runs forward update.
        """
        # Build name->id map once; perform full update in one locked block
        with self._mj_lock:
            q_mj = self.data.qpos.copy()
            for m in _MAPPING:
                mj_name = m['mj_name']
                ridx = m['robot_index']
                scale = m.get('scale', 1.0)
                offset = m.get('offset', 0.0)
                sign = m.get('sign', 1.0)
                jid = self.model.joint(mj_name).id
                # Unit conversion: gripper raw input is in mm and must be converted to m
                raw = q_real[ridx]
                if mj_name in self.gripper_joint_names:
                    raw = raw * 0.001/2  # mm -> m
                val = sign * (raw * scale + offset)
                q_mj[jid] = val
            self.data.qpos[:] = q_mj
            has_range = getattr(self.model, 'jnt_range', None) is not None
            if has_range:
                for j in range(self.model.njnt):
                    lo, hi = self.model.jnt_range[j]
                    if lo < hi:
                        self.data.qpos[j] = float(np.clip(self.data.qpos[j], lo, hi))
            _fwd_pos = getattr(mujoco, "mj_fwdPosition", None)
            if _fwd_pos is not None:
                _fwd_pos(self.model, self.data)
            if sync_mocap:
                self._sync_mocap_targets_locked()
    
    def update_configureation_to_home(self):
        """Update configuration from a named keyframe."""
        with self._mj_lock:
            # 1. Apply keyframe
            self.configuration.update_from_keyframe("homepos")
            # 2. Forward update
            _fwd = getattr(mujoco, "mj_forward", None)
            if _fwd is not None:
                _fwd(self.model, self.data)
            else:
                _fwd_pos = getattr(mujoco, "mj_fwdPosition", None)
                if _fwd_pos is not None:
                    _fwd_pos(self.model, self.data)
            # 3. Posture & torso orientation
            self.posture_task.set_target_from_configuration(self.configuration)
            self.torso_task.set_target_from_configuration(self.configuration)
            # 4. COM task target (keep at current COM position)
            if hasattr(self.data, 'subtree_com'):
                com_current = self.data.subtree_com[1].copy()
                self.com_task.set_target(com_current)
                self.com_base = self.data.subtree_com[1].copy()

    def update_viewer(self, viewer=None):
        """Update MuJoCo derived data and optionally sync the viewer."""
        with self._mj_lock:
            _fwd_pos = getattr(mujoco, "mj_fwdPosition", None)
            if _fwd_pos is not None:
                _fwd_pos(self.model, self.data)
            _sensor_pos = getattr(mujoco, "mj_sensorPos", None)
            if _sensor_pos is not None:
                _sensor_pos(self.model, self.data)
            # Keep COM mocap marker at current model COM for visualization
            if hasattr(self.data, 'subtree_com'):
                try:
                    self.data.mocap_pos[self.com_mid] = self.data.subtree_com[1]
                except Exception:
                    pass
            # Keep torso mocap following current torso_site pose for visualization
            try:
                mink.move_mocap_to_frame(self.model, self.data, "torso_site_target", "torso_site", "site")
            except Exception:
                pass
            if viewer is not None:
                viewer.sync()
            
    def launch_viewer(self, hand_amp=0.5, hand_freq=0.5):
        with mujoco.viewer.launch_passive(
            model=self.model, data=self.data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            _free_cam = getattr(mujoco, "mjv_defaultFreeCamera", None)
            if _free_cam is not None:
                _free_cam(self.model, viewer.cam)
            t = 0.0
            rate = RateLimiter(frequency=200.0, warn=False)
            tgt_hand_pose7 =[np.hstack([self.hands_base[0].copy(),self.hands_base_quat[0].copy()]),
                             np.hstack([self.hands_base[1].copy(),self.hands_base_quat[1].copy()])]
            while viewer.is_running():
                # Drive simple periodic mocap motions so IK produces movement.
                t += rate.dt
                # meters, Hz (COM trajectory optional)
                # com_amp, com_freq = 0.2, 0.25

                w_hand = 2.0 * math.pi * hand_freq
                for i in range(2):
                    phase = math.pi if (i % 2 == 1) else 0.0  # alternate left/right
                    offset_x = hand_amp * math.sin(w_hand * t + phase)
                    new_pos = self.hands_base[i].copy()
                    new_pos[0] = new_pos[0] + offset_x
                    tgt_hand_pose7[i][0:3] = new_pos

                # Solve IK for current targets and update viewer
                self.solve_ik(tgt_hand_pose7)
                self.update_viewer(viewer)
                rate.sleep()
    
    # Test function for this class
    def test_traj(self, duration=5.0, viewer=False, hand_amp=0.5, hand_freq=0.5):
        """Run a simple sinusoidal hand trajectory.

        If viewer=False, only perform kinematic IK updates without rendering.
        """
        if viewer:
            return self.launch_viewer(hand_amp=hand_amp, hand_freq=hand_freq)

        # No viewer: run purely kinematic loop
        t = 0.0
        rate = RateLimiter(frequency=200.0, warn=False)
        tgt_hand_pose7 =[np.hstack([self.hands_base[0].copy(),self.hands_base_quat[0].copy()]),
                         np.hstack([self.hands_base[1].copy(),self.hands_base_quat[1].copy()])]
        steps = int(duration * 200)
        for _ in range(steps):
            t += rate.dt
            w_hand = 2.0 * math.pi * hand_freq
            for i in range(2):
                phase = math.pi if (i % 2 == 1) else 0.0
                offset_x = hand_amp * math.sin(w_hand * t + phase)
                new_pos = self.hands_base[i].copy()
                new_pos[2] = new_pos[2] + offset_x
                tgt_hand_pose7[i][0:3] = new_pos
                # Example: keep orientation constant; could add slow yaw rotation here if desired
            self.solve_ik(tgt_hand_pose7)
            self.update_viewer(viewer=None)
            rate.sleep()
        return True

    # Teleoperation and simulation interaction function
    def apply_mapped_qpos(self, q_mapped, sync_mocap: bool = False):
        """Apply a mapped qpos array (ordered like `_MAPPING`) into the MuJoCo sim.

        Args:
            q_mapped: iterable of floats with length == len(_MAPPING).
            sync_mocap: if True, move mocap targets (hands, COM) to match current frames
                        after updating qpos; useful for passive visualization without IK.
        """
        q = np.asarray(q_mapped, dtype=float).reshape(-1)
        if q.shape[0] < 1:
            return
        # Structure:
        # First 18 values: 4 (torso) + 7 (left arm) + 7 (right arm)
        # Optional 19th/20th values: two gripper scalars (left, right) in mm.
        # If provided, they are copied into 4 gripper joints in the model with mm->m conversion
        # (and the same /2 scaling convention as set_real_qpos).
        with self._mj_lock:
            # Write torso and arm values (no unit conversion involved)
            base_count = min(18, q.shape[0])
            for i in range(base_count):
                m = _MAPPING[i]
                jid = self.model.joint(m['mj_name']).id
                self.data.qpos[jid] = float(q[i])
            # Handle gripper (if 2 parameters are provided)
            if q.shape[0] >= 20:
                left_grip_mm = float(np.clip(q[18], 0, 100.0))
                right_grip_mm = float(np.clip(q[19], 0, 100.0))
                # mm -> m, and keep /2 scaling as in set_real_qpos (split between two fingers)
                left_grip_m = left_grip_mm * 0.001 / 2.0
                right_grip_m = right_grip_mm * 0.001 / 2.0
                # Mapping indices: 18,19 for left fingers; 20,21 for right fingers
                if len(_MAPPING) >= 22:
                    try:
                        jid_l1 = self.model.joint(_MAPPING[18]['mj_name']).id
                        jid_l2 = self.model.joint(_MAPPING[19]['mj_name']).id
                        self.data.qpos[jid_l1] = left_grip_m
                        self.data.qpos[jid_l2] = left_grip_m
                        jid_r1 = self.model.joint(_MAPPING[20]['mj_name']).id
                        jid_r2 = self.model.joint(_MAPPING[21]['mj_name']).id
                        self.data.qpos[jid_r1] = right_grip_m
                        self.data.qpos[jid_r2] = right_grip_m
                    except Exception:
                        pass
            _fwd_pos = getattr(mujoco, "mj_fwdPosition", None)
            if _fwd_pos is not None:
                _fwd_pos(self.model, self.data)
            if sync_mocap:
                self._sync_mocap_targets_locked()

    def get_sim_feedback(self):
        """Return a minimal feedback dict from the current MuJoCo state.

        The dict contains qpos/qvel split into `torso`, `left_arm`, `right_arm`,
        plus placeholder gripper values. All joint lists are numpy arrays (radians).
        """
        # Build list of q values in the same order as _MAPPING
        with self._mj_lock:
            qlist = [float(self.data.qpos[self.model.joint(m['mj_name']).id]) for m in _MAPPING]
            vlist = [float(self.data.qvel[self.model.joint(m['mj_name']).id]) for m in _MAPPING]
        qarr = np.array(qlist, dtype=float)
        varr = np.array(vlist, dtype=float)
        # Gripper feedback: use average of two fingers (if available); otherwise 0.0
        left_grip_fb = 0.0
        right_grip_fb = 0.0
        if qarr.shape[0] >= 22:
            left_grip_fb = float((qarr[18] + qarr[19]) * 1000) # convert to mm
            right_grip_fb = float((qarr[20] + qarr[21]) * 1000) # convert to mm
        feedback = {
            'qpos': {
                'torso': qarr[0:4].copy(),
                'left_arm': qarr[4:11].copy(),
                'right_arm': qarr[11:18].copy(),
                'left_gripper': [left_grip_fb],
                'right_gripper': [right_grip_fb],
                'chassis': []
            },
            'qvel': {
                'torso': varr[0:4].copy(),
                'left_arm': varr[4:11].copy(),
                'right_arm': varr[11:18].copy(),
            }
        }
        return feedback

    def _sync_mocap_targets_locked(self):
        """Assumes caller holds `self._mj_lock`.

        Move mocap bodies (hands targets and COM target) to the current frames so that
        visual markers match the robot state. This is helpful when not running IK.
        """
        for hand in self.hands:
            # Move the corresponding mocap body to the site's current pose
            mink.move_mocap_to_frame(self.model, self.data, f"{hand}_target", hand, "site")
        # Keep COM target at current COM of the torso subtree (body 1)
        if hasattr(self.data, 'subtree_com'):
            self.data.mocap_pos[self.com_mid] = self.data.subtree_com[1]
        # Keep torso target following the actual torso site
        try:
            mink.move_mocap_to_frame(self.model, self.data, "torso_site_target", "torso_site", "site")
        except Exception:
            pass

    def start_viewer_thread(self, hand_amp=0.0, hand_freq=0.0, drive_mocap: bool = False, viewer_hz: float = 60.0, compute_hz: float = 200.0):
        """Start a background thread that runs a passive viewer.

        Call `stop_viewer_thread()` to stop it. This is a convenience wrapper
        useful when integrating simulation with an external control loop.
        """
        if hasattr(self, "_viewer_thread") and getattr(self, "_viewer_thread", None) is not None:
            return
        import threading
        self._viewer_stop = threading.Event()

        def _run():
            try:
                # Use a separate render buffer so viewer never reads live self.data
                with mujoco.viewer.launch_passive(model=self.model, data=self._render_data, show_left_ui=False, show_right_ui=False) as viewer:
                    _free_cam = getattr(mujoco, "mjv_defaultFreeCamera", None)
                    if _free_cam is not None:
                        _free_cam(self.model, viewer.cam)
                    rate = RateLimiter(frequency=max(1.0, float(compute_hz)), warn=False)
                    # viewer sync limiter (lower rate than compute)
                    viewer_interval = 1.0 / max(1.0, float(viewer_hz))
                    last_view_sync_t = time.time()
                    while viewer.is_running() and not self._viewer_stop.is_set():
                        # simple periodic mocap motion if requested
                        t = time.time()
                        w_hand = 2.0 * math.pi * hand_freq
                        # Update live data and copy to render buffer under lock
                        with self._mj_lock:
                            if drive_mocap and hand_amp != 0.0:
                                for i in range(2):
                                    phase = math.pi if (i % 2 == 1) else 0.0
                                    offset_x = hand_amp * math.sin(w_hand * t + phase)
                                    new_pos = self.hands_base[i].copy()
                                    new_pos[0] = new_pos[0] + offset_x
                                    self.data.mocap_pos[self.hands_mid[i]] = new_pos
                            # Compute derived quantities on live data (fast loop)
                            _fwd = getattr(mujoco, "mj_forward", None)
                            if _fwd is not None:
                                _fwd(self.model, self.data)
                            else:
                                _fwd_pos = getattr(mujoco, "mj_fwdPosition", None)
                                if _fwd_pos is not None:
                                    _fwd_pos(self.model, self.data)
                                _sensor_pos = getattr(mujoco, "mj_sensorPos", None)
                                if _sensor_pos is not None:
                                    _sensor_pos(self.model, self.data)
                            # Update COM mocap marker to follow current COM
                            if hasattr(self.data, 'subtree_com'):
                                try:
                                    self.data.mocap_pos[self.com_mid] = self.data.subtree_com[1]
                                except Exception:
                                    pass
                            # Keep torso target following the actual torso site
                            try:
                                mink.move_mocap_to_frame(self.model, self.data, "torso_site_target", "torso_site", "site")
                            except Exception:
                                pass
                            # Copy live data to render buffer safely
                            _copy = getattr(mujoco, "mj_copyData", None)
                            if _copy is not None:
                                _copy(self._render_data, self.model, self.data)
                            else:
                                # Fallback: minimal fields
                                self._render_data.qpos[:] = self.data.qpos
                                self._render_data.qvel[:] = self.data.qvel
                                self._render_data.mocap_pos[:] = self.data.mocap_pos
                                self._render_data.mocap_quat[:] = self.data.mocap_quat
                                _fwd2 = getattr(mujoco, "mj_forward", None)
                                if _fwd2 is not None:
                                    _fwd2(self.model, self._render_data)
                        # Sync the viewer at throttled rate, reading only render buffer
                        now_t = time.time()
                        if (now_t - last_view_sync_t) >= viewer_interval:
                            viewer.sync()
                            last_view_sync_t = now_t
                        rate.sleep()
            except Exception:
                pass

        self._viewer_thread = threading.Thread(target=_run, daemon=True)
        self._viewer_thread.start()

    def stop_viewer_thread(self):
        if hasattr(self, "_viewer_stop") and self._viewer_stop is not None:
            self._viewer_stop.set()
        if hasattr(self, "_viewer_thread") and self._viewer_thread is not None:
            self._viewer_thread.join(timeout=1.0)
            self._viewer_thread = None

    def check_collision_pairs(self):
        """Check collisions across configured geom groups.

        Returns a flat result: {"colliding": bool, "min_dist": [d0, d1, ...]}
        where each element in min_dist is the smallest contact distance for the
        corresponding configured pair (negative means penetration). The overall
        colliding is True if any pair has penetration.
        """
        min_dists = []
        scores = []
        any_colliding = False
        with self._mj_lock:
            _fwd = getattr(mujoco, "mj_forward", None)
            if _fwd is not None:
                _fwd(self.model, self.data)
            pairs = [(["left_hand_collision", "right_hand_collision", "left_elbow_collision", "right_elbow_collision"],
                      ["torso_collision"])]
            for groupA, groupB in pairs:
                idsA = []
                idsB = []
                for name in groupA:
                    try:
                        idsA.append(self.model.geom(name).id)
                    except Exception:
                        pass
                for name in groupB:
                    try:
                        idsB.append(self.model.geom(name).id)
                    except Exception:
                        pass
                min_dist = float("inf")
                ncon = int(self.data.ncon)
                pair_colliding = False
                for ci in range(ncon):
                    con = self.data.contact[ci]
                    g1 = int(con.geom1)
                    g2 = int(con.geom2)
                    if (g1 in idsA and g2 in idsB) or (g1 in idsB and g2 in idsA):
                        dist = float(con.dist)
                        if dist < min_dist:
                            min_dist = dist
                        if dist < 0.005:
                            pair_colliding = True
                if min_dist == float("inf"):
                    # No contact constraints matched this pair. This is normal when geoms are separated,
                    # especially if geom margins are 0. Use mj_geomDistance to get actual separation.
                    _gd = getattr(mujoco, "mj_geomDistance", None)
                    if _gd is not None and len(idsA) > 0 and len(idsB) > 0:
                        distmax = 1.0  # meters; increase if your model has larger separations
                        best = float("inf")
                        best_pair = None
                        for ga in idsA:
                            for gb in idsB:
                                try:
                                    dsep = float(_gd(self.model, self.data, int(ga), int(gb), float(distmax), None))
                                    if dsep < best:
                                        best = dsep
                                        best_pair = (int(ga), int(gb))
                                except Exception:
                                    pass
                        if best != float("inf"):
                            min_dist = best
                        else:
                            min_dist = float("nan")
                    else:
                        min_dist = float("nan")
                min_dists.append(min_dist)
                # Score mapping: >0.045 -> 100; [0.005,0.045] -> linear to 0; <0 -> 0; NaN -> 100
                if np.isnan(min_dist):
                    score = 100.0
                elif min_dist >= 0.045:
                    score = 100.0
                elif min_dist <= 0.005:
                    score = 0.0
                else:
                    score = 100.0 * ((min_dist-0.005) / 0.04)
                scores.append(float(score))
                any_colliding = any_colliding or pair_colliding
            overall_score = float(min(scores)) if len(scores) > 0 else 100.0
            return {"colliding": bool(any_colliding), "min_dist": min_dists, "score": overall_score}

    def get_hand_target_pose_error(self):
        """Compute 6D error between hand target mocaps and actual hand sites.

        Returns a list with two dicts (left, right):
        - pos_error: Euclidean distance between mocap pos and site pos (meters)
        - ori_error_rad: rotation angle between mocap quat and site quat (radians)
        - pos_delta: 3-vector difference (target - actual)
        - ori_axis_angle: 3-vector axis * angle (angle-axis) representing orientation error
        """
        errs = []
        with self._mj_lock:
            # Make sure derived quantities are current
            _fwd_pos = getattr(mujoco, "mj_fwdPosition", None)
            if _fwd_pos is not None:
                _fwd_pos(self.model, self.data)
            for i, hand in enumerate(self.hands):
                # Target from mocap
                tpos = self.data.mocap_pos[self.hands_mid[i]].copy()
                tquat = self.data.mocap_quat[self.hands_mid[i]].copy()
                # Actual site pose
                spos, squat = self.forward_kinematics(hand, "site", sync_mocap=False)
                if squat is None:
                    squat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
                # Position error
                dp = tpos - spos
                pos_err = float(np.linalg.norm(dp))
                # Orientation error: q_err = q_target * inv(q_actual)
                qw_t, qx_t, qy_t, qz_t = tquat
                qw_a, qx_a, qy_a, qz_a = squat
                # inverse actual
                q_inv_a = np.array([qw_a, -qx_a, -qy_a, -qz_a], dtype=float)
                q_err = smath.quat_multiply(np.array([qw_t, qx_t, qy_t, qz_t], dtype=float), q_inv_a)
                # angle from quaternion using shortest rotation (q and -q are equivalent)
                w = float(q_err[0])
                w = max(-1.0, min(1.0, w))
                ang = 2.0 * math.acos(abs(w))
                # normalize axis from vector part; handle near-zero rotation robustly
                v = np.array([q_err[1], q_err[2], q_err[3]], dtype=float)
                v_norm = float(np.linalg.norm(v))
                if v_norm > 1e-12:
                    axis = v / v_norm
                else:
                    axis = np.array([0.0, 0.0, 0.0], dtype=float)
                ori_axis_angle = axis * ang
                errs.append({
                    "pos_error": pos_err,
                    "ori_error_rad": float(ang),
                    "pos_delta": dp,
                    "ori_axis_angle": ori_axis_angle,
                })
        return errs
    
    def check_out_workspace(self):
        """Determine whether workspace limits are exceeded based on 6D error between target and actual hand pose.

        Rule: if either hand has `pos_error` or `ori_error_rad` > 0.03, mark as out-of-workspace.
        Returns: {"any_out": bool, "left": {...}, "right": {...}}
        left/right include: pos_error, ori_error_rad, pos_delta, ori_axis_angle, out.
        """
        errs = self.get_hand_target_pose_error()
        result_per_hand = []
        any_out = False
        overall_score = 100.0
        for e in errs:
            out = (float(e.get("pos_error", 0.0)) > 0.01) or (float(e.get("ori_error_rad", 0.0)) > 2.0/180.0*math.pi) # and not self._is_com_inside_support() ?
            # Scoring rules:
            # - Position error p: 0 -> 100, 0.045 -> 60; above threshold, use exponential decay asymptotically toward 0
            # - Angular error a_deg: 0 -> 100, 9 deg -> 60; above threshold, use exponential decay asymptotically toward 0
            p = float(e.get("pos_error", 0.0))
            a_rad = float(e.get("ori_error_rad", 0.0))
            a_deg = a_rad * 180.0 / math.pi
            if p<=0.005:
                score_pos = 100.0
            elif p <= 0.045:
                score_pos = 100.0 - 40.0 * ((p-0.005) / 0.04)
            else:
                # Progressive decay: continuous at p=0.045 with score 60, then exponentially decays toward 0
                delta_p = p - 0.045
                decay_scale_p = 0.1  # Decay scale of about 10 cm
                score_pos = 60.0 * math.exp(-max(0.0, delta_p) / max(1e-6, decay_scale_p))
            if a_deg <= 1.0:
                score_ori = 100.0
            elif a_deg <= 9.0:
                score_ori = 100.0 - 40.0 * ((a_deg - 1.0) / 8.0)
            else:
                # Progressive decay: continuous at a=9 degrees with score 60, then exponentially decays toward 0
                delta_a = a_deg - 9.0
                decay_scale_a = 20.0  # Decay scale of about 20 degrees
                score_ori = 60.0 * math.exp(-max(0.0, delta_a) / max(1e-6, decay_scale_a))
            score = min(score_pos, score_ori)
            res = {
                "pos_error": float(e.get("pos_error", 0.0)),
                "ori_error_rad": float(e.get("ori_error_rad", 0.0)),
                "pos_delta": np.asarray(e.get("pos_delta", np.zeros(3, dtype=float)), dtype=float),
                "ori_axis_angle": np.asarray(e.get("ori_axis_angle", np.zeros(3, dtype=float)), dtype=float),
                "out": bool(out),
                "score": float(max(0.0, score)),
            }
            result_per_hand.append(res)
            any_out = any_out or out
            overall_score = min(overall_score, res["score"]) if result_per_hand else res["score"]
        left = result_per_hand[0] if len(result_per_hand) > 0 else None
        right = result_per_hand[1] if len(result_per_hand) > 1 else None
        return {"any_out": bool(any_out), "left": left, "right": right, "score": float(overall_score if result_per_hand else 100.0)}
    
                
if __name__ == "__main__":
    r1pro = R1ProMink()
    r1pro.test_traj(duration=10.0, viewer=True, hand_amp=0.3, hand_freq=1.0)
    # r1pro.replay_txt_traj(viewer=True)
