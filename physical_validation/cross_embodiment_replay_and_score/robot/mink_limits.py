import numpy as np
import mujoco
import mink
from mink.limits.limit import Limit, Constraint


class FixedJointLimit(Limit):
    """Hard equality constraint that freezes specified 1-DoF joints in the QP.

    Enforces, in the tangent space used by mink:

      Δq_j = 0

    implemented as two inequalities:

      Δq_j ≤ 0
      -Δq_j ≤ 0

    Notes:
    - This constrains the *increment* during one solve step. To fix the joint to an
      absolute constant value, set `qpos` to that value before solving each step.
    - Only supports hinge/slide joints (1 DoF). Floating joints are not supported.
    """

    def __init__(self, model: mujoco.MjModel, joint_names):
        super().__init__()
        self.model = model
        names = list(joint_names) if joint_names is not None else []
        self.joint_names = names
        self.indices = []
        for name in names:
            jid = self.model.joint(str(name)).id
            dofadr = int(self.model.jnt_dofadr[jid])

            # MuJoCo Python bindings differ across versions; `jnt_dofnum` may not exist.
            # We only support 1-DoF joints (hinge/slide). Ball/free joints are rejected.
            jtype = int(self.model.jnt_type[jid])
            if jtype not in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            ):
                raise ValueError(
                    f"FixedJointLimit only supports 1-DoF hinge/slide joints, got {name!r} jnt_type={jtype}"
                )
            self.indices.append(dofadr)

        if len(self.indices) <= 0:
            self.projection_matrix = None
        else:
            nb = len(self.indices)
            self.projection_matrix = np.zeros((nb, int(self.model.nv)), dtype=float)
            for i, dof in enumerate(self.indices):
                self.projection_matrix[i, int(dof)] = 1.0

    def compute_qp_inequalities(self, configuration: mink.Configuration, dt: float) -> Constraint:
        del configuration, dt
        if self.projection_matrix is None:
            return Constraint()
        G = np.vstack([self.projection_matrix, -self.projection_matrix])
        h = np.zeros((G.shape[0],), dtype=float)
        return Constraint(G=G, h=h)
