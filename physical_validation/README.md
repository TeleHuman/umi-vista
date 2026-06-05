# Physical Validation

This directory will contain the VISTA physical-validation pipeline for UMI trajectories.

The planned pipeline follows the paper structure:

1. **Data-completeness pre-check**
   - Detect missing files, empty files, frame drops, and malformed trajectory records.

2. **Trajectory continuity scoring**
   - Score the smoothness of raw gripper motion before robot-specific replay.

3. **Self-collision risk scoring**
   - Replay trajectories with target robot kinematics and score collision-pair distances.

4. **Execution-fidelity scoring**
   - Compare desired UMI end-effector poses with feasible robot replay poses.

5. **Overall trajectory scoring**
   - Aggregate continuity, collision, and fidelity scores with embodiment-conditioned weighting.

Current status: placeholder only. Implementation code will be added after cleanup.
