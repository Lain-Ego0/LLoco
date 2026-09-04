# Go2 source migration matrix

The registry in `legged_gym/envs/__init__.py` is authoritative. It contains 14
tasks; directory names and unregistered configs are not counted as tasks.

| Source task | LLoco task | Source config | Source environment | Status |
|---|---|---|---|---|
| `go2_trot` | `Unitree-Go2-Trot-Flat` | `Go2_MoB/Go2_Trot/Go2_Trot_Config.py` | `Go2_MoB/Go2_Trot/Go2_Trot.py` | stage-1 runnable; latency/symmetry gaps below |
| `go2_jump` | `Unitree-Go2-Jump-Flat` | `Go2_MoB/Go2_Jump/Go2_Jump_Config.py` | `Go2_MoB/Go2_Jump/Go2_Jump.py` | pending |
| `go2_handstand` | `Unitree-Go2-Handstand-Flat` | `Go2_Stand/Go2_Handstand/Go2_Handstand_Config.py` | `Go2_Stand/Go2_Handstand/Go2_Handstand.py` | pending |
| `go2_leggedstand` | `Unitree-Go2-Legged-Stand-Flat` | `Go2_Stand/Go2_Leggedstand/Go2_Leggedstand_Config.py` | `Go2_Stand/Go2_Leggedstand/Go2_Leggedstand.py` | pending |
| `go2_spring_jump` | `Unitree-Go2-Spring-Jump-Flat` | `Go2_Flip/Go2_Spring_Jump/Go2_Spring_Jump_Config.py` | `Go2_Flip/Go2_Spring_Jump/Go2_Spring_Jump.py` | pending |
| `go2_backflip` | `Unitree-Go2-Backflip-Flat` | `Go2_Flip/Go2_BackFlip/Go2_BackFlip_Config.py` | `Go2_Flip/Go2_BackFlip/Go2_BackFlip.py` | pending |
| `go2_dreamwaq` | `Unitree-Go2-DreamWaQ-Rough` | `Go2_DreamWaQ/Go2_DreamWaQ_Config.py` | `Go2_DreamWaQ/Go2_DreamWaQ.py` | pending |
| `go2_amp_dreamwaq` | `Unitree-Go2-AMP-DreamWaQ-Rough` | `Go2_AMP_DreamWaQ/Go2_AMP_DreamWaQ_Config.py` | `Go2_AMP_DreamWaQ/Go2_AMP_DreamWaQ.py` | pending |
| `go2_cts` | `Unitree-Go2-CTS-Rough` | `Go2_Cts/Go2_Cts_Config.py` | `Go2_Cts/Go2_Cts.py` | pending |
| `go2_amp_cts` | `Unitree-Go2-AMP-CTS-Rough` | `Go2_AMP_Cts/Go2_AMP_Cts_Config.py` | `Go2_AMP_Cts/Go2_AMP_Cts.py` | pending |
| `go2_amp_ts` | `Unitree-Go2-AMP-TS-Teacher-Rough` | `Go2_AMP_Ts/Go2_AMP_Ts_Config.py` | `base/legged_robot_amp_ts.py` | pending |
| `go2_amp_ts_student` | `Unitree-Go2-AMP-TS-Student-Rough` | `Go2_AMP_Ts/Go2_AMP_Ts_Student_Config.py` | `base/legged_robot_amp_ts.py` | pending |
| `go2_ts` | `Unitree-Go2-TS-Teacher-Rough` | `Go2_TS/Go2_TS_Config.py` | `base/legged_robot_amp_ts.py` | pending |
| `go2_ts_student` | `Unitree-Go2-TS-Student-Rough` | `Go2_TS/Go2_TS_Student_Config.py` | `base/legged_robot_amp_ts.py` | pending |

## Trot parity table

| Concern | Isaac Gym source | mjlab implementation |
|---|---|---|
| Actor observation | `[phase sin/cos, command, delayed IMU, delayed q/dq, action]`, 47 × 10 | one frame-major history term, 470 dimensions |
| Critic observation | command/phase, q-relative, q, dq, action, base velocities, Euler, stance and 4 contacts, 68 × 3 | one frame-major history term, 204 dimensions |
| History reset | zero all frames, then append current frame | custom history term with the same zero-fill behavior |
| Noise | uniform per-field amplitudes, actor only; each noisy frame is retained | noise is applied to the new 47-element frame before it enters history |
| Action | default pose + `0.25 * action`, 1–3 physics-substep lag | shared per-environment physics-substep delay action term |
| Control | explicit PD, Kp 20, Kd 0.5, URDF effort limits | mjlab ideal PD with the same gains and limits |
| Command | uniform ±1, every 5 s; 5% all zero and independent 5% XY zero | custom native command term |
| Reward | 16 source terms, including batch-mean trot gate | same formulas, weights, gates, and dt scaling |
| Reset/termination | q offset ±0.1; fixed root state; base force > 1 N | native reset events and a dedicated base contact term |
| Randomization | friction, base/link mass, COM, gains, motor zero, 4 s velocity overwrite | native startup events and exact overwrite push event |
| PPO | seed 1, 24 steps, 15k iterations, LR 1e-5, 512/256/128 ELU | same supported rsl_rl 5.4.2 settings |

Three source details are not silently claimed as exact:

1. Isaac Gym updates motor/IMU observation latency inside each of four physics
   substeps. mjlab's Observation Manager samples after decimation, so the current
   implementation retains the observation fields but does not yet model this
   substep sensor pipeline. A simulation hook is required before parity can be
   marked complete for latency.
2. The old fork's `sym_loss` PPO option is absent from upstream rsl_rl 5.4.2.
   Its observation/action permutations are recorded in the source config, but a
   compatible augmentation hook remains pending. No old rsl_rl code is vendored.
3. mjlab's public `body_mass` randomizer changes mass without recomputing body
   inertia, while the Isaac Gym source requests inertia recomputation after its
   mass edits. A native mjlab inertia-safe event is still needed for strict
   dynamic parity.
