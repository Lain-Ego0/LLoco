# Go2 source migration matrix

The registry in `legged_gym/envs/__init__.py` is authoritative. It contains 14
tasks; directory names and unregistered configs are not counted as tasks.

| Source task | LLoco task | Source config | Source environment | Status |
|---|---|---|---|---|
| `go2_trot` | `Unitree-Go2-Trot-Flat` | `Go2_MoB/Go2_Trot/Go2_Trot_Config.py` | `Go2_MoB/Go2_Trot/Go2_Trot.py` | stage-1 runnable; latency/symmetry gaps below |
| `go2_jump` | `Unitree-Go2-Jump-Flat` | `Go2_MoB/Go2_Jump/Go2_Jump_Config.py` | `Go2_MoB/Go2_Jump/Go2_Jump.py` | stage-1 accepted; corrected-contact 2048 × 1000 training, checkpoint/ONNX validation and Viser inspection passed |
| `go2_handstand` | `Unitree-Go2-Rear-Stand-Flat` | `Go2_Stand/Go2_Handstand/Go2_Handstand_Config.py` | `Go2_Stand/Go2_Handstand/Go2_Handstand.py` | accepted as Rear Stand; 4096 × 2000 training and Viser validation passed |
| `go2_leggedstand` | `Unitree-Go2-Handstand-Flat` | `Go2_Stand/Go2_Leggedstand/Go2_Leggedstand_Config.py` | `Go2_Stand/Go2_Leggedstand/Go2_Leggedstand.py` | accepted; 2048 x 800 zero-initialized training and deterministic playback passed |
| `go2_spring_jump` | `Unitree-Go2-Spring-Jump-Flat` | `Go2_Flip/Go2_Spring_Jump/Go2_Spring_Jump_Config.py` | `Go2_Flip/Go2_Spring_Jump/Go2_Spring_Jump.py` | stage-1 runnable; one-shot state-machine and source training assist migrated |
| `go2_backflip` | `Unitree-Go2-Backflip-Flat` | `Go2_Flip/Go2_BackFlip/Go2_BackFlip_Config.py` | `Go2_Flip/Go2_BackFlip/Go2_BackFlip.py` | incomplete; implementation retained outside registry: 2048×1000 verifies runtime only, not stable takeoff/landing |
| `go2_dreamwaq` | `Unitree-Go2-DreamWaQ-Rough` | `Go2_DreamWaQ/Go2_DreamWaQ_Config.py` | `Go2_DreamWaQ/Go2_DreamWaQ.py` | accepted; smoke, 2048 × 1000 from-zero validation, 1000-round continuation and Viser playback passed |
| `go2_amp_dreamwaq` | `Unitree-Go2-AMP-DreamWaQ-Rough` | `Go2_AMP_DreamWaQ/Go2_AMP_DreamWaQ_Config.py` | `Go2_AMP_DreamWaQ/Go2_AMP_DreamWaQ.py` | corrected AMP pipeline; velocity-recovery continuation reached 2000 iterations, directional tracking improved; longer acceptance still pending |
| `go2_cts` | `Unitree-Go2-CTS-Rough` | `Go2_Cts/Go2_Cts_Config.py` | `Go2_Cts/Go2_Cts.py` | pending |
| `go2_amp_cts` | `Unitree-Go2-AMP-CTS-Rough` | `Go2_AMP_Cts/Go2_AMP_Cts_Config.py` | `Go2_AMP_Cts/Go2_AMP_Cts.py` | pending |
| `go2_amp_ts` | `Unitree-Go2-AMP-TS-Teacher-Rough` | `Go2_AMP_Ts/Go2_AMP_Ts_Config.py` | `base/legged_robot_amp_ts.py` | pending |
| `go2_amp_ts_student` | `Unitree-Go2-AMP-TS-Student-Rough` | `Go2_AMP_Ts/Go2_AMP_Ts_Student_Config.py` | `base/legged_robot_amp_ts.py` | pending |
| `go2_ts` | `Unitree-Go2-TS-Teacher-Rough` | `Go2_TS/Go2_TS_Config.py` | `base/legged_robot_amp_ts.py` | pending |
| `go2_ts_student` | `Unitree-Go2-TS-Student-Rough` | `Go2_TS/Go2_TS_Student_Config.py` | `base/legged_robot_amp_ts.py` | pending |

## AMP-DreamWaQ parity table

| Concern | Isaac Gym source | mjlab implementation |
|---|---|---|
| Terminal transition | captures AMP state before reset and substitutes it for done environments | pre-reset recorder exports the same 31-field terminal state; reset observations only seed the next episode |
| AMP normalization | shared RunningMeanStd for policy/expert states, clipped to [-10, 10] | checkpointed float64 running moments and identical clipping/update order |
| Policy data | 1,000,000-transition ring replay sampled across iterations | fixed-size 1,000,000-transition device replay buffer |
| Expert data | 2,000,000 transitions evaluated at continuous `t` and `t + 0.02 s`, using each motion file's `FrameDuration` | same weighted time sampling and linear interpolation of discriminator fields; no adjacent-frame assumption |
| Optimization | PPO and discriminator share one Adam step; trunk/head decay 1e-4/1e-2 | same combined loss, optimizer groups, gradient penalty and decay |
| Exploration | per-joint minimum std is 5% of the softened joint range | same vector floor applied after every combined optimizer step |
| Observation noise | joint position amplitude 0.01 | AMP actor overrides DreamWaQ's 0.02 amplitude with 0.01 |
| Command distribution | lateral range +/-0.6, earlier speed schedule, 5% standing and independent 5% turn-in-place | dedicated AMP command term with the same distribution |

The exact source reward has a zero terminal cost. Under MuJoCo, a random AMP
policy reaches base-contact termination after roughly 15 steps and learns to
use termination to avoid the net-negative early shaping return; a corrected
2048-environment diagnostic run remained at 18 steps through iteration 360.
The runnable migration therefore adds a +1 alive term and a one-time -5
terminal cost as an explicit backend adaptation. These terms address the
termination loophole while leaving every Gym reward component, target motion,
observation, command, AMP objective, and final policy interface unchanged.

The source configuration declares `reference_state_initialization_prob=0.85`,
but its `reset_idx()` never reads that probability and always performs the
ordinary randomized reset. The migration therefore deliberately does not add
motion-reference initialization. The old `model_100.pt` predates the corrected
AMP transition pipeline and must not be resumed or used for acceptance.

### AMP-DreamWaQ acceptance result

The corrected run started from random network weights with 2,048 environments,
the full 2,000,000-transition expert preload and 1,000,000-transition policy
replay. It completed 1,000 iterations without a numerical failure. From the
first to the final iteration, mean episode length rose from 16.82 to 936.35,
mean reward from -5.50 to 26.79, and the base-contact termination metric fell
from 50.54 to 0.41. Linear/angular tracking terms reached 0.380/0.315. AMP loss
settled at 0.188 with policy/expert discriminator predictions -0.623/+0.623,
rather than collapsing toward the near-zero loss seen in the rejected run.

`model_999.pt` contains the actor, critic, discriminator, AMP running moments,
joint PPO/discriminator optimizer and VAE optimizer, and all saved tensors are
finite. In deterministic inference with play-mode observation noise and pushes
disabled, 64 randomized environments were each run for 1,000 steps: 63 reached
the time limit, only 2 non-timeout terminations occurred, and completed episode
length averaged 982.85 steps. The accepted run is stored under
`logs/rsl_rl/go2_amp_dreamwaq/2026-09-07_16-50-47_corrected_amp_survival_2048x1000`.

The initial checkpoint was then continued from `model_999.pt` with the alive
bridge reduced from `1.0` to `0.1` while retaining the `-5` terminal cost. The
resulting `model_1999.pt` is stored under
`logs/rsl_rl/go2_amp_dreamwaq/2026-09-07_18-16-24_velocity_recovery_alive01_to2000`.
In a fixed-command 300-step steady-state test, it reached `+0.875 m/s` for a
`+1.0 m/s` forward command, `-0.879 m/s` for `-1.0 m/s`, `+0.506 m/s` for
`+0.6 m/s` lateral, and `+0.772 rad/s` for `+1.0 rad/s` yaw. This improves
substantially over the first checkpoint, but residual tracking error remains,
so the task is not yet marked fully Gym-equivalent.

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

## Jump parity table

| Concern | Isaac Gym source | mjlab implementation |
|---|---|---|
| Actor observation | phase sin/cos, scaled command, delayed angular velocity/Euler, delayed q/dq and action; 47 × 10 | same field order and scaling in a frame-major 470-vector |
| Critic observation | actor-independent state plus friction, a zero-valued `body_mass` buffer, two stance flags and four contacts; 70 × 3 | same 70-field order and source's intentionally zero mass label, 210 dimensions |
| Phase | `episode_length * dt / 1.5`, not wrapped; stance before 0.6 and flight after 0.6 | same unwrapped phase and one-time stance transition |
| Action/control | default pose + `0.25 * action`; episode-sampled 1–3 physics-substep lag; Kp 20/Kd 0.5 | same action mapping, lag range and gains |
| Command | uniform XYZ velocity command every 5 s, 5% all-zero and independent 5% XY-zero | same custom command sampler |
| Rewards | 18 nonzero source terms, including stateful filtered foot air time and gated jump/contact rewards | separate Jump equations and state variables; weights retain source dt scaling |
| Reset/termination | fixed root state, joint offset ±0.1, base force > 1 N, 24 s timeout | same reset ranges and termination threshold |
| Randomization | 256 friction buckets, mass/COM/gain/encoder perturbations, 4 s velocity overwrite | same bucket count/ranges and perturbation schedule |
| PPO | seed 1, 24 steps, 15k iterations, LR 1e-4, 512/256/128 ELU | same supported rsl_rl 5.4.2 settings |

Jump retains the same three cross-backend limitations listed for Trot: the
substep observation-latency hook and old-fork symmetry loss are not available,
and mass edits do not yet reproduce Isaac Gym's `recomputeInertia=True` exactly.

## Spring Jump parity table

| Concern | Isaac Gym source | mjlab implementation |
|---|---|---|
| Observation | `zeros(2)`, target-X/Y/jump flag, delayed IMU, delayed q/dq and action: 47 × 10; privileged 65 × 3 | Same source order and frame-major zero-filled history; critic retains all four contacts and `has_jumped` despite the source's stale “2 contacts” comment |
| Command/state | Target X ∈ [0.8, 1.2], fixed Y, a jump flag set on randomized policy frame 50–59 | Dedicated one-shot command term, with `was_in_flight`, `has_jumped`, first landing position and max-height state reset per environment |
| Rewards | 20 nonzero reward scales, including flight/landing gates and root-frame foot-clearance | Source equations and weights, including body-frame foot rotation and source signed 47-field mirror mapping applied independently to each of 10 frames |
| Reset/termination | fixed default joints/root state; base collision or z ≤ 0.15; 5 s timeout | Exact reset ranges and both termination conditions |
| Randomization/assist | 64 friction buckets, mass/COM/PD/encoder variations, 4 s velocity overwrite; probabilistic one-shot upward assist fading after 9,600 policy steps | Same native events and a state-machine assist using the source probability schedule |
| PPO | seed 1, 24 steps, 50k iterations, LR 1e-5, 512/256/128 ELU, source mirror loss | Same rsl_rl 5.4.2 settings and the existing gradient-compatible symmetry adapter |

The registered source calls `self.cfg.rewards.stance_reward_sigma` after a
landing, but neither its task config nor its inherited base config defines that
attribute. The migration makes its repair explicit as
`exp(-5 * abs(height - 0.35))`; this is the only Spring Jump formula that
cannot be mechanically copied from the checked-in source. Like the other
tasks, substep IMU/motor latency and Isaac-Gym inertia recomputation remain
backend limitations.

## Backflip parity table

| Concern | Isaac Gym source | mjlab implementation |
|---|---|---|
| Actor/critic observation | `[zeros(2), one-shot command, delayed angular velocity/projected gravity, delayed q/dq, action]`: 47 × 10; critic has the 50-field state frame × 3 | Same field order, scales, frame-major zero reset and actor-only uniform noise; projected gravity is retained rather than replaced by Euler angles |
| Command/state | Command begins `[0,0,0]`; the final flag turns to 1 at random policy frame 50–59; tracks filtered contact flight, first landing, and maximum absolute pitch rate | Dedicated no-velocity one-shot command and per-environment flight/landing/max-pitch state |
| Rewards | 20 nonzero terms: upward velocity, positive pitch rate before/during flight, height, rotation-complete orientation, origin landing, symmetry and penalties | Same weights, gates and source state variables; no symmetry-loss PPO extension because source `sym_loss=False` |
| Auxiliary force | On trigger, source gives an 80%-initially-probable upward velocity addition 2.0–3.5; after flight, gives an angular-Y addition 2.0–2.5; probability fades to zero after 9,600 reward steps | Same two one-shot state transitions and source probability schedule |
| Reset/termination | Exact nominal q/root reset; base contact or z ≤ 0.1; 4 s timeout | Same reset range, height and contact termination rules |
| Randomization | 64 friction buckets [0.2, 1.25], base/link mass, COM, gains, encoder offset and 4 s velocity overwrite | Same native events and parameter ranges |
| PPO | seed 1, 24 steps, 50k iterations, LR 1e-5, 512/256/128 ELU | Same rsl_rl 5.4.2-supported configuration |

As in the checked-in Spring Jump source, Backflip's post-landing term refers
to an undefined `stance_reward_sigma`. The migration makes its compatibility
repair explicit as `exp(-5 * abs(base_height - 0.35))`; it is the only
non-mechanical formula repair. The source's substep observation latency and
Isaac Gym `recomputeInertia=True` behavior likewise have no exact public
mjlab equivalent.

### Backflip runtime-validation result

The seed-1, 2,048-environment, 1,000-iteration run completed without a
numerical/runtime failure and saved `model_999.pt` under
`logs/rsl_rl/go2_backflip/2026-09-07_13-48-20_validation_2048x1000`; Viser
also started with that checkpoint. This is **not** an acceptance result: the
policy does not yet make a stable takeoff and landing, and the source base
contact termination resets some attempts during takeoff. Backflip is therefore
deliberately unregistered and remains incomplete; its implementation is kept
only for diagnosis and future source-faithful repair.

## Rear Stand parity table (Gym source: `go2_handstand`)

| Concern | Isaac Gym source | mjlab implementation |
|---|---|---|
| Actor observation | angular velocity, projected gravity, scaled command, relative q, dq and action; 45 × 1 | identical 45-field order/scaling and uniform per-field noise |
| Critic observation | body linear velocity + the already-noisy actor frame + 34 domain labels + 4 contacts; 86 × 1 | identical concatenation, including the duplicated restitution label |
| Action/control | default pose + `0.25 * action`; 0–3 substep switch delay resampled each policy step; Kp 40/Kd 1 and 90% effort limits | same mapping, delay schedule, gains and effort limits |
| Command | 10 s heading command, X in [-0.2, 0.6], Y zero; 20% all-zero and independent 10% XY-zero | custom sampler and source-specific heading controller with [-1, 1] yaw clipping |
| Rewards | 23 nonzero terms and a persistent batch-mean height gate at 0.70 | same formulas, ordering, weights, gate state and dt scaling |
| Reset/termination | q = default × U(0.5, 1.5), six root velocities in [-0.5, 0.5], base force > 1 N, 20 s timeout | matching reset events and termination threshold |
| Randomization | friction/restitution, mass/COM, gains/encoder, joint friction/damping/armature, 8 s velocity overwrite | matching ranges and privileged labels; native MuJoCo fields where available |
| PPO | seed 1, 24 steps, 15k iterations, LR 1e-3, 512/256/128 ELU; signed observation/action mirror loss with coefficient 1.0 and gradients through both branches | same settings/permutation through rsl_rl's symmetry extension plus a thin source-gradient adapter |

Rear Stand's action latency and PPO mirror loss are implemented. It retains the
mass-inertia limitation above. PhysX restitution has no one-to-one MuJoCo
scalar; its source sample and both critic-label entries are preserved, but
collision restitution dynamics cannot be claimed exact without a validated
`solref`/`solimp` mapping.

## Handstand parity table (Gym source: `go2_leggedstand`)

The source raises `RL_foot` and `RR_foot`, keeps `FL_foot` and `FR_foot` as the
support pair, and targets projected gravity `[1, 0, 0]`. Therefore its migrated
semantic name is Handstand rather than Legged Stand.

| Concern | Isaac Gym source | mjlab implementation |
|---|---|---|
| Actor/critic observation | successful bundled policy/checkpoint uses 48 actor fields: `zeros(2)`, constant-zero `stand_command`, then the later 45-field layout; critic is 3 + 48 + 34 + 4 = 89 | identical 48/89 dimensions, order, scaling, noise and duplicated restitution label; legacy constants retained because they define the trained artifact interface |
| Action/control | default pose + `0.25 * action` + motor zero offset; 0–3 substep switch delay each policy step, with zero `last_actions` after reset; Kp 40/Kd 1 and 90% effort limits | same target sign and delay schedule; reset rows are seeded with a zero-action delay frame rather than backfilled with their first new action |
| Command | every 5 s; X/Yaw in [-0.4, 0.4], Y fixed zero; 20% all-zero and independent 10% XY-zero; no heading control | same effective sampler; unused Gym heading range maps to `None` because mjlab rejects a heading range when heading control is disabled |
| Rewards | 22 nonzero terms in source order; batch-wide height gate uses `mean(exp(-10*error)) > 0.78`, while height reward uses exponent 5 | all 22 source terms retain the same formulas, signs, weights, ordering, stateful two-front-foot air time and dt scaling; the zero-initialized MuJoCo training adaptation below adds three explicit terms |
| Pose/contact semantics | rear feet target 0.67 m and off-ground; front feet alternate support; projected-gravity target `[1, 0, 0]` | same rear/front selection and target orientation |
| Reset/termination | q = default × U(0.5, 1.5), all root velocities in [-0.5, 0.5], base contact > 1 N, 20 s | same reset and termination rules |
| Randomization/push | friction/restitution, masses/COM, gains/encoder, joint friction/damping multipliers and armature; additive velocity push every 8 s | same ranges and labels; dedicated multiplier events and additive push |
| PPO | seed 1, 24 steps, 15k iterations, LR 1e-3, 512/256/128 ELU, `sym_loss=False` | same supported settings with no symmetry extension |

The checked-in Gym source was later changed to 45/86 and explicitly comments
that the three constant fields were deleted. Its bundled `policy_1.pt` and
`model_10600.pt` remain 48/89, proving that the successful training artifact
predates that edit. The migrated task follows the successful artifact contract.

The source config sets observation and command/action latency flags, but
`Go2_Leggedstand.py` never reads those fields. Only its independently implemented
0–3 physics-substep action switch delay is effective and therefore migrated.
The mass-inertia and restitution backend limitations described above still
apply.

### Handstand cross-backend validation

The Gym `model_10600.pt` policy was evaluated directly in both backends with
the recovered 48-field actor contract. It survived 400 policy steps in 128/128
PhysX environments. In MuJoCo it survived about half of randomized environments,
71% with startup domain randomization disabled, and 92% from the exact default
state with startup randomization disabled. This isolates the residual gap to
the PhysX-to-MuJoCo contact/initial-transient change rather than observation or
joint ordering.

Training only the source reward from a random actor converged to the source
task's zero-cost early-termination loophole. The accepted mjlab task therefore
adds an alive reward, a terminal cost, and a dense moving-target curriculum for
projected gravity, rear-foot height, and base height. The moving target reaches
the exact source objective after 400 PPO iterations and its auxiliary reward
fades completely to zero after 600 iterations.

The acceptance run started from seed-1 random network weights with 2048
environments and no resume, load-run, or checkpoint option. At iteration 799,
after 200 iterations with zero curriculum reward, mean episode length was
985/1000 steps, rear-foot height reward was 3.87, and base-height reward was
0.84. Deterministic playback confirmed a four-foot start followed by a stable,
front-foot-supported handstand. The run is stored under
`logs/rsl_rl/go2_handstand/2026-09-07_10-37-41_zero_init_complete_curriculum_2048x800`.

The Gym checkpoint experiment remains diagnostic evidence for backend parity;
it is not used to initialize or accept the mjlab policy.
