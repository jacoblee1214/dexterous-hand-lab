# Idle oscillation diagnosis

The updated right and left URDFs were first built without collision exclusions.
At the reference pose MuJoCo immediately reported 12 right-hand and 13 left-hand
self contacts. The repeated offending body/geom pairs were:

- `Palm` ↔ `thumb_0`, `thumb_1`
- `Palm` ↔ `index_0`, `index_1`
- `Palm` ↔ `middle_0`
- `Palm` ↔ `ring_0`, `ring_1`
- `Palm` ↔ `pinky_0`, `pinky_1`

The CAD shells overlap around their mechanical pivots/base sleeves by roughly
1.3–27.7 mm. Treating every high-resolution visual mesh as an independent convex
collision hull made the solver repel mechanically connected parts. In the first
five-second baseline, maximum hand-joint speed reached 55.15 rad/s on the right
and 38.63 rad/s on the left; neither hand settled. This identified unintended
self collision as the cause, not GUI noise, gravity, actuator-target mismatch,
or insufficient solver stiffness.

The builder now excludes only parent-child body pairs and the palm-to-first-
flexion-link sleeve pairs. Object-to-hand and unrelated-link collisions remain
enabled. Position gains (`kp=2.0`, `kv=0.18`), joint damping (`0.08`), armature
(`0.001`), timestep (`0.002 s`), and the implicit-fast integrator were not
increased to mask the issue.

After the fix, the five-second `--idle-diagnostics` audit gives both hands:

| Scenario | initial max ctrl error | max contacts | max self contacts | final max abs qvel |
|---|---:|---:|---:|---:|
| Right, gravity on | 0 | 0 | 0 | 1.960e-16 rad/s |
| Right, gravity off | 0 | 0 | 0 | 0 rad/s |
| Left, gravity on | 0 | 0 | 0 | 1.963e-16 rad/s |
| Left, gravity off | 0 | 0 | 0 | 0 rad/s |

With gravity enabled, the position servos settle with a maximum static position
error of about 0.00704 rad. That is gravity-induced equilibrium offset rather
than periodic twitching. Visual no-twitch confirmation remains a manual test.
