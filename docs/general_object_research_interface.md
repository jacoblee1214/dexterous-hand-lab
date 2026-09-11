# General-object visuo-tactile interface and future motion hooks

The static portion of this interface is now implemented by General Object SDF
v1. Neural decoding, measured moving-object pose tracking, regrasp planning and
object-pose oracle inputs remain unimplemented.

## Object-centric observation contract

The general-object solver consumes a timestamped bundle containing calibrated RGB
features and rays; explicit foreground, background, unknown/occluded, and
unreliable visibility; named scalar tactile observations with finite contact
regions and uncertainty; named synchronized joint state; calibration versions;
and, when available, a **measured** object-pose estimate with covariance, source,
frame IDs, and timestamp. It must not require a known radius or object CAD mesh.

All transforms must state their direction. A sample accumulated over time must
be mapped from robot/world into one declared object frame using an object pose
valid at that sample's source timestamp. Missing, stale, unobservable, or
ambiguous pose makes temporal object-frame accumulation invalid. MuJoCo truth
may remain evaluator-only and cannot silently fill the pose field.

The existing camera calibration, visibility representation, named joint/tactile
preprocessing, finite sensor regions, uncertainty fields, timestamp
synchronization, replay provider, hardware read-only boundary, and separate
evaluation path can remain. Sphere-specific tangent-cone residuals, the fixed
radius prior, algebraic sphere-center initialization, center-only optimizer, and
sphere metrics must be replaced by an object representation, calibrated ray/
surface likelihoods, object-frame data association, and geometry-appropriate
regularization and metrics.

## Future motion experiment levels

The experiment manifest must distinguish:

1. **Static grasp:** object pose is fixed; historical observations may share the
   fixed object frame.
2. **Controlled reorientation:** an externally commanded, measured motion with a
   synchronized pose estimate and uncertainty maps every sample into object
   coordinates.
3. **Assisted regrasp:** an external fixture or approved controller changes
   contact while pose tracking remains independently observed.
4. **Free in-hand regrasp:** full six-degree object motion, contact changes, and
   possible slip require validated pose/contact tracking, safety supervision,
   and observability checks.

Each level needs source timestamps, pose latency/jitter, covariance, dropout and
reset behavior, frame conventions, motion-segment IDs, and an independent
evaluation reference. Advancing levels does not authorize motor control; real
actuation still requires the actual transport, limits, watchdog, emergency stop,
and a separate safety approval.
