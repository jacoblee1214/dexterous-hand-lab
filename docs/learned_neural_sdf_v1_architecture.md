# Lightweight Learned Visuo-Tactile Neural SDF v1

This milestone is a preliminary, single-training-seed experiment. It asks
whether sparse scalar tactile geometry improves a compact RGB-conditioned
implicit surface specifically where the hand occludes the object. It does not
claim arbitrary-object reconstruction or edge deployment.

## Information and coordinate boundary

Inference receives one synchronized `GeneralObjectObservation`: clean Research
RGB, four input visibility masks, camera intrinsics and `camera_from_object`,
named joint observations, named scalar tactile readings, calibrated indentation,
finite sensor patches, sensing directions, uncertainty, coverage, timestamps,
and an object-frame initialization. The v1 decoder uses the image/camera and
derived tactile tensors; the raw joint vector and timestamp are retained by the
interface as future kinematics/temporal hooks but are not decoder features.
Object family, instance ID, dimensions,
radius, CAD, GT SDF/surface, true MuJoCo contact point/normal, and evaluation
region labels are kept in a separate supervision dictionary and are absent from
`NeuralSDFModel.forward`.

The benchmark object frame is **not an inferred free-object pose**. It is
`declared_static_object` in `world`, with source
`declared_static_fixture`. `world_from_object` is the declared static fixture
transform. Each sensor's permitted transform is composed from the declared
frame and calibrated/kinematic sensor geometry to form
`T_object_from_sensor`. Simulation truth generates labels only. This static
fixture assumption must later be replaced by a measured tracker transform
before object reorientation; observations from different poses are not fused in
v1.

## Dataset and split

`experiments/neural_sdf/protocol_v1.yaml` deterministically creates 84 object
instances and 432 observations:

- train: 48 instances / 288 observations from sphere, cylinder, cuboid, and
  rounded box;
- validation: 12 disjoint instances / 48 observations from the same families;
- held-out test: 16 disjoint seen-family instances plus 8 asymmetric-L
  instances / 96 observations total. Asymmetric L is absent from train and
  validation.

Instances vary in dimensions/aspect ratio, declared orientation, occlusion,
appearance, lighting, contact count (0/1/2/4/8), and clustered versus diverse
contact placement. Training also applies whole-modality and per-contact dropout.
Test dimensions extend beyond the training ranges declared in the frozen
protocol. The manifest is immutable and SHA-256 checked. Family, instance,
parameters, filename, and sample order never enter a model tensor.

## Model

Let an object-frame query be `x`. The camera projection is

`p_c = T_camera_from_object [x, 1]^T`,

`u = fx p_cx/p_cz + cx`, `v = fy p_cy/p_cz + cy`.

The clean RGB image is normalized with ImageNet statistics and passed through a
frozen, hash-verified ResNet-18 through `layer2`. Its 128 channels are reduced by
a fixed group mean to a 16-channel 28×28 map. The local visual vector is sampled
at `(u,v)` by differentiable bilinear interpolation. Nearest-sampled input masks
encode object, known background, hand-occluded/unknown, and unreliable pixels;
an additional flag marks out-of-view queries. A normalized camera ray and depth
are supplied. Hand-occluded pixels are never changed into known background.

For tactile observation `i`, the global element encoder receives calibrated
representative position, sensing direction, scalar magnitude, estimated
indentation, region sigma, and stable named sensor/finger embeddings. A shared
MLP produces `h_i`. The set representation is permutation invariant:

`h_global = 0.5 (mean_i h_i + max_i h_i)` over valid observations.

For local fusion, `r_i=x-p_i`, representative distance `d_i=||r_i||`, normal
alignment, scalar/indentation/sigma, and a near-contact weight are computed. In
finite-patch modes,

`d_patch_i(x) = min_{p in calibrated patch_i} ||x-p||`;

otherwise `d_patch_i=d_i`. A shared local MLP is aggregated with

`a_i(x)=exp[-0.5 (d_patch_i/(0.012+sigma_i))^2]`.

L4 additionally multiplies `a_i` by the observable-only uncertainty factor
`1/(1+(sigma_i/0.014)^2)` and spatial coverage. No GT error is used by the gate.
The final compact MLP receives Fourier-encoded `x` (three frequency bands), RGB,
visibility/ray features, global and local tactile features, and coverage, and
predicts `f_theta(x)` in metres. Each learned head has 27,889 trainable
parameters; the frozen visual backbone is replaceable.

## Ablations

- L0: learned RGB only;
- L1: RGB plus global permutation-invariant tactile feature;
- L2: RGB plus representative-contact local feature and representative loss;
- L3: RGB plus finite-patch local feature and finite-patch loss;
- L4: L3 plus observable uncertainty/coverage gating.

All five heads have the same decoder capacity and training schedule. Thus L0–L4
are feature/constraint ablations rather than capacity comparisons.

## Training objective

The frozen schedule uses 128 queries per observation: 25% sampled near visible
GT surface, 25% near hand-occluded GT surface, and 50% uniform in the object
volume. Region labels stratify training/evaluation only and never enter the
network. The objective is

`L = SmoothL1_beta=3mm(f_theta(x), sdf_GT(x))`
`  + 0.2 mean_{s in GT surface}|f_theta(s)|`
`  + 0.1 L_contact`.

For L1/L2, `L_contact` evaluates the permitted representative estimate. For
L3/L4 it uses all 25 calibrated patch samples and the differentiable soft
minimum

`-tau log mean_j exp(-|f_theta(p_ij)|/tau)`, `tau=2 mm`.

It never selects a patch point using the MuJoCo true contact. L4 weights this
term only by observable sigma and coverage. The contact term is evaluated every
four minibatches for the fixed CPU budget. Eikonal loss was deliberately not
added in v1: signed-distance supervision already constrains field values, and
the experiment keeps the important tactile-loss comparison transparent.

Training uses AdamW (`lr=1e-3`, weight decay `1e-4`), batch 16, gradient clipping
at 2, five epochs, 20% full tactile dropout, and 20% per-contact dropout. The
best checkpoint is selected solely by mean absolute predicted SDF on validation
hand-occluded GT surface points. The held-out test is opened only afterward.

## Physical interpretation and remaining sim-only assumptions

Scalar tactile is a magnitude-like calibrated response, not a measured 3D force
vector. Representative position, indentation, finite patch, and uncertainty are
calibrated estimates. The model interface is hardware-shaped, but v1 still uses
procedural RGB/masks, a declared static fixture frame, simulated kinematics and
calibrated-contact estimates, and synthetic supervision. No real-hardware
validation, temporal fusion, regrasp, pose tracking, motor control, or deployment
claim is made.
