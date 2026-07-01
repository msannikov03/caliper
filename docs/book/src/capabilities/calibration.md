# Calibration

`caliper-calib` implements the verifiable core of kinematic calibration:
**joint-offset (zero) calibration**.

## The problem

A real robot's encoders read joint angles in a frame whose zero is offset from
the kinematic model's zero by an unknown constant vector `δ` (from mechanical
assembly, homing, or encoder mounting). Given observations
`{(commanded qₖ, measured tip pose Tₖ)}`, the crate estimates `δ` such that
`FK(qₖ + δ) ≈ Tₖ` for every observation.

## The method

It solves by **damped Gauss–Newton least squares**. For each observation the
residual is the body-frame error twist

```text
rₖ = log6( FK(qₖ + δ)⁻¹ · Tₖ )      ∈ se(3),   stored [v; ω]
```

At the true offset `δ*`, `FK(qₖ + δ*) = Tₖ`, the error pose is the identity, and
every `rₖ = 0`. Differentiating to first order gives `d rₖ / d δ = −J_b(qₖ + δ)`,
where `J_b` is the **LOCAL (body) geometric manipulator Jacobian** of the target
frame — exactly the Jacobian `caliper-kinematics` already computes. So
calibration reuses the same FK and Jacobian machinery the rest of the engine
depends on, which is why it inherits their (Pinocchio-validated) correctness for
the forward evaluation.

## Scope

This is the **joint-offset** slice of calibration — the part with a clean,
verifiable formulation. Fuller kinematic calibration (link-length / DH-parameter
identification, etc.) is not claimed here.
