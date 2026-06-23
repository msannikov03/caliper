import * as THREE from "three";

/**
 * URDF is Z-up; three.js / the drei <Grid> floor is Y-up (the XZ plane).
 *
 * We render the whole robot inside ONE group whose matrix is `DISPLAY_UP`, a
 * −90° rotation about X that maps URDF +Z → three +Y. Engine (get_frames) poses
 * are in URDF world; we set them as each frame group's LOCAL matrix, so a frame's
 * three-world matrix becomes `DISPLAY_UP · T_urdf` and it sits correctly on the
 * Y-up floor.
 *
 * The IK gizmo lives inside this SAME group, so:
 *   - the controlled `matrix` we feed it is the tip pose in URDF world (T_tip),
 *   - the world matrix it reports on drag is `DISPLAY_UP · T_target`,
 *   - so to recover the URDF-world target we left-multiply by `DISPLAY_UP⁻¹`.
 * Keeping the engine in pure URDF coordinates is what makes IK convention-safe.
 */
export const DISPLAY_UP = new THREE.Matrix4().makeRotationX(-Math.PI / 2);
export const DISPLAY_UP_INV = DISPLAY_UP.clone().invert();
