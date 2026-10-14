// SOMA 77-joint skeleton data needed to IMPORT upstream constraints.json files
// written with 77-joint `local_joints_rot` (upstream save converts SOMA30 to 77,
// kimodo/constraints.py). From NVIDIA Kimodo (Apache-2.0)
// kimodo/skeleton/definitions.py (SOMASkeleton77 bone order).
//
// SOMASkeleton30.from_SOMASkeleton77(L77) == L77[:, SOMA30_FROM_77]: a plain
// selection of LOCAL rotations by joint name. Export always writes the 30-joint
// control skeleton, so the relaxed-hands fill needed for 30 -> 77 is not here.
//
// Exports: SOMA77_NAMES, SOMA30_FROM_77, isSoma30(skeleton), soma77ToSoma30(rows77).

export const SOMA77_NAMES = Object.freeze([
  'Hips', 'Spine1', 'Spine2', 'Chest', 'Neck1', 'Neck2',
  'Head', 'HeadEnd', 'Jaw', 'LeftEye', 'RightEye', 'LeftShoulder',
  'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3',
  'LeftHandThumbEnd', 'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3', 'LeftHandIndex4', 'LeftHandIndexEnd',
  'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3', 'LeftHandMiddle4', 'LeftHandMiddleEnd', 'LeftHandRing1',
  'LeftHandRing2', 'LeftHandRing3', 'LeftHandRing4', 'LeftHandRingEnd', 'LeftHandPinky1', 'LeftHandPinky2',
  'LeftHandPinky3', 'LeftHandPinky4', 'LeftHandPinkyEnd', 'RightShoulder', 'RightArm', 'RightForeArm',
  'RightHand', 'RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3', 'RightHandThumbEnd', 'RightHandIndex1',
  'RightHandIndex2', 'RightHandIndex3', 'RightHandIndex4', 'RightHandIndexEnd', 'RightHandMiddle1', 'RightHandMiddle2',
  'RightHandMiddle3', 'RightHandMiddle4', 'RightHandMiddleEnd', 'RightHandRing1', 'RightHandRing2', 'RightHandRing3',
  'RightHandRing4', 'RightHandRingEnd', 'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3', 'RightHandPinky4',
  'RightHandPinkyEnd', 'LeftLeg', 'LeftShin', 'LeftFoot', 'LeftToeBase', 'LeftToeEnd',
  'RightLeg', 'RightShin', 'RightFoot', 'RightToeBase', 'RightToeEnd',
]);

/** For each SOMA30 joint (in soma30 order), its index in the 77-joint skeleton. */
export const SOMA30_FROM_77 = Object.freeze([
  0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 18, 28, 39, 40, 41, 42, 46, 56, 67, 68, 69, 70, 72, 73, 74, 75,
]);

/** True when `skeleton` is the 30-joint SOMA control skeleton. */
export function isSoma30(skeleton) {
  return skeleton.J === 30 && (skeleton.key === 'soma30' || skeleton.key === 'somaskel30');
}

/** Selects the 30 control-joint rows (any per-joint values) out of 77 rows. */
export function soma77ToSoma30(rows77) {
  if (rows77.length !== 77) throw new Error(`soma77ToSoma30: expected 77 joints, got ${rows77.length}`);
  return SOMA30_FROM_77.map((i) => rows77[i]);
}
