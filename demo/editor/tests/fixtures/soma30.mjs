// soma30 /api/models entry used by the tests. names/parents/offsets mirror
// demo/skeletons_extra.go (NVIDIA Kimodo, Apache-2.0); hips/chains follow
// upstream kimodo/skeleton/definitions.py.
export const SOMA30_MODEL = Object.freeze({
  id: 'soma-rp-v1.1',
  skeleton_key: 'soma30',
  names: ['Hips', 'Spine1', 'Spine2', 'Chest', 'Neck1', 'Neck2', 'Head', 'Jaw', 'LeftEye', 'RightEye',
    'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftHandThumbEnd', 'LeftHandMiddleEnd',
    'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RightHandThumbEnd', 'RightHandMiddleEnd',
    'LeftLeg', 'LeftShin', 'LeftFoot', 'LeftToeBase', 'RightLeg', 'RightShin', 'RightFoot', 'RightToeBase'],
  parents: [-1, 0, 1, 2, 3, 4, 5, 6, 6, 6, 3, 10, 11, 12, 13, 13, 3, 16, 17, 18, 19, 19, 0, 22, 23, 24, 0, 26, 27, 28],
  offsets: [[0, 0, 0], [-0.00013727, 0.0500376256, -0.00053726669], [-1.86574103e-9, 0.0712530139, -0.000298248546],
    [-5.75188398e-9, 0.0755006305, -0.00815970992], [-0.00181676517, 0.263112953, -0.00553348292],
    [-2.85102231e-8, 0.0770939664, 0.0230258546], [-4.5975437e-8, 0.0612891595, 0.0195370861],
    [2.63687901e-5, 0.0047559225, 0.0309494062], [0.0320638079, 0.0538020513, 0.0758688308],
    [-0.0322244017, 0.05361869, 0.0755823359], [0.0162165175, 0.232371641, 0.0511341324],
    [0.149198457, 2.19397873e-8, -0.0550232576], [0.287393078, 2.50268389e-9, -2.58787737e-5],
    [0.270939812, -7.06625108e-9, 2.60897248e-5], [0.122686267, -0.0322017573, 0.0483306876],
    [0.190119595, -0.00312878387, -0.000339570373], [-0.0138011824, 0.231803086, 0.0521415786],
    [-0.150371962, 1.17387901e-7, -0.0554560437], [-0.287366393, 1.87628082e-8, -2.59709359e-5],
    [-0.271336198, -1.16767401e-9, 2.61269368e-5], [-0.122642483, -0.0321145448, 0.0480403904],
    [-0.190005945, -0.00306615542, -0.0003157343], [0.10043214, -0.0843452671, 0.0259565473],
    [-1e-8, -0.432217537, -0.00802912805], [1e-8, -0.421550959, -0.0348152298],
    [0, -0.0505947206, 0.132315294], [-0.10047278, -0.0829525995, 0.0262031695],
    [1e-8, -0.433622059, -0.00805555828], [2e-8, -0.421173943, -0.0347839785],
    [-3.42907669e-9, -0.0507960932, 0.132841956]],
  root_index: 0,
  hips: [26, 22],
  chains: {
    LeftHand: ['LeftHand', 'LeftHandMiddleEnd'],
    RightHand: ['RightHand', 'RightHandMiddleEnd'],
    LeftFoot: ['LeftFoot', 'LeftToeBase'],
    RightFoot: ['RightFoot', 'RightToeBase'],
  },
});

// A small synthetic skeleton for hand-checkable FK:
//   0 root, 1 child (+1 along x), 2 grandchild (+1 along y), 3 second child of root (-1 along x).
export const TOY_MODEL = Object.freeze({
  skeleton_key: 'toy4',
  names: ['root', 'a', 'b', 'c'],
  parents: [-1, 0, 1, 0],
  offsets: [[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0]],
  root_index: 0,
  hips: ['c', 'a'], // right = c (-x), left = a (+x): rest heading 0 (facing +Z)
  chains: { LeftHand: ['b'], RightHand: ['c'], LeftFoot: ['a', 'b'], RightFoot: ['c'] },
});
