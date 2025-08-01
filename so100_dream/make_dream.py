import json, cv2, math, numpy as np, pathlib, tqdm, os
from urdfpy import URDF

# ---- config ----------------------------------------------------------------
URDF_PATH        = 'SO100_urdf/so100.urdf'
JOINT_LOG        = '20episodes-frameobs.csv'
IMG_DIR          = pathlib.Path('images')
OUT_DIR          = pathlib.Path('dream_json')
OUT_DIR.mkdir(exist_ok=True)
LINK_NAMES       = ['base', 'shoulder', 'upper_arm', 'lower_arm', 'wrist', 'gripper', 'jaw']
# ---------------------------------------------------------------------------

# 1. load calibration
K_json = json.load(open('intrinsics.json'))
fx, fy, cx, cy = K_json['camera_matrix'][0][0], K_json['camera_matrix'][1][1], \
                 K_json['camera_matrix'][0][2], K_json['camera_matrix'][1][2]
K      = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=float)
dist   = np.array(K_json['dist_coeffs'])   # keep if you’ll undistort

E_json = json.load(open('extrinsics.json'))
R_cam2rob = np.array(E_json['rotation_mat'])     # 3×3
t_cam2rob = np.array(E_json['translation_m'])    # 3,
# make 4×4 homogeneous
T_cam2rob        = np.eye(4)
T_cam2rob[:3,:3] = R_cam2rob
T_cam2rob[:3, 3] = t_cam2rob
T_rob2cam = np.linalg.inv(T_cam2rob)

# 2. load robot model
robot = URDF.load(URDF_PATH)

def fk(q):
    """dict(link → 4×4 pose) in robot base frame"""
    return robot.link_fk(cfg={j:float(q[i]) for i,j in enumerate(robot.actuated_joint_names)})

# 3. read joint log (no header)
with open(JOINT_LOG) as f:
    joint_rows = [list(map(float, row.strip().split(','))) for row in f if row.strip()]

# 4. get sorted image paths
image_paths = sorted(IMG_DIR.glob('*.jpg'))  # Change to '*.png' if needed

if len(joint_rows) != len(image_paths):
    print(f"❌ Mismatch: {len(joint_rows)} joint states, {len(image_paths)} images.")
    exit(1)

for idx, (q, img_path) in enumerate(tqdm.tqdm(zip(joint_rows, image_paths), total=len(joint_rows), desc='frames')):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"❌ Could not read image {img_path}")
        continue

    # FK → link poses
    T_links = fk(q)

    keypoints = {}
    for name in LINK_NAMES:
        p_rob = T_links[robot.link_map[name]][:3,3]                 # 3×1
        p_cam = (T_rob2cam @ np.append(p_rob,1))[:3]                # to cam
        uv    = (K @ p_cam)[:2] / p_cam[2]                          # project

        keypoints[name] = {
            'location':            p_cam.tolist(),
            'projected_location':  uv.tolist()
        }

    data = {
        'objects':[{
            'class': 'so100',
            'location': keypoints['base']['location'],   # base link
            'keypoints': keypoints
        }],
        'sim_state': {
            'joint_names': robot.actuated_joint_names,
            'position':    q,
        },
        'frame_index': idx
    }

    with open(OUT_DIR/f'{idx:06}.json', 'w') as fp:
        json.dump(data, fp, indent=2)

print('DONE – files in', OUT_DIR)
