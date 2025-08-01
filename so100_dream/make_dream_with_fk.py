import json, cv2, math, numpy as np, pathlib, tqdm, os
import warnings

# Suppress numpy deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---- config ----------------------------------------------------------------
URDF_PATH        = 'SO100_urdf/so100.urdf'
JOINT_LOG        = '20episodes-frameobs.csv'
IMG_DIR          = pathlib.Path('images')
OUT_DIR          = pathlib.Path('dream_json')
OUT_DIR.mkdir(exist_ok=True)
LINK_NAMES       = ['base', 'shoulder', 'upper_arm', 'lower_arm', 'wrist', 'gripper', 'jaw']
# ---------------------------------------------------------------------------

# SO100 Robot DH Parameters (approximate - you may need to adjust these)
# Link lengths in meters
L1 = 0.1  # Base to shoulder
L2 = 0.2  # Shoulder to elbow  
L3 = 0.2  # Elbow to wrist
L4 = 0.1  # Wrist to gripper

def dh_matrix(theta, d, a, alpha):
    """Create DH transformation matrix"""
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    
    return np.array([
        [ct, -st*ca, st*sa, a*ct],
        [st, ct*ca, -ct*sa, a*st],
        [0, sa, ca, d],
        [0, 0, 0, 1]
    ])

def so100_forward_kinematics(q):
    """
    Forward kinematics for SO100 robot
    q: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
    Returns: dict of link poses in base frame
    """
    # Convert joint angles to radians
    q_rad = np.array(q[:5]) * np.pi / 180.0  # First 5 joints in degrees
    
    # DH parameters for SO100 (approximate)
    dh_params = [
        (q_rad[0], L1, 0, -np.pi/2),      # shoulder_pan
        (q_rad[1], 0, L2, 0),             # shoulder_lift  
        (q_rad[2], 0, L3, 0),             # elbow_flex
        (q_rad[3], 0, 0, -np.pi/2),       # wrist_flex
        (q_rad[4], L4, 0, 0),             # wrist_roll
    ]
    
    # Compute transformation matrices
    T = np.eye(4)
    transforms = {'base': T.copy()}
    
    for i, (theta, d, a, alpha) in enumerate(dh_params):
        T = T @ dh_matrix(theta, d, a, alpha)
        
        if i == 0:
            transforms['shoulder'] = T.copy()
        elif i == 1:
            transforms['upper_arm'] = T.copy()
        elif i == 2:
            transforms['lower_arm'] = T.copy()
        elif i == 3:
            transforms['wrist'] = T.copy()
        elif i == 4:
            transforms['gripper'] = T.copy()
            # Jaw is slightly offset from gripper
            jaw_offset = np.array([0, 0, 0.02, 1])
            transforms['jaw'] = T @ np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0.02], [0,0,0,1]])
    
    return transforms

# 1. load calibration
try:
    K_json = json.load(open('intrinsics.json'))
    fx, fy, cx, cy = K_json['camera_matrix'][0][0], K_json['camera_matrix'][1][1], \
                     K_json['camera_matrix'][0][2], K_json['camera_matrix'][1][2]
    K      = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=float)
    dist   = np.array(K_json['dist_coeffs'])   # keep if you'll undistort

    E_json = json.load(open('extrinsics.json'))
    R_cam2rob = np.array(E_json['rotation_mat'])     # 3×3
    t_cam2rob = np.array(E_json['translation_m'])    # 3,
    # make 4×4 homogeneous
    T_cam2rob        = np.eye(4)
    T_cam2rob[:3,:3] = R_cam2rob
    T_cam2rob[:3, 3] = t_cam2rob
    T_rob2cam = np.linalg.inv(T_cam2rob)
    CALIBRATION_AVAILABLE = True
    print("✅ Camera calibration loaded successfully")
except Exception as e:
    print(f"Warning: Camera calibration not available: {e}")
    CALIBRATION_AVAILABLE = False

# 3. read joint log (no header)
with open(JOINT_LOG) as f:
    joint_rows = [list(map(float, row.strip().split(','))) for row in f if row.strip()]

# 4. get sorted image paths
image_paths = sorted(IMG_DIR.glob('*.jpg'))  # Change to '*.png' if needed

if len(joint_rows) != len(image_paths):
    print(f"❌ Mismatch: {len(joint_rows)} joint states, {len(image_paths)} images.")
    exit(1)

print(f"Processing {len(joint_rows)} frames...")

for idx, (q, img_path) in enumerate(tqdm.tqdm(zip(joint_rows, image_paths), total=len(joint_rows), desc='frames')):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"❌ Could not read image {img_path}")
        continue

    # Compute forward kinematics
    try:
        T_links = so100_forward_kinematics(q)
        
        keypoints = {}
        for name in LINK_NAMES:
            if name in T_links:
                # Get 3D position in robot base frame
                p_rob = T_links[name][:3, 3]
                
                if CALIBRATION_AVAILABLE:
                    # Transform to camera frame
                    p_cam = (T_rob2cam @ np.append(p_rob, 1))[:3]
                    # Project to 2D
                    uv = (K @ p_cam)[:2] / p_cam[2]
                    
                    keypoints[name] = {
                        'location': p_cam.tolist(),
                        'projected_location': uv.tolist()
                    }
                else:
                    # Just use robot frame coordinates
                    keypoints[name] = {
                        'location': p_rob.tolist(),
                        'projected_location': [0.0, 0.0]  # No projection without calibration
                    }
            else:
                keypoints[name] = {
                    'location': [0.0, 0.0, 0.0],
                    'projected_location': [0.0, 0.0]
                }
    except Exception as e:
        print(f"Warning: Forward kinematics failed for frame {idx}: {e}")
        # Create dummy keypoints
        keypoints = {}
        for name in LINK_NAMES:
            keypoints[name] = {
                'location': [0.0, 0.0, 0.0],
                'projected_location': [0.0, 0.0]
            }

    # Create data structure
    data = {
        'objects':[{
            'class': 'so100',
            'location': keypoints['base']['location'],   # base link
            'keypoints': keypoints
        }],
        'sim_state': {
            'joint_names': ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper'],
            'position':    q,
        },
        'frame_index': idx
    }

    with open(OUT_DIR/f'{idx:06}.json', 'w') as fp:
        json.dump(data, fp, indent=2)

print('DONE – files in', OUT_DIR) 