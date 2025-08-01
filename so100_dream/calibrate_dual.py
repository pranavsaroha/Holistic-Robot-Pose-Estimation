import numpy as np
if not hasattr(np, 'float'):
    np.float = float
import cv2, csv, glob, json
from urdfpy import URDF
import os


# ---- CONFIG --------------------------------------------------------------
PATTERN        = (6, 8)      # inner corners (rows, cols)
SQUARE         = 0.030       # metres (30 mm)
URDF_PATH      = 'SO100_urdf/so100.urdf'
IMG_GLOB       = 'checkerboard_images/*.jpg'
JOINT_LOG      = 'initial-episode-states.csv'
LINK_TCP       = 'gripper'      # link name that holds the checkerboard
TCP_to_board   = np.eye(4)   # adjust if board offset from TCP
# -------------------------------------------------------------------------

print('[1] loading robot …')
robot = URDF.load(URDF_PATH)

def fk(joint_vals):
    cfg = dict(zip(robot.actuated_joint_names, joint_vals))
    return robot.link_fk(cfg)[robot.link_map[LINK_TCP]]

# ----- collect image points & object points ------------------------------
objp  = np.zeros((PATTERN[0]*PATTERN[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE

obj_pts, img_pts, robot_T_board = [], [], []

# load joints.csv into dict(frame → q list)
qdict = {}
with open(JOINT_LOG) as f:
    for idx, row in enumerate(csv.reader(f)):
        qdict[idx] = [float(val) for val in row]

image_paths = sorted(glob.glob(IMG_GLOB))
image_paths_valid = []
for idx, path in enumerate(image_paths):
    if idx not in qdict:
        print(f'skip {path} (no joints)')
        continue
    img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print(f'skip {path} (could not read image)')
        continue
    print(f"{path}: shape={img_gray.shape}, dtype={img_gray.dtype}")
    ok, corners = cv2.findChessboardCorners(img_gray, PATTERN, cv2.CALIB_CB_ADAPTIVE_THRESH)
    if not ok:
        print(f'skip {path} (no pattern)')
        continue
    cv2.cornerSubPix(img_gray, corners, (11,11), (-1,-1),
                     (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))

    obj_pts.append(objp)
    img_pts.append(corners)
    image_paths_valid.append(path)

    T_base_tcp   = fk(qdict[idx])           # base → TCP
    robot_T_board.append(T_base_tcp @ TCP_to_board)

print(f'[2] detected {len(obj_pts)} valid views')

# ----- intrinsic calibration ---------------------------------------------
print('[3] calibrating intrinsics …')
shape = img_gray.shape[::-1]       # (w,h)
N_intr = min(50, len(obj_pts))
np.random.seed(42)  # for reproducibility
indices = np.random.choice(len(obj_pts), N_intr, replace=False)
calib_obj_pts = [obj_pts[i] for i in indices]
calib_img_pts = [img_pts[i] for i in indices]

print(f"[3] calibrating intrinsics using {N_intr} randomly selected valid views …")

# Calibrate intrinsics
_, K, dist, _, _ = cv2.calibrateCamera(
    calib_obj_pts, calib_img_pts, shape, None, None)

instr = {
    'camera_matrix': K.tolist(),
    'dist_coeffs':   dist.ravel().tolist()
}
json.dump(instr, open('intrinsics.json', 'w'), indent=2)
print('    -> intrinsics.json written')

# ----- hand‑eye (extrinsic) ----------------------------------------------
print('[4] calibrating hand–eye …')

R_gripper2base = []
t_gripper2base = []
R_target2cam = []
t_target2cam = []

rvecs = []
tvecs = []
for i in range(len(obj_pts)):
    # robot_T_board[i] is base→board (4x4)
    T = robot_T_board[i]
    R = T[:3, :3]
    t = T[:3, 3]
    # Invert to get board→base (gripper→base)
    R_inv = R.T
    t_inv = -R.T @ t
    R_gripper2base.append(R_inv)
    t_gripper2base.append(t_inv)

    # Use solvePnP to get checkerboard pose in camera frame
    ret, rvec, tvec = cv2.solvePnP(obj_pts[i], img_pts[i], K, dist)
    rvecs.append(rvec)
    tvecs.append(tvec)
    R_cb2cam, _ = cv2.Rodrigues(rvec)
    t_cb2cam = tvec
    R_target2cam.append(R_cb2cam)
    t_target2cam.append(t_cb2cam)

R, t = cv2.calibrateHandEye(
    R_gripper2base, t_gripper2base,
    R_target2cam, t_target2cam,
    method=cv2.CALIB_HAND_EYE_TSAI)

extr = {
    'rotation_mat':  R.tolist(),
    'translation_m': t.flatten().tolist()
}
json.dump(extr, open('extrinsics.json', 'w'), indent=2)
print('    -> extrinsics.json written')

# quick quality check
mean_err = 0
for obj, img, rvec, tvec in zip(obj_pts, img_pts, rvecs, tvecs):
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    mean_err += cv2.norm(img, proj, cv2.NORM_L2) / len(proj)
print(f'[✓] mean reprojection error ≈ {mean_err/len(obj_pts):.3f} px')

# After collecting image_paths_valid (list of valid image paths):
with open('valid_images.txt', 'w') as f:
    for path in image_paths_valid:
        f.write(f"{os.path.basename(path)}\n")

