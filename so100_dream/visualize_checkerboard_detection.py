import cv2

image_path = 'checkerboard_images/000040.png'  # Change to any sample image
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Try both pattern orders with both detectors
patterns = [(6, 8), (8, 6)]
for pattern in patterns:
    # Classic detector
    ok, corners = cv2.findChessboardCorners(gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH)
    print(f"Pattern {pattern} found (classic):", ok)
    vis_img = img.copy()
    cv2.drawChessboardCorners(vis_img, pattern, corners, ok)
    cv2.imwrite(f'detected_{pattern[0]}x{pattern[1]}.png', vis_img)

    # SB detector
    ok_sb, corners_sb = cv2.findChessboardCornersSB(gray, pattern)
    print(f"Pattern {pattern} found (SB):", ok_sb)
    vis_img_sb = img.copy()
    cv2.drawChessboardCorners(vis_img_sb, pattern, corners_sb, ok_sb)
    cv2.imwrite(f'detected_SB_{pattern[0]}x{pattern[1]}.png', vis_img_sb) 