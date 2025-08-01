import cv2
import pathlib

img_dir = pathlib.Path('images')
for png_path in img_dir.glob('*.png'):
    img = cv2.imread(str(png_path))
    if img is not None:
        jpg_path = png_path.with_suffix('.jpg')
        cv2.imwrite(str(jpg_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        print(f'Converted {png_path} -> {jpg_path}')
    else:
        print(f'Could not read {png_path}') 