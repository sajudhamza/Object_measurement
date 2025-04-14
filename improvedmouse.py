import cv2
import numpy as np
from ultralytics import YOLO

# === STEP 1: Camera Calibration Parameters (replace with actual ZED values) ===
focal_length_px = 306.0  # Replace with fx from ZED Explorer or pyzed.sl
baseline_m = 0.12        # Replace with ZED baseline in meters

# === STEP 2: Load the stitched stereo image ===
image = cv2.imread("Test72.png")
if image is None:
    print("❌ Image not found. Check path.")
    exit()

# === STEP 3: Split into left and right images ===
height, width, _ = image.shape
half_width = width // 2
left_img = image[:, :half_width].copy()
right_img = image[:, half_width:]

# === STEP 4: Convert to grayscale ===
gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

# === STEP 5: Preprocess: Contrast enhance + blur ===
clahe = cv2.createCLAHE(clipLimit=3.0)
gray_left = clahe.apply(gray_left)
gray_right = clahe.apply(gray_right)

gray_left = cv2.GaussianBlur(gray_left, (5, 5), 0)
gray_right = cv2.GaussianBlur(gray_right, (5, 5), 0)

# === STEP 6: Create StereoSGBM matcher ===
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16 * 5,
    blockSize=5,
    P1=8 * 3 * 5 ** 2,
    P2=32 * 3 * 5 ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# === STEP 7: Compute disparity map ===
disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

# === STEP 8: Load your custom YOLO model ===
model = YOLO("/Users/cyril/Downloads/Pace Phd/output/mouse-dateset/runs/detect/train5/weights/best.pt")

# === STEP 9: Detect the mouse in the image ===
results = model(left_img)
mouse_detected = False

# === STEP 10: Get bounding box of the detected mouse ===
for result in results[0].boxes:
    xyxy = result.xyxy[0].numpy()
    conf = result.conf[0].item()
    cls = result.cls[0].item()

    mouse_detected = True
    x1, y1, x2, y2 = map(int, xyxy)
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    cv2.rectangle(left_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.circle(left_img, (center_x, center_y), 5, (0, 0, 255), 3)

    print(f"✅ Mouse detected at ({center_x}, {center_y}) with confidence: {conf:.3f}")
    break

if not mouse_detected:
    print("❌ No mouse detected.")
else:
    # === STEP 11: Get depth using average disparity from 11x11 patch ===
    x, y = center_x, center_y
    patch = disparity[max(0, y - 5): y + 6, max(0, x - 5): x + 6]
    valid_disp = patch[patch > 0]

    if valid_disp.size > 0:
        disparity_value = np.mean(valid_disp)
        depth = (focal_length_px * baseline_m) / disparity_value
        print(f"✅ Depth of mouse at ({x}, {y}): {depth:.3f} meters (Disparity: {disparity_value:.2f})")
    else:
        print(f"❌ Invalid disparity at ({x}, {y}) — no valid values in patch.")
        print(f"Patch shape: {patch.shape}, min: {patch.min():.2f}, max: {patch.max():.2f}")

# === STEP 12: Display outputs ===
cv2.imshow("Detected Mouse", left_img)

# Show disparity for debugging
disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
disp_vis = np.uint8(disp_vis)
cv2.imshow("Disparity Map", disp_vis)
cv2.imwrite("disparity_debug.png", disp_vis)

cv2.waitKey(0)
cv2.destroyAllWindows()


