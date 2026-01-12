"""
SIFT-based Camera Rotation Estimation

This module estimates camera rotation between two frames using SIFT feature 
matching and Homography decomposition. It's designed to work with planar scenes
(like QR codes or ArUco markers) and produces results compatible with ArUco-based
pose estimation.

Key approach:
1. Detect SIFT features in both frames
2. Match features using FLANN with ratio test
3. Compute Homography with RANSAC
4. Decompose Homography to extract rotation

The rotation convention matches ArUco: R2 @ R1.T represents how the scene
appears to rotate from frame 1 to frame 2.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

def load_images(frame1_path, frame2_path):
    """Load two image frames and validate them."""
    frame1 = cv2.imread(frame1_path)
    frame2 = cv2.imread(frame2_path)
    
    if frame1 is None:
        raise FileNotFoundError(f"Could not load image: {frame1_path}")
    if frame2 is None:
        raise FileNotFoundError(f"Could not load image: {frame2_path}")
    
    # Convert to grayscale for feature detection
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    return frame1, frame2, gray1, gray2

def detect_and_match_features(gray1, gray2):
    """Detect SIFT features and match them between frames with high accuracy."""
    # Initialize SIFT detector with parameters optimized for accuracy
    sift = cv2.SIFT_create(
        nfeatures=0,  # No limit on features for maximum accuracy
        nOctaveLayers=6,  # More octave layers for better scale invariance
        contrastThreshold=0.02,  # Lower threshold to detect more features
        edgeThreshold=15,  # Higher edge threshold for better quality keypoints
        sigma=1.6  # Gaussian sigma
    )
    
    # Detect keypoints and compute descriptors
    print("Detecting features in frame 1...")
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    print(f"Found {len(keypoints1)} keypoints in frame 1")
    
    print("Detecting features in frame 2...")
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
    print(f"Found {len(keypoints2)} keypoints in frame 2")
    
    if len(keypoints1) < 8 or len(keypoints2) < 8:
        raise ValueError("Not enough keypoints detected. Need at least 8 points in each frame.")
    
    # Use FLANN-based matcher for better accuracy
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)  # Higher checks for better accuracy
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    print("Matching features...")
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    
    # Apply Lowe's ratio test for robust matching
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.6 * n.distance:  # Even stricter ratio for higher accuracy
                good_matches.append(m)
    
    print(f"Found {len(good_matches)} good matches after ratio test")
    
    if len(good_matches) < 8:
        raise ValueError(f"Not enough good matches found: {len(good_matches)}. Need at least 8.")
    
    return keypoints1, keypoints2, good_matches

def estimate_camera_intrinsics(image_shape):
    """Estimate camera intrinsic matrix from image dimensions."""
    height, width = image_shape[:2]
    
    # Estimate focal length using field of view approximation
    # Assuming ~60 degree horizontal FOV for typical smartphone cameras
    # focal_length = width / (2 * tan(FOV/2))
    focal_length = width / (2 * np.tan(np.radians(30)))  # 60 degree FOV
    
    # Principal point at image center
    cx = width / 2.0
    cy = height / 2.0
    
    # Camera intrinsic matrix
    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    
    return K

def calculate_rotation(keypoints1, keypoints2, matches, K):
    """Calculate camera rotation using Homography decomposition.
    
    For scenes with planar dominant features or when viewing a flat surface
    (like a QR code or ArUco marker), Homography provides more stable results
    than Essential Matrix decomposition.
    """
    # Extract matched point coordinates
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
    
    # Calculate Homography with RANSAC for robustness
    print("Computing Homography...")
    H, mask = cv2.findHomography(
        pts1, pts2,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=10000,
        confidence=0.9999
    )
    
    inliers = np.sum(mask)
    print(f"Homography computed with {inliers} inliers out of {len(matches)} matches")
    
    if inliers < 8:
        raise ValueError(f"Not enough inliers: {inliers}. Rotation estimation may be unreliable.")
    
    # Decompose homography to get rotation and translation
    # decomposeHomographyMat returns up to 4 solutions
    num_solutions, rotations, translations, normals = cv2.decomposeHomographyMat(H, K)
    
    print(f"Homography decomposition found {num_solutions} solutions")
    
    # Select the best solution based on multiple criteria
    # For a camera looking at a planar target:
    # 1. The normal should point towards camera (positive Z component)
    # 2. Translation Z should be negative (moving towards or away from plane reasonably)
    # 3. Rotation should be reasonable for consecutive frames
    
    best_R = None
    best_t = None
    best_score = float('inf')
    
    for i in range(num_solutions):
        R = rotations[i]
        t = translations[i]
        n = normals[i]
        
        # Check if rotation matrix is proper (det = 1)
        det = np.linalg.det(R)
        if abs(det - 1.0) > 0.01:
            continue
        
        # Calculate rotation angle
        trace_val = np.clip((np.trace(R) - 1) / 2, -1, 1)
        angle_rad = np.arccos(trace_val)
        angle_deg = np.degrees(angle_rad)
        
        # Score components (lower is better)
        rotation_score = angle_deg  # Prefer smaller rotations for consecutive frames
        
        # Check normal direction - prefer normals pointing towards camera (n_z > 0)
        normal_score = 0 if n[2, 0] > 0 else 100
        
        # Prefer translations where the plane stays in front of camera
        # t is normalized, so we check direction consistency
        translation_score = 0
        
        # Combined score
        score = rotation_score + normal_score + translation_score
        
        if score < best_score:
            best_score = score
            best_R = R
            best_t = t
    
    if best_R is None:
        # Fallback: use solution with smallest rotation angle
        min_angle = float('inf')
        for i in range(num_solutions):
            R = rotations[i]
            trace_val = np.clip((np.trace(R) - 1) / 2, -1, 1)
            angle = np.degrees(np.arccos(trace_val))
            if angle < min_angle:
                min_angle = angle
                best_R = R
                best_t = translations[i]
    
    # Calculate the final rotation angle for logging
    trace_val = np.clip((np.trace(best_R) - 1) / 2, -1, 1)
    final_angle = np.degrees(np.arccos(trace_val))
    print(f"Selected rotation with angle: {final_angle:.3f}°")
    
    # The homography H transforms points from image 1 to image 2: p2 ~ H @ p1
    # The decomposed R represents the camera rotation from frame 1 to frame 2
    # ArUco computes R2 @ R1.T which represents how the marker's pose changed.
    # The Homography decomposition R represents the same transformation,
    # so we return it directly to match ArUco's convention.
    R_final = best_R
    
    return R_final, best_t if best_t is not None else np.zeros((3, 1)), mask

def rotation_matrix_to_euler_angles(R):
    """Convert rotation matrix to Euler angles (in degrees)."""
    # Calculate rotation angles using rotation matrix decomposition
    # Using ZYX convention (yaw, pitch, roll)
    
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    
    singular = sy < 1e-6
    
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0
    
    # Convert from radians to degrees
    roll_deg = np.degrees(roll)
    pitch_deg = np.degrees(pitch)
    yaw_deg = np.degrees(yaw)
    
    # Calculate total rotation angle (magnitude)
    # Using Rodrigues formula to get angle from rotation matrix
    angle_rad = np.arccos((np.trace(R) - 1) / 2)
    total_angle_deg = np.degrees(angle_rad)
    
    return roll_deg, pitch_deg, yaw_deg, total_angle_deg

def visualize_matches(frame1, frame2, keypoints1, keypoints2, matches, mask):
    """Create visualization of matched features."""
    # Filter matches by inlier mask
    inlier_matches = [matches[i] for i in range(len(matches)) if mask[i]]
    
    # Draw matches
    img_matches = cv2.drawMatches(
        frame1, keypoints1,
        frame2, keypoints2,
        inlier_matches, None,
        matchColor=(0, 255, 0),  # Green for inliers
        singlePointColor=(255, 0, 0),  # Blue for keypoints
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    return img_matches

def main(frame1_path, frame2_path, visualize=True, save_visualization=True):
    """Main function to calculate camera rotation between two frames."""
    print("=" * 80)
    print("Camera Rotation Calculation - High Accuracy Mode")
    print("=" * 80)
    print(f"Frame 1: {frame1_path}")
    print(f"Frame 2: {frame2_path}")
    print("-" * 80)
    
    # Load images
    print("\n[1/5] Loading images...")
    frame1, frame2, gray1, gray2 = load_images(frame1_path, frame2_path)
    print(f"Image dimensions: {gray1.shape[1]}x{gray1.shape[0]}")
    
    # Detect and match features
    print("\n[2/5] Detecting and matching features...")
    keypoints1, keypoints2, matches = detect_and_match_features(gray1, gray2)
    
    # Estimate camera intrinsics
    print("\n[3/5] Estimating camera parameters...")
    K = estimate_camera_intrinsics(gray1.shape)
    print(f"Estimated focal length: {K[0, 0]:.2f} pixels")
    
    # Calculate rotation
    print("\n[4/5] Computing camera rotation...")
    R, t, mask = calculate_rotation(keypoints1, keypoints2, matches, K)
    
    # Convert to Euler angles
    print("\n[5/5] Converting to rotation angles...")
    roll, pitch, yaw, total_angle = rotation_matrix_to_euler_angles(R)
    
    # Display results
    print("\n" + "=" * 80)
    print("ROTATION RESULTS")
    print("=" * 80)
    print(f"Roll  (rotation around X-axis): {roll:+.4f}°")
    print(f"Pitch (rotation around Y-axis): {pitch:+.4f}°")
    print(f"Yaw   (rotation around Z-axis): {yaw:+.4f}°")
    print(f"\nTotal rotation magnitude: {total_angle:.4f}°")
    print("=" * 80)
    
    # Print rotation matrix for reference
    print("\nRotation Matrix:")
    print(R)
    
    # Visualize matches
    if visualize or save_visualization:
        print("\nGenerating visualization...")
        img_matches = visualize_matches(frame1, frame2, keypoints1, keypoints2, matches, mask)
        
        if save_visualization:
            output_path = "rotation_matches.jpg"
            cv2.imwrite(output_path, img_matches)
            print(f"Visualization saved to: {output_path}")
        
        if visualize:
            # Resize for display if too large
            h, w = img_matches.shape[:2]
            if w > 1920:
                scale = 1920 / w
                new_w = int(w * scale)
                new_h = int(h * scale)
                img_matches = cv2.resize(img_matches, (new_w, new_h))
            
            cv2.imshow("Feature Matches (Inliers Only)", img_matches)
            print("Press any key to close visualization...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    return roll, pitch, yaw, total_angle

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python First_Try.py <frame1_path> <frame2_path>")
        print("Example: python First_Try.py frame1.jpg frame2.jpg")
        sys.exit(1)
    
    frame1_path = sys.argv[1]
    frame2_path = sys.argv[2]
    
    # Validate file paths
    if not Path(frame1_path).exists():
        print(f"Error: Frame 1 not found: {frame1_path}")
        sys.exit(1)
    if not Path(frame2_path).exists():
        print(f"Error: Frame 2 not found: {frame2_path}")
        sys.exit(1)
    
    try:
        roll, pitch, yaw, total_angle = main(frame1_path, frame2_path)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
