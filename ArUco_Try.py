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
    
    return frame1, frame2

def estimate_camera_intrinsics(image_shape):
    """Estimate camera intrinsic matrix from image dimensions."""
    height, width = image_shape[:2]
    
    # Estimate focal length using field of view approximation
    # Assuming ~60 degree horizontal FOV for typical smartphone cameras
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
    
    # Assume no distortion for simplicity
    dist_coeffs = np.zeros((5, 1))
    
    return K, dist_coeffs

def detect_aruco_markers(frame, aruco_dict, parameters):
    """Detect ArUco markers in a frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect markers
    corners, ids, rejected = cv2.aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters
    )
    
    return corners, ids, rejected

def estimate_marker_pose(corners, marker_size, camera_matrix, dist_coeffs):
    """Estimate the pose of a marker."""
    # Define 3D points of the marker corners in marker coordinate system
    # Marker is at z=0, with corners at (+/- size/2, +/- size/2, 0)
    object_points = np.array([
        [-marker_size/2, marker_size/2, 0],
        [marker_size/2, marker_size/2, 0],
        [marker_size/2, -marker_size/2, 0],
        [-marker_size/2, -marker_size/2, 0]
    ], dtype=np.float32)
    
    # Estimate pose using solvePnP with iterative refinement
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        corners,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE  # Iterative method for better accuracy
    )
    
    # Refine the pose estimation
    if success:
        rvec, tvec = cv2.solvePnPRefineLM(object_points, corners, camera_matrix, dist_coeffs, rvec, tvec)
    
    return rvec, tvec

def rotation_vector_to_euler_angles(rvec):
    """Convert rotation vector to Euler angles in degrees."""
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    
    # Extract Euler angles (ZYX convention)
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
    angle_rad = np.linalg.norm(rvec)
    total_angle_deg = np.degrees(angle_rad)
    
    return roll_deg, pitch_deg, yaw_deg, total_angle_deg

def calculate_rotation_difference(rvec1, rvec2):
    """Calculate the rotation difference between two rotation vectors.
    
    This computes how the marker's pose changed from frame 1 to frame 2.
    R_diff = R2 @ R1.T represents the rotation to transform from orientation 1 to orientation 2.
    """
    # Convert rotation vectors to rotation matrices
    R1, _ = cv2.Rodrigues(rvec1)
    R2, _ = cv2.Rodrigues(rvec2)
    
    # Calculate relative rotation: R_diff = R2 * R1^T
    # This represents how the marker appears to have rotated from frame 1 to frame 2
    R_diff = R2 @ R1.T
    
    # Convert back to rotation vector
    rvec_diff, _ = cv2.Rodrigues(R_diff)
    
    return R_diff, rvec_diff

def visualize_markers(frame1, frame2, corners1, ids1, corners2, ids2, 
                     camera_matrix, dist_coeffs, marker_size):
    """Visualize detected ArUco markers with axes."""
    vis1 = frame1.copy()
    vis2 = frame2.copy()
    
    if ids1 is not None and len(ids1) > 0:
        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(vis1, corners1, ids1)
        
        # Draw axes for each marker
        for i in range(len(ids1)):
            rvec, tvec = estimate_marker_pose(
                corners1[i][0], marker_size, camera_matrix, dist_coeffs
            )
            cv2.drawFrameAxes(vis1, camera_matrix, dist_coeffs, rvec, tvec, marker_size * 0.5)
    
    if ids2 is not None and len(ids2) > 0:
        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(vis2, corners2, ids2)
        
        # Draw axes for each marker
        for i in range(len(ids2)):
            rvec, tvec = estimate_marker_pose(
                corners2[i][0], marker_size, camera_matrix, dist_coeffs
            )
            cv2.drawFrameAxes(vis2, camera_matrix, dist_coeffs, rvec, tvec, marker_size * 0.5)
    
    # Combine images side by side
    combined = np.hstack([vis1, vis2])
    
    return combined

def main(frame1_path, frame2_path, marker_size=0.1, aruco_dict_type='DICT_6X6_250'):
    """Main function to calculate rotation using ArUco markers."""
    print("=" * 80)
    print("ArUco Marker Rotation Calculation - High Accuracy Mode")
    print("=" * 80)
    print(f"Frame 1: {frame1_path}")
    print(f"Frame 2: {frame2_path}")
    print(f"ArUco Dictionary: {aruco_dict_type}")
    print(f"Marker Size: {marker_size} meters")
    print("-" * 80)
    
    # Load images
    print("\n[1/5] Loading images...")
    frame1, frame2 = load_images(frame1_path, frame2_path)
    print(f"Image dimensions: {frame1.shape[1]}x{frame1.shape[0]}")
    
    # Estimate camera parameters
    print("\n[2/5] Estimating camera parameters...")
    camera_matrix, dist_coeffs = estimate_camera_intrinsics(frame1.shape)
    print(f"Estimated focal length: {camera_matrix[0, 0]:.2f} pixels")
    
    # Initialize ArUco detector
    print("\n[3/5] Detecting ArUco markers...")
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_dict_type))
    parameters = cv2.aruco.DetectorParameters()
    
    # Optimize detection parameters for accuracy
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 23
    parameters.adaptiveThreshWinSizeStep = 10
    parameters.adaptiveThreshConstant = 7
    parameters.minMarkerPerimeterRate = 0.03
    parameters.maxMarkerPerimeterRate = 4.0
    parameters.polygonalApproxAccuracyRate = 0.03  # Higher accuracy for corner detection
    parameters.minCornerDistanceRate = 0.05
    parameters.minDistanceToBorder = 3
    parameters.minMarkerDistanceRate = 0.05
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # Subpixel refinement
    parameters.cornerRefinementWinSize = 5
    parameters.cornerRefinementMaxIterations = 30
    parameters.cornerRefinementMinAccuracy = 0.01
    
    # Detect markers in both frames
    corners1, ids1, rejected1 = detect_aruco_markers(frame1, aruco_dict, parameters)
    corners2, ids2, rejected2 = detect_aruco_markers(frame2, aruco_dict, parameters)
    
    if ids1 is None or len(ids1) == 0:
        print(f"❌ No ArUco markers detected in frame 1!")
        print(f"   Rejected candidates: {len(rejected1)}")
        print("\nTip: Make sure you have ArUco markers in your images.")
        print("     If you have QR codes instead, use First_Try.py for feature-based tracking.")
        return None
    
    if ids2 is None or len(ids2) == 0:
        print(f"❌ No ArUco markers detected in frame 2!")
        print(f"   Rejected candidates: {len(rejected2)}")
        return None
    
    print(f"✓ Detected {len(ids1)} markers in frame 1: {ids1.flatten().tolist()}")
    print(f"✓ Detected {len(ids2)} markers in frame 2: {ids2.flatten().tolist()}")
    
    # Find common markers between frames
    common_ids = set(ids1.flatten()) & set(ids2.flatten())
    
    if len(common_ids) == 0:
        print("❌ No common markers found between frames!")
        return None
    
    print(f"✓ Found {len(common_ids)} common marker(s): {list(common_ids)}")
    
    # Calculate rotation for each common marker
    print("\n[4/5] Calculating marker poses and rotations...")
    
    all_rotations = []
    
    for marker_id in common_ids:
        # Get corners for this marker in both frames
        idx1 = np.where(ids1 == marker_id)[0][0]
        idx2 = np.where(ids2 == marker_id)[0][0]
        
        corners_1 = corners1[idx1][0]
        corners_2 = corners2[idx2][0]
        
        # Estimate pose in both frames
        rvec1, tvec1 = estimate_marker_pose(corners_1, marker_size, camera_matrix, dist_coeffs)
        rvec2, tvec2 = estimate_marker_pose(corners_2, marker_size, camera_matrix, dist_coeffs)
        
        # Calculate rotation difference
        R_diff, rvec_diff = calculate_rotation_difference(rvec1, rvec2)
        
        # Convert to Euler angles
        roll, pitch, yaw, total_angle = rotation_vector_to_euler_angles(rvec_diff)
        
        all_rotations.append({
            'id': marker_id,
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
            'total': total_angle,
            'rvec1': rvec1,
            'rvec2': rvec2,
            'R_diff': R_diff
        })
        
        print(f"\nMarker ID {marker_id}:")
        print(f"  Roll:  {roll:+.4f}°")
        print(f"  Pitch: {pitch:+.4f}°")
        print(f"  Yaw:   {yaw:+.4f}°")
        print(f"  Total: {total_angle:.4f}°")
    
    # Average rotation if multiple markers
    print("\n[5/5] Computing final results...")
    
    if len(all_rotations) > 1:
        avg_roll = np.mean([r['roll'] for r in all_rotations])
        avg_pitch = np.mean([r['pitch'] for r in all_rotations])
        avg_yaw = np.mean([r['yaw'] for r in all_rotations])
        avg_total = np.mean([r['total'] for r in all_rotations])
        
        print("\n" + "=" * 80)
        print("AVERAGED ROTATION RESULTS (from multiple markers)")
        print("=" * 80)
        print(f"Roll  (rotation around X-axis): {avg_roll:+.4f}°")
        print(f"Pitch (rotation around Y-axis): {avg_pitch:+.4f}°")
        print(f"Yaw   (rotation around Z-axis): {avg_yaw:+.4f}°")
        print(f"\nTotal rotation magnitude: {avg_total:.4f}°")
        print("=" * 80)
    else:
        rot = all_rotations[0]
        print("\n" + "=" * 80)
        print("ROTATION RESULTS")
        print("=" * 80)
        print(f"Roll  (rotation around X-axis): {rot['roll']:+.4f}°")
        print(f"Pitch (rotation around Y-axis): {rot['pitch']:+.4f}°")
        print(f"Yaw   (rotation around Z-axis): {rot['yaw']:+.4f}°")
        print(f"\nTotal rotation magnitude: {rot['total']:.4f}°")
        print("=" * 80)
        print("\nRotation Matrix:")
        print(rot['R_diff'])
    
    # Visualize markers
    print("\nGenerating visualization...")
    vis = visualize_markers(frame1, frame2, corners1, ids1, corners2, ids2,
                           camera_matrix, dist_coeffs, marker_size)
    
    output_path = "aruco_detection.jpg"
    cv2.imwrite(output_path, vis)
    print(f"Visualization saved to: {output_path}")
    
    # Display
    h, w = vis.shape[:2]
    if w > 1920:
        scale = 1920 / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        vis = cv2.resize(vis, (new_w, new_h))
    
    cv2.imshow("ArUco Marker Detection", vis)
    print("Press any key to close visualization...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return all_rotations

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python ArUco_Try.py <frame1_path> <frame2_path> [marker_size] [dict_type]")
        print("\nExamples:")
        print("  python ArUco_Try.py frame1.jpg frame2.jpg")
        print("  python ArUco_Try.py frame1.jpg frame2.jpg 0.05")
        print("  python ArUco_Try.py frame1.jpg frame2.jpg 0.05 DICT_4X4_50")
        print("\nAvailable ArUco dictionaries:")
        print("  DICT_4X4_50, DICT_4X4_100, DICT_4X4_250, DICT_4X4_1000")
        print("  DICT_5X5_50, DICT_5X5_100, DICT_5X5_250, DICT_5X5_1000")
        print("  DICT_6X6_50, DICT_6X6_100, DICT_6X6_250, DICT_6X6_1000")
        print("  DICT_7X7_50, DICT_7X7_100, DICT_7X7_250, DICT_7X7_1000")
        print("\nNote: Marker size is in meters (default: 0.1m = 10cm)")
        sys.exit(1)
    
    frame1_path = sys.argv[1]
    frame2_path = sys.argv[2]
    marker_size = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    dict_type = sys.argv[4] if len(sys.argv) > 4 else 'DICT_6X6_250'
    
    # Validate file paths
    if not Path(frame1_path).exists():
        print(f"Error: Frame 1 not found: {frame1_path}")
        sys.exit(1)
    if not Path(frame2_path).exists():
        print(f"Error: Frame 2 not found: {frame2_path}")
        sys.exit(1)
    
    try:
        results = main(frame1_path, frame2_path, marker_size, dict_type)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
