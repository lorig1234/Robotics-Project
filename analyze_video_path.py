"""
Video Rotation Analysis with Walking Path Visualization

This script:
1. Takes a video as input
2. Extracts frames at specified intervals
3. Runs both SIFT and ArUco methods on each frame pair
4. Calculates agreement scores between methods
5. Visualizes the walking path for both methods

Usage:
    python analyze_video_path.py <video_path> [sample_rate]
    
Example:
    python analyze_video_path.py qr_code_video2.mp4 30
"""

import cv2
import numpy as np
import sys
import os
import tempfile
from pathlib import Path
from datetime import datetime

# Import the detection methods
import First_Try
import ArUco_Try

# Try to import matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found. Install it for visualizations: pip install matplotlib")


def extract_frames_from_video(video_path, output_dir, sample_rate=30):
    """Extract frames from video at specified sample rate."""
    print(f"\n{'='*80}")
    print("EXTRACTING FRAMES FROM VIDEO")
    print(f"{'='*80}")
    print(f"Video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Sample rate: Every {sample_rate} frame(s)")
    print(f"Expected output: ~{total_frames // sample_rate} frames")
    
    os.makedirs(output_dir, exist_ok=True)
    
    frame_paths = []
    timestamps = []
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % sample_rate == 0:
            timestamp = frame_count / fps if fps > 0 else frame_count
            frame_path = os.path.join(output_dir, f"frame_{saved_count:04d}_t{timestamp:.3f}s.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            timestamps.append(timestamp)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_count} frames")
    
    return frame_paths, timestamps, fps


def run_sift_method(frame1_path, frame2_path, verbose=True):
    """Run SIFT feature matching method."""
    try:
        frame1, frame2, gray1, gray2 = First_Try.load_images(frame1_path, frame2_path)
        keypoints1, keypoints2, matches = First_Try.detect_and_match_features(gray1, gray2)
        K = First_Try.estimate_camera_intrinsics(gray1.shape)
        R, t, mask = First_Try.calculate_rotation(keypoints1, keypoints2, matches, K)
        roll, pitch, yaw, total_angle = First_Try.rotation_matrix_to_euler_angles(R)
        
        return {
            'success': True,
            'method': 'SIFT',
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
            'total': total_angle,
            'rotation_matrix': R,
            'features': len(matches),
            'inliers': int(np.sum(mask)) if mask is not None else len(matches)
        }
    except Exception as e:
        return {
            'success': False,
            'method': 'SIFT',
            'error': str(e)
        }


def run_aruco_method(frame1_path, frame2_path, marker_size=0.1, dict_type='DICT_6X6_250', verbose=True):
    """Run ArUco marker tracking method."""
    try:
        frame1, frame2 = ArUco_Try.load_images(frame1_path, frame2_path)
        camera_matrix, dist_coeffs = ArUco_Try.estimate_camera_intrinsics(frame1.shape)
        
        aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_type))
        parameters = cv2.aruco.DetectorParameters()
        
        corners1, ids1, _ = ArUco_Try.detect_aruco_markers(frame1, aruco_dict, parameters)
        corners2, ids2, _ = ArUco_Try.detect_aruco_markers(frame2, aruco_dict, parameters)
        
        if ids1 is None or ids2 is None or len(ids1) == 0 or len(ids2) == 0:
            return {'success': False, 'method': 'ArUco', 'error': 'No markers detected'}
        
        common_ids = set(ids1.flatten()) & set(ids2.flatten())
        
        if len(common_ids) == 0:
            return {'success': False, 'method': 'ArUco', 'error': 'No common markers'}
        
        all_rotations = []
        
        for marker_id in common_ids:
            idx1 = np.where(ids1 == marker_id)[0][0]
            idx2 = np.where(ids2 == marker_id)[0][0]
            
            corners_1 = corners1[idx1][0]
            corners_2 = corners2[idx2][0]
            
            rvec1, tvec1 = ArUco_Try.estimate_marker_pose(corners_1, marker_size, camera_matrix, dist_coeffs)
            rvec2, tvec2 = ArUco_Try.estimate_marker_pose(corners_2, marker_size, camera_matrix, dist_coeffs)
            
            R_diff, rvec_diff = ArUco_Try.calculate_rotation_difference(rvec1, rvec2)
            roll, pitch, yaw, total_angle = ArUco_Try.rotation_vector_to_euler_angles(rvec_diff)
            
            all_rotations.append({
                'roll': roll, 'pitch': pitch, 'yaw': yaw,
                'total': total_angle, 'R_diff': R_diff
            })
        
        avg_roll = np.mean([r['roll'] for r in all_rotations])
        avg_pitch = np.mean([r['pitch'] for r in all_rotations])
        avg_yaw = np.mean([r['yaw'] for r in all_rotations])
        avg_total = np.mean([r['total'] for r in all_rotations])
        avg_R_diff = all_rotations[0]['R_diff']
        
        return {
            'success': True,
            'method': 'ArUco',
            'roll': avg_roll,
            'pitch': avg_pitch,
            'yaw': avg_yaw,
            'total': avg_total,
            'rotation_matrix': avg_R_diff,
            'markers': len(common_ids)
        }
    except Exception as e:
        return {'success': False, 'method': 'ArUco', 'error': str(e)}


def compare_rotation_matrices(R1, R2):
    """Compare two rotation matrices."""
    frobenius_norm = np.linalg.norm(R1 - R2, 'fro')
    
    R_rel = R2 @ R1.T
    trace_val = np.clip((np.trace(R_rel) - 1) / 2, -1, 1)
    angle_diff_deg = np.degrees(np.arccos(trace_val))
    
    matrix_agreement = max(0, 100 - (angle_diff_deg * 5))
    
    return {
        'frobenius_norm': frobenius_norm,
        'angle_difference': angle_diff_deg,
        'matrix_agreement': matrix_agreement
    }


def calculate_agreement(sift_result, aruco_result):
    """Calculate agreement between both methods."""
    if not (sift_result['success'] and aruco_result['success']):
        return None
    
    roll_diff = abs(sift_result['roll'] - aruco_result['roll'])
    pitch_diff = abs(sift_result['pitch'] - aruco_result['pitch'])
    yaw_diff = abs(sift_result['yaw'] - aruco_result['yaw'])
    
    rms_diff = np.sqrt((roll_diff**2 + pitch_diff**2 + yaw_diff**2) / 3)
    euler_agreement = max(0, 100 - (rms_diff * 10))
    
    matrix_comp = compare_rotation_matrices(
        sift_result['rotation_matrix'],
        aruco_result['rotation_matrix']
    )
    
    return {
        'roll_diff': roll_diff,
        'pitch_diff': pitch_diff,
        'yaw_diff': yaw_diff,
        'euler_agreement': euler_agreement,
        'matrix_agreement': matrix_comp['matrix_agreement'],
        'frobenius_norm': matrix_comp['frobenius_norm'],
        'angle_difference': matrix_comp['angle_difference']
    }


def euler_to_rotation_matrix(roll, pitch, yaw):
    """Convert Euler angles (in degrees) to rotation matrix."""
    roll, pitch, yaw = np.radians(roll), np.radians(pitch), np.radians(yaw)
    
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    
    return Rz @ Ry @ Rx


def compute_walking_path(rotations, timestamps):
    """Compute walking path from rotation data."""
    positions = [np.array([0.0, 0.0, 0.0])]
    orientations = [np.eye(3)]
    forward = np.array([0.0, 0.0, 1.0])
    step_size = 0.5
    
    cumulative_roll, cumulative_pitch, cumulative_yaw = 0, 0, 0
    
    path_data = []
    
    for i, rot in enumerate(rotations):
        if rot is None:
            # Use zero rotation if method failed
            roll, pitch, yaw = 0, 0, 0
        else:
            roll, pitch, yaw = rot['roll'], rot['pitch'], rot['yaw']
        
        cumulative_roll += roll
        cumulative_pitch += pitch
        cumulative_yaw += yaw
        
        R = euler_to_rotation_matrix(roll, pitch, yaw)
        current_orientation = orientations[-1] @ R
        orientations.append(current_orientation)
        
        current_forward = current_orientation @ forward
        new_position = positions[-1] + step_size * current_forward
        positions.append(new_position)
        
        path_data.append({
            'timestamp': timestamps[i] if i < len(timestamps) else i * 0.5,
            'roll': roll, 'pitch': pitch, 'yaw': yaw,
            'cumulative_roll': cumulative_roll,
            'cumulative_pitch': cumulative_pitch,
            'cumulative_yaw': cumulative_yaw,
            'position': new_position.copy(),
            'forward': current_forward.copy()
        })
    
    return np.array(positions), orientations, path_data


def print_frame_analysis(frame_idx, timestamp, sift_result, aruco_result, agreement):
    """Print detailed frame-by-frame analysis."""
    print(f"\n[Frame {frame_idx:02d}] t={timestamp:.2f}s")
    print("-" * 60)
    
    # SIFT Results
    if sift_result['success']:
        print(f"  SIFT:  Roll={sift_result['roll']:+7.2f}° Pitch={sift_result['pitch']:+7.2f}° "
              f"Yaw={sift_result['yaw']:+7.2f}° Total={sift_result['total']:6.2f}°")
        print(f"         Features: {sift_result.get('features', 'N/A')}, Inliers: {sift_result.get('inliers', 'N/A')}")
    else:
        print(f"  SIFT:  FAILED - {sift_result.get('error', 'Unknown error')}")
    
    # ArUco Results
    if aruco_result['success']:
        print(f"  ArUco: Roll={aruco_result['roll']:+7.2f}° Pitch={aruco_result['pitch']:+7.2f}° "
              f"Yaw={aruco_result['yaw']:+7.2f}° Total={aruco_result['total']:6.2f}°")
        print(f"         Markers detected: {aruco_result.get('markers', 'N/A')}")
    else:
        print(f"  ArUco: FAILED - {aruco_result.get('error', 'Unknown error')}")
    
    # Agreement
    if agreement:
        print(f"  Agreement:")
        print(f"    Euler: {agreement['euler_agreement']:.1f}% | Matrix: {agreement['matrix_agreement']:.1f}%")
        print(f"    Diff: Roll={agreement['roll_diff']:.2f}° Pitch={agreement['pitch_diff']:.2f}° Yaw={agreement['yaw_diff']:.2f}°")
        
        # Interpretation
        if agreement['euler_agreement'] > 85:
            print(f"    Status: GOOD MATCH ✓")
        elif agreement['euler_agreement'] > 70:
            print(f"    Status: MODERATE MATCH ~")
        else:
            print(f"    Status: LOW MATCH - methods disagree ✗")


def print_walking_path(method_name, path_data, color_code):
    """Print walking path reconstruction for a method."""
    print(f"\n{'='*80}")
    print(f"{color_code}WALKING PATH RECONSTRUCTION - {method_name}")
    print(f"{'='*80}")
    
    for i, data in enumerate(path_data):
        total_rot = np.sqrt(data['roll']**2 + data['pitch']**2 + data['yaw']**2)
        
        print(f"[t={data['timestamp']:5.1f}s] Frame {i:02d}")
        print(f"    Rotation:   Roll={data['roll']:+6.2f}°  Pitch={data['pitch']:+6.2f}°  Yaw={data['yaw']:+6.2f}°  (Total={total_rot:5.2f}°)")
        print(f"    Cumulative: Roll={data['cumulative_roll']:+7.2f}°  Pitch={data['cumulative_pitch']:+7.2f}°  Yaw={data['cumulative_yaw']:+7.2f}°")
        print(f"    Position:   X={data['position'][0]:+6.2f}  Y={data['position'][1]:+6.2f}  Z={data['position'][2]:+6.2f}")
        print(f"    Facing:     X={data['forward'][0]:+5.2f}  Y={data['forward'][1]:+5.2f}  Z={data['forward'][2]:+5.2f}")
        
        # Movement interpretation
        if abs(data['yaw']) > 3:
            direction = "LEFT" if data['yaw'] > 0 else "RIGHT"
            print(f"    Movement:   Turning {direction} ({abs(data['yaw']):.1f}°)")
        elif abs(data['pitch']) > 4:
            direction = "UP" if data['pitch'] > 0 else "DOWN"
            print(f"    Movement:   Looking {direction} ({abs(data['pitch']):.1f}°)")
        elif abs(data['roll']) > 3:
            direction = "LEFT" if data['roll'] > 0 else "RIGHT"
            print(f"    Movement:   Tilting {direction} ({abs(data['roll']):.1f}°)")
        else:
            print(f"    Movement:   Walking forward (steady)")


def print_path_summary(method_name, path_data, positions):
    """Print summary of walking path."""
    if not path_data:
        return
        
    final = path_data[-1]
    
    print(f"\n{'-'*40}")
    print(f"{method_name} PATH SUMMARY")
    print(f"{'-'*40}")
    print(f"Total time: {final['timestamp']:.1f} seconds")
    print(f"Final cumulative rotation:")
    print(f"    Roll (tilt):     {final['cumulative_roll']:+.2f}°")
    print(f"    Pitch (up/down): {final['cumulative_pitch']:+.2f}°")
    print(f"    Yaw (left/right):{final['cumulative_yaw']:+.2f}°")
    print(f"Estimated displacement:")
    print(f"    X (left/right):  {final['position'][0]:+.2f} units")
    print(f"    Y (up/down):     {final['position'][1]:+.2f} units")
    print(f"    Z (forward):     {final['position'][2]:+.2f} units")
    
    # Interpretation
    print(f"\nInterpretation:")
    if final['cumulative_yaw'] < -10:
        print(f"  → Turned RIGHT overall")
    elif final['cumulative_yaw'] > 10:
        print(f"  → Turned LEFT overall")
    else:
        print(f"  → Walked mostly STRAIGHT")
    
    if final['cumulative_pitch'] < -20:
        print(f"  → Camera was tilting DOWN")
    elif final['cumulative_pitch'] > 20:
        print(f"  → Camera was tilting UP")


def plot_dual_paths(sift_positions, aruco_positions, sift_orientations, aruco_orientations, timestamps, output_file):
    """Create visualization comparing both walking paths."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping visualization")
        return
    
    sift_pos = np.array(sift_positions)
    aruco_pos = np.array(aruco_positions)
    
    fig = plt.figure(figsize=(16, 10))
    
    # 3D comparison plot
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(sift_pos[:, 0], sift_pos[:, 2], sift_pos[:, 1], 'b-', linewidth=2, label='SIFT Path')
    ax1.plot(aruco_pos[:, 0], aruco_pos[:, 2], aruco_pos[:, 1], 'r-', linewidth=2, label='ArUco Path')
    ax1.scatter(0, 0, 0, c='green', s=100, marker='o', label='Start')
    ax1.scatter(sift_pos[-1, 0], sift_pos[-1, 2], sift_pos[-1, 1], c='blue', s=100, marker='s')
    ax1.scatter(aruco_pos[-1, 0], aruco_pos[-1, 2], aruco_pos[-1, 1], c='red', s=100, marker='s')
    ax1.set_xlabel('X (Left/Right)')
    ax1.set_ylabel('Z (Forward)')
    ax1.set_zlabel('Y (Up/Down)')
    ax1.set_title('3D Walking Path Comparison')
    ax1.legend()
    
    # Top-down view
    ax2 = fig.add_subplot(222)
    ax2.plot(sift_pos[:, 0], sift_pos[:, 2], 'b-', linewidth=2, label='SIFT')
    ax2.plot(aruco_pos[:, 0], aruco_pos[:, 2], 'r-', linewidth=2, label='ArUco')
    ax2.scatter(0, 0, c='green', s=100, marker='o', label='Start')
    ax2.scatter(sift_pos[-1, 0], sift_pos[-1, 2], c='blue', s=80, marker='s')
    ax2.scatter(aruco_pos[-1, 0], aruco_pos[-1, 2], c='red', s=80, marker='s')
    
    # Add orientation arrows
    for i in range(0, len(sift_orientations), max(1, len(sift_orientations)//8)):
        if i < len(sift_pos):
            forward = sift_orientations[i] @ np.array([0, 0, 0.3])
            ax2.annotate('', xy=(sift_pos[i, 0] + forward[0], sift_pos[i, 2] + forward[2]),
                        xytext=(sift_pos[i, 0], sift_pos[i, 2]),
                        arrowprops=dict(arrowstyle='->', color='blue', alpha=0.5))
    for i in range(0, len(aruco_orientations), max(1, len(aruco_orientations)//8)):
        if i < len(aruco_pos):
            forward = aruco_orientations[i] @ np.array([0, 0, 0.3])
            ax2.annotate('', xy=(aruco_pos[i, 0] + forward[0], aruco_pos[i, 2] + forward[2]),
                        xytext=(aruco_pos[i, 0], aruco_pos[i, 2]),
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.5))
    
    ax2.set_xlabel('X (Left/Right)')
    ax2.set_ylabel('Z (Forward)')
    ax2.set_title("Top-Down View (Bird's Eye)")
    ax2.legend()
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)
    
    # Side view (Z-Y plane)
    ax3 = fig.add_subplot(223)
    ax3.plot(sift_pos[:, 2], sift_pos[:, 1], 'b-', linewidth=2, label='SIFT')
    ax3.plot(aruco_pos[:, 2], aruco_pos[:, 1], 'r-', linewidth=2, label='ArUco')
    ax3.scatter(0, 0, c='green', s=100, marker='o', label='Start')
    ax3.set_xlabel('Z (Forward)')
    ax3.set_ylabel('Y (Up/Down)')
    ax3.set_title('Side View')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Deviation over time
    ax4 = fig.add_subplot(224)
    min_len = min(len(sift_pos), len(aruco_pos))
    deviations = [np.linalg.norm(sift_pos[i] - aruco_pos[i]) for i in range(min_len)]
    time_axis = timestamps[:min_len] if len(timestamps) >= min_len else list(range(min_len))
    ax4.plot(time_axis, deviations, 'g-', linewidth=2)
    ax4.fill_between(time_axis, deviations, alpha=0.3, color='green')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Path Deviation (units)')
    ax4.set_title('SIFT vs ArUco Path Deviation Over Time')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"\nVisualization saved to: {output_file}")
    plt.show()


def analyze_video(video_path, sample_rate=30, marker_size=0.1, dict_type='DICT_6X6_250'):
    """Main function to analyze video and visualize walking path."""
    
    print("=" * 80)
    print("VIDEO ROTATION ANALYSIS WITH WALKING PATH VISUALIZATION")
    print("=" * 80)
    print(f"Video: {video_path}")
    print(f"Sample rate: Every {sample_rate} frames")
    print(f"ArUco marker size: {marker_size}m")
    print(f"ArUco dictionary: {dict_type}")
    
    # Create temp directory for frames
    temp_dir = tempfile.mkdtemp(prefix="rotation_analysis_")
    
    try:
        # Extract frames
        frame_paths, timestamps, fps = extract_frames_from_video(video_path, temp_dir, sample_rate)
        
        if len(frame_paths) < 2:
            print("Error: Not enough frames extracted")
            return
        
        # Process frame pairs
        print(f"\n{'='*80}")
        print(f"PROCESSING {len(frame_paths)-1} FRAME PAIRS")
        print(f"{'='*80}")
        
        sift_rotations = []
        aruco_rotations = []
        all_results = []
        
        sift_successes = 0
        aruco_successes = 0
        both_successes = 0
        
        for i in range(len(frame_paths) - 1):
            frame1 = frame_paths[i]
            frame2 = frame_paths[i + 1]
            timestamp = timestamps[i]
            
            # Run both methods (suppress verbose output)
            import io
            import sys as _sys
            old_stdout = _sys.stdout
            _sys.stdout = io.StringIO()
            
            sift_result = run_sift_method(frame1, frame2, verbose=False)
            aruco_result = run_aruco_method(frame1, frame2, marker_size, dict_type, verbose=False)
            
            _sys.stdout = old_stdout
            
            # Calculate agreement
            agreement = calculate_agreement(sift_result, aruco_result)
            
            # Print frame analysis
            print_frame_analysis(i, timestamp, sift_result, aruco_result, agreement)
            
            # Track successes
            if sift_result['success']:
                sift_successes += 1
                sift_rotations.append(sift_result)
            else:
                sift_rotations.append({'roll': 0, 'pitch': 0, 'yaw': 0, 'success': False})
            
            if aruco_result['success']:
                aruco_successes += 1
                aruco_rotations.append(aruco_result)
            else:
                aruco_rotations.append({'roll': 0, 'pitch': 0, 'yaw': 0, 'success': False})
            
            if sift_result['success'] and aruco_result['success']:
                both_successes += 1
            
            all_results.append({
                'frame_idx': i,
                'timestamp': timestamp,
                'sift': sift_result,
                'aruco': aruco_result,
                'agreement': agreement
            })
        
        # Summary statistics
        print(f"\n{'='*80}")
        print("PROCESSING SUMMARY")
        print(f"{'='*80}")
        print(f"Total frame pairs: {len(frame_paths)-1}")
        print(f"SIFT successful:   {sift_successes} ({sift_successes/(len(frame_paths)-1)*100:.1f}%)")
        print(f"ArUco successful:  {aruco_successes} ({aruco_successes/(len(frame_paths)-1)*100:.1f}%)")
        print(f"Both successful:   {both_successes} ({both_successes/(len(frame_paths)-1)*100:.1f}%)")
        
        if both_successes > 0:
            avg_euler_agreement = np.mean([r['agreement']['euler_agreement'] 
                                           for r in all_results if r['agreement']])
            avg_matrix_agreement = np.mean([r['agreement']['matrix_agreement'] 
                                            for r in all_results if r['agreement']])
            print(f"\nAverage Euler Agreement:  {avg_euler_agreement:.1f}%")
            print(f"Average Matrix Agreement: {avg_matrix_agreement:.1f}%")
        
        # Compute walking paths for both methods
        sift_positions, sift_orientations, sift_path_data = compute_walking_path(sift_rotations, timestamps)
        aruco_positions, aruco_orientations, aruco_path_data = compute_walking_path(aruco_rotations, timestamps)
        
        # Print walking paths
        print_walking_path("SIFT", sift_path_data, "\033[94m")  # Blue
        print_path_summary("SIFT", sift_path_data, sift_positions)
        
        print_walking_path("ArUco", aruco_path_data, "\033[91m")  # Red
        print_path_summary("ArUco", aruco_path_data, aruco_positions)
        
        # Create visualization
        output_file = f"walking_path_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plot_dual_paths(sift_positions, aruco_positions, 
                       sift_orientations, aruco_orientations, 
                       timestamps, output_file)
        
        # Save results to file
        results_file = f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        save_results(results_file, video_path, all_results, sift_path_data, aruco_path_data)
        print(f"Results saved to: {results_file}")
        
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*80}")
        
    finally:
        # Cleanup
        print("\nCleaning up temporary files...")
        for frame_path in frame_paths:
            try:
                os.remove(frame_path)
            except:
                pass
        try:
            os.rmdir(temp_dir)
        except:
            pass


def save_results(filename, video_path, results, sift_path, aruco_path):
    """Save analysis results to file."""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("Video Rotation Analysis Results\n")
        f.write("=" * 80 + "\n")
        f.write(f"Video: {video_path}\n")
        f.write(f"Total frame pairs: {len(results)}\n\n")
        
        f.write("Frame-by-Frame Results:\n")
        f.write("-" * 80 + "\n")
        
        for r in results:
            f.write(f"\n[Frame {r['frame_idx']:02d}] t={r['timestamp']:.2f}s\n")
            
            if r['sift']['success']:
                f.write(f"  SIFT:  Roll={r['sift']['roll']:+7.2f}° Pitch={r['sift']['pitch']:+7.2f}° "
                       f"Yaw={r['sift']['yaw']:+7.2f}° Total={r['sift']['total']:6.2f}°\n")
            else:
                f.write(f"  SIFT:  FAILED - {r['sift'].get('error', 'Unknown')}\n")
            
            if r['aruco']['success']:
                f.write(f"  ArUco: Roll={r['aruco']['roll']:+7.2f}° Pitch={r['aruco']['pitch']:+7.2f}° "
                       f"Yaw={r['aruco']['yaw']:+7.2f}° Total={r['aruco']['total']:6.2f}°\n")
            else:
                f.write(f"  ArUco: FAILED - {r['aruco'].get('error', 'Unknown')}\n")
            
            if r['agreement']:
                f.write(f"  Agreement: Euler={r['agreement']['euler_agreement']:.1f}% "
                       f"Matrix={r['agreement']['matrix_agreement']:.1f}%\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Walking Path Summary\n")
        f.write("-" * 80 + "\n")
        
        if sift_path:
            final = sift_path[-1]
            f.write(f"\nSIFT Path:\n")
            f.write(f"  Final position: X={final['position'][0]:.2f} Y={final['position'][1]:.2f} Z={final['position'][2]:.2f}\n")
            f.write(f"  Cumulative rotation: Roll={final['cumulative_roll']:.2f}° "
                   f"Pitch={final['cumulative_pitch']:.2f}° Yaw={final['cumulative_yaw']:.2f}°\n")
        
        if aruco_path:
            final = aruco_path[-1]
            f.write(f"\nArUco Path:\n")
            f.write(f"  Final position: X={final['position'][0]:.2f} Y={final['position'][1]:.2f} Z={final['position'][2]:.2f}\n")
            f.write(f"  Cumulative rotation: Roll={final['cumulative_roll']:.2f}° "
                   f"Pitch={final['cumulative_pitch']:.2f}° Yaw={final['cumulative_yaw']:.2f}°\n")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python analyze_video_path.py <video_path> [sample_rate] [marker_size]")
        print("\nExamples:")
        print("  python analyze_video_path.py video.mp4")
        print("  python analyze_video_path.py video.mp4 30")
        print("  python analyze_video_path.py video.mp4 30 0.05")
        sys.exit(1)
    
    video_path = sys.argv[1]
    sample_rate = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    marker_size = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    
    if not Path(video_path).exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)
    
    try:
        analyze_video(video_path, sample_rate, marker_size)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
