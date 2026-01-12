import cv2
import numpy as np
import sys
from pathlib import Path
import os
import tempfile
from datetime import datetime

# Import the main functions from both methods
import First_Try
import ArUco_Try

def extract_frames_from_video(video_path, output_dir, sample_rate=1):
    """
    Extract frames from video at specified sample rate.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save extracted frames
        sample_rate: Extract every Nth frame (1 = all frames, 30 = 1 per second at 30fps)
    
    Returns:
        List of frame file paths
    """
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video info:")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Sample rate: Every {sample_rate} frame(s)")
    print(f"  Expected output: ~{total_frames // sample_rate} frames")
    
    os.makedirs(output_dir, exist_ok=True)
    
    frame_paths = []
    frame_count = 0
    saved_count = 0
    
    print("\nExtracting frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % sample_rate == 0:
            # Save frame
            timestamp = frame_count / fps if fps > 0 else frame_count
            frame_path = os.path.join(output_dir, f"frame_{saved_count:04d}_t{timestamp:.3f}s.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            saved_count += 1
            
            if saved_count % 10 == 0:
                print(f"  Extracted {saved_count} frames...", end='\r')
        
        frame_count += 1
    
    cap.release()
    print(f"\n✓ Extracted {saved_count} frames to {output_dir}")
    
    return frame_paths

def run_sift_method(frame1_path, frame2_path):
    """Run SIFT feature matching method."""
    try:
        # Load images
        frame1, frame2, gray1, gray2 = First_Try.load_images(frame1_path, frame2_path)
        
        # Detect and match features
        keypoints1, keypoints2, matches = First_Try.detect_and_match_features(gray1, gray2)
        
        # Estimate camera intrinsics
        K = First_Try.estimate_camera_intrinsics(gray1.shape)
        
        # Calculate rotation
        R, t, mask = First_Try.calculate_rotation(keypoints1, keypoints2, matches, K)
        
        # Convert to Euler angles
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
            'inliers': int(np.sum(mask))
        }
    except Exception as e:
        return {
            'success': False,
            'method': 'SIFT',
            'error': str(e)
        }

def run_aruco_method(frame1_path, frame2_path, marker_size=0.1, dict_type='DICT_6X6_250'):
    """Run ArUco marker tracking method."""
    try:
        # Load images
        frame1, frame2 = ArUco_Try.load_images(frame1_path, frame2_path)
        
        # Estimate camera parameters
        camera_matrix, dist_coeffs = ArUco_Try.estimate_camera_intrinsics(frame1.shape)
        
        # Initialize ArUco detector
        aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_type))
        parameters = cv2.aruco.DetectorParameters()
        
        # Detect markers
        corners1, ids1, rejected1 = ArUco_Try.detect_aruco_markers(frame1, aruco_dict, parameters)
        corners2, ids2, rejected2 = ArUco_Try.detect_aruco_markers(frame2, aruco_dict, parameters)
        
        if ids1 is None or ids2 is None or len(ids1) == 0 or len(ids2) == 0:
            return {
                'success': False,
                'method': 'ArUco',
                'error': 'No markers detected'
            }
        
        # Find common markers
        common_ids = set(ids1.flatten()) & set(ids2.flatten())
        
        if len(common_ids) == 0:
            return {
                'success': False,
                'method': 'ArUco',
                'error': 'No common markers'
            }
        
        # Calculate rotation for each common marker
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
                'roll': roll,
                'pitch': pitch,
                'yaw': yaw,
                'total': total_angle,
                'R_diff': R_diff
            })
        
        # Average if multiple markers
        avg_roll = np.mean([r['roll'] for r in all_rotations])
        avg_pitch = np.mean([r['pitch'] for r in all_rotations])
        avg_yaw = np.mean([r['yaw'] for r in all_rotations])
        avg_total = np.mean([r['total'] for r in all_rotations])
        
        # Use first marker's rotation matrix as representative
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
        return {
            'success': False,
            'method': 'ArUco',
            'error': str(e)
        }

def compare_rotation_matrices(R1, R2):
    """Compare two rotation matrices and calculate difference metrics."""
    # Frobenius norm (element-wise difference)
    frobenius_norm = np.linalg.norm(R1 - R2, 'fro')
    
    # Rotation angle difference using trace
    # For rotation matrices R1 and R2, the relative rotation is R_rel = R2 @ R1.T
    # The angle is: theta = arccos((trace(R_rel) - 1) / 2)
    R_rel = R2 @ R1.T
    trace_val = np.trace(R_rel)
    # Clamp to [-1, 1] to avoid numerical errors with arccos
    trace_val = np.clip((trace_val - 1) / 2, -1, 1)
    angle_diff_rad = np.arccos(trace_val)
    angle_diff_deg = np.degrees(angle_diff_rad)
    
    # Geodesic distance on SO(3)
    geodesic_distance = angle_diff_rad
    
    # Matrix agreement score (0-100%)
    # Perfect match = 100%, 10° difference = 50%, 20°+ difference = 0%
    matrix_agreement = max(0, 100 - (angle_diff_deg * 5))
    
    return {
        'frobenius_norm': frobenius_norm,
        'angle_difference': angle_diff_deg,
        'geodesic_distance': geodesic_distance,
        'relative_rotation': R_rel,
        'matrix_agreement': matrix_agreement
    }

def compare_results(sift_result, aruco_result):
    """Compare results from both methods."""
    comparison = {
        'both_successful': sift_result['success'] and aruco_result['success'],
        'sift_only': sift_result['success'] and not aruco_result['success'],
        'aruco_only': aruco_result['success'] and not sift_result['success'],
        'both_failed': not sift_result['success'] and not aruco_result['success']
    }
    
    if comparison['both_successful']:
        # Calculate differences in Euler angles
        comparison['roll_diff'] = abs(sift_result['roll'] - aruco_result['roll'])
        comparison['pitch_diff'] = abs(sift_result['pitch'] - aruco_result['pitch'])
        comparison['yaw_diff'] = abs(sift_result['yaw'] - aruco_result['yaw'])
        comparison['total_diff'] = abs(sift_result['total'] - aruco_result['total'])
        
        # Compare rotation matrices if available
        if 'rotation_matrix' in sift_result and 'rotation_matrix' in aruco_result:
            matrix_comparison = compare_rotation_matrices(
                sift_result['rotation_matrix'],
                aruco_result['rotation_matrix']
            )
            comparison.update(matrix_comparison)
        
        # Calculate Euler agreement using RMS of individual axis differences
        # This properly accounts for large differences in any single axis
        rms_diff = np.sqrt((comparison['roll_diff']**2 + comparison['pitch_diff']**2 + comparison['yaw_diff']**2) / 3)
        # 0° diff = 100%, 5° RMS = 50%, 10°+ RMS = 0%
        comparison['agreement_score'] = max(0, 100 - (rms_diff * 10))
    
    return comparison

def process_video(video_path, sample_rate=30, marker_size=0.1, dict_type='DICT_6X6_250'):
    """
    Process video and compare rotation detection methods.
    
    Args:
        video_path: Path to video file
        sample_rate: Extract every Nth frame
        marker_size: Size of ArUco markers in meters
        dict_type: ArUco dictionary type
    """
    print("=" * 80)
    print("Video Rotation Analysis - Method Comparison")
    print("=" * 80)
    print(f"Video: {video_path}")
    print(f"Sample rate: Every {sample_rate} frame(s)")
    print(f"ArUco marker size: {marker_size}m")
    print(f"ArUco dictionary: {dict_type}")
    print("=" * 80)
    
    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp(prefix="rotation_analysis_")
    print(f"\nTemporary directory: {temp_dir}")
    
    try:
        # Extract frames
        frame_paths = extract_frames_from_video(video_path, temp_dir, sample_rate)
        
        if len(frame_paths) < 2:
            print("❌ Not enough frames extracted. Need at least 2 frames.")
            return
        
        # Process consecutive frame pairs
        print(f"\n{'=' * 80}")
        print(f"Processing {len(frame_paths) - 1} frame pairs")
        print(f"{'=' * 80}\n")
        
        results = []
        sift_successes = 0
        aruco_successes = 0
        both_successes = 0
        
        for i in range(len(frame_paths) - 1):
            frame1 = frame_paths[i]
            frame2 = frame_paths[i + 1]
            
            frame1_name = Path(frame1).name
            frame2_name = Path(frame2).name
            
            print(f"[{i+1}/{len(frame_paths)-1}] {frame1_name} → {frame2_name}")
            
            # Run both methods
            sift_result = run_sift_method(frame1, frame2)
            aruco_result = run_aruco_method(frame1, frame2, marker_size, dict_type)
            
            # Compare results
            comparison = compare_results(sift_result, aruco_result)
            
            # Display results
            if comparison['both_successful']:
                print(f"  ✓ SIFT:  Roll={sift_result['roll']:+7.3f}° Pitch={sift_result['pitch']:+7.3f}° Yaw={sift_result['yaw']:+7.3f}° Total={sift_result['total']:7.3f}°")
                print(f"  ✓ ArUco: Roll={aruco_result['roll']:+7.3f}° Pitch={aruco_result['pitch']:+7.3f}° Yaw={aruco_result['yaw']:+7.3f}° Total={aruco_result['total']:7.3f}°")
                print(f"  → Euler Angle Diff: Roll={comparison['roll_diff']:.3f}° Pitch={comparison['pitch_diff']:.3f}° Yaw={comparison['yaw_diff']:.3f}° Total={comparison['total_diff']:.3f}°")
                print(f"  → Euler Agreement: {comparison['agreement_score']:.1f}%")
                if 'frobenius_norm' in comparison:
                    print(f"  → Matrix Diff: Frobenius={comparison['frobenius_norm']:.6f} Angle={comparison['angle_difference']:.3f}°")
                    print(f"  → Matrix Agreement: {comparison['matrix_agreement']:.1f}%")
                both_successes += 1
                sift_successes += 1
                aruco_successes += 1
            elif comparison['sift_only']:
                print(f"  ✓ SIFT:  Roll={sift_result['roll']:+7.3f}° Pitch={sift_result['pitch']:+7.3f}° Yaw={sift_result['yaw']:+7.3f}° Total={sift_result['total']:7.3f}°")
                print(f"  ✗ ArUco: {aruco_result.get('error', 'Failed')}")
                sift_successes += 1
            elif comparison['aruco_only']:
                print(f"  ✗ SIFT:  {sift_result.get('error', 'Failed')}")
                print(f"  ✓ ArUco: Roll={aruco_result['roll']:+7.3f}° Pitch={aruco_result['pitch']:+7.3f}° Yaw={aruco_result['yaw']:+7.3f}° Total={aruco_result['total']:7.3f}°")
                aruco_successes += 1
            else:
                print(f"  ✗ SIFT:  {sift_result.get('error', 'Failed')}")
                print(f"  ✗ ArUco: {aruco_result.get('error', 'Failed')}")
            
            print()
            
            results.append({
                'frame1': frame1_name,
                'frame2': frame2_name,
                'sift': sift_result,
                'aruco': aruco_result,
                'comparison': comparison
            })
        
        # Summary statistics
        print("=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        print(f"Total frame pairs processed: {len(results)}")
        print(f"SIFT successful: {sift_successes} ({sift_successes/len(results)*100:.1f}%)")
        print(f"ArUco successful: {aruco_successes} ({aruco_successes/len(results)*100:.1f}%)")
        print(f"Both methods successful: {both_successes} ({both_successes/len(results)*100:.1f}%)")
        
        if both_successes > 0:
            # Calculate average differences when both methods work
            avg_diffs = {
                'roll': np.mean([r['comparison']['roll_diff'] for r in results if r['comparison']['both_successful']]),
                'pitch': np.mean([r['comparison']['pitch_diff'] for r in results if r['comparison']['both_successful']]),
                'yaw': np.mean([r['comparison']['yaw_diff'] for r in results if r['comparison']['both_successful']]),
                'total': np.mean([r['comparison']['total_diff'] for r in results if r['comparison']['both_successful']])
            }
            avg_agreement = np.mean([r['comparison']['agreement_score'] for r in results if r['comparison']['both_successful']])
            
            print("\nAverage differences (when both methods work):")
            print(f"  Euler Angles:")
            print(f"    Roll:  {avg_diffs['roll']:.3f}°")
            print(f"    Pitch: {avg_diffs['pitch']:.3f}°")
            print(f"    Yaw:   {avg_diffs['yaw']:.3f}°")
            print(f"    Total: {avg_diffs['total']:.3f}°")
            
            # Calculate average matrix differences if available
            successful_comparisons = [r['comparison'] for r in results if r['comparison']['both_successful']]
            if successful_comparisons and 'frobenius_norm' in successful_comparisons[0]:
                avg_frobenius = np.mean([c['frobenius_norm'] for c in successful_comparisons])
                avg_angle_diff = np.mean([c['angle_difference'] for c in successful_comparisons])
                avg_matrix_agreement = np.mean([c['matrix_agreement'] for c in successful_comparisons])
                print(f"  Rotation Matrix:")
                print(f"    Frobenius Norm: {avg_frobenius:.6f}")
                print(f"    Angle Difference: {avg_angle_diff:.3f}°")
                print(f"    Matrix Agreement: {avg_matrix_agreement:.1f}%")
            
            print(f"\nAverage Euler agreement score: {avg_agreement:.1f}%")
        
        # Save results to file
        output_file = f"comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Video Rotation Analysis - Method Comparison\n")
            f.write("=" * 80 + "\n")
            f.write(f"Video: {video_path}\n")
            f.write(f"Sample rate: Every {sample_rate} frame(s)\n")
            f.write(f"Total frame pairs: {len(results)}\n\n")
            
            for i, result in enumerate(results):
                f.write(f"[{i+1}] {result['frame1']} → {result['frame2']}\n")
                
                if result['comparison']['both_successful']:
                    f.write(f"  SIFT:  Roll={result['sift']['roll']:+7.3f}° Pitch={result['sift']['pitch']:+7.3f}° ")
                    f.write(f"Yaw={result['sift']['yaw']:+7.3f}° Total={result['sift']['total']:7.3f}°\n")
                    f.write(f"  ArUco: Roll={result['aruco']['roll']:+7.3f}° Pitch={result['aruco']['pitch']:+7.3f}° ")
                    f.write(f"Yaw={result['aruco']['yaw']:+7.3f}° Total={result['aruco']['total']:7.3f}°\n")
                    f.write(f"  Euler Angle Diff: Roll={result['comparison']['roll_diff']:.3f}° Pitch={result['comparison']['pitch_diff']:.3f}° ")
                    f.write(f"Yaw={result['comparison']['yaw_diff']:.3f}° Total={result['comparison']['total_diff']:.3f}°\n")
                    if 'frobenius_norm' in result['comparison']:
                        f.write(f"  Matrix Diff: Frobenius={result['comparison']['frobenius_norm']:.6f} ")
                        f.write(f"Angle={result['comparison']['angle_difference']:.3f}°\n")
                    f.write(f"  Euler Agreement: {result['comparison']['agreement_score']:.1f}%\n")
                    if 'matrix_agreement' in result['comparison']:
                        f.write(f"  Matrix Agreement: {result['comparison']['matrix_agreement']:.1f}%\n")
                    
                    # Write rotation matrices
                    if 'rotation_matrix' in result['sift']:
                        f.write(f"  SIFT Rotation Matrix:\n")
                        for row in result['sift']['rotation_matrix']:
                            f.write(f"    [{row[0]:+.6f} {row[1]:+.6f} {row[2]:+.6f}]\n")
                    if 'rotation_matrix' in result['aruco']:
                        f.write(f"  ArUco Rotation Matrix:\n")
                        for row in result['aruco']['rotation_matrix']:
                            f.write(f"    [{row[0]:+.6f} {row[1]:+.6f} {row[2]:+.6f}]\n")
                else:
                    if result['sift']['success']:
                        f.write(f"  SIFT: Total={result['sift']['total']:7.3f}°\n")
                    else:
                        f.write(f"  SIFT: Failed - {result['sift'].get('error', 'Unknown')}\n")
                    
                    if result['aruco']['success']:
                        f.write(f"  ArUco: Total={result['aruco']['total']:7.3f}°\n")
                    else:
                        f.write(f"  ArUco: Failed - {result['aruco'].get('error', 'Unknown')}\n")
                
                f.write("\n")
        
        print(f"\n✓ Results saved to: {output_file}")
        print("=" * 80)
    finally:
        # Clean up temporary files
        print(f"\nCleaning up temporary files...")
        for frame_path in frame_paths:
            try:
                os.remove(frame_path)
            except:
                pass
        try:
            os.rmdir(temp_dir)
            print("✓ Temporary files cleaned up")
        except:
            print(f"⚠ Could not remove temporary directory: {temp_dir}")

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python Compare_Methods.py <video_path> [sample_rate] [marker_size] [dict_type]")
        print("\nExamples:")
        print("  python Compare_Methods.py video.mp4")
        print("  python Compare_Methods.py video.mp4 30")
        print("  python Compare_Methods.py video.mp4 30 0.05")
        print("  python Compare_Methods.py video.mp4 30 0.05 DICT_6X6_250")
        print("\nParameters:")
        print("  sample_rate: Extract every Nth frame (default: 30)")
        print("  marker_size: ArUco marker size in meters (default: 0.1)")
        print("  dict_type: ArUco dictionary type (default: DICT_6X6_250)")
        sys.exit(1)
    
    video_path = sys.argv[1]
    sample_rate = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    marker_size = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    dict_type = sys.argv[4] if len(sys.argv) > 4 else 'DICT_6X6_250'
    
    # Validate video path
    if not Path(video_path).exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)
    
    try:
        process_video(video_path, sample_rate, marker_size, dict_type)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
