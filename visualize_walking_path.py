import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Actual SIFT rotation data from comparison_results_20260105_181945.txt
# Format: (roll, pitch, yaw) in degrees
sift_rotations = [
    (0, 0, 0),  # Frame 0 - starting point (no rotation data for first pair)
    (-1.150, -4.970, -0.287),   # Frame 1->2
    (+2.637, -7.072, -0.816),   # Frame 2->3
    (-3.590, +4.309, -6.073),   # Frame 3->4
    (+0.172, -4.854, -1.743),   # Frame 4->5
    (-0.721, -3.267, -0.154),   # Frame 5->6
    (+0.326, -2.506, -1.401),   # Frame 6->7
    (-1.807, -2.060, -1.264),   # Frame 7->8
    (-2.061, +0.122, -3.757),   # Frame 8->9
    (-1.317, -3.150, -1.611),   # Frame 9->10
    (-1.552, +0.595, -2.308),   # Frame 10->11
    (-4.400, -1.806, -1.165),   # Frame 11->12
    (+0.691, +1.648, -3.034),   # Frame 12->13
    (-0.228, -1.225, -1.494),   # Frame 13->14
    (-0.570, -4.129, -1.728),   # Frame 14->15
    (+1.401, -0.442, -0.963),   # Frame 15->16
    (-1.343, -3.968, -2.160),   # Frame 16->17
    (+0.345, -1.602, -0.798),   # Frame 17->18
    (-0.663, -2.592, -0.361),   # Frame 18->19
    (+0.248, +1.493, +1.073),   # Frame 19->20
    (+2.993, +5.901, +1.528),   # Frame 20->21
    (+2.050, +5.541, +2.322),   # Frame 21->22
    (+0.258, -0.470, +6.306),   # Frame 22->23
]

# Timestamps
timestamps = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 
              6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0]

def euler_to_rotation_matrix(roll, pitch, yaw):
    """Convert Euler angles (in degrees) to rotation matrix."""
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)
    
    # Rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    return Rz @ Ry @ Rx

def simulate_walk():
    """Simulate walking path based on rotation transformations."""
    
    print("=" * 80)
    print("WALKING PATH RECONSTRUCTION FROM ROTATION MATRICES")
    print("=" * 80)
    print()
    print("Using SIFT rotation estimates from video analysis")
    print("Video: qr_code_video2.mp4")
    print("Total frames analyzed: 23 pairs")
    print()
    print("-" * 80)
    print("FRAME-BY-FRAME CAMERA ORIENTATION TRACKING")
    print("-" * 80)
    print()
    
    # Track camera position and orientation
    positions = [np.array([0.0, 0.0, 0.0])]  # Starting position
    orientations = [np.eye(3)]  # Starting orientation (identity)
    
    # Camera forward direction (initially looking along +Z)
    forward = np.array([0.0, 0.0, 1.0])
    
    # Cumulative rotation tracking
    cumulative_roll = 0
    cumulative_pitch = 0
    cumulative_yaw = 0
    
    # Assume constant walking speed (arbitrary units)
    step_size = 0.5
    
    for i, (roll, pitch, yaw) in enumerate(sift_rotations):
        t = timestamps[i]
        
        # Update cumulative angles
        cumulative_roll += roll
        cumulative_pitch += pitch
        cumulative_yaw += yaw
        
        # Compute rotation matrix for this frame
        R = euler_to_rotation_matrix(roll, pitch, yaw)
        
        # Update cumulative orientation
        current_orientation = orientations[-1] @ R
        orientations.append(current_orientation)
        
        # Get current forward direction
        current_forward = current_orientation @ forward
        
        # Update position (walking forward in the direction camera is facing)
        new_position = positions[-1] + step_size * current_forward
        positions.append(new_position)
        
        # Print frame info
        total_rotation = np.sqrt(roll**2 + pitch**2 + yaw**2)
        
        print(f"[t={t:5.1f}s] Frame {i:02d}")
        print(f"    Rotation:  Roll={roll:+6.2f}°  Pitch={pitch:+6.2f}°  Yaw={yaw:+6.2f}°  (Total={total_rotation:5.2f}°)")
        print(f"    Cumulative: Roll={cumulative_roll:+7.2f}°  Pitch={cumulative_pitch:+7.2f}°  Yaw={cumulative_yaw:+7.2f}°")
        print(f"    Position:   X={new_position[0]:+6.2f}  Y={new_position[1]:+6.2f}  Z={new_position[2]:+6.2f}")
        print(f"    Facing:     X={current_forward[0]:+5.2f}  Y={current_forward[1]:+5.2f}  Z={current_forward[2]:+5.2f}")
        
        # Interpret movement
        if abs(yaw) > 3:
            direction = "LEFT" if yaw > 0 else "RIGHT"
            print(f"    Movement:   Turning {direction} ({abs(yaw):.1f}°)")
        elif abs(pitch) > 4:
            direction = "UP" if pitch > 0 else "DOWN"
            print(f"    Movement:   Looking {direction} ({abs(pitch):.1f}°)")
        elif abs(roll) > 3:
            direction = "LEFT" if roll > 0 else "RIGHT"
            print(f"    Movement:   Tilting {direction} ({abs(roll):.1f}°)")
        else:
            print(f"    Movement:   Walking forward (steady)")
        print()
    
    print("-" * 80)
    print("SUMMARY")
    print("-" * 80)
    print(f"Total time:        {timestamps[-1]:.1f} seconds")
    print(f"Final cumulative rotation:")
    print(f"    Roll (tilt):   {cumulative_roll:+.2f}°")
    print(f"    Pitch (up/down): {cumulative_pitch:+.2f}°")
    print(f"    Yaw (left/right): {cumulative_yaw:+.2f}°")
    print()
    
    final_pos = positions[-1]
    print(f"Estimated displacement from start:")
    print(f"    X (left/right): {final_pos[0]:+.2f} units")
    print(f"    Y (up/down):    {final_pos[1]:+.2f} units")
    print(f"    Z (forward):    {final_pos[2]:+.2f} units")
    print()
    
    # Interpret overall movement pattern
    print("INTERPRETATION:")
    if cumulative_yaw < -10:
        print("  → You turned RIGHT overall during the video")
    elif cumulative_yaw > 10:
        print("  → You turned LEFT overall during the video")
    else:
        print("  → You walked mostly STRAIGHT")
    
    if cumulative_pitch < -20:
        print("  → Camera was tilting DOWN (looking at ground/phone)")
    elif cumulative_pitch > 20:
        print("  → Camera was tilting UP (looking at ceiling/sky)")
    else:
        print("  → Camera stayed mostly level vertically")
    
    print("=" * 80)
    
    return positions, orientations

def plot_path(positions, orientations):
    """Create 3D visualization of the walking path."""
    
    positions = np.array(positions)
    
    fig = plt.figure(figsize=(14, 5))
    
    # 3D path plot
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(positions[:, 0], positions[:, 2], positions[:, 1], 'b-', linewidth=2, label='Path')
    ax1.scatter(positions[0, 0], positions[0, 2], positions[0, 1], c='green', s=100, marker='o', label='Start')
    ax1.scatter(positions[-1, 0], positions[-1, 2], positions[-1, 1], c='red', s=100, marker='s', label='End')
    
    # Draw orientation arrows every few frames
    for i in range(0, len(orientations), 3):
        pos = positions[i]
        R = orientations[i]
        forward = R @ np.array([0, 0, 0.3])
        ax1.quiver(pos[0], pos[2], pos[1], forward[0], forward[2], forward[1], 
                   color='orange', alpha=0.7, arrow_length_ratio=0.3)
    
    ax1.set_xlabel('X (Left/Right)')
    ax1.set_ylabel('Z (Forward)')
    ax1.set_zlabel('Y (Up/Down)')
    ax1.set_title('3D Walking Path')
    ax1.legend()
    
    # Top-down view (X-Z plane)
    ax2 = fig.add_subplot(132)
    ax2.plot(positions[:, 0], positions[:, 2], 'b-', linewidth=2)
    ax2.scatter(positions[0, 0], positions[0, 2], c='green', s=100, marker='o', label='Start')
    ax2.scatter(positions[-1, 0], positions[-1, 2], c='red', s=100, marker='s', label='End')
    
    # Add time markers
    for i in range(0, len(positions), 5):
        ax2.annotate(f'{timestamps[min(i, len(timestamps)-1)]:.1f}s', 
                     (positions[i, 0], positions[i, 2]), fontsize=8)
    
    ax2.set_xlabel('X (Left/Right)')
    ax2.set_ylabel('Z (Forward)')
    ax2.set_title('Top-Down View (Bird\'s Eye)')
    ax2.legend()
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)
    
    # Side view (Z-Y plane)
    ax3 = fig.add_subplot(133)
    ax3.plot(positions[:, 2], positions[:, 1], 'b-', linewidth=2)
    ax3.scatter(positions[0, 2], positions[0, 1], c='green', s=100, marker='o', label='Start')
    ax3.scatter(positions[-1, 2], positions[-1, 1], c='red', s=100, marker='s', label='End')
    ax3.set_xlabel('Z (Forward)')
    ax3.set_ylabel('Y (Up/Down)')
    ax3.set_title('Side View')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('walking_path_visualization.png', dpi=150)
    print("\nVisualization saved to: walking_path_visualization.png")
    plt.show()


if __name__ == '__main__':
    positions, orientations = simulate_walk()
    
    # Try to plot if matplotlib is available
    try:
        plot_path(positions, orientations)
    except Exception as e:
        print(f"\nCould not create plot: {e}")
        print("Install matplotlib for visualization: pip install matplotlib")
