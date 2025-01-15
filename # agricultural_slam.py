import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from datetime import datetime
import os

class AgriSLAM:
    def __init__(self):
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")
            
        # Set camera resolution for better processing
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # Create output directories
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"field_mapping_{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/frames", exist_ok=True)
        
        # Initialize ORB detector
        self.orb = cv2.ORB_create(
            nfeatures=2000,        
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_FAST_SCORE,
            patchSize=31
        )
        
        # Initialize matcher with ratio test parameters
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Storage for mapping
        self.map_points = np.empty((0, 3))
        self.trajectory = np.empty((0, 3))
        self.keyframes = []
        
        # Camera parameters (will be updated in initialization)
        self.camera_matrix = np.array([
            [640, 0, 320],  # Focal length and principal point
            [0, 640, 240],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Initialize 3D visualization
        plt.ion()
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.setup_3d_plot()
        
    def setup_3d_plot(self):
        """Initialize the 3D plot settings"""
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_zlabel('Z (meters)')
        self.ax.set_title(' Field Mapping')
        plt.show(block=False)
        
    def match_features(self, desc1, desc2):
        """Perform robust feature matching with ratio test"""
        if desc1 is None or desc2 is None:
            return []
            
        # Find two nearest matches for ratio test
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        good_matches = []
        
        # Apply ratio test
        for match_pair in matches:
            if len(match_pair) == 2:  # Ensure we have two matches for ratio test
                m, n = match_pair
                if m.distance < 0.7 * n.distance:  # Ratio test
                    good_matches.append(m)
                    
        return good_matches
        
    def get_matched_points(self, kpts1, kpts2, matches):
        """Extract matched point coordinates"""
        if len(matches) < 8:  # Need at least 8 points for essential matrix
            return None, None
            
        pts1 = np.float32([kpts1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kpts2[m.trainIdx].pt for m in matches])
        
        return pts1, pts2
        
    def triangulate_points(self, pts1, pts2, R, t):
        """Triangulate 3D points from matched feature points"""
        if pts1 is None or pts2 is None:
            return None
            
        # Create projection matrices
        P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = np.hstack((R, t))
        
        P1 = self.camera_matrix @ P1
        P2 = self.camera_matrix @ P2
        
        # Reshape points for triangulation
        pts1_reshaped = pts1.T
        pts2_reshaped = pts2.T
        
        # Triangulate points
        points_4d = cv2.triangulatePoints(P1, P2, pts1_reshaped, pts2_reshaped)
        points_3d = points_4d[:3, :] / points_4d[3, :]
        
        return points_3d.T
        
    def update_3d_display(self):
        """Update the 3D visualization"""
        self.ax.clear()
        
        if len(self.map_points) > 0:
            # Plot field features
            self.ax.scatter(
                self.map_points[:, 0],
                self.map_points[:, 1],
                self.map_points[:, 2],
                c='g', marker='.', s=1, alpha=0.6,
                label='Field Features'
            )
            
        if len(self.trajectory) > 1:
            # Plot camera path
            self.ax.plot(
                self.trajectory[:, 0],
                self.trajectory[:, 1],
                self.trajectory[:, 2],
                'r-', linewidth=2,
                label='Camera Path'
            )
            
        self.ax.legend()
        self.setup_3d_plot()
        plt.draw()
        plt.pause(0.001)
        
    def run_mapping(self):
        """Main mapping loop"""
        prev_frame = None
        prev_kpts = None
        prev_desc = None
        cumulative_R = np.eye(3)
        cumulative_t = np.zeros((3, 1))
        
        print("Starting field mapping... Press 'q' to stop.")
        frame_count = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                    
                # Process current frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                kpts, desc = self.orb.detectAndCompute(gray, None)
                
                # Draw current keypoints
                frame_viz = cv2.drawKeypoints(frame, kpts, None, 
                    color=(0, 255, 0), 
                    flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
                )
                
                if prev_desc is not None:
                    # Match features
                    matches = self.match_features(desc, prev_desc)
                    
                    if len(matches) >= 8:
                        # Get matched points
                        pts1, pts2 = self.get_matched_points(kpts, prev_kpts, matches)
                        
                        if pts1 is not None and pts2 is not None:
                            # Estimate essential matrix
                            E, mask = cv2.findEssentialMat(
                                pts1, pts2, self.camera_matrix,
                                method=cv2.RANSAC,
                                prob=0.999,
                                threshold=1.0
                            )
                            
                            if E is not None:
                                # Recover pose
                                _, R, t, mask = cv2.recoverPose(
                                    E, pts1, pts2, self.camera_matrix, mask=mask
                                )
                                
                                # Update cumulative transformation
                                cumulative_R = R @ cumulative_R
                                cumulative_t = cumulative_t + cumulative_R @ t
                                
                                # Add to trajectory
                                self.trajectory = np.vstack((
                                    self.trajectory,
                                    cumulative_t.T
                                ))
                                
                                # Triangulate points using matched points
                                inlier_pts1 = pts1[mask.ravel() == 1]
                                inlier_pts2 = pts2[mask.ravel() == 1]
                                
                                if len(inlier_pts1) >= 8:
                                    points_3d = self.triangulate_points(
                                        inlier_pts1, inlier_pts2, R, t
                                    )
                                    
                                    if points_3d is not None:
                                        # Filter points by depth and distance
                                        valid_points = points_3d[points_3d[:, 2] > 0]
                                        if len(valid_points) > 0:
                                            self.map_points = np.vstack((
                                                self.map_points, valid_points
                                            ))
                                
                                # Update 3D visualization
                                self.update_3d_display()
                                
                                # Save keyframe
                                if frame_count % 30 == 0:
                                    frame_path = f"{self.output_dir}/frames/frame_{frame_count}.jpg"
                                    cv2.imwrite(frame_path, frame)
                
                # Update previous frame data
                prev_frame = frame
                prev_kpts = kpts
                prev_desc = desc
                
                # Show current frame
                cv2.imshow('Field Mapping', frame_viz)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
                frame_count += 1
                
        except Exception as e:
            print(f"Error during mapping: {str(e)}")
            
        finally:
            # Cleanup
            self.cap.release()
            cv2.destroyAllWindows()
            plt.close('all')
            
            # Save results
            np.savetxt(f"{self.output_dir}/trajectory.txt", self.trajectory)
            np.savetxt(f"{self.output_dir}/map_points.txt", self.map_points)
            print(f"Mapping complete. Results saved in {self.output_dir}")

if __name__ == "__main__":
    try:
        slam = AgriSLAM()
        slam.run_mapping()
    except Exception as e:
        print(f"Error: {str(e)}")