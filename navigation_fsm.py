"""
Navigation Finite State Machine for Viam Rover

This module implements a clean FSM-based navigation system with states:
- IDLE: Waiting for command
- SEARCHING: Spinning to find target
- APPROACHING: 3-phase navigation (ACQUIRE→ROTATE→DRIVE)
- ARRIVED: At target, stopped
- AVOIDING: Backing up from obstacle
- RETURNING: Navigating back to start position
"""

import asyncio
import time
import numpy as np


class NavigationState:
    """Navigation FSM States"""
    IDLE = "IDLE"
    SEARCHING = "SEARCHING"
    APPROACHING = "APPROACHING"
    ARRIVED = "ARRIVED"
    AVOIDING = "AVOIDING"
    RETURNING = "RETURNING"


class ApproachPhase:
    """Sub-phases within APPROACHING state"""
    ACQUIRE = "ACQUIRE"
    ROTATE = "ROTATE"
    DRIVE = "DRIVE"


class NavigationConfig:
    """Configuration for navigation behavior"""
    # Target distance
    target_distance_cm: float = 5.0   # Stop this close to target (gripper range)
    dist_threshold_cm: float = 2.0    # Tolerance (+/- 2cm)
    
    # Bearing thresholds (radians)
    bearing_threshold: float = 0.20       # ~11.5° - relaxed to reduce oscillation
    bearing_hysteresis: float = 0.15      # ~8.5° - prevents rapid switching
    large_turn_threshold: float = 0.35    # ~20° - use tank turn above this
    
    # Motor speeds (higher = fewer small movements = fewer API calls)
    rotate_speed: float = 0.28            # Tank turn speed
    pivot_speed: float = 0.25             # Pivot turn speed
    drive_speed: float = 0.35             # Forward drive speed (increased from 0.22)
    search_speed: float = 0.30            # Search rotation speed (was 0.22)
    backup_speed: float = 0.25            # Backup speed for avoiding
    
    # Camera (IMX708 - Pi Camera Module 3)
    camera_hfov_deg: float = 66.0  # IMX708 standard FOV (102° for wide version)
    frame_width: int = 1280        # Must match camera resolution in server_native.py
    
    # Acquire samples
    acquire_count: int = 3
    
    # Obstacle avoidance
    obstacle_min_distance_cm: float = 20.0  # Back up if closer than this
    backup_duration_sec: float = 0.8
    
    # Return navigation
    auto_return: bool = True              # Automatically return after reaching target
    return_distance_threshold: float = 15.0  # cm - close enough to start position
    
    # Motor drift compensation (negative = reduce left motor, positive = reduce right motor)
    drift_compensation: float = -0.10     # 10% reduction on LEFT motor (left is faster)


class NavigationFSM:
    """
    Finite State Machine for robot navigation.
    
    Receives detection and lidar data each frame, outputs motor commands.
    """
    
    def __init__(self, left_motor, right_motor, camera=None, imu=None, config: NavigationConfig = None):
        self.left_motor = left_motor
        self.right_motor = right_motor
        self.camera = camera
        self.imu = imu  # IMU for precision turns
        self.config = config or NavigationConfig()
        
        # State
        self.state = NavigationState.IDLE
        self.approach_phase = ApproachPhase.ACQUIRE
        
        # Approach phase data
        self.acquire_samples = []
        self.target_distance = 0.0
        self.target_bearing = 0.0
        self.target_imu_rotation = 0.0  # IMU: Stores exact angle to turn
        self.last_turn_dir = 0
        
        # Avoiding state data
        self.avoid_start_time = 0.0
        
        # Motor command coalescing (prevents Viam API flooding)
        self._last_left_power = None
        self._last_right_power = None
        self._last_motor_time = 0.0
        self._MOTOR_INTERVAL = 0.10  # 100ms = 10Hz max (was 20Hz, reduced to avoid API overflow)
        self._POWER_DEADBAND = 0.02  # 2% deadband
        self._MOTOR_TIMEOUT = 2.5    # Timeout for motor commands
        
        # Start position tracking for RETURNING state
        self.start_x = 0.0
        self.start_y = 0.0
        self.start_theta = 0.0
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_theta = 0.0
        self.return_phase = "ROTATE"  # ROTATE or DRIVE
        self.return_target_heading = 0.0
        
        # Callbacks for state changes (optional)
        self.on_state_change = None
        self.on_arrived = None
        self.on_returned = None  # Called when returned to start
        
        # Blind drive mode (for when target is lost at close range)
        self.blind_drive_start_pos = None  # (x, y) when blind drive started
        self.blind_drive_target_dist = 0.0  # How far to drive blindly
        self.last_valid_distance = 0.0  # Last known target distance
        self.BLIND_THRESHOLD_CM = 20.0  # If lost within this distance, drive blind
        self.target_lost_time = 0.0  # When we lost the target
        self.COAST_TIME_LIMIT = 0.3  # How long to wait before blind approach
    
    @property
    def state_summary(self) -> str:
        """Human-readable state summary"""
        if self.state == NavigationState.APPROACHING:
            return f"{self.state}/{self.approach_phase}"
        return self.state
    
    def _set_state(self, new_state: str, phase: str = None):
        """Change state with optional callback"""
        old_state = self.state
        self.state = new_state
        if phase:
            self.approach_phase = phase
        
        if old_state != new_state:
            print(f"Nav: {old_state} → {new_state}")
            if self.on_state_change:
                self.on_state_change(old_state, new_state)
    
    async def start(self, start_pose: dict = None):
        """Start navigation - enters SEARCHING state
        
        Args:
            start_pose: Optional dict with 'x', 'y', 'theta' for return navigation
        """
        self.acquire_samples.clear()
        self.target_distance = 0.0
        self.target_bearing = 0.0
        self.last_turn_dir = 0
        
        # Store start position for RETURNING
        if start_pose:
            self.start_x = start_pose.get('x', 0.0)
            self.start_y = start_pose.get('y', 0.0)
            self.start_theta = start_pose.get('theta', 0.0)
        else:
            self.start_x = 0.0
            self.start_y = 0.0
            self.start_theta = 0.0
        
        self._set_state(NavigationState.SEARCHING)
        print(f"🚀 Navigation started - SEARCHING for target (start: x={self.start_x:.1f}, y={self.start_y:.1f})")
    
    async def start_approach(self):
        """Start direct approach - skips SEARCHING (for when target already visible)"""
        self.acquire_samples.clear()
        self._set_state(NavigationState.APPROACHING, ApproachPhase.ACQUIRE)
        print("🎯 Approaching target - ACQUIRING")
    
    async def stop(self):
        """Stop navigation and motors"""
        was_active = self.state != NavigationState.IDLE
        self._set_state(NavigationState.IDLE)
        await self._stop_motors()
        if was_active:
            print("⏹ Navigation stopped")
    
    def update_motors(self, left_motor, right_motor, imu=None):
        """Update motor references (call after reconnection)"""
        self.left_motor = left_motor
        self.right_motor = right_motor
        if imu is not None:
            self.imu = imu
        # Reset coalescing state
        self._last_left_power = None
        self._last_right_power = None
        self._last_motor_time = 0.0
        print("✓ NavigationFSM motors updated")
    
    async def update(self, detection: dict = None, lidar_min_distance_cm: float = None, current_pose: dict = None):
        """
        Called each frame to update navigation.
        
        Args:
            detection: Dict with 'distance_cm' and 'center_x' from YOLO
            lidar_min_distance_cm: Minimum distance from lidar (for obstacle avoidance)
            current_pose: Dict with 'x', 'y', 'theta' for return navigation
        """
        # Update current pose for RETURNING
        if current_pose:
            self.current_x = current_pose.get('x', 0.0)
            self.current_y = current_pose.get('y', 0.0)
            self.current_theta = current_pose.get('theta', 0.0)
        
        if self.state == NavigationState.IDLE:
            return
        
        if self.state == NavigationState.ARRIVED:
            # Check if auto-return is enabled
            if self.config.auto_return:
                self._start_return()
            return
        
        # Check for obstacles first (safety) - but not during RETURNING
        if self.state != NavigationState.RETURNING:
            if lidar_min_distance_cm is not None and lidar_min_distance_cm < self.config.obstacle_min_distance_cm:
                if self.state != NavigationState.AVOIDING:
                    self._set_state(NavigationState.AVOIDING)
                    self.avoid_start_time = time.time()
                    print(f"⚠️ Obstacle detected at {lidar_min_distance_cm:.1f}cm - backing up")
        
        # Handle current state
        if self.state == NavigationState.SEARCHING:
            await self._handle_searching(detection)
        elif self.state == NavigationState.APPROACHING:
            await self._handle_approaching(detection)
        elif self.state == NavigationState.AVOIDING:
            await self._handle_avoiding()
        elif self.state == NavigationState.RETURNING:
            await self._handle_returning()
    
    # =========================================================================
    # STATE HANDLERS
    # =========================================================================
    
    async def _handle_searching(self, detection: dict):
        """SEARCHING: Spin slowly looking for target"""
        if detection and detection.get('distance_cm'):
            # Found target! Switch to APPROACHING
            self._set_state(NavigationState.APPROACHING, ApproachPhase.ACQUIRE)
            self.acquire_samples.clear()
            print("🎯 Target found! → ACQUIRING")
            await self._stop_motors()
        else:
            # Keep spinning slowly
            await self._set_motor_power(self.config.search_speed, -self.config.search_speed)
    
    async def _handle_approaching(self, detection: dict):
        """APPROACHING: 3-phase approach (ACQUIRE → ROTATE → DRIVE)"""
        
        # Check if we have a valid detection
        if not detection or not detection.get('distance_cm'):
            # Lost target!
            time_since_loss = time.time() - self.target_lost_time if self.target_lost_time else 0
            
            # If we just lost it, start timing
            if self.target_lost_time == 0:
                self.target_lost_time = time.time()
                await self._stop_motors()
                return
            
            # Wait a short time before deciding (target might reappear)
            if time_since_loss < self.COAST_TIME_LIMIT:
                await self._stop_motors()
                return
            
            # Target lost for too long - check if we should blind drive
            if self.last_valid_distance > 0 and self.last_valid_distance < self.BLIND_THRESHOLD_CM:
                print(f"  🙈 Blind Approach Activated! Target too close to see (was {self.last_valid_distance:.1f}cm)")
                
                # Calculate how much further we need to go
                # e.g. Last saw it at 18cm, we want to stop at 10cm. Drive 8cm more.
                remaining_dist = self.last_valid_distance - self.config.target_distance_cm
                
                if remaining_dist > 0:
                    await self._execute_blind_drive(remaining_dist)
                    return
                else:
                    # Already close enough!
                    print(f"✓ TARGET REACHED (blind)! Last distance: {self.last_valid_distance:.1f}cm")
                    self._set_state(NavigationState.ARRIVED)
                    await self._stop_motors()
                    if self.on_arrived:
                        self.on_arrived()
                    return
            
            # Not close enough for blind drive, just stop
            await self._stop_motors()
            return
        
        # We have a detection - reset lost timer and update last known distance
        self.target_lost_time = 0
        det_distance = detection['distance_cm']
        self.last_valid_distance = det_distance

        # ZONE FOCUSING LOGIC
        # Only set focus if we have the new camera driver
        if hasattr(self.camera, 'set_focus'):
            if det_distance > 80:
                self.camera.set_focus(0.0)  # Infinity (Search/Far)
            elif det_distance > 30:
                self.camera.set_focus(4.0)  # Mid-range
            else:
                self.camera.set_focus(8.0)  # Macro (Grab)

        det_center_x = detection.get('center_x', self.config.frame_width / 2)
        
        # Calculate bearing from center
        frame_center = self.config.frame_width / 2
        pixel_offset = det_center_x - frame_center
        det_bearing = pixel_offset * (self.config.camera_hfov_deg / self.config.frame_width) * (np.pi / 180.0)
        
        # Sub-phase logic
        if self.approach_phase == ApproachPhase.ACQUIRE:
            await self._handle_acquire(det_distance, det_bearing)
        elif self.approach_phase == ApproachPhase.ROTATE:
            await self._handle_rotate(det_bearing)
        elif self.approach_phase == ApproachPhase.DRIVE:
            await self._handle_drive(det_distance, det_bearing)
    
    async def _handle_acquire(self, distance: float, bearing: float):
        """ACQUIRE sub-phase: Collect samples, average, and LOCK IN IMU TARGET"""
        self.acquire_samples.append((distance, bearing))
        print(f"  Sample {len(self.acquire_samples)}/{self.config.acquire_count}: dist={distance:.1f}cm")
        
        if len(self.acquire_samples) >= self.config.acquire_count:
            # Average the samples
            avg_dist = sum(s[0] for s in self.acquire_samples) / len(self.acquire_samples)
            avg_bearing = sum(s[1] for s in self.acquire_samples) / len(self.acquire_samples)
            self.target_distance = avg_dist
            self.target_bearing = avg_bearing
            
            print(f"✓ Target acquired: dist={avg_dist:.1f}cm, bearing={np.degrees(avg_bearing):.1f}°")
            
            # Decide next phase
            if abs(avg_bearing) > self.config.bearing_threshold:
                self.approach_phase = ApproachPhase.ROTATE
                
                # IMU Pre-calculation: Lock in turn amount
                if self.imu:
                    # Reset IMU "zero" to now. We want to turn exactly 'avg_bearing' amount.
                    # Camera bearing: positive = target is RIGHT
                    # Motor logic: positive remaining_turn = turn RIGHT
                    # IMU heading: positive = counterclockwise (LEFT) rotation
                    # So target_imu_rotation should match bearing sign
                    self.imu.reset_heading()
                    self.target_imu_rotation = avg_bearing
                    print(f"  → ROTATE (IMU Precision Turn: {np.degrees(avg_bearing):.1f}°)")
                else:
                    print(f"  → ROTATE (Camera Reactive Turn: {np.degrees(avg_bearing):.1f}°)")
            else:
                self.approach_phase = ApproachPhase.DRIVE
                print("  → DRIVE")
    
    async def _handle_rotate(self, bearing: float):
        """
        ROTATE sub-phase: SMOOTH PIVOT TURN (1-Wheel Locked)
        """
        
        # 1. Determine how much is left to turn
        # First check for overshoot: if camera bearing flipped sign from original target,
        # we've turned too far and need to reverse direction
        overshot = (np.sign(bearing) != np.sign(self.target_bearing) and 
                    np.sign(self.target_bearing) != 0 and
                    abs(bearing) > self.config.bearing_hysteresis)
        
        if overshot:
            # We overshot! Use current camera bearing to correct
            # This overrides IMU calculation since camera shows actual target position
            remaining_turn = bearing
            threshold = self.config.bearing_hysteresis
            print(f"  ⚠ Overshoot detected! Target now at {np.degrees(bearing):.1f}° - reversing")
        elif self.imu:
            # IMU: Calculate remaining angle based on gyro integration
            # Sign convention analysis:
            # - Camera bearing: positive = target is RIGHT of center
            # - Motor logic: positive remaining_turn = turn RIGHT (left wheel forward)
            # - IMU heading: positive = counterclockwise (LEFT), negative = clockwise (RIGHT)
            # 
            # When target is RIGHT (positive bearing), we turn RIGHT (clockwise).
            # Turning RIGHT makes heading go NEGATIVE.
            # We want remaining_turn to approach 0 as we reach the target.
            # 
            # remaining_turn = target + heading
            # - Start: target = +X, heading = 0, remaining = +X (turn right)
            # - Mid:   target = +X, heading = -X/2, remaining = +X/2 (keep turning)
            # - End:   target = +X, heading = -X, remaining = 0 (done!)
            current_heading = self.imu.get_heading()
            remaining_turn = self.target_imu_rotation + current_heading
            threshold = 0.05  # ~3 degrees precision for IMU
        else:
            # Fallback: Use current camera bearing for reactive turn control
            # The bearing tells us where the target currently is relative to center
            remaining_turn = bearing
            threshold = self.config.bearing_hysteresis

        # 2. Check if we are done (target is centered)
        if abs(bearing) <= threshold:  # Use camera bearing for final check, not IMU
            await self._stop_motors()
            self.approach_phase = ApproachPhase.DRIVE
            self.last_turn_dir = 0
            print("✓ Aligned! → DRIVE")
            # Small pause to let chassis settle before driving
            await asyncio.sleep(0.2)
            return
        
        # 3. Calculate Smooth Speed (Ramp Down)
        # "Deliberate" movement: Start at pivot_speed, slow down as we get closer.
        # MIN_MOVING_POWER ensures we don't stall due to friction.
        
        base_speed = self.config.pivot_speed  # Default 0.25
        MIN_MOVING_POWER = 0.24  # Calibrated value (overcomes friction)
        
        # Simple P-Control: Scale speed based on remaining error 
        dynamic_speed = abs(remaining_turn) * 1.5 
        
        # Clamp speed between stall-threshold and max pivot speed
        target_speed = max(MIN_MOVING_POWER, min(base_speed, dynamic_speed))
        
        # 4. Execute Pivot Turn (One Wheel Locked)
        if remaining_turn > 0:
            # Turn RIGHT -> Lock RIGHT wheel, Drive LEFT wheel forward
            # Pivot point is the right wheel
            l_pow = target_speed
            r_pow = 0.0
        else:
            # Turn LEFT -> Lock LEFT wheel, Drive RIGHT wheel forward
            # Pivot point is the left wheel
            l_pow = 0.0
            r_pow = target_speed
            
        await self._set_motor_power(l_pow, r_pow)
    
    async def _handle_drive(self, distance: float, bearing: float):
        """DRIVE sub-phase: Drive straight to target"""
        # Check if bearing drifted too much
        if abs(bearing) > self.config.bearing_threshold * 2:
            print(f"⚠ Target drifted ({np.degrees(bearing):.1f}°) - re-rotating")
            self.approach_phase = ApproachPhase.ROTATE
            await self._stop_motors()
            return
        
        # Check if arrived
        if distance <= self.config.target_distance_cm + self.config.dist_threshold_cm:
            print(f"✓ TARGET REACHED! Distance: {distance:.1f}cm")
            self._set_state(NavigationState.ARRIVED)
            await self._stop_motors()
            if self.on_arrived:
                self.on_arrived()
            return
        
        # Drive forward
        await self._set_motor_power(self.config.drive_speed, self.config.drive_speed)
    
    async def _execute_blind_drive(self, distance_cm: float):
        """
        Blind Drive: Use odometry to drive forward a specified distance without camera.
        Called when target is lost at close range (under our camera's view).
        """
        # 1. Initialize start position if this is the first blind frame
        if self.blind_drive_start_pos is None:
            # Save current robot X/Y from odometry
            self.blind_drive_start_pos = (self.current_x, self.current_y)
            self.blind_drive_target_dist = distance_cm
            print(f"  🙈 Blind Drive: Starting from ({self.current_x:.1f}, {self.current_y:.1f}), target {distance_cm:.1f}cm")
        
        # 2. Calculate distance traveled since start
        dx = self.current_x - self.blind_drive_start_pos[0]
        dy = self.current_y - self.blind_drive_start_pos[1]
        traveled = np.sqrt(dx*dx + dy*dy)
        
        remaining = self.blind_drive_target_dist - traveled
        print(f"  🙈 Blind Drive: {remaining:.1f}cm to go")
        
        # 3. Check if we arrived
        if remaining <= 0:
            print("  ✓ Blind Arrival Complete!")
            self._set_state(NavigationState.ARRIVED)
            await self._stop_motors()
            self.blind_drive_start_pos = None  # Reset for next time
            self.last_valid_distance = 0.0
            if self.on_arrived:
                self.on_arrived()
        else:
            # Drive straight forward
            await self._set_motor_power(self.config.drive_speed, self.config.drive_speed)
    
    async def _handle_avoiding(self):
        """AVOIDING: Back up from obstacle"""
        elapsed = time.time() - self.avoid_start_time
        
        if elapsed < self.config.backup_duration_sec:
            # Still backing up
            await self._set_motor_power(-self.config.backup_speed, -self.config.backup_speed)
        else:
            # Done backing up - return to SEARCHING
            await self._stop_motors()
            self._set_state(NavigationState.SEARCHING)
            print("↩ Backup complete → SEARCHING")
    
    def _start_return(self):
        """Initialize return to start position"""
        # Calculate bearing to start position
        dx = self.start_x - self.current_x
        dy = self.start_y - self.current_y
        
        # Distance to start
        distance_to_start = np.sqrt(dx**2 + dy**2)
        
        if distance_to_start < self.config.return_distance_threshold:
            # Already close enough to start
            print("✓ Already at start position - navigation complete")
            self._set_state(NavigationState.IDLE)
            if self.on_returned:
                self.on_returned()
            return
        
        # Calculate target heading (angle to start)
        self.return_target_heading = np.arctan2(dy, dx)
        self.return_phase = "ROTATE"
        
        # Reset IMU for precision turn
        if self.imu:
            self.imu.reset_heading()
            # Calculate how much we need to turn
            heading_diff = self.return_target_heading - self.current_theta
            # Normalize to [-pi, pi]
            while heading_diff > np.pi:
                heading_diff -= 2 * np.pi
            while heading_diff < -np.pi:
                heading_diff += 2 * np.pi
            self.target_imu_rotation = heading_diff
        
        self._set_state(NavigationState.RETURNING)
        print(f"↩ RETURNING to start (dist={distance_to_start:.1f}cm, heading={np.degrees(self.return_target_heading):.1f}°)")
    
    async def _handle_returning(self):
        """RETURNING: Navigate back to start position"""
        # Calculate distance to start
        dx = self.start_x - self.current_x
        dy = self.start_y - self.current_y
        distance_to_start = np.sqrt(dx**2 + dy**2)
        
        # Check if arrived at start
        if distance_to_start < self.config.return_distance_threshold:
            await self._stop_motors()
            self._set_state(NavigationState.IDLE)
            print(f"✓ RETURNED to start! Distance: {distance_to_start:.1f}cm")
            if self.on_returned:
                self.on_returned()
            return
        
        if self.return_phase == "ROTATE":
            # Rotate to face start position
            if self.imu:
                current_heading = self.imu.get_heading()
                remaining_turn = self.target_imu_rotation - current_heading
                threshold = 0.08  # ~5 degrees
            else:
                # Fallback: use odometry heading
                heading_diff = self.return_target_heading - self.current_theta
                while heading_diff > np.pi:
                    heading_diff -= 2 * np.pi
                while heading_diff < -np.pi:
                    heading_diff += 2 * np.pi
                remaining_turn = heading_diff
                threshold = 0.15
            
            if abs(remaining_turn) <= threshold:
                await self._stop_motors()
                self.return_phase = "DRIVE"
                print("  ✓ Aligned → DRIVING to start")
                await asyncio.sleep(0.2)
                return
            
            # Turn toward start
            MIN_MOVING_POWER = 0.24
            target_speed = max(MIN_MOVING_POWER, min(self.config.pivot_speed, abs(remaining_turn) * 1.5))
            
            if remaining_turn > 0:
                await self._set_motor_power(target_speed, 0.0)
            else:
                await self._set_motor_power(0.0, target_speed)
        
        elif self.return_phase == "DRIVE":
            # Drive toward start
            # Recalculate bearing in case of drift
            target_heading = np.arctan2(dy, dx)
            heading_error = target_heading - self.current_theta
            while heading_error > np.pi:
                heading_error -= 2 * np.pi
            while heading_error < -np.pi:
                heading_error += 2 * np.pi
            
            # If drifted too much, go back to ROTATE
            if abs(heading_error) > 0.5:  # ~30 degrees
                self.return_phase = "ROTATE"
                if self.imu:
                    self.imu.reset_heading()
                    self.target_imu_rotation = heading_error
                print(f"  ⚠ Drift detected ({np.degrees(heading_error):.1f}°) → re-rotating")
                return
            
            # Drive forward
            await self._set_motor_power(self.config.drive_speed, self.config.drive_speed)
    
    # =========================================================================
    # MOTOR HELPERS
    # =========================================================================
    
    async def _set_motor_power(self, left: float, right: float):
        """Set motor power with coalescing and timeout handling"""
        now = time.time()
        
        # Coalescing: Skip if power values haven't changed significantly
        # and minimum interval hasn't elapsed
        left_changed = self._last_left_power is None or abs(left - self._last_left_power) > self._POWER_DEADBAND
        right_changed = self._last_right_power is None or abs(right - self._last_right_power) > self._POWER_DEADBAND
        interval_elapsed = (now - self._last_motor_time) >= self._MOTOR_INTERVAL
        is_stop = (left == 0 and right == 0)
        
        # Only send if something changed OR interval elapsed OR it's a stop command
        if not (left_changed or right_changed or interval_elapsed or is_stop):
            return
        
        try:
            if self.left_motor and self.right_motor:
                # Apply drift compensation (reduce one motor to correct for drift)
                comp = self.config.drift_compensation
                if comp < 0:
                    # Negative = reduce LEFT motor
                    left_adj = left * (1.0 + comp)  # e.g., 0.9 for -0.10
                    right_adj = right
                else:
                    # Positive = reduce RIGHT motor
                    left_adj = left
                    right_adj = right * (1.0 - comp)
                
                # Handle synchronous motors (server_native)
                if not asyncio.iscoroutinefunction(self.left_motor.set_power):
                    self.left_motor.set_power(left_adj)
                    self.right_motor.set_power(right_adj)
                else:
                    # Handle async motors (Viam SDK)
                    await asyncio.wait_for(
                        asyncio.gather(
                            self.left_motor.set_power(left_adj),
                            self.right_motor.set_power(right_adj)
                        ),
                        timeout=self._MOTOR_TIMEOUT
                    )
                
                # Update state on success
                self._last_left_power = left
                self._last_right_power = right
                self._last_motor_time = now
        except asyncio.TimeoutError:
            print(f"⚠ Nav motor timeout (L={left:.2f}, R={right:.2f})")
        except Exception as e:
            print(f"Motor error: {e}")
    
    async def _stop_motors(self):
        """Stop both motors"""
        await self._set_motor_power(0, 0)
