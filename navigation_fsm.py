"""
Navigation Finite State Machine for Viam Rover

This module implements a clean FSM-based navigation system with states:
- IDLE: Waiting for command
- SEARCHING: Spinning to find target
- APPROACHING: 3-phase navigation (ACQUIRE→ROTATE→DRIVE)
- ARRIVED: At target, stopped
- AVOIDING: Backing up from obstacle
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


class ApproachPhase:
    """Sub-phases within APPROACHING state"""
    ACQUIRE = "ACQUIRE"
    ROTATE = "ROTATE"
    DRIVE = "DRIVE"


class NavigationConfig:
    """Configuration for navigation behavior"""
    # Target distance
    target_distance_cm: float = 15.0
    dist_threshold_cm: float = 3.0
    
    # Bearing thresholds (radians)
    bearing_threshold: float = 0.12       # ~7° - aligned enough to drive
    bearing_hysteresis: float = 0.08      # ~4.5° - prevents oscillation
    large_turn_threshold: float = 0.35    # ~20° - use tank turn above this
    
    # Motor speeds (higher = fewer small movements = fewer API calls)
    rotate_speed: float = 0.28            # Tank turn speed
    pivot_speed: float = 0.25             # Pivot turn speed
    drive_speed: float = 0.35             # Forward drive speed (increased from 0.22)
    search_speed: float = 0.22            # Search rotation speed
    backup_speed: float = 0.25            # Backup speed for avoiding
    
    # Camera
    camera_hfov_deg: float = 76.5
    frame_width: int = 640
    
    # Acquire samples
    acquire_count: int = 3
    
    # Obstacle avoidance
    obstacle_min_distance_cm: float = 20.0  # Back up if closer than this
    backup_duration_sec: float = 0.8


class NavigationFSM:
    """
    Finite State Machine for robot navigation.
    
    Receives detection and lidar data each frame, outputs motor commands.
    """
    
    def __init__(self, left_motor, right_motor, config: NavigationConfig = None):
        self.left_motor = left_motor
        self.right_motor = right_motor
        self.config = config or NavigationConfig()
        
        # State
        self.state = NavigationState.IDLE
        self.approach_phase = ApproachPhase.ACQUIRE
        
        # Approach phase data
        self.acquire_samples = []
        self.target_distance = 0.0
        self.target_bearing = 0.0
        self.last_turn_dir = 0
        
        # Avoiding state data
        self.avoid_start_time = 0.0
        
        # Motor command coalescing (prevents Viam API flooding)
        self._last_left_power = None
        self._last_right_power = None
        self._last_motor_time = 0.0
        self._MOTOR_INTERVAL = 0.05  # 50ms = 20Hz max
        self._POWER_DEADBAND = 0.02  # 2% deadband
        self._MOTOR_TIMEOUT = 2.5    # Timeout for motor commands
        
        # Callbacks for state changes (optional)
        self.on_state_change = None
        self.on_arrived = None
    
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
    
    async def start(self):
        """Start navigation - enters SEARCHING state"""
        self.acquire_samples.clear()
        self.target_distance = 0.0
        self.target_bearing = 0.0
        self.last_turn_dir = 0
        self._set_state(NavigationState.SEARCHING)
        print("🚀 Navigation started - SEARCHING for target")
    
    async def start_approach(self):
        """Start direct approach - skips SEARCHING (for when target already visible)"""
        self.acquire_samples.clear()
        self._set_state(NavigationState.APPROACHING, ApproachPhase.ACQUIRE)
        print("🎯 Approaching target - ACQUIRING")
    
    async def stop(self):
        """Stop navigation and motors"""
        self._set_state(NavigationState.IDLE)
        await self._stop_motors()
        print("⏹ Navigation stopped")
    
    def update_motors(self, left_motor, right_motor):
        """Update motor references (call after reconnection)"""
        self.left_motor = left_motor
        self.right_motor = right_motor
        # Reset coalescing state
        self._last_left_power = None
        self._last_right_power = None
        self._last_motor_time = 0.0
        print("✓ NavigationFSM motors updated")
    
    async def update(self, detection: dict = None, lidar_min_distance_cm: float = None):
        """
        Called each frame to update navigation.
        
        Args:
            detection: Dict with 'distance_cm' and 'center_x' from YOLO
            lidar_min_distance_cm: Minimum distance from lidar (for obstacle avoidance)
        """
        if self.state == NavigationState.IDLE:
            return
        
        if self.state == NavigationState.ARRIVED:
            return
        
        # Check for obstacles first (safety)
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
        if not detection or not detection.get('distance_cm'):
            # Lost target - stop motors but don't change state yet
            await self._stop_motors()
            return
        
        det_distance = detection['distance_cm']
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
        """ACQUIRE sub-phase: Collect samples and average"""
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
                print(f"  → ROTATE (need to turn {np.degrees(avg_bearing):.1f}°)")
            else:
                self.approach_phase = ApproachPhase.DRIVE
                print("  → DRIVE")
    
    async def _handle_rotate(self, bearing: float):
        """ROTATE sub-phase: Turn toward target"""
        # Check if aligned (with hysteresis)
        if abs(bearing) <= self.config.bearing_hysteresis:
            await self._stop_motors()
            self.approach_phase = ApproachPhase.DRIVE
            self.last_turn_dir = 0
            print("✓ Aligned! → DRIVE")
            return
        
        if abs(bearing) > self.config.bearing_threshold:
            turn_dir = 1 if bearing > 0 else -1
            
            # Oscillation detection
            speed_mult = 0.7 if (self.last_turn_dir != 0 and turn_dir != self.last_turn_dir) else 1.0
            self.last_turn_dir = turn_dir
            
            if abs(bearing) > self.config.large_turn_threshold:
                # Large turn - tank turn (both wheels opposite)
                l_pow = self.config.rotate_speed * turn_dir * speed_mult
                r_pow = -self.config.rotate_speed * turn_dir * speed_mult
            else:
                # Small turn - pivot turn (one wheel only)
                if turn_dir > 0:
                    l_pow = self.config.pivot_speed * speed_mult
                    r_pow = 0.0
                else:
                    l_pow = 0.0
                    r_pow = self.config.pivot_speed * speed_mult
            
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
                await asyncio.wait_for(
                    asyncio.gather(
                        self.left_motor.set_power(left),
                        self.right_motor.set_power(right)
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
