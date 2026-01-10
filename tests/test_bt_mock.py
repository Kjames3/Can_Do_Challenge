
import unittest
import asyncio
import sys
import os
from unittest.mock import MagicMock, AsyncMock

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from navigation_fsm import NavigationFSM, NodeStatus

class TestBT(unittest.IsolatedAsyncioTestCase):
    async def test_structure(self):
        """Verify tree is built"""
        fsm = NavigationFSM(None, None)
        self.assertIsNotNone(fsm.tree)
        
    async def test_search_default(self):
        """Should be SEARCHING by default"""
        fsm = NavigationFSM(AsyncMock(), AsyncMock())
        await fsm.start()
        await fsm.update()
        self.assertEqual(fsm.state, "SEARCHING")

    async def test_obstacle_avoidance(self):
        """Obstacle should trigger AVOIDING state (Highest Priority)"""
        fsm = NavigationFSM(AsyncMock(), AsyncMock())
        await fsm.start()
        
        # Inject obstacle
        await fsm.update(lidar_min_distance_cm=10.0)
        
        self.assertEqual(fsm.state, "AVOIDING")
        
        # Verify backing up
        fsm.ctx.left_motor.set_power.assert_called_with(-0.25)

    async def test_return_home_trigger(self):
        """Arrived + AutoReturn -> RETURNING"""
        fsm = NavigationFSM(AsyncMock(), AsyncMock())
        await fsm.start()
        fsm.ctx.nav_state = "ARRIVED"
        fsm.ctx.config.auto_return = True
        
        await fsm.update()
        
        self.assertEqual(fsm.state, "RETURNING")

    async def test_approach_goal(self):
        """Goal set -> APPROACHING"""
        fsm = NavigationFSM(AsyncMock(), AsyncMock())
        await fsm.start()
        
        # Set a goal manually
        fsm.ctx.goal_x = 100
        fsm.ctx.goal_y = 0
        
        await fsm.update(current_pose={'x':0, 'y':0, 'theta':0})
        
        self.assertEqual(fsm.state, "APPROACHING")
        self.assertEqual(fsm.ctx.approach_phase, "DRIVE")

if __name__ == '__main__':
    unittest.main()
