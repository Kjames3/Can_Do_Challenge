"""
Test Script for Yahboom X3 Motor Mapping
Run this to identify which motor is M1, M2, M3, M4.

Usage:
    python test_x3_motors.py
"""
import time
import logging
from drivers_x3 import Rosmaster

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MotorTest")

def main():
    logger.info("Initializing ROSMASTER Serial Connection...")
    bot = Rosmaster(port="/dev/ttyUSB0")
    time.sleep(1) # Allow connection settle
    
    logger.info("Starting Motor Test Sequence.")
    logger.info("Ensure robot is on a stand/blocks!")
    
    try:
        # Test M1
        logger.info("Testing M1 (Front Left?) - Forward 50%")
        bot.set_motor(0.5, 0, 0, 0)
        time.sleep(2)
        bot.stop()
        time.sleep(1)

        # Test M2
        logger.info("Testing M2 (Front Right?) - Forward 50%")
        bot.set_motor(0, 0.5, 0, 0)
        time.sleep(2)
        bot.stop()
        time.sleep(1)

        # Test M3
        logger.info("Testing M3 (Rear Left?) - Forward 50%")
        bot.set_motor(0, 0, 0.5, 0)
        time.sleep(2)
        bot.stop()
        time.sleep(1)

        # Test M4
        logger.info("Testing M4 (Rear Right?) - Forward 50%")
        bot.set_motor(0, 0, 0, 0.5)
        time.sleep(2)
        bot.stop()
        time.sleep(1)
        
        logger.info("Test Complete.")
        
    except KeyboardInterrupt:
        logger.info("Test Interrupted!")
    finally:
        bot.cleanup()

if __name__ == "__main__":
    main()
