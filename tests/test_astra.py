import asyncio
import time
from viam.robot.client import RobotClient
from viam.components.camera import Camera
from viam.rpc.dial import DialOptions
from viam.components.power_sensor import PowerSensor

ROBOT_ADDRESS = "yeep-main.ayzp4fw8rj.viam.cloud"
API_KEY_ID = "d55dcbc4-6c31-4d78-97d9-57792293a0b7"
API_KEY = "3u88u6fsuowp1wv4inpyebnv13k6dkhn"

# TODO: Update this name after discovery
ASTRA_NAME = "astra" 

async def main():
    print("Connecting to robot...")
    opts = RobotClient.Options.with_api_key(api_key=API_KEY, api_key_id=API_KEY_ID)
    
    try:
        robot = await RobotClient.at_address(ROBOT_ADDRESS, opts)
        print("✓ Connected.")
        
        # Try to get the camera
        try:
            astra = Camera.from_robot(robot, ASTRA_NAME)
            print(f"✓ Found camera: {ASTRA_NAME}")
        except Exception:
            print(f"✗ Camera '{ASTRA_NAME}' not found on robot.")
            # List available just in case
            print("Available resources:")
            for name in robot.resource_names:
                print(f" - {name.name}")
            await robot.close()
            return

        print("\nTesting Data Streams...")
        
        # Test 1: Get Image (RGB/Color)
        try:
            print("Fetching RGB Image...")
            # Viam default get_image is often RGB
            img = await astra.get_image(mime_type="image/jpeg")
            print(f"✓ Received Image: {img.width}x{img.height}, {img.mime_type}")
        except Exception as e:
            print(f"✗ Failed to get RGB image: {e}")

        # Test 2: Get Point Cloud (Depth)
        try:
            print("Fetching Point Cloud...")
            pc, _ = await astra.get_point_cloud()
            # pc is bytes
            print(f"✓ Received Point Cloud: {len(pc)} bytes")
        except Exception as e:
            print(f"✗ Failed to get point cloud: {e}")
            
        print("\nTest Complete.")
        await robot.close()
        
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
