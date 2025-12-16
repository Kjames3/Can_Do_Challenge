import asyncio
from viam.robot.client import RobotClient
from viam.rpc.dial import DialOptions

ROBOT_ADDRESS = "yeep-main.ayzp4fw8rj.viam.cloud"
API_KEY_ID = "d55dcbc4-6c31-4d78-97d9-57792293a0b7"
API_KEY = "3u88u6fsuowp1wv4inpyebnv13k6dkhn"

async def main():
    opts = RobotClient.Options.with_api_key(api_key=API_KEY, api_key_id=API_KEY_ID)
    robot = await RobotClient.at_address(ROBOT_ADDRESS, opts)
    
    print("Connected resources:")
    for name in robot.resource_names:
        print(f" - {name.name} ({name.subtype})")
        
    await robot.close()

if __name__ == "__main__":
    asyncio.run(main())
