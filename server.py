import asyncio
import json
import websockets
from viam.robot.client import RobotClient
from viam.rpc.dial import DialOptions
from viam.components.motor import Motor
from viam.components.camera import Camera
import base64
import io
import struct
import numpy as np

# --- !!! EDIT THESE VALUES !!! ---
# Replace with your robot's connection details
ROBOT_ADDRESS = "rover-2-main.ayzp4fw8rj.viam.cloud"
API_KEY_ID = "4ed5a063-a693-49a1-9212-2f04c7a50dd6"
API_KEY = "q2h3tup8scoj1dnhneolqnoa3c5pw8br"
# ---------------------------------

# --- Component Names ---
LEFT_MOTOR_NAME = "left"
RIGHT_MOTOR_NAME = "right"
CAMERA_NAME = "cam"
LIDAR_NAME = "rplidar"
# ---------------------------------

# Global variables
robot: RobotClient = None
left_motor: Motor = None
right_motor: Motor = None
camera: Camera = None
lidar: Camera = None

connected_clients = set()

def parse_pcd(data: bytes):
    """
    Parses a PCD (Point Cloud Data) file in binary format.
    Returns a list of [x, y] points.
    """
    try:
        # Find the end of the header
        header_end_index = data.find(b"DATA binary\n")
        if header_end_index == -1:
            # Try binary_compressed or ascii? For now, assume binary.
            # print("PCD is not DATA binary, skipping.")
            return []
        
        header = data[:header_end_index].decode('ascii')
        raw_data = data[header_end_index + 12:] # 12 is len("DATA binary\n")

        # Parse header to find fields and count
        lines = header.split('\n')
        fields = []
        count = 0
        size = []
        type_ = []
        
        for line in lines:
            if line.startswith("FIELDS"):
                fields = line.split()[1:]
            elif line.startswith("POINTS"):
                count = int(line.split()[1])
            elif line.startswith("SIZE"):
                size = [int(s) for s in line.split()[1:]]
            elif line.startswith("TYPE"):
                type_ = line.split()[1:]

        # We expect x, y, z at least.
        # Assuming float32 (SIZE 4, TYPE F) for x, y, z.
        # We need to calculate the point step (stride).
        
        point_step = 0
        for s in size:
            point_step += s
            
        # Extract x and y.
        # Assuming x, y are the first two fields and are float32.
        # This is a simplification but works for most Viam PCDs.
        
        points = []
        
        # Use numpy for faster parsing
        # Create a dtype based on fields
        # This is a bit manual, but standard Viam rplidar is usually x, y, z, quality (all floats or ints)
        # Let's try to interpret as an array of structs if possible, or just flat floats.
        
        # Simplest approach: Read all as float32 if they are all 4 bytes
        if all(s == 4 for s in size) and all(t == 'F' for t in type_):
             # It's all floats
             num_floats = len(fields)
             arr = np.frombuffer(raw_data, dtype=np.float32)
             # Reshape
             if arr.size % num_floats == 0:
                 arr = arr.reshape(-1, num_floats)
                 # Get x, y (columns 0 and 1)
                 # Filter out zeros or distant points if needed
                 # For now just return all
                 points = arr[:, :2].tolist()
                 return points
        
        # Fallback to manual struct unpack if numpy fails or types differ
        # (Not implemented for brevity, assuming standard float32 PCD)
        return []

    except Exception as e:
        print(f"Error parsing PCD: {e}")
        return []

async def connect_to_robot():
    global robot, left_motor, right_motor, camera, lidar
    print("Attempting to connect to robot...")
    try:
        dial_options = DialOptions.with_api_key(API_KEY, API_KEY_ID)
        robot = await RobotClient.at_address(ROBOT_ADDRESS, dial_options)
        print(f"Successfully connected to {robot.name}")

        left_motor = Motor.from_robot(robot, LEFT_MOTOR_NAME)
        right_motor = Motor.from_robot(robot, RIGHT_MOTOR_NAME)
        
        try:
            camera = Camera.from_robot(robot, CAMERA_NAME)
            print("Camera initialized.")
        except:
            print("Camera not found.")
            
        try:
            lidar = Camera.from_robot(robot, LIDAR_NAME)
            print("Lidar initialized.")
        except:
            print("Lidar not found.")

        return True
    except Exception as e:
        print(f"Failed to connect to robot: {e}")
        robot = None
        return False

async def producer_task():
    global robot, left_motor, right_motor, camera, lidar, connected_clients
    while True:
        if robot and connected_clients:
            try:
                # 1. Motor Data
                left_pos, left_power_data, right_pos, right_power_data = await asyncio.gather(
                    left_motor.get_position(),
                    left_motor.is_powered(),
                    right_motor.get_position(),
                    right_motor.is_powered()
                )

                data = {
                    "type": "readout",
                    "left_pos": left_pos,
                    "left_power": left_power_data[1],
                    "right_pos": right_pos,
                    "right_power": right_power_data[1]
                }

                # 2. Camera Data (if connected)
                if camera:
                    try:
                        # Get image as bytes (JPEG)
                        img_bytes = await camera.get_image(mime_type="image/jpeg")
                        # Convert to base64
                        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                        data["image"] = img_b64
                    except Exception as e:
                        # print(f"Cam error: {e}")
                        pass

                # 3. Lidar Data (if connected)
                if lidar:
                    try:
                        # Get point cloud
                        pc_bytes, _ = await lidar.get_point_cloud()
                        points = parse_pcd(pc_bytes)
                        # Downsample if too many points to save bandwidth?
                        # For now send all.
                        data["lidar_points"] = points
                    except Exception as e:
                        # print(f"Lidar error: {e}")
                        pass

                # Broadcast
                message = json.dumps(data)
                send_tasks = [client.send(message) for client in connected_clients]
                if send_tasks:
                    await asyncio.wait(send_tasks)

            except Exception as e:
                print(f"Error in producer task: {e}")
        
        # Rate limit: 10Hz for smooth video/lidar
        await asyncio.sleep(0.1)

async def consumer_task(websocket):
    global robot, left_motor, right_motor
    async for message in websocket:
        try:
            data = json.loads(message)
            msg_type = data.get("type")

            if not robot:
                continue

            if msg_type == "set_power":
                motor = data.get("motor")
                power = float(data.get("power", 0.0))
                if motor == "left":
                    await left_motor.set_power(power)
                elif motor == "right":
                    await right_motor.set_power(power)
            
            elif msg_type == "stop":
                await left_motor.set_power(0)
                await right_motor.set_power(0)

        except Exception as e:
            print(f"Error in consumer task: {e}")

async def handler(websocket, path):
    global connected_clients
    print(f"Client connected: {websocket.remote_address}")
    connected_clients.add(websocket)
    try:
        await consumer_task(websocket)
    finally:
        print(f"Client disconnected: {websocket.remote_address}")
        connected_clients.remove(websocket)

async def main():
    if not await connect_to_robot():
        return

    asyncio.create_task(producer_task())

    server_address = "localhost"
    server_port = 8081
    print(f"Starting WebSocket server on ws://{server_address}:{server_port}")
    
    async with websockets.serve(handler, server_address, server_port):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())