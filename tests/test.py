import asyncio

from viam.robot.client import RobotClient
from viam.components.board import Board
from viam.components.motor import Motor
from viam.components.base import Base
from viam.components.camera import Camera
from viam.components.encoder import Encoder
from viam.components.movement_sensor import MovementSensor
from viam.components.power_sensor import PowerSensor

async def connect():
    opts = RobotClient.Options.with_api_key(
         
        api_key='ab5lhwuctyf0t34wt7mu9gq24kgx8azh',
        
        api_key_id='164902f7-7737-4675-85d6-151fedb70a82'
    )
    return await RobotClient.at_address('rover-3-main.ayzp4fw8rj.viam.cloud', opts)

async def main():
    machine = await connect()

    print('Resources:')
    print(machine.resource_names)
    
    # Note that the pin supplied is a placeholder. Please change this to a valid pin you are using.
    # local
    local = Board.from_robot(machine, "local_1")
    local_return_value = await local.gpio_pin_by_name("16")
    print(f"local gpio_pin_by_name return value: {local_return_value}")

    # right
    right = Motor.from_robot(machine, "right")
    right_return_value = await right.is_moving()
    print(f"right is_moving return value: {right_return_value}")

    # left
    left = Motor.from_robot(machine, "left")
    left_return_value = await left.is_moving()
    print(f"left is_moving return value: {left_return_value}")

    # viam_base
    viam_base = Base.from_robot(machine, "viam_base")
    viam_base_return_value = await viam_base.is_moving()
    print(f"viam_base is_moving return value: {viam_base_return_value}")

    # cam
    cam = Camera.from_robot(machine, "cam")
    cam_return_value = await cam.get_image()
    print(f"cam get_image return value: {cam_return_value}")

    # left-enc
    left_enc = Encoder.from_robot(machine, "left-enc")
    left_enc_return_value = await left_enc.get_position()
    print(f"left-enc get_position return value: {left_enc_return_value}")

    # right-enc
    right_enc = Encoder.from_robot(machine, "right-enc")
    right_enc_return_value = await right_enc.get_position()
    print(f"right-enc get_position return value: {right_enc_return_value}")

    # imu
    imu = MovementSensor.from_robot(machine, "imu")
    imu_return_value = await imu.get_linear_acceleration()
    print(f"imu get_linear_acceleration return value: {imu_return_value}")

    # ina219
    ina_219 = PowerSensor.from_robot(machine, "ina219")
    ina_219_return_value = await ina_219.get_power()
    print(f"ina219 get_power return value: {ina_219_return_value}")

    # Don't forget to close the machine when you're done!
    await machine.close()

if __name__ == '__main__':
    asyncio.run(main())
