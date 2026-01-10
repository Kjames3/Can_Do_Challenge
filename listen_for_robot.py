# Save as listen_for_robot.py on your LAPTOP
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
sock.bind(("", 50000))

sock.settimeout(5.0)

print("👂 Listening for Robot IP on UDP Port 50000...")
while True:
    try:
        data, addr = sock.recvfrom(1024)
        msg = data.decode()
        if "ROBOT_IP" in msg:
            print(f"🚀 FOUND ROBOT! IP is: {msg.split(':')[1]}")
            break
    except socket.timeout:
        print("... still listening (make sure robot is on and announce_ip.py is running) ...")
