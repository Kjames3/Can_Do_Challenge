import socket
import time

def announce():
    # UDP Broadcast to Port 50000
    BROADCAST_IP = '<broadcast>'
    PORT = 50000

    # Create socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    print(f"📢 Announcing IP on port {PORT}...")

    while True:
        try:
            # Get current IP
            # We connect to a dummy external IP to get our true LAN IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                my_ip = s.getsockname()[0]

            message = f"ROBOT_IP:{my_ip}".encode('utf-8')
            sock.sendto(message, (BROADCAST_IP, PORT))
            time.sleep(2) # Announce every 2 seconds
        except Exception as e:
            # If network is down, wait and retry
            time.sleep(5)

if __name__ == "__main__":
    time.sleep(10) # Wait for WiFi to connect on boot
    announce()
