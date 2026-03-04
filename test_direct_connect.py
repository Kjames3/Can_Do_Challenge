import socket

def test_ip(ip, port, timeout=2.0):
    try:
        print(f"Testing {ip}:{port} with timeout {timeout}s...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((ip, port))
        sock.close()
        print("Success! Connection established.")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_ip("192.168.137.91", 8081)
