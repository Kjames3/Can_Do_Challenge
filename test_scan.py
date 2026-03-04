import sys
import socket
import concurrent.futures

ROBOT_PORT = 8081
TIMEOUT = 0.5 

def test_ip(ip):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(TIMEOUT)
        sock.connect((ip, ROBOT_PORT))
        sock.close()
        return ip, True, "Success"
    except socket.timeout:
        return ip, False, "Timeout"
    except Exception as e:
        return ip, False, str(e)

ips = [f"10.13.244.{i}" for i in range(1, 255)] + [f"10.13.246.{i}" for i in range(1, 255)]
print(f"Scanning {len(ips)} IPs...")
found = []
errors = {}
with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
    for ip, success, msg in executor.map(test_ip, ips):
        if success:
            found.append(ip)
        else:
            errors[msg] = errors.get(msg, 0) + 1

if found:
    print("FOUND ROBOT AT:", found)
else:
    print("No robot found.")
print("Errors breakdown:", errors)
