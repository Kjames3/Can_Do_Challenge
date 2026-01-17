import socket
import threading
import ipaddress
import time
from queue import Queue

# Configuration
ROBOT_PORT = 8081
TIMEOUT = 0.4  # Fast timeout
THREAD_COUNT = 1000  # Higher thread count for wider scan

def get_local_ip():
    """Finds the local IP address of this computer."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"

def scan_ip(ip, result_queue):
    """Tries to connect to the robot port on a specific IP."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(TIMEOUT)
        result = sock.connect_ex((str(ip), ROBOT_PORT))
        if result == 0:
            result_queue.put(str(ip))
        sock.close()
    except:
        pass

def worker(queue, result_queue):
    while True:
        ip = queue.get()
        if ip is None:
            break
        scan_ip(ip, result_queue)
        queue.task_done()

def main():
    print(f"\n🚀 STARTING WIDE NETWORK SCANNER (Port {ROBOT_PORT})")
    print("="*50)
    
    local_ip = get_local_ip()
    print(f"📍 Your IP:     {local_ip}")

    # Determine Subnet to Scan 
    # Scanning the entire /16 subnet (10.13.x.x) to ensure we find the robot
    # regardless of which specific block it's in.
    try:
        octets = list(map(int, local_ip.split('.')))
        network_prefix = ".".join(map(str, octets[:2])) # e.g. "10.13"
        
        # Scan all subnets in the /16 range (0-255)
        block_start = 0
        block_end = 255
        
        print(f" Target Range: {network_prefix}.0.1  --->  {network_prefix}.255.254")
        print(f" Scanning approx {256 * 255} IPs with {THREAD_COUNT} threads...")
        
    except Exception as e:
        print(f"Error parsing IP: {e}")
        return

    ip_queue = Queue()
    result_queue = Queue()
    threads = []

    # Start threads
    for _ in range(THREAD_COUNT):
        t = threading.Thread(target=worker, args=(ip_queue, result_queue))
        t.daemon = True
        t.start()
        threads.append(t)

    # Queue IPs
    total_queued = 0
    for third in range(block_start, block_end + 1):
        for fourth in range(1, 255):
            target_ip = f"{network_prefix}.{third}.{fourth}"
            if target_ip != local_ip:
                ip_queue.put(target_ip)
                total_queued += 1

    # Wait
    print(f"⏳ Scanning... (This may take ~10-15 seconds)")
    ip_queue.join()

    # Stop threads
    for _ in range(THREAD_COUNT):
        ip_queue.put(None)

    # Results
    print("\n" + "="*50)
    found_ips = []
    while not result_queue.empty():
        found_ips.append(result_queue.get())
    
    if found_ips:
        print(f"✅ FOUND {len(found_ips)} DEVICE(S):")
        for ip in found_ips:
            print(f"   🤖 ROBOT FOUND AT: {ip}")
            print(f"      -> Enter {ip} in GUI")
            
            # Additional check: Try to fetch hostname if possible
            try:
                hostname = socket.gethostbyaddr(ip)[0]
                print(f"         (Hostname: {hostname})")
            except:
                pass
    else:
        print("❌ NO ROBOT FOUND.")
        print("   Tips:")
        print("   1. Verify robot is powered on.")
        print("   2. Verify 'rover_server.service' is active on Pi.")
        print("   3. Try connecting via Hotspot if this network blocked ports.")
    print("="*50)

    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()
