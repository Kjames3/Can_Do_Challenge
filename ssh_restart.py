import paramiko
import sys
import time

try:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    print("Connecting to Pi...")
    client.connect('192.168.137.91', username='besto', password='1', timeout=5)
    
    print("Restarting server...")
    # First try to restart the service (this might prompt for sudo password)
    stdin, stdout, stderr = client.exec_command("echo '1' | sudo -S systemctl restart rover_server.service")
    exit_status = stdout.channel.recv_exit_status()
    
    print("Checking status...")
    stdin, stdout, stderr = client.exec_command("echo '1' | sudo -S systemctl status rover_server.service")
    status_out = stdout.read().decode()
    print(status_out)
    
    # If the service isn't found or failed, let's just start the script in the background
    if "Loaded: not-found" in status_out or exit_status != 0:
        print("Service issue, trying to run script directly in background...")
        client.exec_command("nohup python3 Desktop/viam_projects/server_native.py > server.log 2>&1 &")
        client.exec_command("nohup python3 viam_projects/server_native.py > server.log 2>&1 &")
        print("Spawned background python script.")
        
    client.close()
    print("Done")
except Exception as e:
    print(f"Error: {e}")
