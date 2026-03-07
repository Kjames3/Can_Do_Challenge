import paramiko
import os

host = "192.168.137.91"
port = 22
username = "besto"
password = "1"

local_path = r"c:\Users\besto\OneDrive\Documents\Viam Rover 2 projects\viam_projects\server_native.py"
remote_path = "/home/besto/Can_Do_Challenge/server_native.py"

try:
    transport = paramiko.Transport((host, port))
    transport.connect(username=username, password=password)
    sftp = paramiko.SFTPClient.from_transport(transport)
    print(f"Uploading {local_path} to {remote_path}...")
    sftp.put(local_path, remote_path)
    
    # Also remotely restart the service
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port, username, password)
    print("Restarting rover_server.service...")
    stdin, stdout, stderr = ssh.exec_command("sudo systemctl restart rover_server.service")
    print(stdout.read().decode())
    print(stderr.read().decode())
    
    sftp.close()
    transport.close()
    ssh.close()
    
    print("Upload and restart successful!")
except Exception as e:
    print(f"Error: {e}")
