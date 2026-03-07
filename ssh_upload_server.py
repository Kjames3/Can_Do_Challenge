import paramiko
import os

print("--- Uploading true server_native.py to the Pi ---")
local_path = 'server_native.py'
remote_path = '/home/besto/Can_Do_Challenge/server_native.py'

transport = paramiko.Transport(('192.168.137.91', 22))
transport.connect(username='besto', password='1')
sftp = paramiko.SFTPClient.from_transport(transport)

# Upload the file
sftp.put(local_path, remote_path)
print("File uploaded successfully!")

sftp.close()
transport.close()

# Start the server
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('192.168.137.91', username='besto', password='1', timeout=5)

client.exec_command("echo '1' | sudo -S systemctl restart rover_server.service")
print("Restarted rover_server.service!")

client.close()
