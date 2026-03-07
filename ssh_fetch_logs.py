import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('192.168.137.91', username='besto', password='1', timeout=5)

# Verify the file is actually NOT completely empty on the Pi
print("--- Server file stats on Pi ---")
stdin, stdout, stderr = client.exec_command("ls -la ~/Can_Do_Challenge/server_native.py")
print(stdout.read().decode())

print("--- Print the very bottom of the file where main is called ---")
stdin, stdout, stderr = client.exec_command("tail -n 20 ~/Can_Do_Challenge/server_native.py")
print(stdout.read().decode())

client.close()
