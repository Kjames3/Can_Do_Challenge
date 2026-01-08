# Setup Guide: Auto-Start Announce IP on Boot

This guide explains how to set up your Raspberry Pi to automatically run the `announce_ip.py` script when it powers on.

## Prerequisites

1.  **Transfer Files**: You need to move the following files from this folder to your Raspberry Pi into the `/home/besto/Can_Do_Challenge/` directory:
    *   `announce_ip.py`
    *   `announce_ip.service`

2.  **Verify File Existence**:
    Run this command on your Pi to make sure the file is there:
    ```bash
    ls -l /home/besto/Can_Do_Challenge/announce_ip.py
    ls -l /home/besto/Can_Do_Challenge/server_native.py
    ```

## Setup Steps

1.  **Edit the Service Files (If Needed)**:
    Open `announce_ip.service` and `rover_server.service` to verify the paths match your setup.
    
    > **IMPORTANT**: 
    > 1. `ExecStart` must point to the python inside your virtual environment (e.g., `/home/besto/Can_Do_Challenge/.venv/bin/python`) so it can find your installed libraries.
    > 2. `ExecStart` is a **configuration setting** inside the file. It is **NOT** a command you type into the terminal.

2.  **Copy the Service Files**:
    Move both service files to the systemd directory.
    ```bash
    sudo cp /home/besto/Can_Do_Challenge/announce_ip.service /etc/systemd/system/
    sudo cp /home/besto/Can_Do_Challenge/rover_server.service /etc/systemd/system/
    ```

3.  **Reload Systemd**:
    Tell systemd to read the new files.
    ```bash
    sudo systemctl daemon-reload
    ```

4.  **Enable the Services**:
    This makes them start automatically on boot.
    ```bash
    sudo systemctl enable announce_ip.service
    sudo systemctl enable rover_server.service
    ```

5.  **Start the Services**:
    To test them without rebooting:
    ```bash
    sudo systemctl start announce_ip.service
    sudo systemctl start rover_server.service
    ```

## Verification

1.  **Check Status**:
    To see if they are running correctly:
    ```bash
    sudo systemctl status announce_ip.service
    sudo systemctl status rover_server.service
    ```
    You should both say `Active: active (running)`.

2.  **View Logs**:
    If it fails, check the logs (press `q` to exit):
    ```bash
    journalctl -u announce_ip.service -f
    # OR
    journalctl -u rover_server.service -f
    ```
