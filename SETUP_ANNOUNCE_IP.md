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
    ```
    *If it says "No such file or directory", you need to fix the path or copy the file there first.*

## Setup Steps

1.  **Edit the Service File (If Needed)**:
    Open `announce_ip.service` on your Windows machine or on the Pi. Look for the `ExecStart` line.
    
    > **IMPORTANT**: `ExecStart` is a **configuration setting** inside the file. It is **NOT** a command you type into the terminal.

    Ensure it matches your actual path:
    ```ini
    ExecStart=/usr/bin/python3 /home/besto/Can_Do_Challenge/announce_ip.py
    User=besto
    ```

2.  **Copy the Service File**:
    Move the service file to the systemd directory.
    ```bash
    sudo cp /home/besto/Can_Do_Challenge/announce_ip.service /etc/systemd/system/
    ```

3.  **Reload Systemd**:
    Tell systemd to read the new file.
    ```bash
    sudo systemctl daemon-reload
    ```

4.  **Enable the Service**:
    This makes it start automatically on boot.
    ```bash
    sudo systemctl enable announce_ip.service
    ```

5.  **Start the Service**:
    To test it without rebooting:
    ```bash
    sudo systemctl start announce_ip.service
    ```

## Verification

1.  **Check Status**:
    To see if it is running correctly:
    ```bash
    sudo systemctl status announce_ip.service
    ```
    You should see `Active: active (running)`.

2.  **View Logs**:
    If it fails, check the logs (press `q` to exit):
    ```bash
    journalctl -u announce_ip.service -f
    ```
