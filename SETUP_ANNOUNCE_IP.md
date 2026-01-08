# Setup Guide: Auto-Start Announce IP on Boot

This guide explains how to set up your Raspberry Pi to automatically run the `announce_ip.py` script when it powers on.

## Prerequisites

1.  **Transfer Files**: You need to move the following files from this folder to your Raspberry Pi (e.g., using `scp`, `rsync`, or a flash drive):
    *   `announce_ip.py`
    *   `announce_ip.service`
2.  **Location**: This guide assumes you place the files in `/home/pi/viam_projects/`. If you put them elsewhere, you **MUST** edit the `announce_ip.service` file to match your path.

## Setup Steps

1.  **Connect to your Raspberry Pi** via SSH or open a terminal on the Pi.

2.  **Verify Paths**:
    Ensure your script needs no special environment. If you usually run it with a virtual environment (like Viam's venv), you might need to change the `ExecStart` line in `announce_ip.service` to point to that python executable (e.g., `/home/pi/viam_projects/venv/bin/python`).
    
    Current default in `announce_ip.service`:
    ```ini
    ExecStart=/usr/bin/python3 /home/pi/viam_projects/announce_ip.py
    ```

3.  **Copy the Service File**:
    Move the service file to the systemd directory.
    ```bash
    sudo cp announce_ip.service /etc/systemd/system/
    ```

4.  **Reload Systemd**:
    Tell systemd to read the new file.
    ```bash
    sudo systemctl daemon-reload
    ```

5.  **Enable the Service**:
    This makes it start automatically on boot.
    ```bash
    sudo systemctl enable announce_ip.service
    ```

6.  **Start the Service Immediately** (Optional):
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
    If it fails, check the logs:
    ```bash
    journalctl -u announce_ip.service -f
    ```

## Troubleshooting

-   **"File not found"**: Double-check the path in `ExecStart` inside the `.service` file.
-   **Permission Denied**: Ensure `announce_ip.py` is readable (e.g., `chmod +x announce_ip.py`).
-   **Network Issues**: The service is set to wait for the network (`After=network-online.target`), but if it starts too early, it might fail. The script already has a retry mechanism, so it should recover.
