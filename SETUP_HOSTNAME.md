# Setup Guide: Enabling `besto.local`

To connect to your robot using `besto.local` instead of searching for the IP address every time, we need to configure the Raspberry Pi's hostname and ensure the mDNS service (Avahi) is running.

## Step 1: Set the Hostname to `besto`

By default, Raspberry Pis are named `raspberrypi` (accessible via `raspberrypi.local`). We need to change this to `besto`.

1.  SSH into your Raspberry Pi or open a terminal on it.
2.  Run the configuration tool:
    ```bash
    sudo raspi-config
    ```
3.  Navigate to:
    *   **1 System Options** -> **S4 Hostname**
4.  Delete the existing name and type: `besto`
5.  Select **OK**.
6.  The system will ask to reboot. Select **No** for now (we have one more step), or **Yes** if you want to skip Step 2 and test immediately (but Step 2 is recommended).

## Step 2: Install/Verify mDNS Service (Avahi)

The `avahi-daemon` is what broadcasts the `.local` address to your network. It is usually installed by default, but let's make sure.

1.  In the terminal, update your package list:
    ```bash
    sudo apt-get update
    ```
2.  Install Avahi:
    ```bash
    sudo apt-get install avahi-daemon -y
    ```
3.  Ensure it is enabled and running:
    ```bash
    sudo systemctl enable avahi-daemon
    sudo systemctl start avahi-daemon
    ```

## Step 3: Reboot and Verify

1.  Reboot the Pi:
    ```bash
    sudo reboot
    ```
2.  Wait for the Pi to restart (about 30-60 seconds).
3.  **On your Windows PC**: Open Command Prompt (cmd) or PowerShell and try to ping the robot:
    ```powershell
    ping besto.local
    ```
    *   **Success**: You should see replies from an IP address (e.g., `Reply from 192.168.1.161...`).
    *   **Failure**: If it says "Ping request could not find host", wait another minute and try again.

## Step 4: Connect via GUI

Once the ping works:
1.  Open `GUI.html` or `debug_gui.html`.
2.  You should now be able to use `besto.local` in the connection box.
    *   Note: `GUI.html` is already configured to default to `besto.local`.

---

### Troubleshooting Windows mDNS
If `ping besto.local` fails but you are sure the Pi is set up correctly:
*   Ensure your Windows PC and the Pi are on the **same Wi-Fi network**.
*   Some aggressive firewalls or "Public" network settings in Windows might block mDNS.
