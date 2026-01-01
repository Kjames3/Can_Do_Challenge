"""
Viam Rover GUI Launcher Script
==============================
Opens the GUI in your browser and syncs captured images when you're done.

Usage:
    python launcher.py
    
Configuration:
    Edit the settings below to match your setup.
"""

import os
import platform
import subprocess
import webbrowser
import time
import argparse

# ================= Configuration =================
# Update these to match your setup

ROBOT_IP = "192.168.1.161"             # Replace with your Pi's IP address
ROBOT_USER = "besto"                   # Your Pi username
REMOTE_PATH = "~/viam_projects/viam_projects/training_images"  # Folder on Pi with captured images
LOCAL_PATH = "./downloaded_images"     # Folder on your laptop to save images
GUI_FILE_PATH = "GUI.html"             # Path to your local GUI file

# =================================================


def get_image_count_remote():
    """Try to get image count from remote Pi (optional, may fail if SSH isn't set up)."""
    try:
        ssh_cmd = f'ssh {ROBOT_USER}@{ROBOT_IP} "find {REMOTE_PATH} -name \'*.jpg\' 2>/dev/null | wc -l"'
        result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            count = int(result.stdout.strip())
            return count
    except:
        pass
    return None


def sync_images():
    """Detects OS and runs the appropriate SCP command."""
    current_os = platform.system()
    print(f"\n🖥️  Detected OS: {current_os}")
    print("🔄 Starting image sync...")

    # Ensure local directory exists
    if not os.path.exists(LOCAL_PATH):
        os.makedirs(LOCAL_PATH)
        print(f"   Created directory: {LOCAL_PATH}")

    # Check if there are images to download (optional)
    remote_count = get_image_count_remote()
    if remote_count is not None:
        if remote_count == 0:
            print("⚠️  No images found on the rover. Nothing to sync.")
            return False
        else:
            print(f"   Found {remote_count} images on rover")

    # Build SCP command
    # -r = recursive (for subdirectories like close/, medium/, far/)
    scp_cmd = f'scp -r {ROBOT_USER}@{ROBOT_IP}:"{REMOTE_PATH}/*" "{LOCAL_PATH}"'
    
    # On Windows, you might need to adjust path separators
    if current_os == "Windows":
        # Convert forward slashes to backslashes for local path
        local_path_win = LOCAL_PATH.replace("/", "\\")
        scp_cmd = f'scp -r {ROBOT_USER}@{ROBOT_IP}:"{REMOTE_PATH}/*" "{local_path_win}"'

    try:
        print(f"   Executing: {scp_cmd}")
        print("   (You may be prompted for your Pi's password)")
        print()
        
        result = subprocess.run(scp_cmd, shell=True)
        
        if result.returncode == 0:
            # Count downloaded files
            downloaded = 0
            for root, dirs, files in os.walk(LOCAL_PATH):
                downloaded += len([f for f in files if f.endswith(('.jpg', '.jpeg', '.png'))])
            
            print(f"\n✅ Sync Complete! {downloaded} images saved to: {os.path.abspath(LOCAL_PATH)}")
            return True
        else:
            print("\n❌ Sync Failed. Check your connection or credentials.")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Sync Failed: {e}")
        return False
    except FileNotFoundError:
        print("\n❌ Error: 'scp' command not found.")
        print("   On Windows, make sure OpenSSH is installed (Windows 10+ has it built-in)")
        print("   Try: Settings > Apps > Optional Features > OpenSSH Client")
        return False


def main():
    parser = argparse.ArgumentParser(description='Viam Rover GUI Launcher')
    parser.add_argument('--ip', type=str, help='Override robot IP address')
    parser.add_argument('--local-path', type=str, help='Override local download path')
    parser.add_argument('--skip-gui', action='store_true', help='Skip opening GUI, just sync')
    args = parser.parse_args()
    
    global ROBOT_IP, LOCAL_PATH
    if args.ip:
        ROBOT_IP = args.ip
    if args.local_path:
        LOCAL_PATH = args.local_path
    
    print("=" * 50)
    print("🤖 Viam Rover GUI Launcher")
    print("=" * 50)
    print(f"   Robot IP:     {ROBOT_IP}")
    print(f"   Remote Path:  {REMOTE_PATH}")
    print(f"   Local Path:   {LOCAL_PATH}")
    print("=" * 50)

    if not args.skip_gui:
        print(f"\n🚀 Launching GUI in browser...")
        
        # Open the GUI in default browser
        gui_abs_path = os.path.abspath(GUI_FILE_PATH)
        
        if not os.path.exists(gui_abs_path):
            print(f"⚠️  Warning: GUI file not found at {gui_abs_path}")
            print("   Make sure GUI.html is in the same folder as this script")
        else:
            webbrowser.open(f"file://{gui_abs_path}")

        print("\n" + "=" * 50)
        print("   GUI is running in your browser.")
        print("   ")
        print("   🎮 Drive the rover with arrow keys or WASD")
        print("   📸 Press SPACE to capture images")
        print("   💾 Click 'Download All' for quick download")
        print("   ")
        print("   When finished, come back here.")
        print("=" * 50)

        # Wait for user trigger
        try:
            input("\n🔴 Press [ENTER] here when you are done to Sync & Exit... ")
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            print("\n👋 Goodbye!")
            return

    # Run the sync
    print("\n" + "-" * 50)
    success = sync_images()
    print("-" * 50)
    
    if success:
        print(f"\n📁 Images saved to: {os.path.abspath(LOCAL_PATH)}")
    
    print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()
