
import sys
import traceback

print(f"Python Executable: {sys.executable}")
print("Attempting to import picamera2...")

try:
    import picamera2
    print(f"✅ picamera2 imported successfully: {picamera2.__file__}")
except ImportError:
    print("❌ Failed to import picamera2")
    traceback.print_exc()
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")
    traceback.print_exc()

print("\nAttempting to import libcamera...")
try:
    import libcamera
    print(f"✅ libcamera imported successfully")
except ImportError:
    print("❌ Failed to import libcamera")
    traceback.print_exc()
