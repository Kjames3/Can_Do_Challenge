
import time
from gpiozero import PWMOutputDevice, DigitalOutputDevice

# Pin Config (Must match server_native.py)
# Left
L_IN1 = 19
L_IN2 = 13
L_PWM = 26

# Right
R_IN1 = 6
R_IN2 = 5
R_PWM = 22

def test_motor(name, pwm_pin, in1_pin, in2_pin):
    print(f"Testing {name}...")
    try:
        pwm = PWMOutputDevice(pwm_pin)
        in1 = DigitalOutputDevice(in1_pin)
        in2 = DigitalOutputDevice(in2_pin)
        
        print(f"  -> Forward 50%")
        in1.on()
        in2.off()
        pwm.value = 0.5
        time.sleep(2)
        
        print(f"  -> Stop")
        pwm.value = 0
        in1.off()
        in2.off()
        time.sleep(1)
        
        pwm.close()
        in1.close()
        in2.close()
        print(f"  {name} Done.")
    except Exception as e:
        print(f"  ERROR testing {name}: {e}")

print("=== MOTOR HARDWARE TEST ===")
print("Keep robot raised/wheels off ground!")
time.sleep(1)

test_motor("LEFT MOTOR", L_PWM, L_IN1, L_IN2)
print("-" * 20)
test_motor("RIGHT MOTOR", R_PWM, R_IN1, R_IN2)
