from pybricks.hubs import PrimeHub
from pybricks.pupdevices import Motor, UltrasonicSensor
from pybricks.parameters import Color, Port
from pybricks.tools import wait, StopWatch
from pybricks import version
from usys import stdin, stdout
from uselect import poll

hub = PrimeHub()

motor2 = Motor(Port.A)  
motor1 = Motor(Port.B) 
motor3 = Motor(Port.D)  

ultrasonic = UltrasonicSensor(Port.C) 

all_motors = [motor2, motor1, motor3]

class MotorController:
    def __init__(self):
        self.motor1_motor2_speed = 400
        self.motor3_speed = 150
        self.stopwatch = StopWatch()
        self.command_count = 0
        
    def execute_command(self, command):
        cmd = command.lower().strip()
        self.command_count += 1
        timestamp = self.stopwatch.time()
        
        if cmd == '1':
            #print(f"[{timestamp}ms] Motor1 (Port B) - 1 rotation forward")
            hub.light.on(Color.ORANGE)
            motor1.run_angle(self.motor1_motor2_speed, 360, wait=False)
            #print("OK")
            stdout.buffer.write(b"Command 1 executed - Motor1 forward\n")
            return "motor1_forward"
            
        elif cmd == '2':
            #print(f"[{timestamp}ms] Motor1 (Port B) - 1 rotation backward") 
            hub.light.on(Color.ORANGE)
            motor1.run_angle(self.motor1_motor2_speed, -360, wait=False)
            #print("OK")
            stdout.buffer.write(b"Command 2 executed - Motor1 backward\n")
            return "motor1_backward"
            
        elif cmd == '4':
            #print(f"[{timestamp}ms] Motor2 (Port A) - 1 rotation forward")
            hub.light.on(Color.GREEN)
            motor2.run_angle(self.motor1_motor2_speed, 360, wait=False)
            #print("OK")
            stdout.buffer.write(b"Command 4 executed - Motor2 forward\n")
            return "motor2_forward"
            
        elif cmd == '5':
            #print(f"[{timestamp}ms] Motor2 (Port A) - 1 rotation backward")
            hub.light.on(Color.GREEN) 
            motor2.run_angle(self.motor1_motor2_speed, -360, wait=False)
            #print("OK")
            stdout.buffer.write(b"Command 5 executed - Motor2 backward\n")
            return "motor2_backward"
            
        elif cmd == '7':
            #print(f"[{timestamp}ms] Motor3 (Port D) forward (continuous)")
            hub.light.on(Color.VIOLET)
            motor3.run(self.motor3_speed)
            #print("OK")
            stdout.buffer.write(b"Command 7 executed - Motor3 forward continuous\n")
            return "motor3_forward_continuous"
            
        elif cmd == '8':
            #print(f"[{timestamp}ms] Motor3 (Port D) backward (continuous)")
            hub.light.on(Color.VIOLET)
            motor3.run(-self.motor3_speed)
            #print("OK")
            stdout.buffer.write(b"Command 8 executed - Motor3 backward continuous\n")
            return "motor3_backward_continuous"
            
        elif cmd == '9':
            #print(f"[{timestamp}ms] Stopping Motor3 (Port D)")
            hub.light.on(Color.VIOLET)
            motor3.stop()
            #print("OK")
            stdout.buffer.write(b"Command 9 executed - Motor3 stopped\n")
            return "motor3_stop"
            
        elif cmd == '0':
            #print(f"[{timestamp}ms] Emergency stop - stopping all motors...")
            hub.light.on(Color.RED)
            self.stop_all_motors()
            #print("OK")
            stdout.buffer.write(b"Command 0 executed - Emergency stop\n")
            return "emergency_stop"
            
        elif cmd == 's':
            distance = self.get_distance()
            #print("OK")
            # Convert distance to bytes without using .encode()
            distance_str = str(distance)
            distance_bytes = bytes(distance_str, 'utf-8')
            stdout.buffer.write(b"Sensor reading: " + distance_bytes + b"mm\n")
            return f"sensor_reading_{distance}"
            
        elif cmd.startswith('q') or cmd.startswith('stop') or cmd == 'bye':
            print("Stop command received from PC")
            #print("OK")
            stdout.buffer.write(b"Quit command received - shutting down\n")
            return "quit"
            
        else:
            if cmd.strip():
                #print(f"[{timestamp}ms] Unknown command from PC: '{cmd}'")
                #print("ERROR")
                # Convert command to bytes without using .encode()
                cmd_bytes = bytes(cmd, 'utf-8')
                stdout.buffer.write(b"Unknown command: '" + cmd_bytes + b"'\n")
            return "unknown"
    
    def stop_all_motors(self):
        for motor in all_motors:
            motor.stop()
    
    # def get_distance(self):
    #     try:
    #         distance_mm = ultrasonic.distance()
    #         timestamp = self.stopwatch.time()
    #         print(f"[{timestamp}ms] Distance: {distance_mm} mm")
    #         if distance_mm < 100:
    #             print(f"[{timestamp}ms] ALERT: Object detected at {distance_mm} mm!")
    #         return distance_mm
    #     except Exception as e:
    #         timestamp = self.stopwatch.time()
    #         print(f"[{timestamp}ms] Error reading ultrasonic sensor: {e}")
    #         return -1

controller = MotorController()

# Set up polling for non-blocking input
keyboard = poll()
keyboard.register(stdin)

# Signal ready and wait for commands
hub.light.on(Color.CYAN)

while True:
    try:
        # Let the PC know we are ready for a command
        stdout.buffer.write(b"rdy\n")
        
        # Wait for input to be available (non-blocking check)
        while not keyboard.poll(0):
            # Do something here while waiting - check sensors, etc.
            wait(10)
        
        # Read 2 bytes (1 command character + newline)
        data = stdin.buffer.read(2)
        
        if data and len(data) >= 1:
            # Get the command character (first byte)
            command_byte = data[0:1]
            
            # Send confirmation that command was received
            stdout.buffer.write(b"Received command: '" + command_byte + b"'\n")
            
            try:
                command_char = chr(data[0])
                
                result = controller.execute_command(command_char)
                
                if result == "quit":
                    controller.stop_all_motors()
                    hub.light.off()
                    #print("Hub shutting down")
                    break
            except:
                stdout.buffer.write(b"Invalid command received\n")
        
    except KeyboardInterrupt:
        #print("Program interrupted")
        controller.stop_all_motors()
        hub.light.off()
        break
    except Exception as e:
        #print(f"Error: {e}")
        stdout.buffer.write(b"Hub error occurred\n")
        wait(100)