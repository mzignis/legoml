from pybricks.hubs import PrimeHub
from pybricks.pupdevices import Motor, UltrasonicSensor
from pybricks.parameters import Color, Port
from pybricks.tools import wait, StopWatch
from pybricks import version
from usys import stdin, stdout
from uselect import poll

hub = PrimeHub()

motor2 = Motor(Port.A)  # Second pusher
motor1 = Motor(Port.B)  # Front pusher
motor3 = Motor(Port.D)  # Conveyor belt motor

ultrasonic = UltrasonicSensor(Port.C) 

all_motors = [motor2, motor1, motor3]

class BrickSortingController:
    def __init__(self):
        self.motor1_motor2_speed = 400  # Pusher motor speed
        self.motor3_speed = 250         # Conveyor belt speed
        self.stopwatch = StopWatch()
        self.command_count = 0
        
    def execute_command(self, command):
        cmd = command.lower().strip()
        self.command_count += 1
        timestamp = self.stopwatch.time()
        
        if cmd == '1':
            # Front pusher activation
            hub.light.on(Color.ORANGE)
            motor1.run_angle(self.motor1_motor2_speed, 360, wait=False)
            stdout.buffer.write(b"Front pusher activated (Command 1)\n")
            return "front_pusher_activated"
            
        elif cmd == '2':
            # Front pusher reverse (if needed)
            hub.light.on(Color.ORANGE)
            motor1.run_angle(self.motor1_motor2_speed, -360, wait=False)
            stdout.buffer.write(b"Front pusher reversed (Command 2)\n")
            return "front_pusher_reversed"
            
        elif cmd == '4':
            # Second pusher activation
            hub.light.on(Color.GREEN)
            motor2.run_angle(self.motor1_motor2_speed, 360, wait=False)
            stdout.buffer.write(b"Second pusher activated (Command 4)\n")
            return "second_pusher_activated"
            
        elif cmd == '5':
            # Second pusher reverse (if needed)
            hub.light.on(Color.GREEN) 
            motor2.run_angle(self.motor1_motor2_speed, -360, wait=False)
            stdout.buffer.write(b"Second pusher reversed (Command 5)\n")
            return "second_pusher_reversed"
            
        elif cmd == '7':
            # Conveyor belt forward (alternative direction)
            hub.light.on(Color.VIOLET)
            motor3.run(self.motor3_speed)
            stdout.buffer.write(b"Conveyor belt forward (Command 7)\n")
            return "conveyor_forward"
            
        elif cmd == '8':
            # Conveyor belt backward (main direction for brick processing)
            hub.light.on(Color.VIOLET)
            motor3.run(-self.motor3_speed)
            stdout.buffer.write(b"Conveyor belt running - processing brick (Command 8)\n")
            return "conveyor_processing"
            
        elif cmd == '9':
            # Stop conveyor belt
            hub.light.on(Color.CYAN)  # Return to ready state
            motor3.stop()
            stdout.buffer.write(b"Conveyor belt stopped (Command 9)\n")
            return "conveyor_stopped"
            
        elif cmd == '0':
            # Emergency stop - stopping all motors
            hub.light.on(Color.RED)
            self.stop_all_motors()
            stdout.buffer.write(b"EMERGENCY STOP - All motors stopped (Command 0)\n")
            return "emergency_stop"
            
        elif cmd == 's':
            # Sensor reading (for future ML integration)
            distance = self.get_distance()
            distance_str = str(distance)
            distance_bytes = bytes(distance_str, 'utf-8')
            stdout.buffer.write(b"Sensor reading: " + distance_bytes + b"mm\n")
            return f"sensor_reading_{distance}"
            
        elif cmd.startswith('q') or cmd.startswith('stop') or cmd == 'bye':
            print("Stop command received from PC")
            stdout.buffer.write(b"Shutdown command received\n")
            return "quit"
            
        else:
            if cmd.strip():
                cmd_bytes = bytes(cmd, 'utf-8')
                stdout.buffer.write(b"Unknown command: '" + cmd_bytes + b"'\n")
            return "unknown"
    
    def stop_all_motors(self):
        for motor in all_motors:
            motor.stop()
    
    def get_distance(self):
        try:
            distance_mm = ultrasonic.distance()
            return distance_mm
        except Exception as e:
            stdout.buffer.write(b"Sensor error\n")
            return -1

# Initialize controller
controller = BrickSortingController()

# Set up polling for non-blocking input
keyboard = poll()
keyboard.register(stdin)

# Signal ready state - hub is ready for brick sorting
hub.light.on(Color.CYAN)
stdout.buffer.write(b"Brick sorting system ready\n")

# Main command processing loop
while True:
    try:
        # Let the PC know we are ready for a command
        stdout.buffer.write(b"rdy\n")
        
        # Wait for input to be available (non-blocking check)
        while not keyboard.poll(0):
            # Keep system responsive while waiting
            wait(10)
        
        # Read command input
        data = stdin.buffer.read(2)
        
        if data and len(data) >= 1:
            # Get the command character
            command_byte = data[0:1]
            
            # Acknowledge command receipt
            stdout.buffer.write(b"Received: '" + command_byte + b"'\n")
            
            try:
                command_char = chr(data[0])
                
                result = controller.execute_command(command_char)
                
                if result == "quit":
                    controller.stop_all_motors()
                    hub.light.off()
                    stdout.buffer.write(b"Brick sorting system shutting down\n")
                    break
                    
            except:
                stdout.buffer.write(b"Invalid command format\n")
        
    except KeyboardInterrupt:
        controller.stop_all_motors()
        hub.light.off()
        stdout.buffer.write(b"System interrupted\n")
        break
    except Exception as e:
        stdout.buffer.write(b"System error occurred\n")
        wait(100)

# Clean shutdown
controller.stop_all_motors()
hub.light.off()
stdout.buffer.write(b"Brick sorting system offline\n")