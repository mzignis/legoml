# Bluetooth Protocol & UUIDs

## Overview

This guide covers the implementation of Bluetooth communication between a PC and Pybricks hub using the Pybricks-specific protocol.

**Important**: Use Pybricks-specific protocol, NOT Nordic UART Service directly

## Protocol Configuration

### UUIDs
- **Service UUID**: `c5f50001-8280-46da-89f4-6d8051e4aeef`
- **Command/Event characteristic**: `c5f50002-8280-46da-89f4-6d8051e4aeef`

### Command Format
All commands must be prefixed with `b"\x06"` (write stdin command byte)

## PC Side Implementation (Python with Bleak)

### Key Requirements
- Use `write_gatt_char()` with `response=True` for reliability
- Set up notifications on the command/event characteristic to receive hub responses
- Handle responses by checking if `data[0] == 0x01` (stdout event), then use `data[1:]` as payload

### Command Structure
```python
command = b"\x06" + f"{command}\n".encode('utf-8')
```

### Example Implementation
```python
import asyncio
from bleak import BleakClient

async def send_command(client, command):
    cmd_bytes = b"\x06" + f"{command}\n".encode('utf-8')
    await client.write_gatt_char("c5f50002-8280-46da-89f4-6d8051e4aeef", cmd_bytes, response=True)

def notification_handler(sender, data):
    if data[0] == 0x01:  # stdout event
        payload = data[1:]
        print(f"Hub response: {payload}")
```

## Hub Side Implementation (Pybricks MicroPython)

### Required Imports
```python
from usys import stdin, stdout
from uselect import poll
```

### Critical Guidelines

#### Reading Input
- Use `stdin.buffer.read()` for reading bytes
- **DO NOT** use `input()` or `stdin.read()`

#### Polling Pattern
```python
keyboard = poll()
keyboard.register(stdin)

# Check for input
if keyboard.poll(0):
    data = stdin.buffer.read()
```

#### Ready Signal
Send "rdy" signal before waiting for each command:
```python
stdout.buffer.write(b"rdy\n")
```

#### String Encoding
**Critical**: Strings don't have `.encode()` method in Pybricks
```python
# WRONG
text = "hello"
encoded = text.encode('utf-8')

# CORRECT
text = "hello"
encoded = bytes(text, 'utf-8')
```

#### Byte Operations
Use byte concatenation:
```python
result = b"prefix" + byte_data + b"suffix"
```

#### Reliable Data Reading
Read fixed-length data (e.g., 2 bytes for command + newline) for reliability:
```python
if keyboard.poll(0):
    command_data = stdin.buffer.read(2)  # command + newline
```

## Common Pitfalls

### Error Handling
- `UnicodeDecodeError` doesn't exist - use generic `Exception`
- Always wrap string operations in try/except blocks

### String Methods
- String `.encode()` method doesn't exist - use `bytes()` constructor
- Hub will crash and restart if you use unsupported string methods

### Command Format
- Commands are sent as single characters plus newline, not full words
- Example: `"w\n"` not `"forward\n"`

### Connection State
- Hub must be idle (blinking blue) when PC connects
- Manually start hub program with center button after connection

## Debugging Tips

### Hub Side Debugging
```python
# Add confirmation messages
stdout.buffer.write(b"Command received\n")
stdout.buffer.write(bytes(f"Processing: {command}", 'utf-8') + b"\n")
```

### PC Side Debugging
```python
def debug_notification_handler(sender, data):
    print(f"Raw data: {data}")
    if data[0] == 0x01:
        payload = data[1:].decode('utf-8', errors='ignore')
        print(f"Hub says: {payload}")
```

### Troubleshooting Crashes
If hub crashes repeatedly:
1. Check for string encoding issues first
2. Verify you're using `bytes()` constructor instead of `.encode()`
3. Ensure all string operations are wrapped in exception handling
4. Confirm command format is single character + newline

## Example Workflow

1. **PC connects** to hub (hub should be blinking blue)
2. **Hub sends** "rdy" signal
3. **PC sends** command with proper prefix: `b"\x06w\n"`
4. **Hub receives** and processes command
5. **Hub sends** response via stdout
6. **PC receives** response through notification handler
7. **Hub sends** next "rdy" signal

