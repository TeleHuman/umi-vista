from threading import Thread, Lock
from collections import deque
import time

import serial.tools.list_ports
from pyrobotiqgripper import RobotiqGripper
#pyrobotiqgripper 1.0.1


class SafeGripperController:
    """
    A full-duplex capable gripper controller for Robotiq 2F-85/140.
    
    Architecture:
    - Thread 1 (Monitor): Continuously reads status registers (Position, Object Detection, etc.).
    - Thread 2 (Writer): Sends move commands asynchronously (Non-blocking).
    """
    
    def __init__(self, portname=None, serial_number=None, read_interval=0.02):
        # 1. Initialize Driver
        # Note: RobotiqGripper usually finds port automatically or takes args.
        # Assuming existing code 'RobotiqGripper()' works for this environment.
        resolved_port = self._resolve_port(portname, serial_number)
        if resolved_port:
            self.gripper = RobotiqGripper(portname=resolved_port)
        else:
            self.gripper = RobotiqGripper()
            if hasattr(self.gripper, "_autoConnect"):
                self.gripper._autoConnect()

        # Keep the resolved port for upper-layer debug prints.
        self.portname = resolved_port or getattr(self.gripper, "portname", portname)
        
        # 2. Setup Serial Parameters (Robustness)
        if hasattr(self.gripper, 'serial'):
            # MinimalModbus Instrument
            self.gripper.serial.timeout = 0.2
            self.gripper.serial.write_timeout = 0.2
            
        # 3. Activation / Recovery
        # Some grippers may come up with a latched fault (for example gFLT=14).
        # In that case activate() alone may timeout, and resetActivate() is needed.
        self._ensure_activated()

        # 4. State & Synchronization
        self.lock = Lock()
        self.running = True
        self.read_interval = read_interval
        self.current_pos = 0.0
        self.is_moving = False
        self.obj_detected = False
        
        # 5. Command Queue & Threads
        self.command_queue = deque(maxlen=1)
        
        self.command_thread = Thread(target=self._command_loop, daemon=True)
        self.command_thread.start()
                
        self.monitor_thread = Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def _resolve_port(self, portname, serial_number):
        if serial_number:
            for port in serial.tools.list_ports.comports():
                hwid = getattr(port, "hwid", "") or ""
                if "SER=" not in hwid:
                    continue
                serial_part = hwid.split("SER=")[1].split()[0]
                if serial_part == serial_number:
                    print(f"[Gripper] Serial {serial_number} -> {port.device}")
                    return port.device
            raise RuntimeError(f"No gripper found for serial number {serial_number}")
        return portname

    def _read_status(self):
        try:
            self.gripper.readAll()
            return dict(getattr(self.gripper, "paramDic", {}) or {})
        except Exception:
            return {}

    def _format_status(self, status):
        if not status:
            return "<status unavailable>"
        return (
            f"gACT={status.get('gACT')}, "
            f"gSTA={status.get('gSTA')}, "
            f"gFLT={status.get('gFLT')}, "
            f"gPO={status.get('gPO')}, "
            f"gCU={status.get('gCU')}"
        )

    def _ensure_activated(self):
        status = self._read_status()
        if status.get("gACT") == 1 and status.get("gSTA") == 3:
            print(f"[Gripper] Already active: {self._format_status(status)}")
            return

        print(f"[Gripper] Initial status: {self._format_status(status)}")
        print("Activating Gripper...")
        try:
            self.gripper.activate()
        except Exception as e:
            print(f"[Gripper] activate() failed: {e!r}")
            print("[Gripper] Trying resetActivate() once...")
            self.gripper.resetActivate()

        status = self._read_status()
        if not (status.get("gACT") == 1 and status.get("gSTA") == 3):
            raise RuntimeError(
                f"Robotiq activation failed on {self.portname}. Status: {self._format_status(status)}"
            )
        print(f"[Gripper] Active: {self._format_status(status)}")

    def _monitor_loop(self):
        """
        Periodically polls the gripper status.
        Because we use a lock, this can freely run interleaved with write commands.
        """
        while self.running:
            try:
                start_t = time.time()
                with self.lock:
                    # Update internal state from hardware
                    self.gripper.readAll()
                    
                    # 1. Position Feedback
                    # gPO is the actual position in bits (0-255).
                    # Robotiq 0=Open, 255=Closed.
                    # User 0mm=Closed, 40mm=Open.
                    bit_pos = self.gripper.paramDic.get('gPO', 0)
                    
                    # Mapping: bit=255 -> mm=0. bit=0 -> mm=40.
                    # mm = (255 - bit) * (40.0 / 255.0)
                    self.current_pos = (255 - bit_pos) * (40.0 / 255.0)
                    
                    # 2. Status / Moving State
                    # gOBJ: 0=Moving, 1=Detected(Op), 2=Detected(Cl), 3=Stop(Pos)
                    gOBJ = self.gripper.paramDic.get('gOBJ', 0)
                    self.is_moving = (gOBJ == 0)
                    self.obj_detected = (gOBJ == 1 or gOBJ == 2)
                
                # Sleep to maintain frequency
                elapsed = time.time() - start_t
                rest = self.read_interval - elapsed
                if rest > 0:
                    time.sleep(rest)
                    
            except Exception as e:
                # print(f"[Monitor] Error: {e}")
                time.sleep(0.05)

    def _command_loop(self):
        """
        Consumes commands from the queue.
        Ensures only the latest command is executed (via deque maxlen=1).
        """
        while self.running:
            try:
                if not self.command_queue:
                    time.sleep(0.01)
                    continue
                
                # Fetch latest command
                cmd_func, args, kwargs = self.command_queue.popleft()
                
                # Execute with lock
                with self.lock:
                    cmd_func(*args, **kwargs)
                
                # Small sleep to prevent serial bus saturation if queue is spamming
                time.sleep(0.01)
                
            except Exception as e:
                print(f"[Command] Error: {e}")
                time.sleep(0.05)

    def move(self, pos_mm, speed=255, force=255):
        """
        Public API: Enqueues a move command.
        """
        self.command_queue.append((self._execute_move, (pos_mm, speed, force), {}))

    def _execute_move(self, pos_mm, speed=255, force=255):
        """
        Internal: Writes registers directly.
        Assumes LOCK is held by _command_loop.
        
        pos_mm: Target position in mm (0 = Closed, 40 = Open)
        """
        # Mapping:
        # User 0mm (Closed) -> Robotiq 255 (Closed)
        # User 40mm (Open)  -> Robotiq 0 (Open)
        
        target_bit = int(255 - (pos_mm / 40.0) * 255)
        target_bit = max(0, min(255, target_bit))
        
        speed = max(0, min(255, int(speed)))
        force = max(0, min(255, int(force)))
        
        # Register 1000: Action (0x09) + Reserved (0x00) -> 0x0900
        # Register 1001: Reserved (0x00) + Position (0-255) -> Position (since 0x00FF mask)
        # Register 1002: Speed (High Byte) + Force (Low Byte)
        
        regs = [
            0x0900,  # Act=1, GTO=1
            target_bit, # Reserved(00) | Pos(xx)
            (speed << 8) | force # Speed | Force
        ]
        
        try:
             # Call RobotiqGripper.write_registers directly (inherited from Instrument)
             # NOTE: Lock is already held by _command_loop
             self.gripper.write_registers(1000, regs)
        except Exception as e:
             print(f"Move Error: {e}")

    def get_pos(self):
        return self.current_pos

    # ============================================================
    # API Compatibility with SafeGripperController
    # ============================================================
    
    def open_gripper(self, force=255, speed=255):
        """
        Open gripper to 40mm.
        """
        self.move(40.0, speed, force)

    def close_gripper(self, force=255, speed=255):
        """
        Close gripper to 0mm.
        """
        self.move(0.0, speed, force)
        
    def get_gripper_distance(self):
        """
        Alias for get_pos(). 
        Returns real-time position in mm.
        """
        return self.get_pos()

    def get_gripper_state(self):
        """
        1 = closed, 0 = open
        Threshold < 35.0 mm considered closed (holding something or fully closed)
        """
        pos = self.get_gripper_distance()
        return 1 if pos < 35.0 else 0

    def stop(self):
        self.running = False
        try:
            self.monitor_thread.join(0.5)
            self.command_thread.join(0.5)
        except:
            pass

if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    
    gripper = SafeGripperController(portname="/dev/ttyUSB0")
    
    # parameters
    duration = 10.0
    switch_interval = 0.5 # 1Hz cycle (0.5s open, 0.5s close)
    
    # data recording
    start_time = time.time()
    last_switch_time = start_time - switch_interval # Force immediate switch
    is_closed = False # initial state (will switch to True/Closed immediately)

    times = []
    positions = []
    target_positions = []
    
    print(f"Starting gripper test for {duration} seconds with {switch_interval}s interval...")

    try:
        # while True:
        #     current_time = time.time()
        #     elapsed = current_time - start_time
            
        #     if elapsed >= duration:
        #         break
            
        #     # Switch Logic
        #     if current_time - last_switch_time >= switch_interval:
        #         if is_closed:
        #             # currently closed, switch to open
        #             print(f"[{elapsed:.2f}s] Opening (Target: 40mm)")
        #             gripper.open_gripper() # 40mm
        #             is_closed = False
        #         else:
        #             # currently open, switch to closed
        #             print(f"[{elapsed:.2f}s] Closing (Target: 0mm)")
        #             gripper.close_gripper() # 0mm
        #             is_closed = True
        #         last_switch_time = current_time

        #     # Record
        #     pos = gripper.get_pos()
        #     target = 0.0 if is_closed else 40.0
            
        #     times.append(elapsed)
        #     positions.append(pos)
        #     target_positions.append(target)
            
        #     time.sleep(0.01) # 100Hz sampling loop
        for i in range(1000):
            time1=time.time()
            gripper.move(i%40,255,2)
            time.sleep(0.1)
            time2=time.time()
            a=gripper.get_pos()
            time3=time.time()
            
            print("target:",i%40,",act:",a)

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        print("Stopping gripper...")
        gripper.stop()
        
        # Plotting
        print(f"Plotting {len(times)} data points...")
        plt.figure(figsize=(10, 6))
        plt.plot(times, positions, label='Actual Position')
        plt.plot(times, target_positions, '--', label='Target Position', alpha=0.7)
        plt.xlabel('Time (s)')
        plt.ylabel('Position (mm)')
        plt.title('Gripper Step Response')
        plt.legend()
        plt.grid(True)
        plt.savefig('gripper_response.png')
        print("Plot saved to 'gripper_response.png'")
        # plt.show() # blocking call, maybe better not to block if running headless

    # 256f:c635