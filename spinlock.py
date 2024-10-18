import time

class Spinlock:
    def __init__(self):
        self.locked = False
    
    def acquire(self, timeout=None):
        start_time = time.time()
        while True:
            if not self.locked:
                self.locked = True
                break
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError("Failed to acquire lock within timeout")
            time.sleep(0.001)  # Add a tiny sleep to prevent excessive CPU usage during waiting
    
    def release(self):
        self.locked = False
