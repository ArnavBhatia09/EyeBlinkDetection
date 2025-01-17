import psutil
import datetime

boot_time_timestamp = psutil.boot_time()
boot_time = datetime.datetime.fromtimestamp(boot_time_timestamp)

print(f"Boot time: {boot_time}")

current_time = datetime.datetime.now()
print(f"Current time: {current_time}")

on_time = current_time - boot_time

print(f"On time (in seconds): {on_time}")
