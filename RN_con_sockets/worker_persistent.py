import time
import subprocess

while True:

    try:

        subprocess.run(["python","worker.py", "--host", "", "--port", "500"])

    except:
        pass

    print("Reconnecting in 5s")

    time.sleep(5)