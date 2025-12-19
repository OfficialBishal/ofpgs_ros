import os
import psutil
import signal
import subprocess
import threading


class process_handler:
    @staticmethod
    def start_process(command, capture_output=False):
        """Start a subprocess and return the process object."""
        stdout = subprocess.DEVNULL if not capture_output else subprocess.PIPE
        process = subprocess.Popen(command, shell=True, stdout=stdout, stderr=subprocess.STDOUT, preexec_fn=os.setpgrp)
        print(f"Started: {command}, PID: {process.pid}")

        
        if capture_output:
            # Start a thread to read and print stdout in real-time
            def print_output():
                for line in iter(process.stdout.readline, b''):
                    print(line.decode().strip())  # Print each line in real-time

            output_thread = threading.Thread(target=print_output)
            output_thread.daemon = True  # Daemonize the thread so it ends when the main program ends
            output_thread.start()

        return process

    @staticmethod
    def cleanup(processes, signum=None, frame=None):
        """Cleanup function to terminate all processes and their children."""
        print("Cleaning up...")
        for name, process in processes.items():
            if process and process.poll() is None:  # Check if the process is still running
                print(f"Terminating {name} (PID: {process.pid})")
                try:
                    # Use psutil to terminate the process and its children
                    p = psutil.Process(process.pid)
                    for child in p.children(recursive=True):
                        child.kill()  # Kill the child processes
                    p.kill()  # Kill the parent process
                    print(f"Successfully terminated {name} and its children.")
                except psutil.NoSuchProcess:
                    print(f"Process {name} (PID: {process.pid}) no longer exists.")
                except psutil.AccessDenied:
                    print(f"Permission denied to terminate {name} (PID: {process.pid}).")
                except Exception as e:
                    print(f"Error terminating {name} (PID: {process.pid}): {e}")
        
        print("Cleanup complete.")

    @staticmethod
    def cleanup_exit(processes, signum=None, frame=None):
        process_handler.cleanup(processes)
        exit(0)

    @staticmethod
    def print_process_output_real_time(process):
        """Function to print output of the process in real-time."""
        for line in process.stdout:
            print(line.decode(), end='')  # stdout
        for line in process.stderr:
            print(line.decode(), end='')  # stderr

