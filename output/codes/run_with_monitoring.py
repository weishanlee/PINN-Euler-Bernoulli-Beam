#!/usr/bin/env python3
import subprocess
import sys
import os
import time
import signal

def run_single_harmonic(harmonic, python_executable):
    """Run training for a single harmonic with proper output handling"""
    
    # Clean up any stale lock
    if os.path.exists('/tmp/gpu_training_lock.lock'):
        try:
            os.remove('/tmp/gpu_training_lock.lock')
        except:
            pass
            
    print(f"\n{'='*70}")
    print(f"Training harmonic {harmonic}")
    print(f"{'='*70}")
    
    # Run individual harmonic
    cmd = [python_executable, 'ultra_precision_wave_pinn_GPU.py', '--harmonics', str(harmonic)]
    print(f"Running: {' '.join(cmd)}")
    
    # Create environment with debugging disabled
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    
    start_time = time.time()
    
    try:
        # Run with real-time output
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env=env
        )
        
        # No timeout - let training complete
        # Read output line by line
        import select
        
        while True:
            # Check if process is still running
            if proc.poll() is not None:
                break
            
            # Use select to check if there's data available (with 1 second timeout)
            ready, _, _ = select.select([proc.stdout], [], [], 1.0)
            if ready:
                line = proc.stdout.readline()
                if not line:
                    break
                print(line.rstrip())
            
            # Optional: print periodic status to show it's still running
            elapsed = time.time() - start_time
            if int(elapsed) % 300 == 0 and int(elapsed) > 0:  # Every 5 minutes
                print(f"... Training still running (elapsed: {elapsed/60:.1f} minutes)")
        
        exit_code = proc.wait()
        print(f"\nProcess exited with code: {exit_code}")
        
        return exit_code
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        proc.terminate()
        raise
    except Exception as e:
        print(f"Error running harmonic {harmonic}: {e}")
        return -1

def run_training():
    """Run the training script with monitoring and auto-restart on failure"""
    
    # Check completed harmonics
    completed_file = 'completed_harmonics.txt'
    completed = set()
    if os.path.exists(completed_file):
        with open(completed_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        completed.add(int(line))
                    except ValueError:
                        print(f"Warning: Invalid harmonic value in completed file: {line}")
    
    all_harmonics = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]  # Limited to 50 to avoid segfaults
    remaining = [h for h in all_harmonics if h not in completed]
    
    if not remaining:
        print("All harmonics already completed!")
        return True
    
    print(f"Completed harmonics: {sorted(completed)}")
    print(f"Remaining harmonics: {remaining}")
    
    # Python executable
    python_executable = '/home/wslee/.venv/ml_31226124/bin/python'
    
    # Process each harmonic
    for harmonic in remaining:
        exit_code = run_single_harmonic(harmonic, python_executable)
        
        if exit_code == 0:
            print(f"Harmonic {harmonic} completed successfully!")
            
            # Check if already in completed file
            file_completed = set()
            if os.path.exists(completed_file):
                with open(completed_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                file_completed.add(int(line))
                            except:
                                pass
            
            # Write to file if not already there
            if harmonic not in file_completed:
                try:
                    with open(completed_file, 'a') as f:
                        f.write(f"{harmonic}\n")
                        f.flush()
                        os.fsync(f.fileno())  # Force write to disk
                    print(f"Harmonic {harmonic} written to completed file")
                except Exception as e:
                    print(f"Warning: Could not mark harmonic {harmonic} as completed: {e}")
            
            # Mark as completed in memory
            completed.add(harmonic)
        elif exit_code == -11 or exit_code == 139:
            print(f"SEGMENTATION FAULT detected for harmonic {harmonic}!")
            # Continue with next harmonic
        else:
            print(f"Unexpected exit code: {exit_code}")
            
        # Wait between harmonics and ensure process cleanup
        print("Waiting 15 seconds before next harmonic...")
        # Kill any lingering processes
        try:
            subprocess.run(['pkill', '-f', 'ultra_precision_wave_pinn_GPU.py'], capture_output=True)
        except:
            pass
        time.sleep(15)
    
    # Check final status
    return len(completed) == len(all_harmonics)

def main():
    """Main monitoring loop with restart capability"""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run PINN training with monitoring')
    parser.add_argument('--all', action='store_true',
                        help='Train all harmonic configurations')
    parser.add_argument('--harmonics', type=int, default=None,
                        help='Train specific harmonic (if not using --all)')
    args = parser.parse_args()
    
    # If --all is specified, run all harmonics
    if args.all or args.harmonics is None:
        print("Training all harmonics...")
        max_attempts = 5
        attempt = 0
        
        while attempt < max_attempts:
            attempt += 1
            print(f"\n{'='*70}")
            print(f"Attempt {attempt} of {max_attempts}")
            print(f"{'='*70}\n")
            
            try:
                success = run_training()
                if success:
                    break
            except KeyboardInterrupt:
                print("\nTraining interrupted by user")
                break
            except Exception as e:
                print(f"Error in training: {e}")
                import traceback
                traceback.print_exc()
            
            # Wait before retry
            if attempt < max_attempts:
                print(f"\nWaiting 10 seconds before retry...")
                time.sleep(10)
    else:
        # Train single harmonic
        print(f"Training single harmonic: {args.harmonics}")
        python_executable = '/home/wslee/.venv/ml_31226124/bin/python'
        exit_code = run_single_harmonic(args.harmonics, python_executable)
        if exit_code == 0:
            print(f"Harmonic {args.harmonics} completed successfully!")
        else:
            print(f"Harmonic {args.harmonics} failed with exit code: {exit_code}")
    
    # Final status check
    completed_file = 'completed_harmonics.txt'
    if os.path.exists(completed_file):
        with open(completed_file, 'r') as f:
            completed = set()
            for line in f:
                line = line.strip()
                if line:
                    try:
                        completed.add(int(line))
                    except ValueError:
                        pass
        print(f"\nFinal completed harmonics: {sorted(completed)}")
        
        all_harmonics = set([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])  # Limited to 50 to avoid segfaults
        if completed == all_harmonics:
            print("SUCCESS: All harmonics completed!")
        else:
            missing = all_harmonics - completed
            print(f"Missing harmonics: {sorted(missing)}")

if __name__ == "__main__":
    main()