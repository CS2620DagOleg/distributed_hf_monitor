import argparse
import os
import subprocess
import sys
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Launch the Heart Failure Monitoring System")
    parser.add_argument("--setup", action="store_true", help="Run setup (train model) before starting")
    parser.add_argument("--servers", type=int, default=3, help="Number of server instances to launch")
    parser.add_argument("--clients", type=int, default=2, help="Number of client instances to launch")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host IP")
    parser.add_argument("--base_port", type=int, default=50051, help="Base port for servers")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create directories if they don't exist
    os.makedirs("logs", exist_ok=True)
    
    # Run setup if requested
    if args.setup:
        print("Running setup (training model)...")
        subprocess.run([sys.executable, "train_hf_model.py"])
    
    # Check if model files exist
    if not os.path.exists("heart_failure_model.h5") or not os.path.exists("hf_scaler.gz"):
        print("Error: Model files not found. Run with --setup first.")
        return
    
    # Launch server instances
    print(f"Launching {args.servers} server instances...")
    server_processes = []
    
    for i in range(args.servers):
        server_id = i + 1
        port = args.base_port + i
        initial_leader = "true" if i == 0 else "false"
        
        # Open log file
        log_file = open(f"logs/server_{server_id}.log", "w")
        
        # Start server process
        cmd = [
            sys.executable, "hf_replicated_server.py",
            "--server_id", str(server_id),
            "--server_host", args.host,
            "--server_port", str(port),
            "--initial_leader", initial_leader
        ]
        
        print(f"Starting server {server_id} on {args.host}:{port} (leader={initial_leader})")
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        server_processes.append((proc, log_file))
    
    # Give servers time to start
    print("Waiting for servers to initialize...")
    time.sleep(5)
    
    # Launch client instances
    print(f"Launching {args.clients} client instances...")
    client_processes = []
    
    for i in range(args.clients):
        client_id = i + 1
        
        # Open log file
        log_file = open(f"logs/client_{client_id}.log", "w")
        
        # Start client process
        cmd = [sys.executable, "hf_client.py"]
        
        print(f"Starting client {client_id}")
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        client_processes.append((proc, log_file))
    
    print("\nHeart Failure Monitoring System is running!")
    print("Server logs are being saved to logs/server_*.log")
    print("Client logs are being saved to logs/client_*.log")
    print("Press Ctrl+C to shut down the system")
    
    try:
        # Wait for user to interrupt
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down the system...")
        
        # Terminate client processes
        for proc, log_file in client_processes:
            proc.terminate()
            log_file.close()
        
        # Terminate server processes
        for proc, log_file in server_processes:
            proc.terminate()
            log_file.close()
        
        print("System shutdown complete.")

if __name__ == "__main__":
    main()