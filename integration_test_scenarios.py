#!/usr/bin/env python3
"""
Integration Test Scenarios for Heart Failure Monitoring System

This script runs a series of end-to-end tests that simulate real-world
usage scenarios of the heart failure monitoring system.
"""

import os
import sys
import time
import json
import signal
import sqlite3
import argparse
import subprocess
import threading
from datetime import datetime

# Configuration
TEST_DB_DIR = "test_databases"
SERVER_BASE_PORT = 50071
NUM_TESTS_PER_SCENARIO = 3

# Test scenarios
scenarios = [
    {
        "name": "Normal Operation",
        "description": "Tests system under normal operation with leader and two followers",
        "servers": 3,
        "clients": 2,
        "duration": 30,  # seconds
        "events": []  # No fault injection
    },
    {
        "name": "Leader Failure",
        "description": "Tests failover when leader crashes",
        "servers": 3,
        "clients": 2,
        "duration": 60,
        "events": [
            {"time": 20, "action": "kill_leader"}
        ]
    },
    {
        "name": "Leader Recovery",
        "description": "Tests system when crashed leader rejoins",
        "servers": 3,
        "clients": 2,
        "duration": 90,
        "events": [
            {"time": 20, "action": "kill_leader"},
            {"time": 40, "action": "restart_server", "server_id": 1}
        ]
    },
    {
        "name": "Follower Failure",
        "description": "Tests system when a follower crashes",
        "servers": 3,
        "clients": 2,
        "duration": 60,
        "events": [
            {"time": 20, "action": "kill_server", "server_id": 2}
        ]
    },
    {
        "name": "Multiple Failures",
        "description": "Tests system under multiple component failures",
        "servers": 5,
        "clients": 3,
        "duration": 120,
        "events": [
            {"time": 20, "action": "kill_server", "server_id": 3},
            {"time": 40, "action": "kill_leader"},
            {"time": 60, "action": "kill_server", "server_id": 4},
            {"time": 80, "action": "restart_server", "server_id": 3}
        ]
    },
    {
        "name": "Dynamic Membership",
        "description": "Tests servers joining and leaving the cluster",
        "servers": 3,
        "clients": 2, 
        "duration": 90,
        "events": [
            {"time": 20, "action": "add_server", "server_id": 4},
            {"time": 40, "action": "add_server", "server_id": 5},
            {"time": 60, "action": "kill_server", "server_id": 2}
        ]
    },
    {
        "name": "High Load",
        "description": "Tests system under high request load",
        "servers": 3,
        "clients": 5,
        "duration": 60,
        "events": [
            {"time": 20, "action": "burst_traffic"}
        ]
    }
]

class IntegrationTest:
    def __init__(self, scenario, verbose=False):
        self.scenario = scenario
        self.verbose = verbose
        self.server_processes = {}
        self.client_processes = {}
        self.event_threads = []
        self.leader_id = 1
        
        # Create test directory
        os.makedirs(TEST_DB_DIR, exist_ok=True)
        
        # Store results
        self.results = {
            "scenario": scenario["name"],
            "start_time": None,
            "end_time": None,
            "success": False,
            "events": [],
            "failures": [],
            "metrics": {
                "successful_reports": 0,
                "failed_reports": 0,
                "successful_elections": 0,
                "failed_elections": 0
            }
        }
    
    def log(self, message):
        """Log a message with timestamp"""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"[{timestamp}] {message}")
    
    def create_test_configs(self):
        """Create test configurations for servers and clients"""
        # Server config template
        server_config_template = {
            "server_id": None,
            "server_host": "127.0.0.1",
            "server_port": None,
            "replica_addresses": [],
            "db_file": None,
            "heartbeat_interval": 2,
            "lease_timeout": 5,
            "initial_leader": False
        }
        
        # Create replica address list
        replica_addresses = []
        for i in range(1, self.scenario["servers"] + 1):
            port = SERVER_BASE_PORT + i
            replica_addresses.append(f"127.0.0.1:{port}")
        
        # Create configs for each server
        for i in range(1, self.scenario["servers"] + 1):
            server_config = server_config_template.copy()
            server_config["server_id"] = i
            server_config["server_port"] = SERVER_BASE_PORT + i
            server_config["replica_addresses"] = replica_addresses
            server_config["db_file"] = os.path.join(TEST_DB_DIR, f"test_{self.scenario['name'].replace(' ', '_').lower()}_server_{i}.db")
            server_config["initial_leader"] = (i == 1)  # First server is the leader
            
            # Write config to file
            config_path = f"test_config_server_{i}.json"
            with open(config_path, "w") as f:
                json.dump(server_config, f, indent=2)
        
        # Client config
        client_config = {
            "client_connect_host": "127.0.0.1",
            "client_connect_port": SERVER_BASE_PORT + 1,  # Connect to leader initially
            "replica_addresses": replica_addresses,
            "rpc_timeout": 5,
            "fallback_timeout": 1,
            "overall_leader_lookup_timeout": 5,
            "retry_delay": 1,
            "client_heartbeat_interval": 2,
            "monitoring_interval": 5,  # Shortened for testing
            "green_threshold": 0.30,
            "amber_threshold": 0.60
        }
        
        # Write client config to file
        with open("test_config_client.json", "w") as f:
            json.dump(client_config, f, indent=2)
    
    def cleanup_configs(self):
        """Remove test configuration files"""
        for i in range(1, self.scenario["servers"] + 5):  # +5 for potential dynamic servers
            config_path = f"test_config_server_{i}.json"
            if os.path.exists(config_path):
                os.remove(config_path)
        
        if os.path.exists("test_config_client.json"):
            os.remove("test_config_client.json")
    
    def start_server(self, server_id):
        """Start a server with the given ID"""
        if server_id in self.server_processes:
            self.log(f"Server {server_id} is already running")
            return
        
        config_path = f"test_config_server_{server_id}.json"
        
        if not os.path.exists(config_path):
            self.log(f"Config for server {server_id} does not exist")
            return
        
        command = [
            sys.executable,
            "hf_replicated_server.py",
            f"--config={config_path}"
        ]
        
        log_file = open(f"{TEST_DB_DIR}/server_{server_id}.log", "w")
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        self.server_processes[server_id] = {
            "process": process,
            "log_file": log_file,
            "start_time": datetime.now()
        }
        
        self.log(f"Started server {server_id}")
        time.sleep(1)  # Give the server time to start
    
    def start_client(self, client_id, headless=True):
        """Start a client with the given ID"""
        if client_id in self.client_processes:
            self.log(f"Client {client_id} is already running")
            return
        
        # We'll use a headless version of the client for testing
        command = [
            sys.executable,
            "headless_client.py" if headless else "hf_client_gui.py",
            f"--config=test_config_client.json",
            f"--patient_id=TEST-P{client_id}",
            "--auto_start"
        ]
        
        log_file = open(f"{TEST_DB_DIR}/client_{client_id}.log", "w")
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        self.client_processes[client_id] = {
            "process": process,
            "log_file": log_file,
            "start_time": datetime.now()
        }
        
        self.log(f"Started client {client_id}")
    
    def kill_server(self, server_id):
        """Kill a server with the given ID"""
        if server_id not in self.server_processes:
            self.log(f"Server {server_id} is not running")
            return
        
        process_info = self.server_processes[server_id]
        process = process_info["process"]
        
        try:
            process.terminate()
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
        
        process_info["log_file"].close()
        
        self.log(f"Killed server {server_id}")
        self.server_processes.pop(server_id)
        
        # Record the event
        self.results["events"].append({
            "time": (datetime.now() - self.results["start_time"]).total_seconds(),
            "type": "server_killed",
            "server_id": server_id
        })
    
    def kill_leader(self):
        """Kill the current leader server"""
        self.log(f"Killing leader (server {self.leader_id})")
        self.kill_server(self.leader_id)
        
        # Wait for a new leader to be elected
        time.sleep(10)
        
        # Find the new leader by querying server 1, 2, 3... until we find a running one
        new_leader_found = False
        for i in range(1, self.scenario["servers"] + 5):  # +5 for potential dynamic servers
            if i in self.server_processes:
                try:
                    new_leader_id = self.query_for_leader(i)
                    if new_leader_id:
                        self.leader_id = new_leader_id
                        new_leader_found = True
                        self.log(f"New leader is server {self.leader_id}")
                        
                        # Record successful election
                        self.results["metrics"]["successful_elections"] += 1
                        break
                except Exception as e:
                    self.log(f"Error querying server {i} for leader: {e}")
        
        if not new_leader_found:
            self.log("Failed to find new leader")
            self.results["metrics"]["failed_elections"] += 1
            self.results["failures"].append({
                "time": (datetime.now() - self.results["start_time"]).total_seconds(),
                "type": "leader_election_failed"
            })
    
    def query_for_leader(self, server_id):
        """Query a server to find out who the current leader is"""
        # This is a placeholder - in a real implementation, you would
        # use gRPC to call GetLeaderInfo on the server and parse the response
        # For this test script, we'll simulate this by reading the server log
        log_path = f"{TEST_DB_DIR}/server_{server_id}.log"
        
        if not os.path.exists(log_path):
            return None
        
        with open(log_path, "r") as f:
            log_content = f.read()
            
            # Look for leader info in logs
            # This is an approximation - in production you'd use the gRPC API
            if "I am leader" in log_content:
                return server_id
            
            # Find the most recent leader mentioned in logs
            leader_mentions = []
            for line in log_content.splitlines():
                if "leader_address" in line and "127.0.0.1:" in line:
                    try:
                        port = int(line.split("127.0.0.1:")[1].split()[0].split(',')[0])
                        leader_id = port - SERVER_BASE_PORT
                        leader_mentions.append(leader_id)
                    except:
                        pass
            
            if leader_mentions:
                return leader_mentions[-1]  # Return the most recent mention
        
        return None
    
    def add_server(self, server_id):
        """Add a new server to the cluster"""
        self.log(f"Adding server {server_id} to the cluster")
        
        # Create config for the new server
        server_config_template = {
            "server_id": server_id,
            "server_host": "127.0.0.1",
            "server_port": SERVER_BASE_PORT + server_id,
            "replica_addresses": [],
            "db_file": os.path.join(TEST_DB_DIR, f"test_{self.scenario['name'].replace(' ', '_').lower()}_server_{server_id}.db"),
            "heartbeat_interval": 2,
            "lease_timeout": 5,
            "initial_leader": False
        }
        
        # Get replica addresses from an existing config
        existing_config_path = f"test_config_server_1.json"
        with open(existing_config_path, "r") as f:
            existing_config = json.load(f)
            server_config_template["replica_addresses"] = existing_config["replica_addresses"]
        
        # Add the new server address to the replica list
        new_address = f"127.0.0.1:{SERVER_BASE_PORT + server_id}"
        if new_address not in server_config_template["replica_addresses"]:
            server_config_template["replica_addresses"].append(new_address)
        
        # Write config to file
        config_path = f"test_config_server_{server_id}.json"
        with open(config_path, "w") as f:
            json.dump(server_config_template, f, indent=2)
        
        # Start the server
        self.start_server(server_id)
        
        # Record the event
        self.results["events"].append({
            "time": (datetime.now() - self.results["start_time"]).total_seconds(),
            "type": "server_added",
            "server_id": server_id
        })
    
    def burst_traffic(self):
        """Simulate a burst of traffic by making more frequent client reports"""
        self.log("Simulating traffic burst")
        
        # We'll just add more clients temporarily
        burst_clients = []
        for i in range(10):
            client_id = 100 + i  # Use IDs that won't conflict with existing clients
            self.start_client(client_id)
            burst_clients.append(client_id)
        
        # Let the burst run for 10 seconds
        time.sleep(10)
        
        # Then kill the burst clients
        for client_id in burst_clients:
            if client_id in self.client_processes:
                process_info = self.client_processes[client_id]
                process_info["process"].terminate()
                process_info["log_file"].close()
                self.client_processes.pop(client_id)
        
        self.log("Traffic burst complete")
    
    def schedule_event(self, event):
        """Schedule an event to happen at a specific time"""
        event_time = event["time"]
        action = event["action"]
        
        def event_thread():
            self.log(f"Scheduling event: {action} at time +{event_time}s")
            time.sleep(event_time)
            
            if action == "kill_leader":
                self.kill_leader()
            elif action == "kill_server":
                self.kill_server(event["server_id"])
            elif action == "restart_server":
                self.start_server(event["server_id"])
            elif action == "add_server":
                self.add_server(event["server_id"])
            elif action == "burst_traffic":
                self.burst_traffic()
            else:
                self.log(f"Unknown action: {action}")
        
        thread = threading.Thread(target=event_thread)
        thread.daemon = True
        self.event_threads.append(thread)
    
    def analyze_results(self):
        """Analyze test results by checking logs and databases"""
        try:
            # Count successful reports by looking at databases
            for i in range(1, self.scenario["servers"] + 5):  # +5 for potential dynamic servers
                db_path = os.path.join(TEST_DB_DIR, f"test_{self.scenario['name'].replace(' ', '_').lower()}_server_{i}.db")
                if os.path.exists(db_path):
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    try:
                        cursor.execute("SELECT COUNT(*) FROM risk_reports")
                        count = cursor.fetchone()[0]
                        self.log(f"Server {i} has {count} reports")
                        
                        # Just use the maximum count as our measure
                        self.results["metrics"]["successful_reports"] = max(
                            self.results["metrics"]["successful_reports"],
                            count
                        )
                    except Exception as e:
                        self.log(f"Error querying database {i}: {e}")
                    finally:
                        conn.close()
            
            # Check for any unhandled errors in logs
            error_count = 0
            for server_id in range(1, self.scenario["servers"] + 5):
                log_path = f"{TEST_DB_DIR}/server_{server_id}.log"
                if os.path.exists(log_path):
                    with open(log_path, "r") as f:
                        log_content = f.read()
                        for line in log_content.splitlines():
                            if "ERROR" in line and not any(expected in line for expected in [
                                "Heartbeat to", "Election RPC", "Replication to", "Failed"
                            ]):
                                error_count += 1
                                self.log(f"Unexpected error in server {server_id}: {line}")
            
            # Check client logs for errors
            for client_id in range(1, self.scenario["clients"] + 15):  # +15 for burst clients
                log_path = f"{TEST_DB_DIR}/client_{client_id}.log"
                if os.path.exists(log_path):
                    with open(log_path, "r") as f:
                        log_content = f.read()
                        for line in log_content.splitlines():
                            if "ERROR" in line and not "RPC UNAVAILABLE" in line:
                                error_count += 1
                                self.log(f"Unexpected error in client {client_id}: {line}")
            
            # Mark the test as successful if we have reports and no unexpected errors
            self.results["success"] = (
                self.results["metrics"]["successful_reports"] > 0 and 
                error_count == 0
            )
            
            # Add error count to results
            self.results["metrics"]["unexpected_errors"] = error_count
            
        except Exception as e:
            self.log(f"Error analyzing results: {e}")
            self.results["success"] = False
    
    def run(self):
        """Run the integration test scenario"""
        self.log(f"Running scenario: {self.scenario['name']}")
        self.log(f"Description: {self.scenario['description']}")
        
        try:
            # Record start time
            self.results["start_time"] = datetime.now()
            
            # Create test configs
            self.create_test_configs()
            
            # Schedule events
            for event in self.scenario["events"]:
                self.schedule_event(event)
            
            # Start all event threads
            for thread in self.event_threads:
                thread.start()
            
            # Start servers
            for i in range(1, self.scenario["servers"] + 1):
                self.start_server(i)
            
            # Allow servers to stabilize
            time.sleep(5)
            
            # Start clients
            for i in range(1, self.scenario["clients"] + 1):
                self.start_client(i)
            
            # Let the scenario run for the specified duration
            self.log(f"Scenario running for {self.scenario['duration']} seconds")
            time.sleep(self.scenario['duration'])
            
            # Record end time
            self.results["end_time"] = datetime.now()
            
            # Wait for all event threads to complete
            for thread in self.event_threads:
                thread.join(timeout=1)
            
            # Clean up processes
            for server_id in list(self.server_processes.keys()):
                self.kill_server(server_id)
            
            for client_id, process_info in list(self.client_processes.items()):
                process = process_info["process"]
                process.terminate()
                process_info["log_file"].close()
                self.client_processes.pop(client_id)
            
            # Analyze results
            self.analyze_results()
            
            # Report success or failure
            duration = (self.results["end_time"] - self.results["start_time"]).total_seconds()
            status = "SUCCESS" if self.results["success"] else "FAILED"
            self.log(f"Scenario {self.scenario['name']} {status} in {duration:.1f}s")
            self.log(f"Metrics: {json.dumps(self.results['metrics'], indent=2)}")
            
            return self.results
            
        except Exception as e:
            self.log(f"Error running scenario: {e}")
            self.results["success"] = False
            self.results["failures"].append({
                "time": (datetime.now() - self.results["start_time"]).total_seconds() if self.results["start_time"] else 0,
                "type": "exception",
                "message": str(e)
            })
            return self.results
        finally:
            # Always clean up configs
            self.cleanup_configs()

def run_all_scenarios(args):
    """Run all test scenarios"""
    results = []
    
    for scenario in scenarios:
        if args.scenario and scenario["name"].lower() != args.scenario.lower():
            continue
            
        print(f"\n{'='*80}")
        print(f"SCENARIO: {scenario['name']}")
        print(f"{'='*80}")
        
        # Run the scenario multiple times for more reliable results
        scenario_results = []
        for i in range(1, args.repeat + 1):
            print(f"\nRun {i}/{args.repeat}")
            test = IntegrationTest(scenario, verbose=args.verbose)
            result = test.run()
            scenario_results.append(result)
        
        # Summarize results
        successes = sum(1 for r in scenario_results if r["success"])
        print(f"\nSummary: {successes}/{args.repeat} successful runs")
        
        # Add to overall results
        results.append({
            "scenario": scenario["name"],
            "description": scenario["description"],
            "success_rate": successes / args.repeat,
            "runs": scenario_results
        })
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run integration tests for Heart Failure Monitoring System")
    parser.add_argument("--scenario", help="Run a specific scenario by name")
    parser.add_argument("--repeat", type=int, default=NUM_TESTS_PER_SCENARIO, help="Number of times to repeat each scenario")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--output", help="Save results to JSON file")
    args = parser.parse_args()
    
    results = run_all_scenarios(args)
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()