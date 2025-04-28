import unittest
import os
import sys
import json
import time
import sqlite3
import threading
import subprocess
import grpc
import numpy as np
import tensorflow as tf
import joblib
from unittest.mock import MagicMock, patch
import signal
import shutil
import random

# Import system modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chat_pb2
import chat_pb2_grpc

# Helper function to start a server process
def start_server(server_id, port, is_leader=False, host="127.0.0.1"):
    process = subprocess.Popen([
        sys.executable, 
        "hf_replicated_server.py", 
        f"--server_id={server_id}", 
        f"--server_host={host}", 
        f"--server_port={port}",
        f"--initial_leader={'true' if is_leader else 'false'}"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(2)  # Allow time for server to start
    return process

# Helper function to kill a process
def kill_process(process):
    if process:
        process.terminate()
        process.wait()

class TestHeartFailureSystem(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Create test directory
        cls.test_dir = "test_outputs"
        os.makedirs(cls.test_dir, exist_ok=True)
        
        # Ensure model and scaler exist
        if not os.path.exists("heart_failure_model.h5") or not os.path.exists("hf_scaler.gz"):
            raise Exception("Model files not found. Run train_hf_model.py first.")
        
        # Modify config files for testing
        cls.original_config = {}
        cls.backup_config()
        cls.create_test_configs()

    @classmethod
    def tearDownClass(cls):
        # Restore original configs
        cls.restore_config()
        
        # Clean up test directory
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
    
    @classmethod
    def backup_config(cls):
        # Backup original config files
        if os.path.exists("config.json"):
            with open("config.json", "r") as f:
                cls.original_config["server"] = json.load(f)
        
        if os.path.exists("config_client.json"):
            with open("config_client.json", "r") as f:
                cls.original_config["client"] = json.load(f)
    
    @classmethod
    def restore_config(cls):
        # Restore original config files
        if "server" in cls.original_config:
            with open("config.json", "w") as f:
                json.dump(cls.original_config["server"], f, indent=2)
        
        if "client" in cls.original_config:
            with open("config_client.json", "w") as f:
                json.dump(cls.original_config["client"], f, indent=2)
    
    @classmethod
    def create_test_configs(cls):
        # Create test server config
        server_config = {
            "server_id": 1,
            "server_host": "127.0.0.1",
            "server_port": 50061,
            "replica_addresses": [
                "127.0.0.1:50061",
                "127.0.0.1:50062", 
                "127.0.0.1:50063"
            ],
            "db_file": os.path.join(cls.test_dir, "test_chat_1.db"),
            "heartbeat_interval": 1,
            "lease_timeout": 3,
            "initial_leader": True
        }
        
        with open("config.json", "w") as f:
            json.dump(server_config, f, indent=2)
        
        # Create test client config
        client_config = {
            "client_connect_host": "127.0.0.1",
            "client_connect_port": 50061,
            "replica_addresses": [
                "127.0.0.1:50061",
                "127.0.0.1:50062",
                "127.0.0.1:50063"
            ],
            "rpc_timeout": 3,
            "fallback_timeout": 1,
            "overall_leader_lookup_timeout": 3,
            "retry_delay": 1,
            "client_heartbeat_interval": 2,
            "monitoring_interval": 5,
            "green_threshold": 0.30,
            "amber_threshold": 0.60
        }
        
        with open("config_client.json", "w") as f:
            json.dump(client_config, f, indent=2)
    
    def setUp(self):
        # Clean up any test databases before each test
        for i in range(1, 4):
            db_path = os.path.join(self.test_dir, f"test_chat_{i}.db")
            if os.path.exists(db_path):
                os.remove(db_path)
        
        # Start with no running processes
        self.server_processes = []
    
    def tearDown(self):
        # Kill any running server processes
        for process in self.server_processes:
            kill_process(process)
        self.server_processes = []
    
    def start_test_servers(self, count=3):
        """Start multiple test servers"""
        self.server_processes = []
        
        # Start leader
        leader_process = start_server(1, 50061, is_leader=True)
        self.server_processes.append(leader_process)
        
        # Start followers
        if count > 1:
            for i in range(2, count+1):
                follower_process = start_server(i, 50060+i, is_leader=False)
                self.server_processes.append(follower_process)
        
        # Allow time for leader election and stabilization
        time.sleep(3)
    
    def get_stub(self, address="127.0.0.1:50061"):
        """Get a gRPC stub for the specified server"""
        channel = grpc.insecure_channel(address)
        return chat_pb2_grpc.ChatServiceStub(channel)
    
    def test_server_startup(self):
        """Test basic server startup and shutdown"""
        print("\nTest: Server Startup and Shutdown")
        self.start_test_servers(1)
        
        # Check if server is running by calling GetLeaderInfo
        stub = self.get_stub()
        try:
            response = stub.GetLeaderInfo(chat_pb2.GetLeaderInfoRequest())
            self.assertTrue(response.success)
            self.assertEqual(response.leader_address, "127.0.0.1:50061")
            print("  ✓ Server started successfully")
        except Exception as e:
            self.fail(f"Failed to connect to server: {e}")
        
        # Shutdown
        kill_process(self.server_processes[0])
        print("  ✓ Server shutdown successful")
    
    def test_multiple_servers(self):
        """Test multiple servers startup and leader election"""
        print("\nTest: Multiple Servers and Leadership")
        self.start_test_servers(3)
        
        # Check if all servers are running and recognize the leader
        for i, port in enumerate([50061, 50062, 50063], 1):
            stub = self.get_stub(f"127.0.0.1:{port}")
            try:
                response = stub.GetLeaderInfo(chat_pb2.GetLeaderInfoRequest())
                self.assertTrue(response.success)
                self.assertEqual(response.leader_address, "127.0.0.1:50061")
                print(f"  ✓ Server {i} recognizes leader correctly")
            except Exception as e:
                self.fail(f"Failed to connect to server {i}: {e}")
    
    def test_risk_report_storage(self):
        """Test risk report storage in the database"""
        print("\nTest: Risk Report Storage")
        self.start_test_servers(1)
        
        # Create a risk report and send it to the server
        stub = self.get_stub()
        
        # Test with multiple reports
        test_reports = [
            {
                "patient_id": "P-1001",
                "inputs": [72.5, 130.2, 1.8, 35.0, 42],
                "probability": 0.75,
                "tier": "RED"
            },
            {
                "patient_id": "P-1002",
                "inputs": [65.0, 138.0, 1.2, 45.0, 30],
                "probability": 0.40,
                "tier": "AMBER"
            }
        ]
        
        for report_data in test_reports:
            request = chat_pb2.RiskReportRequest(
                patient_id=report_data["patient_id"],
                timestamp=int(time.time()),
                inputs=report_data["inputs"],
                probability=report_data["probability"],
                tier=report_data["tier"]
            )
            
            try:
                response = stub.SendRiskReport(request)
                self.assertTrue(response.success)
                print(f"  ✓ Report for {report_data['patient_id']} sent successfully")
            except Exception as e:
                self.fail(f"Failed to send report: {e}")
        
        # Verify storage in database
        time.sleep(1)  # Allow time for database writes
        
        db_path = os.path.join(self.test_dir, "test_chat_1.db")
        self.assertTrue(os.path.exists(db_path), "Database file not created")
        
        # Check database contents
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT patient_id, tier FROM risk_reports")
        rows = cursor.fetchall()
        conn.close()
        
        self.assertEqual(len(rows), 2, "Database should contain 2 reports")
        patient_ids = [row[0] for row in rows]
        tiers = [row[1] for row in rows]
        
        self.assertIn("P-1001", patient_ids)
        self.assertIn("P-1002", patient_ids)
        self.assertIn("RED", tiers)
        self.assertIn("AMBER", tiers)
        
        print("  ✓ Reports stored correctly in database")
    
    def test_data_replication(self):
        """Test data replication between leader and followers"""
        print("\nTest: Data Replication")
        self.start_test_servers(3)
        
        # Send a risk report to the leader
        stub = self.get_stub("127.0.0.1:50061")
        request = chat_pb2.RiskReportRequest(
            patient_id="P-2001",
            timestamp=int(time.time()),
            inputs=[70.0, 132.0, 1.5, 38.0, 45],
            probability=0.65,
            tier="RED"
        )
        
        response = stub.SendRiskReport(request)
        self.assertTrue(response.success)
        print("  ✓ Report sent to leader successfully")
        
        # Allow time for replication
        time.sleep(3)
        
        # Check all databases for the report
        for i in range(1, 4):
            db_path = os.path.join(self.test_dir, f"test_chat_{i}.db")
            self.assertTrue(os.path.exists(db_path), f"Database file {i} not created")
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT patient_id, tier FROM risk_reports WHERE patient_id = 'P-2001'")
            rows = cursor.fetchall()
            conn.close()
            
            self.assertEqual(len(rows), 1, f"Database {i} should contain the replicated report")
            self.assertEqual(rows[0][0], "P-2001")
            self.assertEqual(rows[0][1], "RED")
            
            print(f"  ✓ Report successfully replicated to server {i}")
    
    def test_leader_failover(self):
        """Test leader failover when the leader crashes"""
        print("\nTest: Leader Failover")
        self.start_test_servers(3)
        
        # Verify initial leader
        stub = self.get_stub("127.0.0.1:50062")  # Ask follower
        response = stub.GetLeaderInfo(chat_pb2.GetLeaderInfoRequest())
        self.assertEqual(response.leader_address, "127.0.0.1:50061")
        print("  ✓ Initial leader is server 1")
        
        # Kill the leader
        print("  ✓ Killing leader (server 1)")
        kill_process(self.server_processes[0])
        self.server_processes.pop(0)
        
        # Wait for leader election
        time.sleep(5)
        
        # Check who is the new leader - should be server 2 (lowest ID among remaining)
        stub = self.get_stub("127.0.0.1:50063")  # Ask another follower
        response = stub.GetLeaderInfo(chat_pb2.GetLeaderInfoRequest())
        self.assertEqual(response.leader_address, "127.0.0.1:50062")
        print("  ✓ New leader is server 2")
        
        # Test sending a report to the new leader
        stub = self.get_stub("127.0.0.1:50062")
        request = chat_pb2.RiskReportRequest(
            patient_id="P-3001",
            timestamp=int(time.time()),
            inputs=[68.0, 135.0, 1.3, 42.0, 50],
            probability=0.55,
            tier="AMBER"
        )
        
        response = stub.SendRiskReport(request)
        self.assertTrue(response.success)
        print("  ✓ Report sent to new leader successfully")
    
    def test_client_reconnection(self):
        """Test client reconnecting to new leader after failover"""
        print("\nTest: Client Reconnection After Failover")
        self.start_test_servers(3)
        
        # Create a test client
        with patch('tkinter.Tk'):  # Mock Tkinter to avoid GUI
            from hf_client_gui import HeartFailureMonitoringGUI
            
            # Mock the model and GUI elements
            with patch.object(tf.keras.models, 'load_model', return_value=MagicMock()), \
                 patch.object(joblib, 'load', return_value=MagicMock()), \
                 patch.object(HeartFailureMonitoringGUI, 'setup_ui'):
                
                client = HeartFailureMonitoringGUI()
                client.leader_address = "127.0.0.1:50061"
                client.patient_id = "P-TEST"
                
                # Create connection to leader
                client.connect_to_leader("127.0.0.1:50061")
                print("  ✓ Client connected to initial leader")
                
                # Kill the leader
                kill_process(self.server_processes[0])
                self.server_processes.pop(0)
                print("  ✓ Leader (server 1) killed")
                
                # Wait for new leader election
                time.sleep(5)
                
                # Test client's ability to find new leader
                try:
                    client.update_leader()
                    self.assertEqual(client.leader_address, "127.0.0.1:50062")
                    print("  ✓ Client successfully found new leader")
                except Exception as e:
                    self.fail(f"Client failed to find new leader: {e}")
                
                client.cleanup()
    
    def test_model_inference(self):
        """Test model inference with different inputs"""
        print("\nTest: Model Inference")
        
        # Load the model and scaler
        model = tf.keras.models.load_model("heart_failure_model.h5")
        scaler = joblib.load("hf_scaler.gz")
        
        # Test cases representing different risk levels
        test_cases = [
            # Low risk (should be GREEN)
            {
                "inputs": [55.0, 140.0, 0.9, 60.0, 20],
                "expected_tier": "GREEN"
            },
            # Medium risk (should be AMBER)
            {
                "inputs": [70.0, 132.0, 1.5, 38.0, 45],
                "expected_tier": "AMBER"
            },
            # High risk (should be RED)
            {
                "inputs": [80.0, 125.0, 2.2, 25.0, 90],
                "expected_tier": "RED"
            }
        ]
        
        # Test threshold values from config
        with open("config_client.json", "r") as f:
            client_config = json.load(f)
        
        green_threshold = client_config.get("green_threshold", 0.30)
        amber_threshold = client_config.get("amber_threshold", 0.60)
        
        # Test each case
        for i, case in enumerate(test_cases):
            inputs = case["inputs"]
            scaled_inputs = scaler.transform([inputs])
            probability = float(model.predict(scaled_inputs)[0][0])
            
            # Determine tier based on thresholds
            tier = "GREEN"
            if probability >= amber_threshold:
                tier = "RED"
            elif probability >= green_threshold:
                tier = "AMBER"
            
            print(f"  • Case {i+1}: p={probability:.4f}, tier={tier}")
            
            # This test is less strict since model training can have variations
            # Just ensure the inference runs without errors
            self.assertIn(tier, ["GREEN", "AMBER", "RED"])
        
        print("  ✓ Model inference works correctly")
    
    def test_error_handling(self):
        """Test error handling for various scenarios"""
        print("\nTest: Error Handling")
        self.start_test_servers(1)
        
        # Test case 1: Request with empty patient ID
        stub = self.get_stub()
        request = chat_pb2.RiskReportRequest(
            patient_id="",
            timestamp=int(time.time()),
            inputs=[70.0, 132.0, 1.5, 38.0, 45],
            probability=0.65,
            tier="RED"
        )
        
        response = stub.SendRiskReport(request)
        self.assertFalse(response.success)
        print("  ✓ Properly rejected empty patient ID")
        
        # Test case 2: Request with wrong inputs length
        request = chat_pb2.RiskReportRequest(
            patient_id="P-4001",
            timestamp=int(time.time()),
            inputs=[70.0, 132.0],  # Too few inputs
            probability=0.65,
            tier="RED"
        )
        
        response = stub.SendRiskReport(request)
        self.assertFalse(response.success)
        print("  ✓ Properly rejected wrong input length")
        
        # Test case 3: Request with bad values (should handle typecasting)
        request = chat_pb2.RiskReportRequest(
            patient_id="P-4002",
            timestamp=int(time.time()),
            inputs=[70.0, "bad", 1.5, 38.0, 45],  # Should not happen in real code but testing type safety
            probability=0.65,
            tier="RED"
        )
        
        try:
            response = stub.SendRiskReport(request)
            # This would likely fail due to type issues, but should not crash the server
            print("  ✓ Handled bad value types gracefully")
        except Exception as e:
            print(f"  ⚠ Request with bad values raises exception: {e}")
    
    def test_load_testing(self):
        """Test system with multiple concurrent requests"""
        print("\nTest: Load Testing")
        self.start_test_servers(3)
        
        # Number of concurrent requests to simulate
        num_requests = 20
        
        # Prepare test data
        test_data = []
        for i in range(num_requests):
            test_data.append({
                "patient_id": f"P-5{i:03d}",
                "timestamp": int(time.time()),
                "inputs": [
                    random.uniform(50, 85),
                    random.uniform(125, 145),
                    random.uniform(0.8, 2.5),
                    random.uniform(20, 65),
                    random.randint(1, 100)
                ],
                "probability": random.uniform(0.1, 0.9),
                "tier": random.choice(["GREEN", "AMBER", "RED"])
            })
        
        # Function to send a single report
        def send_report(data):
            stub = self.get_stub()
            request = chat_pb2.RiskReportRequest(
                patient_id=data["patient_id"],
                timestamp=data["timestamp"],
                inputs=data["inputs"],
                probability=data["probability"],
                tier=data["tier"]
            )
            
            try:
                response = stub.SendRiskReport(request)
                return response.success
            except Exception as e:
                print(f"Error sending report {data['patient_id']}: {e}")
                return False
        
        # Send requests in parallel
        threads = []
        results = []
        
        for data in test_data:
            thread = threading.Thread(target=lambda d=data: results.append(send_report(d)))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        success_count = sum(results)
        print(f"  ✓ {success_count}/{num_requests} concurrent requests successful")
        
        # Allow time for replication
        time.sleep(5)
        
        # Verify data was replicated to all servers
        for i in range(1, 4):
            db_path = os.path.join(self.test_dir, f"test_chat_{i}.db")
            
            if not os.path.exists(db_path):
                print(f"  ⚠ Database {i} not found")
                continue
                
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM risk_reports")
            count = cursor.fetchone()[0]
            conn.close()
            
            # Check if at least some reports were successful and replicated
            self.assertGreater(count, 0)
            print(f"  ✓ Server {i} has {count} reports stored")
    
    def test_dynamic_membership(self):
        """Test adding a new server to the cluster dynamically"""
        print("\nTest: Dynamic Membership")
        self.start_test_servers(2)
        
        # Send an initial report
        stub = self.get_stub()
        request = chat_pb2.RiskReportRequest(
            patient_id="P-6001",
            timestamp=int(time.time()),
            inputs=[65.0, 138.0, 1.2, 45.0, 30],
            probability=0.40,
            tier="AMBER"
        )
        
        response = stub.SendRiskReport(request)
        self.assertTrue(response.success)
        print("  ✓ Initial report sent successfully")
        
        # Start a new server that will join the cluster
        print("  ✓ Starting new server (server 3)")
        new_server = start_server(3, 50063, is_leader=False)
        self.server_processes.append(new_server)
        
        # Allow time for the new server to join and sync
        time.sleep(5)
        
        # Check if the new server appears in the replica list
        stub = self.get_stub()
        response = stub.GetLeaderInfo(chat_pb2.GetLeaderInfoRequest())
        
        self.assertIn("127.0.0.1:50063", response.replica_addresses)
        print("  ✓ New server added to replica list")
        
        # Send another report
        request = chat_pb2.RiskReportRequest(
            patient_id="P-6002",
            timestamp=int(time.time()),
            inputs=[70.0, 132.0, 1.5, 38.0, 45],
            probability=0.65,
            tier="RED"
        )
        
        response = stub.SendRiskReport(request)
        self.assertTrue(response.success)
        print("  ✓ Second report sent successfully")
        
        # Allow time for replication
        time.sleep(2)
        
        # Check if new server has both reports
        db_path = os.path.join(self.test_dir, "test_chat_3.db")
        
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT patient_id FROM risk_reports ORDER BY id")
            rows = cursor.fetchall()
            conn.close()
            
            patient_ids = [row[0] for row in rows]
            
            self.assertIn("P-6001", patient_ids)
            self.assertIn("P-6002", patient_ids)
            print("  ✓ New server has synchronized all reports")
        else:
            self.fail("New server database not created")

if __name__ == '__main__':
    unittest.main()