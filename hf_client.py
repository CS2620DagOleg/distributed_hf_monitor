import os
import json
import time
import tkinter as tk
from tkinter import messagebox, simpledialog
import grpc
import numpy as np
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import joblib
import tensorflow as tf

import chat_pb2
import chat_pb2_grpc

# Load client config from config_client.json
with open("config_client.json", "r") as config_file:
    client_config = json.load(config_file)

# Force IPv4: use 127.0.0.1 for connection
host = client_config.get("client_connect_host", "127.0.0.1")
if host == "localhost":
    host = "127.0.0.1"
client_config["client_connect_host"] = host


RPC_TIMEOUT = client_config.get("rpc_timeout", 10)
FALLBACK_TIMEOUT = client_config.get("fallback_timeout", 1)
OVERALL_LEADER_LOOKUP_TIMEOUT = client_config.get("overall_leader_lookup_timeout", 5)
RETRY_DELAY = client_config.get("retry_delay", 1)
CLIENT_HEARTBEAT_INTERVAL = client_config.get("client_heartbeat_interval", 5)
MONITORING_INTERVAL = client_config.get("monitoring_interval", 30)  

# Risk thresholds
GREEN_THRESHOLD = 0.30
AMBER_THRESHOLD = 0.60

class HeartFailureMonitoringClient:
    def __init__(self, patient_id):
        self.patient_id = patient_id
        
        # Load model and scaler
        try:
            self.model = tf.keras.models.load_model("heart_failure_model.h5")
            self.scaler = joblib.load("hf_scaler.gz")
            print(f"Model and scaler loaded successfully")
        except Exception as e:
            print(f"Error loading model or scaler: {e}")
            raise
            
       
        self.leader_address = f"{client_config['client_connect_host']}:{client_config['client_connect_port']}"
        self.connect_to_leader(self.leader_address)
        
       
        self.report_queue = []
        
        # Start background thread 
        self.running = True
        threading.Thread(target=self.client_heartbeat_check, daemon=True).start()
        
        # Start monitoring thread
        threading.Thread(target=self.monitoring_loop, daemon=True).start()
    
    def connect_to_leader(self, address):
        
        self.leader_address = address
        self.channel = grpc.insecure_channel(address)
        self.stub = chat_pb2_grpc.ChatServiceStub(self.channel)
        print(f"Connected to leader at {address}")
    
    def update_leader(self):
        
        fallback = client_config.get("replica_addresses", [])
        def query_addr(addr):
            try:
                channel = grpc.insecure_channel(addr)
                stub = chat_pb2_grpc.ChatServiceStub(channel)
                resp = stub.GetLeaderInfo(chat_pb2.GetLeaderInfoRequest(), timeout=FALLBACK_TIMEOUT)
                return addr, resp
            except Exception as ex:
                return addr, None

        with ThreadPoolExecutor(max_workers=len(fallback)) as executor:
            futures = {executor.submit(query_addr, addr): addr for addr in fallback}
            try:
                for future in as_completed(futures, timeout=OVERALL_LEADER_LOOKUP_TIMEOUT):
                    addr, resp = future.result()
                    if resp and resp.success and resp.leader_address and resp.leader_address != "Unknown":
                        print(f"Found leader at {resp.leader_address} via fallback address {addr}")
                        self.connect_to_leader(resp.leader_address)
                        new_list = resp.replica_addresses if resp.replica_addresses else []
                        merged = set(fallback) | set(new_list)
                        client_config["replica_addresses"] = list(merged)
                        print(f"[Client Update] New runtime replica list: {client_config['replica_addresses']}")
                        return
            except Exception as e:
                print("Exception during fallback leader lookup:", e)
        print("Leader lookup failed.")
        time.sleep(RETRY_DELAY)
    
    def client_heartbeat_check(self):
        
        while self.running:
            try:
                resp = self.stub.GetLeaderInfo(chat_pb2.GetLeaderInfoRequest(), timeout=RPC_TIMEOUT)
                if not (resp.success and resp.leader_address and resp.leader_address != "Unknown"):
                    print("Heartbeat check failed: invalid response.")
                    self.update_leader()
                else:
                    new_list = resp.replica_addresses if resp.replica_addresses else []
                    merged = set(client_config.get("replica_addresses", [])) | set(new_list)
                    client_config["replica_addresses"] = list(merged)
                    print(f"[Client Heartbeat] Updated replica list: {client_config['replica_addresses']}")
            except Exception as e:
                print("Heartbeat check failed:", e)
                self.update_leader()
            time.sleep(CLIENT_HEARTBEAT_INTERVAL)
    
    def call_rpc_with_retry(self, func, request, retries=3):
       
        for i in range(retries):
            try:
                return func(request, timeout=RPC_TIMEOUT)
            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.UNAVAILABLE:
                    print("RPC UNAVAILABLE. Updating leader and retrying...")
                    self.update_leader()
                    time.sleep(RETRY_DELAY)
                else:
                    raise
        raise Exception("RPC failed after retries.")
    
    def simulate_vitals(self):
        
        # [age, serum_sodium, serum_creatinine, ejection_fraction, day]
        
        # Fixed patient age
        age = random.uniform(50, 85)
        
        # Normal range for serum sodium: 135-145 mEq/L
        serum_sodium = random.uniform(125, 145)
        
        # Normal range for serum creatinine: 0.6-1.2 mg/dL
        serum_creatinine = random.uniform(0.8, 2.5)
        
        # Normal ejection fraction: 55-70%
        ejection_fraction = random.uniform(20, 65)
        
        # Day count 
        day = int(time.time() / 86400) % 365
        
        return [age, serum_sodium, serum_creatinine, ejection_fraction, day]
    
    def run_model_inference(self, vitals):
        try:
            scaled_vitals = self.scaler.transform([vitals])
            # Get prediction from model (probability of heart failure)
            probability = float(self.model.predict(scaled_vitals)[0][0])
            return probability
        except Exception as e:
            print(f"Error during model inference: {e}")
            return 0.5  # Return medium risk if inference fails
    
    def classify_risk(self, probability):
        if probability < GREEN_THRESHOLD:
            return "GREEN"
        elif probability < AMBER_THRESHOLD:
            return "AMBER"
        else:
            return "RED"
    
    def handle_risk_report(self, vitals, probability, tier):
        if tier == "GREEN":
            # It will only be stored locally here
            print(f"GREEN risk report: p={probability:.2f}. Storing locally only.")
            return
        
        # For AMBER and RED it will be sent to server for alerts
        timestamp = int(time.time())
        
        # Prepare a report
        report = {
            "patient_id": self.patient_id,
            "timestamp": timestamp,
            "inputs": vitals,
            "probability": probability,
            "tier": tier
        }
        
        # Addition of alerts
        if tier == "AMBER":
            print(f"\nALERT: AMBER risk detected (p={probability:.2f})")
            print("Notification: Please hydrate, re-measure vitals.\n")
        elif tier == "RED":
            print(f"\n!!! URGENT ALERT: RED risk detected (p={probability:.2f}) !!!")
            print("Immediate attention required!\n")
        
        # Send the report to the server
        self.send_risk_report(report)
    
    def send_risk_report(self, report):
        # Convertig the report to a rewuest
        request = chat_pb2.RiskReportRequest(
            patient_id=report["patient_id"],
            timestamp=report["timestamp"],
            inputs=report["inputs"],
            probability=report["probability"],
            tier=report["tier"]
        )
        
        try:
            
            response = self.call_rpc_with_retry(self.stub.SendRiskReport, request)
            if response.success:
                print(f"Risk report sent successfully. Alert sent: {response.alert_sent}")
            else:
                print(f"Failed to send risk report: {response.message}")
                # Add to queue for retry
                self.report_queue.append(report)
        except Exception as e:
            print(f"Error sending risk report: {e}")
            self.report_queue.append(report)
    
    def retry_queued_reports(self):
        if not self.report_queue:
            return
        
        print(f"Retrying {len(self.report_queue)} queued risk reports...")
        remaining_queue = []
        
        for report in self.report_queue:
            try:
                self.send_risk_report(report)
            except Exception:
                remaining_queue.append(report)
        
        self.report_queue = remaining_queue
    
    def monitoring_loop(self):
        while self.running:
            # Simulate vitals 
            vitals = self.simulate_vitals()
            print(f"\nCaptured vitals: Age={vitals[0]:.1f}, Na={vitals[1]:.1f}, Creat={vitals[2]:.2f}, EF={vitals[3]:.1f}%, Day={vitals[4]}")
            
            # Run inference
            probability = self.run_model_inference(vitals)
            print(f"Predicted probability: {probability:.4f}")
            
            # Classify risk and work on report
            tier = self.classify_risk(probability)
            self.handle_risk_report(vitals, probability, tier)
            
            # Retrying reports
            if self.report_queue:
                self.retry_queued_reports()
            
            # Waiting for next interval
            time.sleep(MONITORING_INTERVAL)
    
    def cleanup(self):
        self.running = False


def main():
    patient_id = input("Enter patient ID: ")
    if not patient_id:
        patient_id = f"P-{random.randint(1000, 9999)}"
        print(f"Using random patient ID: {patient_id}")
    
    client = HeartFailureMonitoringClient(patient_id)
    
    try:
        # Keep the main thread running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down client...")
        client.cleanup()


if __name__ == "__main__":
    main()
