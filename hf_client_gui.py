import os
import json
import time
import random
import threading
import datetime
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, font
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import grpc
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

# Read parameters
RPC_TIMEOUT = client_config.get("rpc_timeout", 10)
FALLBACK_TIMEOUT = client_config.get("fallback_timeout", 1)
OVERALL_LEADER_LOOKUP_TIMEOUT = client_config.get("overall_leader_lookup_timeout", 5)
RETRY_DELAY = client_config.get("retry_delay", 1)
CLIENT_HEARTBEAT_INTERVAL = client_config.get("client_heartbeat_interval", 5)
MONITORING_INTERVAL = client_config.get("monitoring_interval", 15)  # seconds between readings (shortened for demo)

# Risk thresholds
GREEN_THRESHOLD = client_config.get("green_threshold", 0.30)
AMBER_THRESHOLD = client_config.get("amber_threshold", 0.60)

class HeartFailureMonitoringGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        
        # Set up the main window
        self.title("Heart Failure Monitoring System")
        self.geometry("800x600")
        self.configure(bg="#f0f0f0")
        
        # Initialize attributes
        self.suggested_id = f"P-{random.randint(1000, 9999)}"
        self.patient_id = None
        self.monitoring_active = False
        self.report_queue = []
        self.leader_address = f"{client_config['client_connect_host']}:{client_config['client_connect_port']}"
        self.running = True
        
        # Try to load model and scaler
        try:
            self.model = tf.keras.models.load_model("heart_failure_model.h5")
            self.scaler = joblib.load("hf_scaler.gz")
            print("Model and scaler loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading model or scaler: {e}")
            self.destroy()
            return
        
        # Connect to the leader
        self.connect_to_leader(self.leader_address)
        
        # Set up the UI
        self.setup_ui()
        
        # Start background thread for leader heartbeat check
        threading.Thread(target=self.client_heartbeat_check, daemon=True).start()
    
    def setup_ui(self):
        # Create frames
        self.top_frame = tk.Frame(self, bg="#f0f0f0")
        self.top_frame.pack(fill="x", padx=10, pady=10)
        
        self.patient_frame = tk.Frame(self.top_frame, bg="#f0f0f0")
        self.patient_frame.pack(side="left", padx=10)
        
        self.status_frame = tk.Frame(self.top_frame, bg="#f0f0f0")
        self.status_frame.pack(side="right", padx=10)
        
        self.vitals_frame = tk.LabelFrame(self, text="Vital Signs", bg="#f0f0f0", font=("Arial", 12, "bold"))
        self.vitals_frame.pack(fill="x", padx=10, pady=5)
        
        self.risk_frame = tk.LabelFrame(self, text="Risk Assessment", bg="#f0f0f0", font=("Arial", 12, "bold"))
        self.risk_frame.pack(fill="x", padx=10, pady=5)
        
        self.log_frame = tk.LabelFrame(self, text="Activity Log", bg="#f0f0f0", font=("Arial", 12, "bold"))
        self.log_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Patient ID entry
        tk.Label(self.patient_frame, text="Patient ID:", bg="#f0f0f0", font=("Arial", 12)).pack(side="left")
        self.patient_id_var = tk.StringVar(value=self.suggested_id)
        self.patient_id_entry = tk.Entry(self.patient_frame, textvariable=self.patient_id_var, width=15, font=("Arial", 12))
        self.patient_id_entry.pack(side="left", padx=5)
        
        self.start_button = tk.Button(self.patient_frame, text="Start Monitoring", command=self.start_monitoring, 
                                     bg="#4CAF50", fg="white", font=("Arial", 12))
        self.start_button.pack(side="left", padx=5)
        
        self.status_indicator = tk.Label(self.patient_frame, text="Ready", 
                                        bg="#f0f0f0", fg="blue", font=("Arial", 12, "italic"))
        self.status_indicator.pack(side="left", padx=5)
        
        # Connection status
        tk.Label(self.status_frame, text="Server:", bg="#f0f0f0", font=("Arial", 12)).pack(side="left")
        self.server_status = tk.Label(self.status_frame, text=self.leader_address, bg="#f0f0f0", 
                                     fg="green", font=("Arial", 12, "bold"))
        self.server_status.pack(side="left", padx=5)
        
        # Vital signs indicators
        vital_signs_grid = tk.Frame(self.vitals_frame, bg="#f0f0f0")
        vital_signs_grid.pack(fill="x", padx=10, pady=5)
        
        # Create labels for each vital sign
        vital_labels = ["Age", "Serum Sodium", "Serum Creatinine", "Ejection Fraction", "Days Monitored"]
        self.vital_values = {}
        
        for i, label in enumerate(vital_labels):
            tk.Label(vital_signs_grid, text=f"{label}:", bg="#f0f0f0", font=("Arial", 11)).grid(row=i//3, column=(i%3)*2, sticky="e", padx=5, pady=5)
            value_label = tk.Label(vital_signs_grid, text="--", bg="#f0f0f0", width=8, font=("Arial", 11, "bold"))
            value_label.grid(row=i//3, column=(i%3)*2+1, sticky="w", padx=5, pady=5)
            self.vital_values[label] = value_label
        
        # Risk assessment display
        risk_display = tk.Frame(self.risk_frame, bg="#f0f0f0")
        risk_display.pack(fill="x", padx=10, pady=5)
        
        # Probability indicator
        tk.Label(risk_display, text="Risk Probability:", bg="#f0f0f0", font=("Arial", 12)).grid(row=0, column=0, padx=5, pady=5)
        self.probability_label = tk.Label(risk_display, text="--", bg="#f0f0f0", font=("Arial", 12, "bold"))
        self.probability_label.grid(row=0, column=1, padx=5, pady=5)
        
        # Risk tier indicator
        tk.Label(risk_display, text="Risk Tier:", bg="#f0f0f0", font=("Arial", 12)).grid(row=0, column=2, padx=5, pady=5)
        self.risk_tier_label = tk.Label(risk_display, text="--", bg="#f0f0f0", font=("Arial", 12, "bold"))
        self.risk_tier_label.grid(row=0, column=3, padx=5, pady=5)
        
        # Risk indicator (colored circle)
        self.risk_indicator = tk.Canvas(risk_display, width=40, height=40, bg="#f0f0f0", highlightthickness=0)
        self.risk_indicator.grid(row=0, column=4, padx=10, pady=5)
        self.risk_indicator.create_oval(5, 5, 35, 35, fill="gray", outline="")
        
        # Activity log
        self.log_text = scrolledtext.ScrolledText(self.log_frame, height=10, width=70, font=("Consolas", 10))
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Set column weights
        vital_signs_grid.columnconfigure(1, weight=1)
        vital_signs_grid.columnconfigure(3, weight=1)
        vital_signs_grid.columnconfigure(5, weight=1)
        
        # Initial log message
        self.log(f"System initialized. Suggested patient ID: {self.suggested_id}")
        self.log("Enter patient ID and click 'Start Monitoring' to begin")
    
    def log(self, message, level="INFO"):
        """Add a message to the log with timestamp"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        # Color coding based on level
        tag = None
        if level == "WARNING":
            tag = "warning"
            if not hasattr(self, 'warning_tag_configured'):
                self.log_text.tag_configure("warning", foreground="orange")
                self.warning_tag_configured = True
        elif level == "ERROR":
            tag = "error"
            if not hasattr(self, 'error_tag_configured'):
                self.log_text.tag_configure("error", foreground="red")
                self.error_tag_configured = True
        elif level == "SUCCESS":
            tag = "success"
            if not hasattr(self, 'success_tag_configured'):
                self.log_text.tag_configure("success", foreground="green")
                self.success_tag_configured = True
        elif level == "ALERT":
            tag = "alert"
            if not hasattr(self, 'alert_tag_configured'):
                self.log_text.tag_configure("alert", foreground="red", font=("Consolas", 10, "bold"))
                self.alert_tag_configured = True
        
        # Insert the log message
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n", tag)
        self.log_text.see(tk.END)  # Scroll to the end
        
    def start_monitoring(self):
        """Start the monitoring process for the entered patient ID"""
        if self.monitoring_active:
            return
        
        # Get patient ID
        patient_id = self.patient_id_var.get().strip()
        if not patient_id:
            patient_id = self.suggested_id
            self.patient_id_var.set(patient_id)
            self.log(f"Using suggested patient ID: {patient_id}")
        
        self.patient_id = patient_id
        self.monitoring_active = True
        
        # Disable start button and ID entry
        self.start_button.config(text="Monitoring...", state="disabled", bg="gray")
        self.patient_id_entry.config(state="disabled")
        
        # Update status indicator
        self.status_indicator.config(text="Monitoring Active", fg="green")
        
        # Start monitoring thread
        self.log(f"Starting monitoring for patient {patient_id}", "SUCCESS")
        threading.Thread(target=self.monitoring_loop, daemon=True).start()
    
    def connect_to_leader(self, address):
        """Connect to the leader server at the given address."""
        self.leader_address = address
        self.channel = grpc.insecure_channel(address)
        self.stub = chat_pb2_grpc.ChatServiceStub(channel=self.channel)
        if hasattr(self, 'server_status'):
            self.server_status.config(text=address)
        print(f"Connected to leader at {address}")
    
    def update_leader(self):
        """Update the leader address by querying fallback addresses."""
        fallback = client_config.get("replica_addresses", [])
        
        # Update UI to show reconnecting status
        if hasattr(self, 'server_status'):
            self.server_status.config(text="Reconnecting...", fg="orange")
        
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
                        self.log(f"Found leader at {resp.leader_address} via fallback address {addr}")
                        self.connect_to_leader(resp.leader_address)
                        new_list = resp.replica_addresses if resp.replica_addresses else []
                        merged = set(fallback) | set(new_list)
                        client_config["replica_addresses"] = list(merged)
                        
                        # Update UI to show connected status
                        if hasattr(self, 'server_status'):
                            self.server_status.config(text=resp.leader_address, fg="green")
                        
                        return
            except Exception as e:
                self.log(f"Exception during fallback leader lookup: {e}", "ERROR")
        
        self.log("Leader lookup failed on all fallback addresses; keeping current connection.", "WARNING")
        
        # Update UI to show warning status
        if hasattr(self, 'server_status'):
            self.server_status.config(text=self.leader_address + " (No response)", fg="red")
            
        time.sleep(RETRY_DELAY)
    
    def client_heartbeat_check(self):
        """Periodically check if the leader is still available."""
        while self.running:
            try:
                resp = self.stub.GetLeaderInfo(chat_pb2.GetLeaderInfoRequest(), timeout=RPC_TIMEOUT)
                if not (resp.success and resp.leader_address and resp.leader_address != "Unknown"):
                    self.log("Heartbeat check failed: invalid response.", "WARNING")
                    self.update_leader()
                else:
                    # Update server status if leader changed
                    if resp.leader_address != self.leader_address:
                        self.log(f"Leader changed to {resp.leader_address}", "WARNING")
                        self.connect_to_leader(resp.leader_address)
                    
                    # Update replica list
                    new_list = resp.replica_addresses if resp.replica_addresses else []
                    merged = set(client_config.get("replica_addresses", [])) | set(new_list)
                    client_config["replica_addresses"] = list(merged)
            except Exception as e:
                self.log(f"Heartbeat check failed: {e}", "WARNING")
                self.update_leader()
            time.sleep(CLIENT_HEARTBEAT_INTERVAL)
    
    def call_rpc_with_retry(self, func, request, retries=3):
        """Call an RPC with retry logic if the leader becomes unavailable."""
        for i in range(retries):
            try:
                return func(request, timeout=RPC_TIMEOUT)
            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.UNAVAILABLE:
                    self.log("RPC UNAVAILABLE. Updating leader and retrying...", "WARNING")
                    self.update_leader()
                    time.sleep(RETRY_DELAY)
                else:
                    self.log(f"RPC error: {e}", "ERROR")
                    raise
        raise Exception("RPC failed after retries.")
    
    def simulate_vitals(self):
        """Simulate vital signs for the patient. In a real system, this would read from sensors."""
        # Generate plausible values for a heart failure patient
        # [age, serum_sodium, serum_creatinine, ejection_fraction, day]
        
        # Generate values that might sometimes trigger AMBER or RED alerts
        # For demo purposes, we'll occasionally generate high-risk values
        if random.random() < 0.2:  # 20% chance of high-risk vitals
            # High risk (RED) values
            age = random.uniform(70, 85)
            serum_sodium = random.uniform(125, 130)  # Low sodium (hyponatremia)
            serum_creatinine = random.uniform(1.8, 2.5)  # High creatinine (kidney issues)
            ejection_fraction = random.uniform(20, 30)  # Very low EF (severe heart failure)
        elif random.random() < 0.3:  # 30% chance of medium-risk vitals
            # Medium risk (AMBER) values
            age = random.uniform(65, 75)
            serum_sodium = random.uniform(130, 135)  # Slightly low sodium
            serum_creatinine = random.uniform(1.3, 1.8)  # Moderately high creatinine
            ejection_fraction = random.uniform(30, 40)  # Low EF (moderate heart failure)
        else:
            # Low risk (GREEN) values
            age = random.uniform(50, 70)
            serum_sodium = random.uniform(135, 145)  # Normal sodium
            serum_creatinine = random.uniform(0.8, 1.3)  # Normal to slightly high creatinine
            ejection_fraction = random.uniform(40, 65)  # Normal to slightly low EF
        
        # Day count (time since monitoring began)
        day = int(time.time() / 86400) % 365
        
        return [age, serum_sodium, serum_creatinine, ejection_fraction, day]
    
    def run_model_inference(self, vitals):
        """Run the heart failure prediction model on the given vitals."""
        # Scale the input values using the pre-trained scaler
        try:
            scaled_vitals = self.scaler.transform([vitals])
            # Get prediction from model (probability of heart failure)
            probability = float(self.model.predict(scaled_vitals)[0][0])
            return probability
        except Exception as e:
            self.log(f"Error during model inference: {e}", "ERROR")
            return 0.5  # Return medium risk if inference fails
    
    def classify_risk(self, probability):
        """Classify the risk based on the predicted probability."""
        if probability < GREEN_THRESHOLD:
            return "GREEN"
        elif probability < AMBER_THRESHOLD:
            return "AMBER"
        else:
            return "RED"
    
    def update_vitals_display(self, vitals):
        """Update the vital signs display with new values."""
        labels = ["Age", "Serum Sodium", "Serum Creatinine", "Ejection Fraction", "Days Monitored"]
        formats = [".1f", ".1f", ".2f", ".1f", "d"]
        
        for i, (label, value, fmt) in enumerate(zip(labels, vitals, formats)):
            if fmt == "d":
                formatted_value = f"{int(value)}"
            else:
                formatted_value = f"{value:{fmt}}"
                
            if label == "Ejection Fraction":
                formatted_value += "%"
            elif label == "Serum Sodium":
                formatted_value += " mEq/L"
            elif label == "Serum Creatinine":
                formatted_value += " mg/dL"
                
            self.vital_values[label].config(text=formatted_value)
    
    def update_risk_display(self, probability, tier):
        """Update the risk assessment display."""
        # Update probability
        self.probability_label.config(text=f"{probability:.4f}")
        
        # Update tier and indicator
        self.risk_tier_label.config(text=tier)
        
        # Update colored indicator
        if tier == "GREEN":
            color = "#4CAF50"  # Green
            self.risk_tier_label.config(fg="#4CAF50")
        elif tier == "AMBER":
            color = "#FFC107"  # Amber
            self.risk_tier_label.config(fg="#FFC107")
        else:  # RED
            color = "#F44336"  # Red
            self.risk_tier_label.config(fg="#F44336")
        
        # Update the oval color
        self.risk_indicator.delete("all")
        self.risk_indicator.create_oval(5, 5, 35, 35, fill=color, outline="")
    
    def handle_risk_report(self, vitals, probability, tier):
        """Handle a risk report based on its tier."""
        if tier == "GREEN":
            # Store locally only, no network traffic
            self.log(f"GREEN risk report: p={probability:.2f}. Storing locally only.")
            return
        
        # For AMBER and RED, send to server
        timestamp = int(time.time())
        
        # Prepare the risk report
        report = {
            "patient_id": self.patient_id,
            "timestamp": timestamp,
            "inputs": vitals,
            "probability": probability,
            "tier": tier
        }
        
        # Add visual/audio alerts for high risk
        if tier == "AMBER":
            self.log(f"AMBER risk detected (p={probability:.2f})", "WARNING")
            self.log("Notification: Please hydrate, re-measure vitals.", "WARNING")
            # Flash the window to get attention
            self.bell()
            self.focus_force()
        elif tier == "RED":
            self.log(f"URGENT ALERT: RED risk detected (p={probability:.2f})", "ALERT")
            self.log("Immediate attention required!", "ALERT")
            # Flash the window and make a sound for urgent attention
            self.bell()
            self.focus_force()
            self.attributes("-topmost", True)
            self.after(3000, lambda: self.attributes("-topmost", False))
        
        # Send the report to the server
        self.send_risk_report(report)
    
    def send_risk_report(self, report):
        """Send a risk report to the leader server."""
        # Convert the report to a RiskReportRequest
        request = chat_pb2.RiskReportRequest(
            patient_id=report["patient_id"],
            timestamp=report["timestamp"],
            inputs=report["inputs"],
            probability=report["probability"],
            tier=report["tier"]
        )
        
        try:
            # Call the SendRiskReport RPC with retry
            response = self.call_rpc_with_retry(self.stub.SendRiskReport, request)
            if response.success:
                self.log(f"Risk report sent successfully. Alert sent: {response.alert_sent}", "SUCCESS")
            else:
                self.log(f"Failed to send risk report: {response.message}", "ERROR")
                # Add to queue for retry
                self.report_queue.append(report)
        except Exception as e:
            self.log(f"Error sending risk report: {e}", "ERROR")
            # Add to queue for retry
            self.report_queue.append(report)
    
    def retry_queued_reports(self):
        """Retry sending any queued risk reports."""
        if not self.report_queue:
            return
        
        self.log(f"Retrying {len(self.report_queue)} queued risk reports...")
        remaining_queue = []
        
        for report in self.report_queue:
            try:
                self.send_risk_report(report)
            except Exception:
                remaining_queue.append(report)
        
        self.report_queue = remaining_queue
    
    def monitoring_loop(self):
        """Main monitoring loop that periodically checks vitals and sends reports."""
        while self.running and self.monitoring_active:
            # Simulate vitals data collection
            vitals = self.simulate_vitals()
            self.log(f"Captured vitals: Age={vitals[0]:.1f}, Na={vitals[1]:.1f}, Creat={vitals[2]:.2f}, EF={vitals[3]:.1f}%, Day={vitals[4]}")
            
            # Update vitals display
            self.update_vitals_display(vitals)
            
            # Run model inference
            probability = self.run_model_inference(vitals)
            self.log(f"Predicted probability: {probability:.4f}")
            
            # Classify risk
            tier = self.classify_risk(probability)
            
            # Update risk display
            self.update_risk_display(probability, tier)
            
            # Handle the risk report
            self.handle_risk_report(vitals, probability, tier)
            
            # Retry any queued reports
            if self.report_queue:
                self.retry_queued_reports()
            
            # Wait for the next monitoring interval
            for _ in range(MONITORING_INTERVAL):
                if not (self.running and self.monitoring_active):
                    break
                time.sleep(1)
    
    def cleanup(self):
        """Clean up resources when shutting down."""
        self.running = False
        self.monitoring_active = False
        self.destroy()


def main():
    app = HeartFailureMonitoringGUI()
    app.protocol("WM_DELETE_WINDOW", app.cleanup)
    app.mainloop()


if __name__ == "__main__":
    main()