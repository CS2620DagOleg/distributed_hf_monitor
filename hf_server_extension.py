"""
Heart Failure Monitoring Server Extension

This module extends the replicated_server.py script to add heart failure monitoring functionality.
It adds new RPCs for handling risk reports and sets up the necessary database tables.
"""

import datetime
import json
import logging
import threading
import time
import chat_pb2

# This will be imported into replicated_server.py
# The ReplicatedChatService class will be extended with these methods

def initialize_hf_db(self):
    """Initialize the database tables for heart failure monitoring."""
    self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS risk_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            age REAL,
            serum_sodium REAL,
            serum_creatinine REAL,
            ejection_fraction REAL,
            day INTEGER,
            probability REAL,
            tier TEXT,
            alert_sent INTEGER DEFAULT 0
        )
    ''')
    self.conn.commit()
    logging.info("Heart failure monitoring database initialized")

    # Start the alert monitoring thread if we're the leader
    if self.is_leader:
        threading.Thread(target=self.alert_monitor_loop, daemon=True).start()

def alert_monitor_loop(self):
    """Monitor for new high-risk reports and trigger alerts."""
    last_id = 0
    while True:
        if not self.is_leader:
            # Only the leader should monitor for alerts
            time.sleep(5)
            continue
        
        try:
            # Get all new RED reports
            self.cursor.execute(
                "SELECT id, patient_id, timestamp, probability, tier FROM risk_reports "
                "WHERE id > ? AND tier = 'RED' AND alert_sent = 0",
                (last_id,)
            )
            new_reports = self.cursor.fetchall()
            
            for report in new_reports:
                report_id, patient_id, timestamp, probability, tier = report
                
                # Update the last_id if this is higher
                if report_id > last_id:
                    last_id = report_id
                
                # Log the alert
                timestamp_str = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                alert_msg = f"[ALERT] RED risk detected for patient {patient_id} at {timestamp_str} (p={probability:.2f})"
                logging.warning(alert_msg)
                print("\n" + alert_msg + "\n")
                
                # Here you would add code to send external notifications
                # e.g., SMS, email, webhook to monitoring dashboard, etc.
                
                # Mark as alerted
                self.cursor.execute("UPDATE risk_reports SET alert_sent = 1 WHERE id = ?", (report_id,))
                self.conn.commit()
                
                # Replicate the alert_sent update to followers
                self.replicate_to_followers("update_alert_sent", {"report_id": report_id})
        
        except Exception as e:
            logging.error(f"Error in alert monitoring: {e}")
        
        # Check every second
        time.sleep(1)

def SendRiskReport(self, request, context):
    """Handle a risk report from a client."""
    if not self.is_leader:
        return chat_pb2.RiskReportResponse(
            success=False, 
            message="Not leader. Please contact the leader.",
            alert_sent=False
        )
    
    patient_id = request.patient_id
    timestamp = request.timestamp
    inputs = request.inputs
    probability = request.probability
    tier = request.tier
    
    if not patient_id or len(inputs) != 5:
        return chat_pb2.RiskReportResponse(
            success=False, 
            message="Invalid risk report data",
            alert_sent=False
        )
    
    try:
        # Insert the risk report into the database
        self.cursor.execute(
            "INSERT INTO risk_reports (patient_id, timestamp, age, serum_sodium, serum_creatinine, ejection_fraction, day, probability, tier, alert_sent) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                patient_id, 
                timestamp, 
                inputs[0],  # age
                inputs[1],  # serum_sodium
                inputs[2],  # serum_creatinine
                inputs[3],  # ejection_fraction
                inputs[4],  # day
                probability,
                tier,
                0  # alert_sent (will be updated by the alert_monitor_loop)
            )
        )
        report_id = self.cursor.lastrowid
        self.conn.commit()
        
        # Replicate to followers
        self.replicate_to_followers("risk_report", {
            "patient_id": patient_id,
            "timestamp": timestamp,
            "inputs": inputs,
            "probability": probability,
            "tier": tier
        })
        
        # Determine if an alert was sent (always false initially, the alert_monitor_loop will handle it)
        alert_sent = False
        
        # Log the received report
        logging.info(f"Risk report received: Patient={patient_id}, Tier={tier}, p={probability:.2f}")
        
        return chat_pb2.RiskReportResponse(
            success=True,
            message="Risk report received and stored successfully",
            alert_sent=alert_sent
        )
    
    except Exception as e:
        logging.error(f"Error storing risk report: {e}")
        return chat_pb2.RiskReportResponse(
            success=False,
            message=f"Error storing risk report: {str(e)}",
            alert_sent=False
        )

def ListRiskReports(self, request, context):
    """List risk reports for a given patient."""
    patient_id = request.patient_id
    count = request.count
    
    if not patient_id:
        return chat_pb2.ListRiskReportsResponse(success=False, reports=[])
    
    try:
        if count <= 0:
            # Get all reports for the patient
            self.cursor.execute(
                "SELECT timestamp, tier, probability, age, serum_sodium, serum_creatinine, ejection_fraction "
                "FROM risk_reports WHERE patient_id = ? ORDER BY timestamp DESC",
                (patient_id,)
            )
        else:
            # Get the most recent 'count' reports
            self.cursor.execute(
                "SELECT timestamp, tier, probability, age, serum_sodium, serum_creatinine, ejection_fraction "
                "FROM risk_reports WHERE patient_id = ? ORDER BY timestamp DESC LIMIT ?",
                (patient_id, count)
            )
        
        rows = self.cursor.fetchall()
        
        # Format the reports as strings
        reports = []
        for row in rows:
            timestamp, tier, probability, age, sodium, creatinine, ef = row
            timestamp_str = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            report_str = (
                f"{timestamp_str} - {tier} (p={probability:.2f}) - "
                f"Age: {age:.1f}, Na: {sodium:.1f}, Creat: {creatinine:.2f}, EF: {ef:.1f}%"
            )
            reports.append(report_str)
        
        logging.info(f"Listed {len(reports)} risk reports for patient '{patient_id}'")
        return chat_pb2.ListRiskReportsResponse(success=True, reports=reports)
    
    except Exception as e:
        logging.error(f"Error listing risk reports: {e}")
        return chat_pb2.ListRiskReportsResponse(success=False, reports=[])

# Add a new case to the ReplicateOperation method
def handle_risk_report_replication(self, data):
    """Handle replication of a risk report operation."""
    try:
        self.cursor.execute(
            "INSERT INTO risk_reports (patient_id, timestamp, age, serum_sodium, serum_creatinine, ejection_fraction, day, probability, tier, alert_sent) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                data["patient_id"],
                data["timestamp"],
                data["inputs"][0],  # age
                data["inputs"][1],  # serum_sodium
                data["inputs"][2],  # serum_creatinine
                data["inputs"][3],  # ejection_fraction
                data["inputs"][4],  # day
                data["probability"],
                data["tier"],
                0  # alert_sent
            )
        )
        self.conn.commit()
        return True
    except Exception as e:
        logging.error(f"Error replicating risk report: {e}")
        return False

def handle_update_alert_sent_replication(self, data):
    """Handle replication of an alert_sent update."""
    try:
        report_id = data["report_id"]
        self.cursor.execute("UPDATE risk_reports SET alert_sent = 1 WHERE id = ?", (report_id,))
        self.conn.commit()
        return True
    except Exception as e:
        logging.error(f"Error replicating alert_sent update: {e}")
        return False

# This function should be called from JoinCluster to include risk_reports in state transfer
def get_risk_reports_for_state_transfer(self):
    """Get all risk reports for state transfer to a new server."""
    self.cursor.execute(
        "SELECT patient_id, timestamp, age, serum_sodium, serum_creatinine, ejection_fraction, day, probability, tier, alert_sent "
        "FROM risk_reports"
    )
    risk_reports = [{
        "patient_id": row[0],
        "timestamp": row[1],
        "age": row[2],
        "serum_sodium": row[3],
        "serum_creatinine": row[4],
        "ejection_fraction": row[5],
        "day": row[6],
        "probability": row[7],
        "tier": row[8],
        "alert_sent": row[9]
    } for row in self.cursor.fetchall()]
    return risk_reports

# This function should be called when applying state during JoinCluster
def apply_risk_reports_from_state(self, risk_reports):
    """Apply risk reports from state transfer."""
    for report in risk_reports:
        self.cursor.execute(
            "INSERT INTO risk_reports (patient_id, timestamp, age, serum_sodium, serum_creatinine, ejection_fraction, day, probability, tier, alert_sent) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                report["patient_id"],
                report["timestamp"],
                report["age"],
                report["serum_sodium"],
                report["serum_creatinine"],
                report["ejection_fraction"],
                report["day"],
                report["probability"],
                report["tier"],
                report["alert_sent"]
            )
        )
    self.conn.commit()