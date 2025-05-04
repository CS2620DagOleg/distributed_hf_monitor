# Heart Failure Monitoring System

A distributed, fault-tolerant system for real-time monitoring of heart failure risk using machine learning. This system leverages a leader-follower architecture to provide robust, scalable health monitoring with automatic failover capabilities.

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Installation](#installation)
   - [Prerequisites](#prerequisites)
   - [Environment Setup](#environment-setup)
   - [Compiling Protocol Buffers](#compiling-protocol-buffers)
4. [Configuration](#configuration)
5. [Training the Model](#training-the-model)
6. [Running the System](#running-the-system)
   - [Starting Servers](#starting-servers)
   - [Starting Clients](#starting-clients)
   - [Testing Failover](#testing-failover)
7. [Data Storage and Management](#data-storage-and-management)
   - [Database Schema](#database-schema)
   - [Viewing Stored Reports](#viewing-stored-reports)
   - [Data Replication](#data-replication) 
8. [Technical Details](#technical-details)
   - [Data Flow](#data-flow)
   - [Risk Classification](#risk-classification)
   - [Server Replication](#server-replication)
   - [Fault Tolerance](#fault-tolerance)
9. [GUI Interface](#gui-interface)
10. [Distributed Deployment](#distributed-deployment)
11. [Testing Framework](#testing-framework)
    - [Unit Tests](#unit-tests)
    - [Integration Tests](#integration-tests)
    - [Performance Tests](#performance-tests)
    - [Running Tests](#running-tests)
12. [Troubleshooting](#troubleshooting)
    - [Common Errors](#common-errors)
    - [Debug Tips](#debug-tips)

## Overview

The Heart Failure Monitoring System is designed to predict heart failure risk in patients by analyzing vital signs in real time. It uses a distributed architecture to ensure high availability and fault tolerance, with the following key features:

- **Real-time risk assessment** using machine learning
- **Distributed, replicated storage** across multiple servers
- **Automatic leader election** and failover
- **Alert system** for high-risk patients
- **GUI client** for monitoring multiple patients
- **Data persistence** even during server failures

This system demonstrates how distributed systems principles can be applied to healthcare monitoring, providing a robust platform that can scale to support many patients while maintaining reliable operation even when components fail.

## System Architecture

The system follows a leader-follower architecture with the following components:

1. **Client Nodes**:
   - Simulate vital sign collection (age, serum sodium, serum creatinine, ejection fraction, day count)
   - Run local inference with the heart failure prediction model
   - Classify risk into GREEN, AMBER, or RED tiers
   - Send risk reports to the server cluster for AMBER and RED risks
   - Provide a GUI for monitoring patient status

2. **Leader Server**:
   - Receives risk reports from clients
   - Stores reports in a SQLite database
   - Replicates data to follower servers
   - Sends alerts for high-risk reports
   - Handles leadership duties for the cluster

3. **Follower Servers**:
   - Maintain synchronized copies of all data
   - Monitor for leader failure
   - Participate in leader election when needed
   - Take over as leader when necessary

4. **Alert System**:
   - Monitors the database for high-risk reports
   - Triggers alerts for RED risk reports
   - Can be extended to send notifications via various channels

## Installation

### Prerequisites

- Python 3.7+
- SQLite (included with Python)
- Conda (recommended for environment management)

### Environment Setup

1. Create a new conda environment:
   ```bash
   conda create --name hf_monitoring python=3.9
   conda activate hf_monitoring
   ```

2. Install the required packages:
   ```bash
   # Core dependencies
   pip install grpcio grpcio-tools

   # For machine learning and data processing
   pip install tensorflow scikit-learn joblib numpy pandas
   ```

### Compiling Protocol Buffers

Before running the system, you need to compile the protocol buffer definitions:

```bash
python -m grpc_tools.protoc --proto_path=. --python_out=. --grpc_python_out=. chat.proto
```

This will generate the following files:
- `chat_pb2.py`: Contains message classes
- `chat_pb2_grpc.py`: Contains service classes

## Configuration

Except for initial state for the replica addresses, booting addresses are input into the console as arguments. We included config file implementation here for the faster testing and demonstration purposes. 

The system uses the following configuration files:

1. **config.json**: Server configuration
   ```json
   {
       "server_id": 1,
       "server_host": "127.0.0.1",
       "server_port": 50051,
       "replica_addresses": ["127.0.0.1:50051", "127.0.0.1:50052", "127.0.0.1:50053"],
       "db_file": "chat.db",
       "heartbeat_interval": 3,
       "lease_timeout": 10,
       "initial_leader": true
   }
   ```

2. **config_client.json**: Client configuration
   ```json
   {
     "client_connect_host": "127.0.0.1",
     "client_connect_port": 50051,
     "replica_addresses": [
       "127.0.0.1:50051",
       "127.0.0.1:50052",
       "127.0.0.1:50053"
     ],
     "rpc_timeout": 10,
     "fallback_timeout": 1,
     "overall_leader_lookup_timeout": 6,
     "retry_delay": 1,
     "client_heartbeat_interval": 5,
     "monitoring_interval": 15,
     "green_threshold": 0.30,
     "amber_threshold": 0.60
   }
   ```

## Training the Model

Before running the system, you need to train the heart failure prediction model:

```bash
python train_hf_model.py
```

This script:
1. Loads the heart failure dataset (`hf.csv`)
2. Extracts features (age, serum sodium, serum creatinine, ejection fraction, day)
3. Trains a neural network model to predict heart failure risk
4. Saves the model as `heart_failure_model.h5`
5. Saves the scaler as `hf_scaler.gz`

The model and scaler will be used by the client to perform local inference on vital signs.

## Running the System

### Starting Servers

You need to start at least one server, but for fault tolerance, it's recommended to start multiple servers. Each server should be started in a separate terminal.

1. Start the leader server (Server 1):
   ```bash
   python hf_replicated_server.py --server_id=1 --server_host=127.0.0.1 --server_port=50051 --initial_leader=true
   ```

2. Start follower servers:
   ```bash
   python hf_replicated_server.py --server_id=2 --server_host=127.0.0.1 --server_port=50052 --initial_leader=false
   python hf_replicated_server.py --server_id=3 --server_host=127.0.0.1 --server_port=50053 --initial_leader=false
   ```

### Starting Clients

Start one or more clients in separate terminals:

```bash
python hf_client_gui.py
```

Each client will:
1. Open a GUI window
2. Suggest a random patient ID
3. Wait for you to click "Start Monitoring"
4. Begin simulating vital signs and performing risk assessment
5. Send AMBER and RED risk reports to the server

### Testing Failover

To test the fault tolerance of the system:

1. Start at least 3 servers and 1 client
2. Wait for some risk reports to be processed
3. Kill the leader server (Server 1) by pressing Ctrl+C in its terminal
4. Observe the logs in the remaining servers as they detect the leader failure and elect a new leader
5. Watch the client reconnect to the new leader and continue sending reports

## Data Storage and Management

### Database Schema

The system stores risk reports in SQLite databases on each server. The database schema includes:

```sql
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
```

### Data Storage Locations

Risk reports are stored in different locations based on their risk level:

1. **GREEN Reports** (p < 0.30): 
   - Stored locally in client memory only
   - Not sent to servers
   - Not persisted after client shutdown

2. **AMBER & RED Reports** (p ≥ 0.30):
   - Sent to the leader server
   - Stored in SQLite databases on all servers
   - Database filenames follow the pattern: `chat_<server_id>.db` (e.g., `chat_1.db`)

### Viewing Stored Reports

You can view the stored reports using:

1. **SQLite Command Line**:
   ```bash
   sqlite3 chat_1.db
   > SELECT * FROM risk_reports;
   ```

2. **Custom Python Script**:
   ```python
   import sqlite3
   
   conn = sqlite3.connect("chat_1.db")
   cursor = conn.cursor()
   cursor.execute("SELECT * FROM risk_reports")
   for row in cursor.fetchall():
       print(row)
   ```

3. **Using list_db.py** (if available from your original repository):
   ```bash
   python list_db.py chat_1.db
   ```

### Data Replication

When the leader server receives a risk report:

1. It writes the report to its local database
2. It forwards the report to all follower servers
3. Followers store the report in their local databases
4. This ensures all servers have identical copies of the data

## Technical Details

### Data Flow

1. **Vital Sign Collection**:
   - In a real system, vital signs would come from sensors or manual input
   - In our simulation, the client generates plausible values for:
     - Age (50-85 years)
     - Serum sodium (125-145 mEq/L)
     - Serum creatinine (0.8-2.5 mg/dL)
     - Ejection fraction (20-65%)
     - Day count since monitoring began

2. **Risk Assessment**:
   - The client loads the pre-trained model and scaler
   - It scales the vital signs using the StandardScaler
   - The model predicts the probability of heart failure risk
   - The probability is classified into GREEN, AMBER, or RED tiers

3. **Report Handling**:
   - GREEN reports (p < 0.30): Stored locally only
   - AMBER reports (0.30 ≤ p < 0.60): Sent to server, notification shown
   - RED reports (p ≥ 0.60): Sent to server, urgent alert triggered

4. **Server Processing**:
   - Reports are received by the leader server
   - The leader stores reports in its SQLite database
   - Reports are replicated to all follower servers
   - RED reports trigger alerts through the alert system

### Risk Classification

The system classifies risk into three tiers:

- **GREEN** (p < 0.30): Low risk
  - No immediate action needed
  - Data stored locally only

- **AMBER** (0.30 ≤ p < 0.60): Moderate risk
  - Patient should hydrate and re-measure vitals
  - Data sent to server for monitoring
  - Notification shown in GUI

- **RED** (p ≥ 0.60): High risk
  - Immediate attention required
  - Data sent to server with high priority
  - Alert triggered on server
  - Urgent notification in GUI

### Server Replication

The system uses a leader-follower replication model:

1. **Write Operations**:
   - All writes (risk reports) go to the leader
   - The leader commits the write to its local database
   - The leader replicates the operation to all followers
   - Followers apply the operation to their local databases

2. **Replication Protocol**:
   - Operations are serialized as JSON and sent via gRPC
   - Each follower acknowledges receipt of the operation
   - The leader ensures at least one follower has received the operation before acknowledging to the client

3. **Dynamic Membership**:
   - New servers can join the cluster at any time
   - The leader transfers its complete state to new servers
   - New servers are added to the runtime replica list
   - All servers are notified of the updated replica list

### Fault Tolerance

The system is designed to handle various failure scenarios:

1. **Leader Failure**:
   - Followers detect missing heartbeats from the leader
   - After a timeout period, followers initiate an election
   - The follower with the lowest server_id becomes the new leader
   - The new leader starts sending heartbeats to all followers
   - Clients discover the new leader through their fallback list

2. **Follower Failure**:
   - The leader detects that a follower is not responding
   - The leader continues operation with the remaining followers
   - When the follower recovers, it rejoins the cluster and receives a state transfer

3. **Network Partitions**:
   - The system uses a simple majority voting scheme
   - In case of a network partition, the partition with the majority of servers continues operation
   - When the partition heals, servers rejoin the cluster

4. **Client Connection Loss**:
   - Clients maintain a list of known server addresses
   - If connection to the leader is lost, clients query all known addresses
   - Clients reconnect to the current leader
   - Unsent reports are queued and retried automatically

## GUI Interface

The client GUI provides a comprehensive interface for monitoring patients:

1. **Patient Information**:
   - Patient ID field (auto-populated with a suggestion)
   - Start Monitoring button
   - Status indicator showing the monitoring state

2. **Vital Signs Display**:
   - Real-time values for all monitored vitals
   - Proper units for each measurement (mEq/L, mg/dL, %, etc.)

3. **Risk Assessment**:
   - Numerical probability display
   - Risk tier (GREEN/AMBER/RED)
   - Color-coded visual indicator

4. **Activity Log**:
   - Real-time scrolling log of all activities
   - Color-coded messages by severity
   - Timestamps for all events

5. **Server Connection**:
   - Current leader address
   - Connection status indicator
   - Automatic leader discovery and reconnection

## Distributed Deployment

To deploy the system across multiple machines:

1. **Network Configuration**:
   - Ensure all machines can communicate with each other
   - Open required ports on firewalls (default: 50051-50053)

2. **Server Configuration**:
   - Use `0.0.0.0` as the listening address to accept connections from any interface
   - Update `replica_addresses` with actual IP addresses

3. **Cross-Machine Command**:
   ```bash
   # On Machine 1 (Leader)
   python hf_replicated_server.py --server_id=1 --server_host=0.0.0.0 --server_port=50051 --initial_leader=true

   # On Machine 2 (Follower)
   python hf_replicated_server.py --server_id=2 --server_host=0.0.0.0 --server_port=50051 --initial_leader=false

   # On Machine 3 (Follower)
   python hf_replicated_server.py --server_id=3 --server_host=0.0.0.0 --server_port=50051 --initial_leader=false
   ```

4. **Client Configuration**:
   - Update `config_client.json` with the actual IP addresses of all servers

## Testing Framework

The system includes a comprehensive testing framework to ensure reliability, robustness, and fault-tolerance. This framework includes unit tests, integration tests, and performance testing capabilities.

### Unit Tests

Unit tests validate individual components of the system in isolation:

- **Server Components**: Tests core server functionality (startup, shutdown, leader election)
- **Client Components**: Tests model inference, risk classification, error handling
- **Data Handling**: Tests serialization, deserialization, and database operations
- **Error Recovery**: Tests system behavior with invalid inputs and failure conditions

To run unit tests:

```bash
python test_heart_failure_system.py
```

### Integration Tests

Integration tests verify the correct operation of the complete system:

- **Normal Operation**: Tests the system under normal conditions
- **Leader Failure**: Tests automatic failover when the leader crashes
- **Leader Recovery**: Tests leader rejoining the cluster after a crash
- **Follower Failure**: Tests continued operation when followers crash
- **Multiple Failures**: Tests robustness with multiple component failures
- **Dynamic Membership**: Tests servers joining and leaving the cluster
- **High Load**: Tests system under high request volume

The integration tests use realistic scenarios and inject faults to validate fault tolerance:

```bash
python integration_test_scenarios.py
```

You can run specific scenarios:

```bash
python integration_test_scenarios.py --scenario "Leader Failure"
```

Or repeat tests multiple times:

```bash
python integration_test_scenarios.py --repeat 5
```

### Performance Tests

The system includes load testing capabilities to validate performance under stress:

- **Concurrent Clients**: Tests with many simultaneously connected clients
- **High Frequency Reports**: Tests rapid report generation and processing
- **Long-Running Stability**: Tests system stability over extended periods

These tests help identify bottlenecks and ensure the system can handle real-world usage patterns.

### Headless Client

For automated testing, a headless (non-GUI) client is provided:

```bash
python headless_client.py --patient_id=TEST-001 --auto_start --duration=60
```

This client can be used to:
- Generate synthetic risk reports automatically
- Test client-server communication
- Validate failover behavior programmatically
- Support integration and performance testing

### Running Tests

To run the complete test suite:

1. Make sure you have compiled the protocol buffer definitions
2. Ensure the model and scaler files exist (`heart_failure_model.h5` and `hf_scaler.gz`)
3. Create an empty `test_outputs` directory for test artifacts:
   ```bash
   mkdir -p test_outputs
   ```
4. Run the tests:
   ```bash
   # Run unit tests
   python test_heart_failure_system.py
   
   # Run integration tests
   python integration_test_scenarios.py
   ```

Test results will be displayed in the console, and detailed logs are saved in the `test_outputs` directory.

## Troubleshooting

### Common Errors

1. **JSON Serialization Error**: 
   - Error: `Object of type RepeatedScalarContainer is not JSON serializable`
   - Fix: Make sure all data types are basic Python types (not Protocol Buffer or NumPy types)

2. **SQLite Recursive Cursor Error**:
   - Error: `Recursive use of cursors not allowed`
   - Fix: Use separate database connections for different threads

3. **Command Line Argument Format**:
   - Correct format: `--server_id=1 --server_host=127.0.0.1 --server_port=50051`
   - Common mistake: Missing equals sign (`--server_id 1`)

4. **Module Not Found Errors**:
   - Error: `ModuleNotFoundError: No module named 'chat_pb2'`
   - Fix: Make sure you've compiled the proto file with `protoc`

### Debug Tips

1. **Check Server Logs**: Look for error messages or warnings in the server console

2. **Verify Database Creation**: Check if SQLite database files are being created properly

3. **Test Client-Server Connection**: Use a simple tool like `telnet` to verify connectivity

4. **Monitor Report Queue**: If reports are getting stuck in the queue, check for serialization or connection issues

5. **Inspect Database Content**: Use SQLite tools to inspect the database contents:
   ```bash
   sqlite3 chat_1.db "SELECT * FROM risk_reports ORDER BY timestamp DESC LIMIT 10;"
   ```

6. **Clear Database for Fresh Start**: If needed, remove the database files and let the system recreate them
   ```bash
   rm chat_*.db
   ```

7. **Test Results Analysis**: If tests are failing, check the test logs in `test_outputs` directory for specific errors
