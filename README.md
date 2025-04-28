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
7. [Technical Details](#technical-details)
   - [Data Flow](#data-flow)
   - [Risk Classification](#risk-classification)
   - [Server Replication](#server-replication)
   - [Fault Tolerance](#fault-tolerance)
8. [GUI Interface](#gui-interface)
9. [Troubleshooting](#troubleshooting)

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
   python hf_replicated_server.py --server_id 1 --server_host 127.0.0.1 --server_port 50051 --initial_leader true
   ```

2. Start follower servers:
   ```bash
   python hf_replicated_server.py --server_id 2 --server_host 127.0.0.1 --server_port 50052 --initial_leader false
   python hf_replicated_server.py --server_id 3 --server_host 127.0.0.1 --server_port 50053 --initial_leader false
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

## Troubleshooting

### Common Issues

1. **Module Not Found Errors**:
   - Make sure you've compiled the protocol buffer definitions
   - Verify all required packages are installed
   - Check that all Python files are in the same directory

2. **Connection Errors**:
   - Ensure the server addresses in configuration files are correct
   - Verify the servers are running and accessible
   - Check for firewall or network issues

3. **Model Loading Errors**:
   - Ensure `heart_failure_model.h5` and `hf_scaler.gz` exist
   - If missing, run `train_hf_model.py` to generate them
   - Verify TensorFlow and joblib are installed correctly

4. **Database Errors**:
   - Check if the SQLite database file is accessible and writable
   - Ensure the server has permission to write to the file
   - Try removing the database file and letting the server recreate it

### Logs and Debugging

- Server logs are printed to the console
- Client logs are displayed in the GUI's activity log
- For more detailed logging, you can modify the logging level in the code

If you encounter issues not covered here, please check the server and client logs for error messages, which often provide more specific information about the problem.
