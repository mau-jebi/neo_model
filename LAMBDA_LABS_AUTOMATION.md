# Lambda Labs H100 Training Automation Guide

## Overview

This guide documents the complete automated training pipeline using Lambda Labs Cloud API to provision H100 GPU instances for neo_model RT-DETR training.

**Date**: January 31, 2026
**Instance Type**: 1× H100 (80GB PCIe)
**Estimated Cost**: $75-90 for full 72-epoch training
**Training Duration**: ~30-36 hours (1.2-1.5 days)

---

## Prerequisites

### 1. Lambda Labs API Key

Store your API key securely:
```bash
# Create key file (add to .gitignore!)
echo "your_api_key_here" > mau-neo.txt
chmod 600 mau-neo.txt

# Test API key
API_KEY=$(cat mau-neo.txt)
curl -s -u "$API_KEY:" https://cloud.lambda.ai/api/v1/instance-types | head -50
```

**Expected**: JSON response with available instance types

### 2. Local neo_model Codebase

Ensure you have:
- Complete neo_model repository
- All Phase 1 & Phase 2 files
- Bug fixes applied (see BUGFIXES.md)
- Configuration: `configs/rtdetr_r50_1920x1080.yml`

---

## Lambda Labs Cloud API Reference

### Base Configuration

```bash
API_KEY="your_secret_key_here"
API_BASE="https://cloud.lambda.ai/api/v1"
```

### Core Endpoints

**Authentication**: HTTP Basic Auth (API key as username, password empty)

```bash
curl -u "$API_KEY:" $API_BASE/<endpoint>
```

#### 1. List Instance Types

Find available GPUs and regions:

```bash
curl -s -u "$API_KEY:" https://cloud.lambda.ai/api/v1/instance-types
```

**Response**:
```json
{
  "data": {
    "gpu_1x_h100_pcie": {
      "instance_type": {
        "name": "gpu_1x_h100_pcie",
        "description": "1x H100 (80 GB PCIe)",
        "price_cents_per_hour": 249,
        "specs": {
          "vcpus": 26,
          "memory_gib": 200,
          "storage_gib": 1024,
          "gpus": 1
        }
      },
      "regions_with_capacity_available": [
        {"name": "us-west-3", "description": "Utah, USA"}
      ]
    }
  }
}
```

**Key Info**:
- Instance type name: `gpu_1x_h100_pcie`
- Price: $2.49/hour (249 cents)
- Available regions: `us-west-3`

#### 2. Manage SSH Keys

**List SSH keys**:
```bash
curl -s -u "$API_KEY:" https://cloud.lambda.ai/api/v1/ssh-keys
```

**Create SSH key**:
```bash
# Generate local key
ssh-keygen -t ed25519 -f ~/.ssh/lambda_neo_model -N "" -C "neo_model_training"

# Upload to Lambda Labs
PUBLIC_KEY=$(cat ~/.ssh/lambda_neo_model.pub)
curl -s -u "$API_KEY:" https://cloud.lambda.ai/api/v1/ssh-keys \
  -X POST \
  -H "Content-Type: application/json" \
  -d "{
    \"name\": \"neo_model_training_key\",
    \"public_key\": \"$PUBLIC_KEY\"
  }"
```

**Response**:
```json
{
  "data": {
    "id": "key_id_here",
    "name": "neo_model_training_key",
    "public_key": "ssh-ed25519 AAAA... neo_model_training"
  }
}
```

#### 3. Launch Instance

```bash
curl -s -u "$API_KEY:" https://cloud.lambda.ai/api/v1/instance-operations/launch \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "instance_type_name": "gpu_1x_h100_pcie",
    "region_name": "us-west-3",
    "ssh_key_names": ["neo_model_training_key"],
    "name": "neo-model-h100-training"
  }'
```

**Response**:
```json
{
  "data": {
    "instance_ids": ["9d180768119948deb53fe8bfa7bf75eb"]
  }
}
```

**Instance ID**: Save this for status checks and termination!

#### 4. Check Instance Status

```bash
INSTANCE_ID="9d180768119948deb53fe8bfa7bf75eb"
curl -s -u "$API_KEY:" https://cloud.lambda.ai/api/v1/instances
```

**Response** (excerpt):
```json
{
  "data": [
    {
      "id": "9d180768119948deb53fe8bfa7bf75eb",
      "name": "neo-model-h100-training",
      "status": "active",  // "booting" -> "active"
      "ip": "209.20.157.204",
      "region": {"name": "us-west-3"},
      "instance_type": {"name": "gpu_1x_h100_pcie"}
    }
  ]
}
```

**Status Values**:
- `booting` - Instance starting up (wait)
- `active` - Ready for SSH connection
- `terminated` - Stopped

**Wait Loop** (poll until active):
```bash
while true; do
  STATUS=$(curl -s -u "$API_KEY:" https://cloud.lambda.ai/api/v1/instances | \
    python3 -c "import sys,json; data=json.load(sys.stdin); print([i['status'] for i in data['data'] if i['id']=='$INSTANCE_ID'][0])")

  if [ "$STATUS" == "active" ]; then
    echo "Instance ready!"
    break
  fi
  echo "Status: $STATUS, waiting..."
  sleep 10
done
```

**Typical boot time**: 2-5 minutes

#### 5. Terminate Instance

**IMPORTANT**: Always terminate when done to avoid ongoing charges!

```bash
curl -s -u "$API_KEY:" https://cloud.lambda.ai/api/v1/instance-operations/terminate \
  -X POST \
  -H "Content-Type: application/json" \
  -d "{
    \"instance_ids\": [\"$INSTANCE_ID\"]
  }"
```

**Response**:
```json
{
  "data": {
    "terminated_instances": ["9d180768119948deb53fe8bfa7bf75eb"]
  }
}
```

**Verify Termination**:
```bash
curl -s -u "$API_KEY:" https://cloud.lambda.ai/api/v1/instances | \
  grep "$INSTANCE_ID" || echo "Instance terminated successfully"
```

---

## Complete Automation Script

### Full Training Pipeline Script

Save as `scripts/automate_training.sh`:

```bash
#!/bin/bash
set -e

# Configuration
API_KEY=$(cat mau-neo.txt)
API_BASE="https://cloud.lambda.ai/api/v1"
INSTANCE_TYPE="gpu_1x_h100_pcie"
REGION="us-west-3"
SSH_KEY_NAME="neo_model_training_key"
SSH_KEY_PATH="$HOME/.ssh/lambda_neo_model"

echo "========================================"
echo "Lambda Labs H100 Training Automation"
echo "========================================"
echo ""

# Stage 0: Verify API Key
echo "Stage 0: Verifying Lambda Labs API key..."
TYPES=$(curl -s -u "$API_KEY:" $API_BASE/instance-types)
if echo "$TYPES" | grep -q "error"; then
    echo "ERROR: API key invalid!"
    exit 1
fi
echo "✓ API key valid"
echo ""

# Stage 1: Create SSH Key
echo "Stage 1: Setting up SSH key..."
if [ ! -f "$SSH_KEY_PATH" ]; then
    ssh-keygen -t ed25519 -f "$SSH_KEY_PATH" -N "" -C "neo_model_training_$(date +%Y%m%d)"
    echo "✓ SSH key generated"
fi

PUBLIC_KEY=$(cat ${SSH_KEY_PATH}.pub)

# Check if key exists
KEY_EXISTS=$(curl -s -u "$API_KEY:" $API_BASE/ssh-keys | grep -q "$SSH_KEY_NAME" && echo "yes" || echo "no")

if [ "$KEY_EXISTS" == "no" ]; then
    curl -s -u "$API_KEY:" $API_BASE/ssh-keys \
      -X POST \
      -H "Content-Type: application/json" \
      -d "{\"name\": \"$SSH_KEY_NAME\", \"public_key\": \"$PUBLIC_KEY\"}" > /dev/null
    echo "✓ SSH key uploaded to Lambda Labs"
else
    echo "✓ SSH key already exists"
fi
echo ""

# Stage 2: Launch Instance
echo "Stage 2: Launching H100 instance..."
echo "  Type: $INSTANCE_TYPE"
echo "  Region: $REGION"
echo "  Price: \$2.49/hour"

LAUNCH_RESPONSE=$(curl -s -u "$API_KEY:" $API_BASE/instance-operations/launch \
  -X POST \
  -H "Content-Type: application/json" \
  -d "{
    \"instance_type_name\": \"$INSTANCE_TYPE\",
    \"region_name\": \"$REGION\",
    \"ssh_key_names\": [\"$SSH_KEY_NAME\"],
    \"name\": \"neo-model-h100-training-$(date +%Y%m%d-%H%M%S)\"
  }")

INSTANCE_ID=$(echo "$LAUNCH_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['instance_ids'][0])")
LAUNCH_TIME=$(date +%s)

echo "✓ Instance launched: $INSTANCE_ID"
echo "  Launch time: $(date)"
echo "$LAUNCH_TIME" > /tmp/neo_model_launch_time.txt
echo "$INSTANCE_ID" > /tmp/neo_model_instance_id.txt
echo ""

# Stage 3: Wait for Instance to become Active
echo "Stage 3: Waiting for instance to boot..."
WAIT_COUNT=0
while [ $WAIT_COUNT -lt 60 ]; do
    STATUS=$(curl -s -u "$API_KEY:" $API_BASE/instances | \
      python3 -c "import sys,json; data=json.load(sys.stdin); print([i['status'] for i in data['data'] if i['id']=='$INSTANCE_ID'][0])")

    if [ "$STATUS" == "active" ]; then
        IP=$(curl -s -u "$API_KEY:" $API_BASE/instances | \
          python3 -c "import sys,json; data=json.load(sys.stdin); print([i['ip'] for i in data['data'] if i['id']=='$INSTANCE_ID'][0])")
        echo "✓ Instance active!"
        echo "  IP: $IP"
        echo "  SSH: ssh -i $SSH_KEY_PATH ubuntu@$IP"
        echo "$IP" > /tmp/neo_model_instance_ip.txt
        break
    fi

    echo "  Status: $STATUS ($(($WAIT_COUNT * 10))s elapsed)"
    sleep 10
    WAIT_COUNT=$(($WAIT_COUNT + 1))
done

if [ "$STATUS" != "active" ]; then
    echo "ERROR: Instance didn't become active after 10 minutes"
    exit 1
fi
echo ""

# Stage 4: Upload Code
echo "Stage 4: Uploading neo_model codebase..."
rsync -avz --progress \
    -e "ssh -i $SSH_KEY_PATH -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
    --exclude 'venv' --exclude '__pycache__' --exclude '.git' --exclude 'mau-neo.txt' \
    --exclude '.pytest_cache' --exclude 'checkpoints/*' --exclude 'data/coco/*' \
    ./ ubuntu@$IP:~/neo_model/ | tail -10
echo "✓ Code uploaded"
echo ""

# Stage 5: Setup Environment
echo "Stage 5: Setting up Python environment..."
ssh -i $SSH_KEY_PATH -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$IP "
cd ~/neo_model
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel --quiet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
pip install -r requirements.txt --quiet
echo '✓ Dependencies installed'
python scripts/verify_installation.py
"
echo ""

# Stage 6: Create H100 Config
echo "Stage 6: Creating H100-optimized config..."
ssh -i $SSH_KEY_PATH -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$IP "
cd ~/neo_model
source venv/bin/activate
cp configs/rtdetr_r50_1920x1080.yml configs/rtdetr_r50_1920x1080_h100.yml
python3 -c \"
import yaml
with open('configs/rtdetr_r50_1920x1080_h100.yml', 'r') as f:
    config = yaml.safe_load(f)
config['data']['train']['batch_size'] = 16
config['data']['val']['batch_size'] = 16
config['data']['train']['num_workers'] = 16
config['data']['val']['num_workers'] = 16
config['training']['accumulate_grad_batches'] = 1
with open('configs/rtdetr_r50_1920x1080_h100.yml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
print('✓ H100 config created: batch_size=16, accumulation=1')
\"
"
echo ""

# Stage 7: Download COCO Dataset
echo "Stage 7: Downloading COCO dataset (~21GB, 5-10 minutes)..."
ssh -i $SSH_KEY_PATH -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$IP "
cd ~/neo_model
source venv/bin/activate
python scripts/prepare_data.py --data_dir data/coco --split both
"
echo "✓ COCO dataset downloaded"
echo ""

# Stage 8: Run Overfit Test (CRITICAL)
echo "Stage 8: Running overfit test (5 batches)..."
echo "  Expected: Loss drops from ~1200 to <100 within 10-15 epochs"
ssh -i $SSH_KEY_PATH -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$IP "
cd ~/neo_model
source venv/bin/activate
timeout 300 python scripts/train.py --config configs/rtdetr_r50_1920x1080_h100.yml --overfit_batches 5 2>&1 | tee overfit_test.log
"

# Check if overfit test passed
FINAL_LOSS=$(ssh -i $SSH_KEY_PATH -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$IP "
grep 'Epoch.*completed.*Loss:' ~/neo_model/overfit_test.log | tail -3
")

echo "$FINAL_LOSS"
echo ""

read -p "Overfit test complete. Loss should be decreasing. Proceed with full training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborting. To terminate instance manually:"
    echo "  Instance ID: $INSTANCE_ID"
    echo "  Terminate: curl -u \"\$API_KEY:\" $API_BASE/instance-operations/terminate -X POST -H 'Content-Type: application/json' -d '{\"instance_ids\": [\"$INSTANCE_ID\"]}'"
    exit 1
fi
echo ""

# Stage 9: Launch Full Training
echo "Stage 9: Launching full 72-epoch training in tmux..."
echo "  Duration: ~30-36 hours"
echo "  Cost: \$75-90 total"
ssh -i $SSH_KEY_PATH -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$IP "
cd ~/neo_model
source venv/bin/activate

# Save launch time for cost calculation
date -Iseconds > /tmp/neo_model_launch_time.txt

# Start training in tmux
tmux new-session -d -s training 'python scripts/train.py --config configs/rtdetr_r50_1920x1080_h100.yml 2>&1 | tee training.log'
"

echo "✓ Training started in tmux session"
echo ""

# Stage 10: Start Web Dashboard
echo "Stage 10: Launching web monitoring dashboard..."
ssh -i $SSH_KEY_PATH -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$IP "
cd ~/neo_model/dashboard
source ../venv/bin/activate

# Install Flask if needed
pip list | grep -q Flask || pip install flask

# Start dashboard in background
nohup python app.py > dashboard.log 2>&1 &
sleep 2
echo '✓ Dashboard started on port 8080'
"

echo "✓ Web dashboard launched"
echo ""
echo "========================================"
echo "Training Launched Successfully!"
echo "========================================"
echo "Instance ID: $INSTANCE_ID"
echo "IP Address: $IP"
echo "SSH: ssh -i $SSH_KEY_PATH ubuntu@$IP"
echo ""
echo "Web Dashboard (NEW!):"
echo "  URL: http://$IP:8080"
echo "  Features: Real-time metrics, GPU stats, cost tracking"
echo "  Auto-refreshes every 30 seconds"
echo "  Mobile-friendly, Jebi-branded design"
echo ""
echo "Monitor training:"
echo "  ssh -i $SSH_KEY_PATH ubuntu@$IP 'tail -f ~/neo_model/training.log'"
echo ""
echo "Check GPU usage:"
echo "  ssh -i $SSH_KEY_PATH ubuntu@$IP 'nvidia-smi'"
echo ""
echo "TensorBoard (local):"
echo "  ssh -L 6006:localhost:6006 -i $SSH_KEY_PATH ubuntu@$IP"
echo "  ssh -i $SSH_KEY_PATH ubuntu@$IP 'cd ~/neo_model && source venv/bin/activate && tensorboard --logdir outputs/logs'"
echo "  Open: http://localhost:6006"
echo ""
echo "Estimated completion: $(date -d '+36 hours' 2>/dev/null || date -v +36H)"
echo "Estimated cost: \$75-90"
echo ""
echo "IMPORTANT: Remember to download results and terminate instance when done!"
echo "  Instance ID saved to: /tmp/neo_model_instance_id.txt"
```

**Make executable**:
```bash
chmod +x scripts/automate_training.sh
```

---

## Training Monitoring

### Web Dashboard (Recommended) 🎉

**NEW**: Real-time web-based monitoring with Jebi branding!

Access your training dashboard from any device (phone, tablet, laptop):

```bash
# Get instance IP
IP=$(cat /tmp/neo_model_instance_ip.txt)

# Open in browser
open http://$IP:8080  # macOS
# or visit: http://your-instance-ip:8080
```

**Dashboard Features**:
- ✅ **Real-time metrics**: Epoch progress, loss, AP scores
- ✅ **GPU monitoring**: Utilization, memory, temperature, power
- ✅ **Cost tracking**: Running cost and projected total
- ✅ **Training speed**: Iterations/sec, time per epoch, ETA
- ✅ **Auto-refresh**: Updates every 30 seconds automatically
- ✅ **Mobile-friendly**: Responsive design for phones
- ✅ **Jebi-branded**: Professional design with Jebi identity

**Dashboard Screenshots**:
- Training progress bar with epoch completion
- Large loss display with trend indicators
- GPU metrics in grid layout
- Cost tracker with uptime hours
- Estimated completion time

**Access from Phone**:
1. Open browser on phone
2. Navigate to `http://209.20.157.204:8080` (use your instance IP)
3. Bookmark for quick access
4. Page auto-refreshes, no manual reload needed

**Dashboard API** (for programmatic access):
```bash
curl http://$IP:8080/api/status | python3 -m json.tool
```

### CLI Monitoring (Traditional)

For command-line monitoring:

```bash
# Load instance info
INSTANCE_ID=$(cat /tmp/neo_model_instance_id.txt)
IP=$(cat /tmp/neo_model_instance_ip.txt)
SSH_KEY_PATH="$HOME/.ssh/lambda_neo_model"

# Check training log
ssh -i $SSH_KEY_PATH ubuntu@$IP "tail -50 ~/neo_model/training.log | grep -E 'Epoch|Loss|AP'"

# Check GPU usage
ssh -i $SSH_KEY_PATH ubuntu@$IP "nvidia-smi"

# Check tmux session
ssh -i $SSH_KEY_PATH ubuntu@$IP "tmux ls"

# Check disk space
ssh -i $SSH_KEY_PATH ubuntu@$IP "df -h ~/neo_model"

# View checkpoints
ssh -i $SSH_KEY_PATH ubuntu@$IP "ls -lh ~/neo_model/checkpoints/"
```

### Cost Tracking

```bash
# Calculate running cost
LAUNCH_TIME=$(cat /tmp/neo_model_launch_time.txt)
CURRENT_TIME=$(date +%s)
HOURS_RUNNING=$(( (CURRENT_TIME - LAUNCH_TIME) / 3600 ))
COST=$(echo "$HOURS_RUNNING * 2.49" | bc)

echo "Instance running for: $HOURS_RUNNING hours"
echo "Estimated cost so far: \$$COST"
```

### Expected Progress Timeline (H100)

| Epoch | Time Elapsed | Loss | AP | Status |
|-------|--------------|------|-----|--------|
| 0 | 0h | ~1200 | N/A | Warmup starting |
| 5 | ~2h | ~800 | ~0.05 | Warmup complete |
| 10 | ~4h | ~400-600 | ~0.25-0.35 | Learning |
| 20 | ~8h | ~200-300 | ~0.40-0.45 | Improving |
| 40 | ~16h | ~150-200 | ~0.48-0.52 | LR drop #1 |
| 55 | ~22h | ~120-150 | ~0.50-0.53 | LR drop #2 |
| 72 | ~30-36h | ~100-130 | **>0.52** | Complete |

---

## Post-Training: Download Results

```bash
# Load instance info
INSTANCE_ID=$(cat /tmp/neo_model_instance_id.txt)
IP=$(cat /tmp/neo_model_instance_ip.txt)
SSH_KEY_PATH="$HOME/.ssh/lambda_neo_model"

# Create local results directory
mkdir -p neo_model_results

# Download checkpoints (~2-5GB)
echo "Downloading checkpoints..."
rsync -avz --progress \
    -e "ssh -i $SSH_KEY_PATH -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
    ubuntu@$IP:~/neo_model/checkpoints/ \
    neo_model_results/checkpoints/

# Download logs
echo "Downloading logs..."
rsync -avz --progress \
    -e "ssh -i $SSH_KEY_PATH -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
    ubuntu@$IP:~/neo_model/outputs/logs/ \
    neo_model_results/outputs/logs/

# Download training log
scp -i $SSH_KEY_PATH -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    ubuntu@$IP:~/neo_model/training.log \
    neo_model_results/

echo "✓ Results downloaded to neo_model_results/"
```

---

## Instance Termination

```bash
# Load instance info
INSTANCE_ID=$(cat /tmp/neo_model_instance_id.txt)
API_KEY=$(cat mau-neo.txt)

# Calculate final cost
LAUNCH_TIME=$(cat /tmp/neo_model_launch_time.txt)
TERMINATE_TIME=$(date +%s)
HOURS_RUNNING=$(( (TERMINATE_TIME - LAUNCH_TIME) / 3600 ))
COST=$(echo "$HOURS_RUNNING * 2.49" | bc)

echo "========================================"
echo "Final Cost Report"
echo "========================================"
echo "Instance ID: $INSTANCE_ID"
echo "Running time: $HOURS_RUNNING hours"
echo "Hourly rate: \$2.49"
echo "Total cost: \$$COST"
echo ""

read -p "Terminate instance now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Terminating instance..."
    curl -s -u "$API_KEY:" https://cloud.lambda.ai/api/v1/instance-operations/terminate \
      -X POST \
      -H "Content-Type: application/json" \
      -d "{\"instance_ids\": [\"$INSTANCE_ID\"]}"

    echo ""
    echo "✓ Instance terminated"
    echo "✓ Billing stopped"

    # Cleanup
    rm /tmp/neo_model_instance_id.txt /tmp/neo_model_instance_ip.txt /tmp/neo_model_launch_time.txt 2>/dev/null
    echo "✓ Cleanup complete"
else
    echo "Instance still running. Remember to terminate manually:"
    echo "  curl -u \"\$API_KEY:\" https://cloud.lambda.ai/api/v1/instance-operations/terminate -X POST -H 'Content-Type: application/json' -d '{\"instance_ids\": [\"$INSTANCE_ID\"]}'"
fi
```

---

## Python Alternative (Using lambda-cloud-client)

For programmatic access from Python:

```bash
pip install lambda-cloud-client
```

```python
import lambda_cloud_client
import os
import time

# Configuration
API_KEY = open("mau-neo.txt").read().strip()
config = lambda_cloud_client.Configuration(username=API_KEY, password="")

with lambda_cloud_client.ApiClient(config) as client:
    api = lambda_cloud_client.DefaultApi(client)

    # List instance types
    types = api.instance_types()
    print("Available H100:", types.data.gpu_1x_h100_pcie)

    # Launch instance
    launch_req = lambda_cloud_client.LaunchInstanceRequest(
        instance_type_name="gpu_1x_h100_pcie",
        region_name="us-west-3",
        ssh_key_names=["neo_model_training_key"],
        name="neo-model-h100-training"
    )
    response = api.launch_instance(launch_req)
    instance_id = response.data.instance_ids[0]
    print(f"Launched: {instance_id}")

    # Wait for active status
    while True:
        instances = api.list_instances()
        instance = [i for i in instances.data if i.id == instance_id][0]
        if instance.status == "active":
            print(f"Instance ready! IP: {instance.ip}")
            break
        time.sleep(10)

    # ... training execution ...

    # Terminate when done
    terminate_req = lambda_cloud_client.TerminateInstanceRequest(
        instance_ids=[instance_id]
    )
    api.terminate_instance(terminate_req)
    print("Instance terminated")
```

---

## Troubleshooting

### Instance Won't Boot
- **Issue**: Status stuck on "booting" for >10 minutes
- **Solution**: Wait up to 15 minutes, then contact Lambda Labs support
- **Workaround**: Try different region

### SSH Connection Refused
- **Issue**: Cannot connect even though status is "active"
- **Solution**: Wait 1-2 more minutes after "active" status
- **Check**: Verify SSH key was added correctly

### Out of Memory (OOM)
- **Issue**: CUDA OOM error during training
- **Solution**: Reduce batch_size in H100 config (16 → 8 or 4)
- **Note**: Should not happen with 80GB H100 at batch_size=16

### Training Stops Unexpectedly
- **Issue**: Tmux session died
- **Solution**: Check logs, resume from last checkpoint:
  ```bash
  ssh -i $SSH_KEY_PATH ubuntu@$IP "
  cd ~/neo_model && source venv/bin/activate &&
  tmux new-session -d -s training 'python scripts/train.py --config configs/rtdetr_r50_1920x1080_h100.yml --resume checkpoints/last_checkpoint.pth 2>&1 | tee -a training.log'
  "
  ```

### High Cost Alert
- **Issue**: Instance running longer than expected
- **Check**: Monitor `/tmp/neo_model_launch_time.txt`
- **Action**: Set cost alert at $100:
  ```bash
  if [ $COST -gt 100 ]; then
      echo "WARNING: Cost exceeded \$100! Consider terminating."
  fi
  ```

---

## Cost Optimization

### Best Practices

1. **Run overfit test first** - Validate pipeline before 72-epoch training
2. **Use tmux** - Persist training across SSH disconnects
3. **Terminate immediately when done** - Avoid idle billing
4. **Monitor regularly** - Check progress every 6-12 hours
5. **Set budget alerts** - Stop if cost exceeds threshold

### Cost Comparison

| GPU | VRAM | Hourly Rate | 72 Epochs | Total Cost |
|-----|------|-------------|-----------|------------|
| RTX 3090 | 24GB | $0.50 | ~72h | ~$36 |
| H100 PCIe | 80GB | **$2.49** | **~36h** | **$90** |
| H100 SXM5 | 80GB | $3.29 | ~30h | ~$99 |
| GH200 | 96GB | $1.49 | ~32h | ~$48 |

**Recommendation**: **H100 PCIe** offers best balance of speed ($2.49/hr) and availability.

**Alternative**: **GH200** at $1.49/hr could save ~$40 but availability may be limited.

---

## Automated Monitoring Script

Save as `scripts/monitor_training.sh`:

```bash
#!/bin/bash

INSTANCE_ID=$(cat /tmp/neo_model_instance_id.txt)
IP=$(cat /tmp/neo_model_instance_ip.txt)
SSH_KEY_PATH="$HOME/.ssh/lambda_neo_model"
LAUNCH_TIME=$(cat /tmp/neo_model_launch_time.txt)

while true; do
    clear
    echo "========================================"
    echo "neo_model Training Monitor"
    echo "========================================"
    echo "Instance: $INSTANCE_ID"
    echo "IP: $IP"

    # Calculate cost
    CURRENT_TIME=$(date +%s)
    HOURS=$(( (CURRENT_TIME - LAUNCH_TIME) / 3600 ))
    COST=$(echo "$HOURS * 2.49" | bc)
    echo "Running: ${HOURS}h | Cost: \$$COST"
    echo ""

    # GPU status
    echo "=== GPU Usage ==="
    ssh -i $SSH_KEY_PATH -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$IP "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits" 2>/dev/null || echo "Connection failed"
    echo ""

    # Training progress
    echo "=== Training Progress ==="
    ssh -i $SSH_KEY_PATH -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$IP "tail -20 ~/neo_model/training.log | grep -E 'Epoch.*completed|AP:'" 2>/dev/null | tail -5
    echo ""

    # Checkpoints
    echo "=== Recent Checkpoints ==="
    ssh -i $SSH_KEY_PATH -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$IP "ls -lh ~/neo_model/checkpoints/ 2>/dev/null | tail -5" 2>/dev/null || echo "No checkpoints yet"
    echo ""

    echo "Press Ctrl+C to stop monitoring"
    sleep 300  # Update every 5 minutes
done
```

**Usage**:
```bash
chmod +x scripts/monitor_training.sh
./scripts/monitor_training.sh
```

---

## References

- **Lambda Labs API Docs**: https://docs.lambda.ai/api/cloud
- **Instance Types**: https://cloud.lambda.ai/api/v1/docs
- **Python Client**: https://pypi.org/project/lambda-cloud-client/
- **Support**: https://support.lambdalabs.com/

---

## Quick Reference Card

### Launch Training (One Command)
```bash
./scripts/automate_training.sh
```

### Monitor Training

**Web Dashboard** (Recommended):
```bash
# Open in browser
open http://$(cat /tmp/neo_model_instance_ip.txt):8080
```

**CLI Monitoring**:
```bash
./scripts/monitor_training.sh
```

### Download Results
```bash
rsync -avz -e "ssh -i ~/.ssh/lambda_neo_model" \
    ubuntu@$(cat /tmp/neo_model_instance_ip.txt):~/neo_model/checkpoints \
    ./neo_model_results/
```

### Terminate Instance
```bash
INSTANCE_ID=$(cat /tmp/neo_model_instance_id.txt)
API_KEY=$(cat mau-neo.txt)
curl -u "$API_KEY:" https://cloud.lambda.ai/api/v1/instance-operations/terminate \
  -X POST -H "Content-Type: application/json" \
  -d "{\"instance_ids\": [\"$INSTANCE_ID\"]}"
```

---

## Summary

This automation guide enables:
- ✅ **Fully automated instance provisioning** via Lambda Labs API
- ✅ **One-command training launch** with `automate_training.sh`
- ✅ **Real-time web dashboard** with Jebi branding (NEW!)
- ✅ **Mobile-friendly monitoring** from any device
- ✅ **Continuous CLI monitoring** with `monitor_training.sh`
- ✅ **Cost tracking** throughout execution
- ✅ **Safe termination** to prevent billing overruns

**Total active user time**: <10 minutes
**Total elapsed time**: ~30-36 hours
**Total cost**: $75-90 for complete training

**Dashboard URL**: `http://[instance-ip]:8080` - Access from any browser!
