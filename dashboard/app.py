#!/usr/bin/env python3
"""
Flask Dashboard for neo_model Training Monitoring
Displays real-time training metrics from training.log and GPU stats
"""

import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, render_template, jsonify

app = Flask(__name__)

# Paths
LOG_PATH = Path.home() / 'neo_model' / 'training.log'
LAUNCH_TIME_PATH = Path('/tmp/neo_model_launch_time.txt')
INSTANCE_COST_PER_HOUR = 2.49  # H100 cost


def parse_training_log():
    """Extract latest training metrics from training.log"""
    metrics = {
        'epoch': 0,
        'total_epochs': 72,
        'current_batch': 0,
        'total_batches': 7329,
        'loss': None,
        'ap': None,
        'ap50': None,
        'ap75': None,
        'best_ap': None,
        'learning_rate': None,
        'speed': None,
        'time_per_epoch': None,
        'last_update': None
    }

    if not LOG_PATH.exists():
        return metrics

    try:
        # Read last 1000 lines for recent data
        with open(LOG_PATH, 'r') as f:
            lines = f.readlines()
            recent_lines = lines[-1000:] if len(lines) > 1000 else lines

        # Parse from most recent to oldest
        for line in reversed(recent_lines):
            # Epoch pattern: "Epoch X/72"
            if metrics['epoch'] == 0:
                match = re.search(r'Epoch (\d+)/(\d+)', line)
                if match:
                    metrics['epoch'] = int(match.group(1))
                    metrics['total_epochs'] = int(match.group(2))

            # Batch pattern: "Batch XXXX/7329"
            if metrics['current_batch'] == 0:
                match = re.search(r'Batch (\d+)/(\d+)', line)
                if match:
                    metrics['current_batch'] = int(match.group(1))
                    metrics['total_batches'] = int(match.group(2))

            # Loss pattern: "Loss: X.XXX" or "loss: X.XXX"
            if metrics['loss'] is None:
                match = re.search(r'[Ll]oss:\s*([\d.]+)', line)
                if match:
                    metrics['loss'] = float(match.group(1))

            # AP patterns
            if metrics['ap'] is None:
                match = re.search(r'AP:\s*([\d.]+)', line)
                if match:
                    metrics['ap'] = float(match.group(1))

            if metrics['ap50'] is None:
                match = re.search(r'AP50:\s*([\d.]+)', line)
                if match:
                    metrics['ap50'] = float(match.group(1))

            if metrics['ap75'] is None:
                match = re.search(r'AP75:\s*([\d.]+)', line)
                if match:
                    metrics['ap75'] = float(match.group(1))

            if metrics['best_ap'] is None:
                match = re.search(r'Best AP:\s*([\d.]+)', line)
                if match:
                    metrics['best_ap'] = float(match.group(1))

            # Learning rate
            if metrics['learning_rate'] is None:
                match = re.search(r'LR:\s*([\d.e-]+)', line)
                if match:
                    metrics['learning_rate'] = float(match.group(1))

            # Speed pattern: "X.XX it/s"
            if metrics['speed'] is None:
                match = re.search(r'([\d.]+)\s*it/s', line)
                if match:
                    metrics['speed'] = float(match.group(1))

            # Epoch time pattern
            if metrics['time_per_epoch'] is None:
                match = re.search(r'Epoch completed in ([\d.]+)s', line)
                if match:
                    metrics['time_per_epoch'] = float(match.group(1))

        # Get last modified time of log file
        metrics['last_update'] = datetime.fromtimestamp(LOG_PATH.stat().st_mtime)

    except Exception as e:
        print(f"Error parsing training log: {e}")

    return metrics


def get_gpu_stats():
    """Query GPU stats using nvidia-smi"""
    stats = {
        'utilization': None,
        'memory_used': None,
        'memory_total': None,
        'temperature': None,
        'power_draw': None
    }

    try:
        cmd = [
            'nvidia-smi',
            '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
            '--format=csv,noheader,nounits',
            '--id=0'  # First GPU
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

        if result.returncode == 0:
            values = result.stdout.strip().split(', ')
            if len(values) >= 5:
                stats['utilization'] = int(values[0])
                stats['memory_used'] = float(values[1]) / 1024  # Convert MB to GB
                stats['memory_total'] = float(values[2]) / 1024
                stats['temperature'] = int(values[3])
                stats['power_draw'] = float(values[4])

    except Exception as e:
        print(f"Error getting GPU stats: {e}")

    return stats


def calculate_cost():
    """Calculate instance cost based on uptime"""
    cost_info = {
        'uptime_hours': None,
        'cost_so_far': None,
        'projected_total': None
    }

    try:
        # Try to get launch time from file
        if LAUNCH_TIME_PATH.exists():
            with open(LAUNCH_TIME_PATH, 'r') as f:
                launch_time_str = f.read().strip()
                launch_time = datetime.fromisoformat(launch_time_str)
        else:
            # Fallback: use uptime command
            result = subprocess.run(['uptime', '-s'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                launch_time = datetime.strptime(result.stdout.strip(), '%Y-%m-%d %H:%M:%S')
            else:
                return cost_info

        # Calculate uptime
        uptime = datetime.now() - launch_time
        uptime_hours = uptime.total_seconds() / 3600
        cost_so_far = uptime_hours * INSTANCE_COST_PER_HOUR

        # Project total cost (assuming 72 epochs, ~2 hours per epoch = 144 hours)
        projected_total = 144 * INSTANCE_COST_PER_HOUR

        cost_info['uptime_hours'] = uptime_hours
        cost_info['cost_so_far'] = cost_so_far
        cost_info['projected_total'] = projected_total

    except Exception as e:
        print(f"Error calculating cost: {e}")

    return cost_info


def estimate_completion(metrics):
    """Estimate training completion time"""
    if metrics['epoch'] == 0 or metrics['time_per_epoch'] is None:
        return None

    epochs_remaining = metrics['total_epochs'] - metrics['epoch']
    seconds_remaining = epochs_remaining * metrics['time_per_epoch']

    eta = datetime.now() + timedelta(seconds=seconds_remaining)
    return {
        'hours_remaining': seconds_remaining / 3600,
        'eta': eta.strftime('%Y-%m-%d %H:%M')
    }


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')


@app.route('/api/status')
def api_status():
    """JSON API endpoint for current training status"""
    training = parse_training_log()
    gpu = get_gpu_stats()
    cost = calculate_cost()

    # Calculate completion estimate
    completion = estimate_completion(training)

    # Calculate progress percentage
    if training['total_epochs'] > 0:
        epoch_progress = (training['epoch'] / training['total_epochs']) * 100
    else:
        epoch_progress = 0

    if training['total_batches'] > 0:
        batch_progress = (training['current_batch'] / training['total_batches']) * 100
    else:
        batch_progress = 0

    response = {
        'training': {
            **training,
            'epoch_progress': round(epoch_progress, 1),
            'batch_progress': round(batch_progress, 1)
        },
        'gpu': gpu,
        'cost': cost,
        'completion': completion,
        'timestamp': datetime.now().isoformat(),
        'last_update_ago': _format_time_ago(training.get('last_update'))
    }

    return jsonify(response)


def _format_time_ago(dt):
    """Format datetime as 'X seconds/minutes ago'"""
    if dt is None:
        return 'Unknown'

    delta = datetime.now() - dt
    seconds = delta.total_seconds()

    if seconds < 60:
        return f"{int(seconds)}s ago"
    elif seconds < 3600:
        return f"{int(seconds / 60)}m ago"
    else:
        return f"{int(seconds / 3600)}h ago"


if __name__ == '__main__':
    print("Starting neo_model training dashboard on port 8080...")
    print("Access via: http://209.20.157.204:8080")
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)
