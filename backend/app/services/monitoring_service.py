import psutil
import GPUtil
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import asyncio
import logging
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

logger = logging.getLogger(__name__)

# Prometheus metrics
cpu_usage_gauge = Gauge('system_cpu_usage_percent', 'CPU usage percentage')
memory_usage_gauge = Gauge('system_memory_usage_percent', 'Memory usage percentage')
disk_usage_gauge = Gauge('system_disk_usage_percent', 'Disk usage percentage')
gpu_usage_gauge = Gauge('system_gpu_usage_percent', 'GPU usage percentage', ['gpu_id'])
gpu_memory_gauge = Gauge('system_gpu_memory_usage_percent', 'GPU memory usage percentage', ['gpu_id'])

training_jobs_running_gauge = Gauge('training_jobs_running', 'Number of running training jobs')
training_jobs_queued_gauge = Gauge('training_jobs_queued', 'Number of queued training jobs')
training_jobs_completed_counter = Counter('training_jobs_completed_total', 'Total completed training jobs')
training_jobs_failed_counter = Counter('training_jobs_failed_total', 'Total failed training jobs')

models_total_gauge = Gauge('models_total', 'Total number of models')
model_inference_duration = Histogram('model_inference_duration_seconds', 'Model inference duration', ['model_name'])
model_accuracy_gauge = Gauge('model_accuracy', 'Model accuracy', ['model_name'])


class MonitoringService:
    def __init__(self):
        self.metrics_history = []
        self.max_history_size = 1000

    async def get_system_metrics(self) -> Dict:
        """Get current system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            # Network metrics
            network = psutil.net_io_counters()
            
            # GPU metrics
            gpu_metrics = await self._get_gpu_metrics()
            
            # Process metrics
            process_metrics = await self._get_process_metrics()
            
            metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "cpu": {
                    "usage_percent": cpu_percent,
                    "count": cpu_count,
                    "frequency": {
                        "current": cpu_freq.current if cpu_freq else None,
                        "min": cpu_freq.min if cpu_freq else None,
                        "max": cpu_freq.max if cpu_freq else None
                    }
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "used": memory.used,
                    "usage_percent": memory.percent,
                    "swap": {
                        "total": swap.total,
                        "used": swap.used,
                        "usage_percent": swap.percent
                    }
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "usage_percent": (disk.used / disk.total) * 100
                },
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                },
                "gpu": gpu_metrics,
                "processes": process_metrics
            }
            
            # Store in history
            self._store_metrics(metrics)

            # Update Prometheus metrics
            self._update_prometheus_metrics(metrics)

            return metrics
            
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {"error": str(e)}

    async def _get_gpu_metrics(self) -> List[Dict]:
        """Get GPU metrics"""
        try:
            gpus = GPUtil.getGPUs()
            gpu_metrics = []
            
            for gpu in gpus:
                gpu_info = {
                    "id": gpu.id,
                    "name": gpu.name,
                    "load": gpu.load * 100,  # Convert to percentage
                    "memory": {
                        "total": gpu.memoryTotal,
                        "used": gpu.memoryUsed,
                        "free": gpu.memoryFree,
                        "usage_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100
                    },
                    "temperature": gpu.temperature,
                    "uuid": gpu.uuid
                }
                gpu_metrics.append(gpu_info)
            
            return gpu_metrics
            
        except Exception as e:
            logger.warning(f"Could not get GPU metrics: {e}")
            return []

    async def _get_process_metrics(self) -> Dict:
        """Get process-specific metrics"""
        try:
            # Get current process
            current_process = psutil.Process()
            
            # Get training processes (processes running training scripts)
            training_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
                try:
                    cmdline = proc.info['cmdline']
                    if cmdline and any('training' in arg for arg in cmdline):
                        training_processes.append({
                            "pid": proc.info['pid'],
                            "name": proc.info['name'],
                            "cpu_percent": proc.info['cpu_percent'],
                            "memory_mb": proc.info['memory_info'].rss / 1024 / 1024,
                            "cmdline": ' '.join(cmdline[:3])  # First 3 args
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            return {
                "current_process": {
                    "pid": current_process.pid,
                    "cpu_percent": current_process.cpu_percent(),
                    "memory_mb": current_process.memory_info().rss / 1024 / 1024,
                    "num_threads": current_process.num_threads()
                },
                "training_processes": training_processes,
                "total_processes": len(psutil.pids())
            }
            
        except Exception as e:
            logger.error(f"Error getting process metrics: {e}")
            return {}

    def _store_metrics(self, metrics: Dict):
        """Store metrics in history"""
        self.metrics_history.append(metrics)
        
        # Keep only recent metrics
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]

    async def get_metrics_history(self, hours: int = 1) -> List[Dict]:
        """Get metrics history for the specified number of hours"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        filtered_metrics = []
        for metric in self.metrics_history:
            try:
                metric_time = datetime.fromisoformat(metric['timestamp'].replace('Z', '+00:00'))
                if metric_time >= cutoff_time:
                    filtered_metrics.append(metric)
            except (KeyError, ValueError):
                continue
        
        return filtered_metrics

    async def get_resource_usage_summary(self) -> Dict:
        """Get resource usage summary"""
        current_metrics = await self.get_system_metrics()
        
        if "error" in current_metrics:
            return current_metrics
        
        # Calculate averages from recent history
        recent_metrics = await self.get_metrics_history(hours=1)
        
        if recent_metrics:
            avg_cpu = sum(m.get('cpu', {}).get('usage_percent', 0) for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.get('memory', {}).get('usage_percent', 0) for m in recent_metrics) / len(recent_metrics)
            avg_gpu = 0
            if recent_metrics[0].get('gpu'):
                gpu_loads = [gpu.get('load', 0) for m in recent_metrics for gpu in m.get('gpu', [])]
                avg_gpu = sum(gpu_loads) / len(gpu_loads) if gpu_loads else 0
        else:
            avg_cpu = current_metrics.get('cpu', {}).get('usage_percent', 0)
            avg_memory = current_metrics.get('memory', {}).get('usage_percent', 0)
            avg_gpu = 0
            if current_metrics.get('gpu'):
                avg_gpu = sum(gpu.get('load', 0) for gpu in current_metrics['gpu']) / len(current_metrics['gpu'])
        
        return {
            "current": {
                "cpu_percent": current_metrics.get('cpu', {}).get('usage_percent', 0),
                "memory_percent": current_metrics.get('memory', {}).get('usage_percent', 0),
                "disk_percent": current_metrics.get('disk', {}).get('usage_percent', 0),
                "gpu_percent": avg_gpu
            },
            "averages_1h": {
                "cpu_percent": round(avg_cpu, 2),
                "memory_percent": round(avg_memory, 2),
                "gpu_percent": round(avg_gpu, 2)
            },
            "training_processes": len(current_metrics.get('processes', {}).get('training_processes', [])),
            "total_processes": current_metrics.get('processes', {}).get('total_processes', 0),
            "gpu_count": len(current_metrics.get('gpu', [])),
            "timestamp": current_metrics.get('timestamp')
        }

    async def check_resource_alerts(self) -> List[Dict]:
        """Check for resource usage alerts"""
        alerts = []
        current_metrics = await self.get_system_metrics()
        
        if "error" in current_metrics:
            return [{"type": "error", "message": "Failed to get system metrics"}]
        
        # CPU alert
        cpu_usage = current_metrics.get('cpu', {}).get('usage_percent', 0)
        if cpu_usage > 90:
            alerts.append({
                "type": "warning",
                "resource": "cpu",
                "message": f"High CPU usage: {cpu_usage:.1f}%",
                "value": cpu_usage,
                "threshold": 90
            })
        
        # Memory alert
        memory_usage = current_metrics.get('memory', {}).get('usage_percent', 0)
        if memory_usage > 85:
            alerts.append({
                "type": "warning",
                "resource": "memory",
                "message": f"High memory usage: {memory_usage:.1f}%",
                "value": memory_usage,
                "threshold": 85
            })
        
        # Disk alert
        disk_usage = current_metrics.get('disk', {}).get('usage_percent', 0)
        if disk_usage > 80:
            alerts.append({
                "type": "warning",
                "resource": "disk",
                "message": f"High disk usage: {disk_usage:.1f}%",
                "value": disk_usage,
                "threshold": 80
            })
        
        # GPU alerts
        for gpu in current_metrics.get('gpu', []):
            gpu_load = gpu.get('load', 0)
            gpu_memory = gpu.get('memory', {}).get('usage_percent', 0)
            
            if gpu_load > 95:
                alerts.append({
                    "type": "info",
                    "resource": "gpu",
                    "message": f"GPU {gpu['id']} high load: {gpu_load:.1f}%",
                    "value": gpu_load,
                    "threshold": 95
                })
            
            if gpu_memory > 90:
                alerts.append({
                    "type": "warning",
                    "resource": "gpu_memory",
                    "message": f"GPU {gpu['id']} high memory usage: {gpu_memory:.1f}%",
                    "value": gpu_memory,
                    "threshold": 90
                })
        
        return alerts

    async def get_training_resource_usage(self) -> Dict:
        """Get resource usage specifically for training processes"""
        current_metrics = await self.get_system_metrics()
        
        training_processes = current_metrics.get('processes', {}).get('training_processes', [])
        
        if not training_processes:
            return {
                "active_training_jobs": 0,
                "total_cpu_usage": 0,
                "total_memory_mb": 0,
                "processes": []
            }
        
        total_cpu = sum(proc.get('cpu_percent', 0) for proc in training_processes)
        total_memory = sum(proc.get('memory_mb', 0) for proc in training_processes)
        
        return {
            "active_training_jobs": len(training_processes),
            "total_cpu_usage": round(total_cpu, 2),
            "total_memory_mb": round(total_memory, 2),
            "processes": training_processes
        }

    def _update_prometheus_metrics(self, metrics: Dict):
        """Update Prometheus metrics with current system metrics"""
        try:
            # System metrics
            cpu_usage_gauge.set(metrics.get('cpu', {}).get('usage_percent', 0))
            memory_usage_gauge.set(metrics.get('memory', {}).get('usage_percent', 0))
            disk_usage_gauge.set(metrics.get('disk', {}).get('usage_percent', 0))

            # GPU metrics
            for gpu in metrics.get('gpu', []):
                gpu_id = str(gpu.get('id', 0))
                gpu_usage_gauge.labels(gpu_id=gpu_id).set(gpu.get('load', 0))
                gpu_memory_gauge.labels(gpu_id=gpu_id).set(
                    gpu.get('memory', {}).get('usage_percent', 0)
                )

            # Training process metrics
            training_processes = metrics.get('processes', {}).get('training_processes', [])
            training_jobs_running_gauge.set(len(training_processes))

        except Exception as e:
            logger.error(f"Error updating Prometheus metrics: {e}")

    async def update_training_metrics(self):
        """Update training-related Prometheus metrics"""
        try:
            from app.models.training_job import TrainingJob, TrainingStatus
            from app.models.model import Model

            # Count training jobs by status
            running_jobs = await TrainingJob.find(
                TrainingJob.status == TrainingStatus.RUNNING
            ).count()
            queued_jobs = await TrainingJob.find(
                TrainingJob.status == TrainingStatus.QUEUED
            ).count()

            training_jobs_running_gauge.set(running_jobs)
            training_jobs_queued_gauge.set(queued_jobs)

            # Count total models
            total_models = await Model.find().count()
            models_total_gauge.set(total_models)

        except Exception as e:
            logger.error(f"Error updating training metrics: {e}")

    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format"""
        return generate_latest()
