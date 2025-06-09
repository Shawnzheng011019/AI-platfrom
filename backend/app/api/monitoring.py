from fastapi import APIRouter, Depends, Response
from typing import List, Dict, Optional

from app.models.user import User
from app.api.auth import get_current_active_user
from app.services.monitoring_service import MonitoringService, CONTENT_TYPE_LATEST

router = APIRouter()


@router.get("/system")
async def get_system_metrics(
    current_user: User = Depends(get_current_active_user)
):
    """Get current system metrics"""
    monitoring_service = MonitoringService()
    metrics = await monitoring_service.get_system_metrics()
    return metrics


@router.get("/system/history")
async def get_system_metrics_history(
    hours: int = 1,
    current_user: User = Depends(get_current_active_user)
):
    """Get system metrics history"""
    monitoring_service = MonitoringService()
    history = await monitoring_service.get_metrics_history(hours=hours)
    return {"history": history, "hours": hours}


@router.get("/system/summary")
async def get_resource_usage_summary(
    current_user: User = Depends(get_current_active_user)
):
    """Get resource usage summary"""
    monitoring_service = MonitoringService()
    summary = await monitoring_service.get_resource_usage_summary()
    return summary


@router.get("/alerts")
async def get_resource_alerts(
    current_user: User = Depends(get_current_active_user)
):
    """Get resource usage alerts"""
    monitoring_service = MonitoringService()
    alerts = await monitoring_service.check_resource_alerts()
    return {"alerts": alerts}


@router.get("/training")
async def get_training_resource_usage(
    current_user: User = Depends(get_current_active_user)
):
    """Get resource usage for training processes"""
    monitoring_service = MonitoringService()
    return await monitoring_service.get_training_resource_usage()


@router.get("/metrics")
async def get_prometheus_metrics():
    """Get Prometheus metrics (no authentication required for Prometheus scraping)"""
    monitoring_service = MonitoringService()
    await monitoring_service.update_training_metrics()
    metrics_data = monitoring_service.get_prometheus_metrics()
    return Response(content=metrics_data, media_type=CONTENT_TYPE_LATEST)
