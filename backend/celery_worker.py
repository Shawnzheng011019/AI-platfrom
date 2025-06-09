#!/usr/bin/env python3
"""
Celery Worker Entry Point
"""

from app.core.celery import celery_app

if __name__ == '__main__':
    celery_app.start()
