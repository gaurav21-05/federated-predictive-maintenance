"""
Industrial IoT Federated Predictive Maintenance Dashboard Module
"""

from .service import LiveDashboardService, DashboardService
from .app import app

__all__ = ["LiveDashboardService", "DashboardService", "app"]
