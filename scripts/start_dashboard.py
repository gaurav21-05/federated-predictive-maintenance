"""
CLI Launcher for Industrial IoT Federated Predictive Maintenance Dashboard
Starts the FastAPI backend and serves the interactive web application.
"""

import argparse
import sys
from pathlib import Path
import uvicorn

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Start Industrial IoT Predictive Maintenance Dashboard")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    args = parser.parse_args()

    print("=" * 72)
    print("  INDUSTRIAL IoT FEDERATED PREDICTIVE MAINTENANCE DASHBOARD")
    print("=" * 72)
    print(f"  [+] Local URL: http://{args.host}:{args.port}")
    print(f"  [+] API Docs:  http://{args.host}:{args.port}/docs")
    print(f"  [+] Serving static UI from src/dashboard/static")
    print("=" * 72)

    uvicorn.run(
        "src.dashboard.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
