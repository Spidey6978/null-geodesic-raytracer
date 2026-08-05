"""
Script: scripts/run_public_server.py
Launches the FastAPI raytracer service locally and opens a secure public HTTPS tunnel via ngrok.
Allows remote clients and web users to trigger black hole renders on your local hardware.
"""

import os
import sys
import time
import argparse
import uvicorn
from dotenv import load_dotenv

load_dotenv()


try:
    from pyngrok import ngrok
except ImportError:
    ngrok = None


def start_public_server(port: int = 8000, auth_token: str = None, domain: str = None):
    """Starts local Uvicorn server and establishes ngrok HTTPS tunnel."""
    if ngrok is None:
        print("❌ Error: 'pyngrok' is not installed. Install it via 'pip install pyngrok'.")
        sys.exit(1)

    token = auth_token or os.getenv("NGROK_AUTHTOKEN")
    if token:
        ngrok.set_auth_token(token)

    custom_domain = domain or os.getenv("NGROK_DOMAIN")
    connect_kwargs = {}
    if custom_domain:
        connect_kwargs["domain"] = custom_domain

    print("🚀 Opening public ngrok HTTPS tunnel...")
    try:
        public_url = ngrok.connect(port, "http", **connect_kwargs).public_url
        print("=" * 65)
        print(f"🌐 Public API Base URL:  {public_url}")
        print(f"🖥️ Public KERR-TRACE UI: {public_url}/ui")
        print(f"📖 Public Swagger Docs:  {public_url}/docs")
        print(f"🏥 Health Check URL:    {public_url}/api/v1/health")
        print("=" * 65)
    except Exception as e:
        print(f"⚠️ Could not establish ngrok tunnel: {e}")
        print("Starting server on localhost only...")
        public_url = f"http://127.0.0.1:{port}"

    print(f"\n⚡ Starting FastAPI server on http://127.0.0.1:{port}...")
    try:
        uvicorn.run("api.main:app", host="127.0.0.1", port=port, log_level="info")
    except KeyboardInterrupt:
        print("\n🛑 Shutting down server and closing ngrok tunnel...")
    finally:
        try:
            ngrok.disconnect(public_url)
            ngrok.kill()
            print("✅ ngrok tunnel closed.")
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Null Geodesic Raytracer Public API Server")
    parser.add_argument("--port", type=int, default=8000, help="Local server port (default 8000)")
    parser.add_argument("--token", type=str, default=None, help="Optional ngrok auth token")
    parser.add_argument("--domain", type=str, default=None, help="Optional static/custom ngrok domain name")
    args = parser.parse_args()

    start_public_server(port=args.port, auth_token=args.token, domain=args.domain)
