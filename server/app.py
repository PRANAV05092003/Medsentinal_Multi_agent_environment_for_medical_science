"""
MedSentinel OpenEnv Server
===========================

FastAPI application built with openenv-core's create_app().
Exposes /reset, /step, /state, /schema, /ws endpoints.

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 7860

HuggingFace Spaces:
    The openenv.yaml points to this file as the app entry point.
    HF Spaces runs on port 7860 by default.
"""

from __future__ import annotations

try:
    from openenv.core.env_server.http_server import create_app
except ImportError as e:
    raise ImportError(
        "openenv-core is required. Install with: pip install openenv-core"
    ) from e

try:
    from ..models import MedSentinelAction, MedSentinelObservation
    from .medsentinel_environment import MedSentinelEnvironment
except ImportError:
    from models import MedSentinelAction, MedSentinelObservation
    from server.medsentinel_environment import MedSentinelEnvironment


# create_app wires up all OpenEnv-standard endpoints:
#   POST /reset          -> environment.reset()
#   POST /step           -> environment.step(action)
#   GET  /state          -> environment.state
#   GET  /schema         -> action + observation JSON schemas
#   WS   /ws             -> WebSocket persistent session
app = create_app(
    MedSentinelEnvironment,
    MedSentinelAction,
    MedSentinelObservation,
    env_name="medsentinel",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    main(host=args.host, port=args.port)
