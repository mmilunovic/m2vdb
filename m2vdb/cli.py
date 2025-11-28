"""
Command-line interface for m2vdb server.

Usage:
    m2vdb-server
    m2vdb-server --port 8080
    m2vdb-server --data-dir /var/lib/m2vdb --reload
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    """Main CLI entry point for m2vdb server."""
    parser = argparse.ArgumentParser(
        prog="m2vdb-server",
        description="Start the m2vdb vector database server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind the server to"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory for persistent storage (default: in-memory only)"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code changes (development only)"
    )
    
    args = parser.parse_args()
    
    # Set environment variable for persistence
    if args.data_dir:
        data_dir = Path(args.data_dir).expanduser().resolve()
        os.environ["M2VDB_DATA_DIR"] = str(data_dir)
        print(f"Using persistent storage: {data_dir}")
    
    # Import uvicorn here to avoid slow startup for --help
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is required to run the server", file=sys.stderr)
        sys.exit(1)
    
    # Display startup info
    print(f"Starting m2vdb server on {args.host}:{args.port}")
    if args.reload:
        print("⚠️  Auto-reload enabled (development mode)")
    
    # Run the server
    uvicorn.run(
        "m2vdb.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
