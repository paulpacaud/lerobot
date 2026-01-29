# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Fake PointAct Policy Server for Testing

A standalone ZeroMQ server that mimics the PointAct policy server interface.
Returns the current state as action (robot stays still = safe for testing).

Usage:
```bash
python -m lerobot.async_inference.fake_pointact_server --port=17000 --chunk_size=50

# With verbose logging
python -m lerobot.async_inference.fake_pointact_server --port=17000 --verbose
```

The server:
- Listens on ZMQ REP socket at specified port
- Handles `get_action` endpoint (same protocol as real server)
- Returns action = current state repeated chunk_size times (robot won't move)
- Prints received observation summary
"""

import logging
import time
from typing import Any

import msgpack
import msgpack_numpy
import numpy as np
import zmq
from tap import Tap

msgpack_numpy.patch()

# Default configuration
DEFAULT_PORT = 17000
DEFAULT_CHUNK_SIZE = 50
DEFAULT_ACTION_DIM = 6  # Joint positions for SO100/SO101


class Args(Tap):
    """Arguments for fake PointAct server."""

    port: int = DEFAULT_PORT  # Port to listen on
    chunk_size: int = DEFAULT_CHUNK_SIZE  # Number of actions in each chunk
    action_dim: int = DEFAULT_ACTION_DIM  # Dimension of each action
    verbose: bool = False  # Enable verbose logging of observations
    add_noise: bool = False  # Add small noise to actions (for testing)
    noise_std: float = 0.01  # Standard deviation of noise if enabled


class FakePointActServer:
    """Fake PointAct server for testing robot client."""

    def __init__(
        self,
        port: int = DEFAULT_PORT,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        action_dim: int = DEFAULT_ACTION_DIM,
        verbose: bool = False,
        add_noise: bool = False,
        noise_std: float = 0.01,
    ):
        self.port = port
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.verbose = verbose
        self.add_noise = add_noise
        self.noise_std = noise_std

        self.logger = logging.getLogger("fake_pointact_server")

        # ZeroMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.address = f"tcp://*:{port}"

        # Stats
        self.request_count = 0
        self.start_time = None

    def start(self) -> None:
        """Start the server and begin listening for requests."""
        self.socket.bind(self.address)
        self.start_time = time.time()
        self.logger.info(f"Fake PointAct server listening on port {self.port}")
        self.logger.info(f"  Chunk size: {self.chunk_size}")
        self.logger.info(f"  Action dim: {self.action_dim}")
        self.logger.info(f"  Add noise: {self.add_noise}")
        if self.add_noise:
            self.logger.info(f"  Noise std: {self.noise_std}")
        self.logger.info("Waiting for requests... (Press Ctrl+C to stop)")

    def handle_request(self, request: dict) -> dict:
        """Handle a single request from the client.

        Args:
            request: Deserialized request dict

        Returns:
            Response dict with action
        """
        endpoint = request.get("endpoint")
        data = request.get("data", {})

        if endpoint == "get_action":
            return self._handle_get_action(data)
        else:
            self.logger.warning(f"Unknown endpoint: {endpoint}")
            return {"error": f"Unknown endpoint: {endpoint}"}

    def _handle_get_action(self, data: dict) -> dict:
        """Handle get_action endpoint.

        Args:
            data: Request data containing batch and options

        Returns:
            Response dict with action chunk
        """
        batch = data.get("batch", {})

        # Extract state from batch
        state_list = batch.get("observation.state", [])
        if state_list and len(state_list) > 0:
            state = np.array(state_list[0], dtype=np.float32)
        else:
            state = np.zeros(self.action_dim, dtype=np.float32)

        # Log observation info
        self._log_observation(batch)

        # Generate action: repeat current state chunk_size times
        # This means the robot will stay still (safe for testing)
        action = np.tile(state[:self.action_dim], (self.chunk_size, 1))

        # Optionally add noise
        if self.add_noise:
            noise = np.random.normal(0, self.noise_std, action.shape).astype(np.float32)
            action = action + noise

        return {"action": action}

    def _log_observation(self, batch: dict) -> None:
        """Log information about the received observation."""
        self.request_count += 1

        # Basic logging
        state = batch.get("observation.state", [])
        state_arr = np.array(state[0]) if state else None
        image = batch.get("observation.images.front_image", [])
        image_arr = np.array(image[0]) if image else None
        points = batch.get("observation.points", [])
        points_arr = np.array(points[0]) if points else None
        task = batch.get("task", [""])[0] if batch.get("task") else ""

        state_str = f"[{', '.join(f'{v:.2f}' for v in state_arr)}]" if state_arr is not None else "None"
        image_str = f"{image_arr.shape}" if image_arr is not None else "None"
        points_str = f"{points_arr.shape[0]} pts" if points_arr is not None and len(points_arr) > 0 else "None"

        self.logger.info(
            f"Request #{self.request_count} | "
            f"State: {state_str} | "
            f"Image: {image_str} | "
            f"Points: {points_str}"
        )

        # Verbose logging
        if self.verbose:
            self.logger.info(f"  Task: '{task}'")
            if state_arr is not None:
                self.logger.info(f"  State shape: {state_arr.shape}, dtype: {state_arr.dtype}")
            if image_arr is not None:
                self.logger.info(f"  Image shape: {image_arr.shape}, dtype: {image_arr.dtype}")
                self.logger.info(f"  Image range: [{image_arr.min()}, {image_arr.max()}]")
            if points_arr is not None and len(points_arr) > 0:
                self.logger.info(f"  Points shape: {points_arr.shape}")
                self.logger.info(
                    f"  Points X range: [{points_arr[:, 0].min():.3f}, {points_arr[:, 0].max():.3f}]"
                )
                self.logger.info(
                    f"  Points Y range: [{points_arr[:, 1].min():.3f}, {points_arr[:, 1].max():.3f}]"
                )
                self.logger.info(
                    f"  Points Z range: [{points_arr[:, 2].min():.3f}, {points_arr[:, 2].max():.3f}]"
                )

    def run(self) -> None:
        """Run the server main loop."""
        self.start()

        try:
            while True:
                # Wait for request
                request_bytes = self.socket.recv()

                # Deserialize request
                request = msgpack.unpackb(request_bytes, raw=False)

                # Handle request
                response = self.handle_request(request)

                # Serialize and send response
                response_bytes = msgpack.packb(response)
                self.socket.send(response_bytes)

        except KeyboardInterrupt:
            self.logger.info("\nServer shutting down...")
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the server and clean up resources."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        self.logger.info(f"Processed {self.request_count} requests in {elapsed:.1f}s")
        if self.request_count > 0 and elapsed > 0:
            self.logger.info(f"Average rate: {self.request_count / elapsed:.2f} req/s")

        self.socket.close()
        self.context.term()
        self.logger.info("Server stopped")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    args = Args().parse_args()

    server = FakePointActServer(
        port=args.port,
        chunk_size=args.chunk_size,
        action_dim=args.action_dim,
        verbose=args.verbose,
        add_noise=args.add_noise,
        noise_std=args.noise_std,
    )

    server.run()


if __name__ == "__main__":
    main()
