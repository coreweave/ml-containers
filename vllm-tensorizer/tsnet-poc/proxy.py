#!/usr/bin/env python3

import argparse
import asyncio
import os
from collections.abc import Coroutine
from contextlib import suppress
from ipaddress import IPv4Address, ip_address
from pathlib import Path
from typing import Any

import tailscale

BUFFER_SIZE = 64 * 1024
EXPERIMENT_ACKNOWLEDGEMENT = "this_is_unstable_software"
_connection_tasks: set[asyncio.Task[None]] = set()


def _spawn(coro: Coroutine[Any, Any, None]) -> None:
    task = asyncio.create_task(coro)
    _connection_tasks.add(task)
    task.add_done_callback(_connection_tasks.discard)


async def _send_all(tailnet_stream: Any, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        sent = await tailnet_stream.send(payload[offset:])
        if sent <= 0:
            raise ConnectionError("tailnet stream stopped accepting bytes")
        offset += sent


async def _copy_local_to_tailnet(
    reader: asyncio.StreamReader, tailnet_stream: Any
) -> None:
    while payload := await reader.read(BUFFER_SIZE):
        await _send_all(tailnet_stream, payload)


async def _copy_tailnet_to_local(
    tailnet_stream: Any, writer: asyncio.StreamWriter
) -> None:
    while True:
        payload = await tailnet_stream.recv()
        if not payload:
            return
        writer.write(payload)
        await writer.drain()


async def _bridge(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    tailnet_stream: Any,
) -> None:
    pumps = {
        asyncio.create_task(_copy_local_to_tailnet(reader, tailnet_stream)),
        asyncio.create_task(_copy_tailnet_to_local(tailnet_stream, writer)),
    }

    try:
        _, pending = await asyncio.wait(pumps, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pumps, return_exceptions=True)
    finally:
        writer.close()
        with suppress(Exception):
            await writer.wait_closed()


async def _connect_device(state_file: str, hostname: str, tags: list[str]) -> Any:
    state_path = Path(state_file)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if state_path.exists():
        state_path.chmod(0o600)

    auth_key = os.environ.get("TS_AUTH_KEY")
    if auth_key is None:
        if state_path.exists():
            print(f"Using persisted Tailscale state from {state_path}.", flush=True)
        else:
            print(
                "Using interactive Tailscale authentication. "
                "Open the auth_url emitted below on the first run.",
                flush=True,
            )

    device = await tailscale.connect(
        str(state_path),
        auth_key,
        hostname=hostname,
        tags=tags or None,
    )
    state_path.chmod(0o600)
    return device


async def _serve_tailnet_connection(
    tailnet_stream: Any, backend_host: str, backend_port: int
) -> None:
    try:
        reader, writer = await asyncio.open_connection(backend_host, backend_port)
    except (OSError, ValueError) as error:
        print(f"Backend connection failed: {error}", flush=True)
        return

    await _bridge(reader, writer, tailnet_stream)


async def _run_server(args: argparse.Namespace) -> None:
    device = await _connect_device(args.state_file, args.hostname, args.tag)
    tailnet_ip = await device.ipv4_addr()
    listener = await device.tcp_listen((tailnet_ip, args.tailnet_port))

    print(
        f"Tailnet listener ready at {tailnet_ip}:{args.tailnet_port}; "
        f"forwarding to {args.backend_host}:{args.backend_port}",
        flush=True,
    )

    while True:
        tailnet_stream = await listener.accept()
        _spawn(
            _serve_tailnet_connection(
                tailnet_stream, args.backend_host, args.backend_port
            )
        )


async def _resolve_peer_ipv4(device: Any, peer_name: str) -> IPv4Address:
    peer = await device.peer_by_name(peer_name)
    if peer is None:
        raise RuntimeError(f"tailnet peer not found: {peer_name}")

    for address in peer["tailnet_addresses"]:
        parsed = ip_address(str(address))
        if isinstance(parsed, IPv4Address):
            return parsed

    raise RuntimeError(f"tailnet peer has no IPv4 address: {peer_name}")


async def _serve_local_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    device: Any,
    target_ip: IPv4Address,
    target_port: int,
) -> None:
    try:
        tailnet_stream = await device.tcp_connect((target_ip, target_port))
    except (OSError, ValueError) as error:
        print(f"Tailnet connection failed: {error}", flush=True)
        writer.close()
        with suppress(Exception):
            await writer.wait_closed()
        return

    await _bridge(reader, writer, tailnet_stream)


async def _run_client(args: argparse.Namespace) -> None:
    device = await _connect_device(args.state_file, args.hostname, args.tag)
    if args.target_ip is not None:
        parsed_target = ip_address(args.target_ip)
        if not isinstance(parsed_target, IPv4Address):
            raise ValueError("--target-ip must be an IPv4 address")
        target_ip = parsed_target
    else:
        target_ip = await _resolve_peer_ipv4(device, args.target_name)

    def accept_local(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        _spawn(
            _serve_local_connection(reader, writer, device, target_ip, args.target_port)
        )

    server = await asyncio.start_server(accept_local, args.local_host, args.local_port)
    print(
        f"Local listener ready at {args.local_host}:{args.local_port}; "
        f"forwarding to {target_ip}:{args.target_port} over the tailnet",
        flush=True,
    )

    async with server:
        await server.serve_forever()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bridge local TCP sockets and tailscale-py TCP streams"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    server = subparsers.add_parser("server", help="Expose a local server")
    server.add_argument("--state-file", default=".state/server.json")
    server.add_argument("--hostname", default="vllm-poc-server")
    server.add_argument("--tag", action="append", default=[])
    server.add_argument("--tailnet-port", type=int, default=8080)
    server.add_argument("--backend-host", default="127.0.0.1")
    server.add_argument("--backend-port", type=int, default=8000)
    server.set_defaults(run=_run_server)

    client = subparsers.add_parser("client", help="Create a local forwarder")
    client.add_argument("--state-file", default=".state/client.json")
    client.add_argument("--hostname", default="vllm-poc-client")
    client.add_argument("--tag", action="append", default=[])
    client.add_argument("--local-host", default="127.0.0.1")
    client.add_argument("--local-port", type=int, default=18080)
    target = client.add_mutually_exclusive_group(required=True)
    target.add_argument("--target-name")
    target.add_argument("--target-ip")
    client.add_argument("--target-port", type=int, default=8080)
    client.set_defaults(run=_run_client)

    return parser


def main() -> None:
    os.environ.setdefault("TS_RS_EXPERIMENT", EXPERIMENT_ACKNOWLEDGEMENT)
    args = _parser().parse_args()
    try:
        asyncio.run(args.run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
