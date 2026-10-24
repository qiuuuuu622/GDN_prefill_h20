#!/usr/bin/env python3
"""Clear every registered Dynamo SGLang worker's Radix/KV cache."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

from dynamo.runtime import DistributedRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--namespace",
        default=os.environ.get("DYN_NAMESPACE", "qwen3-6"),
    )
    parser.add_argument("--component", default="backend")
    parser.add_argument("--expected-workers", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def response_data(response: object) -> object:
    data_method = getattr(response, "data", None)
    raw = data_method() if callable(data_method) else str(response)
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


async def clear(args: argparse.Namespace) -> None:
    loop = asyncio.get_running_loop()
    runtime = DistributedRuntime(
        loop,
        os.environ.get("DYN_DISCOVERY_BACKEND", "file"),
        os.environ.get("DYN_REQUEST_PLANE", "tcp"),
        event_plane=os.environ.get("DYN_EVENT_PLANE", "zmq"),
    )

    try:
        endpoint = runtime.endpoint(
            f"{args.namespace}.{args.component}.clear_kv_blocks"
        )
        client = await endpoint.client()
        deadline = time.monotonic() + args.timeout
        instance_ids: list[int] = []
        while time.monotonic() < deadline:
            instance_ids = list(client.instance_ids())
            if len(instance_ids) >= args.expected_workers:
                break
            await asyncio.sleep(1)

        if len(instance_ids) < args.expected_workers:
            raise RuntimeError(
                f"found {len(instance_ids)} clear_kv_blocks instances; "
                f"expected at least {args.expected_workers}"
            )

        failures: list[str] = []
        for instance_id in sorted(instance_ids):
            stream = await client.direct(json.dumps({}), instance_id)
            responses = [response_data(item) async for item in stream]
            print(
                json.dumps(
                    {"instance_id": instance_id, "responses": responses},
                    ensure_ascii=False,
                )
            )
            if not responses:
                failures.append(f"instance {instance_id}: empty response")
                continue
            for response in responses:
                if isinstance(response, dict) and response.get("status") in {
                    "error",
                    "failed",
                }:
                    failures.append(f"instance {instance_id}: {response}")

        if failures:
            raise RuntimeError("; ".join(failures))
    finally:
        runtime.shutdown()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(clear(args))
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to clear Dynamo SGLang KV cache: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
