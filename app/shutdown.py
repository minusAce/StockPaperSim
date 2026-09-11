from __future__ import annotations

import asyncio
import json
import logging
import os

import httpx

logger = logging.getLogger(__name__)
DOCKER_SOCKET = "/var/run/docker.sock"
KILL_LABEL = "stocksim.kill-group=true"


def _docker_client() -> httpx.AsyncClient:
    # Docker's HTTP API replies with Transfer-Encoding: chunked even for a
    # small, one-shot JSON GET over the Unix socket. A hand-rolled reader that
    # just looks for the header/body "\r\n\r\n" boundary treats the chunk
    # framing (size line + trailing "0\r\n\r\n") as part of the body, so
    # json.loads on it always raises. httpx's Unix Domain Socket transport
    # speaks real HTTP/1.1, chunked encoding included, so use it instead of
    # talking to the socket by hand.
    transport = httpx.AsyncHTTPTransport(uds=DOCKER_SOCKET)
    return httpx.AsyncClient(transport=transport, base_url="http://docker", timeout=5.0)


async def _container_ids(client: httpx.AsyncClient) -> list[str]:
    response = await client.get(
        "/containers/json",
        params={"all": "0", "limit": "0", "filters": json.dumps({"label": [KILL_LABEL]})},
    )
    response.raise_for_status()
    return [str(item.get("Id")) for item in response.json() if item.get("Id")]


async def _stop_container(client: httpx.AsyncClient, container_id: str) -> None:
    response = await client.post(f"/containers/{container_id}/stop", params={"t": "5"})
    if response.status_code not in (204, 304):  # 304 == already stopped; both are fine.
        response.raise_for_status()


async def shutdown_docker_stack() -> None:
    """Stop all running containers marked as part of the StockPaperSim stack."""
    if os.path.exists(DOCKER_SOCKET):
        try:
            async with _docker_client() as client:
                ids = await _container_ids(client)
                if ids:
                    await asyncio.gather(*(_stop_container(client, cid) for cid in ids))
                    return
        except Exception:
            logger.exception("Docker stack shutdown via docker.sock failed")
    # Fallback for non-Docker runs: terminate the current application process.
    os._exit(0)
