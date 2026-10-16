"""Capture screenshots of the running Streamlit app with headless Chrome or Edge.

The app must be running (``streamlit run app/streamlit_app.py``) with Neo4j populated.
Pages are opened through URL parameters that pre-fill their inputs, so the screenshots
are reproducible. The browser is driven through the Chrome DevTools Protocol so that the
script can wait until Streamlit has finished running before capturing. A temporary
browser profile is used.

Usage::

    python scripts/capture_screenshots.py --browser "<path to chrome or msedge>"
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets

PAGES = {
    "document_analysis": "?sample=1",
    "relation_extraction": "relations?sample=2",
    "knowledge_graph": "graph",
    "graph_exploration": "exploration?explore=conditional%20random%20field",
    "scientific_queries": "queries?query=methods_for_task&task=translation&run=1",
    "model_evaluation": "evaluation",
}

# True while Streamlit shows its "running" indicator.
RUNNING_JS = "!!document.querySelector('[data-testid=\"stStatusWidget\"]')"


class Cdp:
    def __init__(self, ws):
        self.ws = ws
        self.next_id = 0

    async def call(self, method: str, **params):
        self.next_id += 1
        msg_id = self.next_id
        await self.ws.send(json.dumps({"id": msg_id, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == msg_id:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg.get("result", {})


async def capture(port: int, base_url: str, out: Path, width: int, height: int, settle: float):
    targets = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/json").read())
    page = next(t for t in targets if t["type"] == "page")
    async with websockets.connect(page["webSocketDebuggerUrl"], max_size=None) as ws:
        cdp = Cdp(ws)
        await cdp.call("Page.enable")
        await cdp.call(
            "Emulation.setDeviceMetricsOverride",
            width=width,
            height=height,
            deviceScaleFactor=1,
            mobile=False,
        )
        for name, path in PAGES.items():
            await cdp.call("Page.navigate", url=f"{base_url}/{path}")
            await asyncio.sleep(settle)
            deadline = time.time() + 180
            while time.time() < deadline:
                res = await cdp.call("Runtime.evaluate", expression=RUNNING_JS)
                if not res["result"].get("value"):
                    break
                await asyncio.sleep(1)
            await asyncio.sleep(2)
            shot = await cdp.call("Page.captureScreenshot", format="png")
            target = out / f"{name}.png"
            target.write_bytes(base64.b64decode(shot["data"]))
            print("captured", target)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--browser", required=True)
    p.add_argument("--base-url", default="http://localhost:8501")
    p.add_argument("--out", type=Path, default=Path("docs/images"))
    p.add_argument("--width", type=int, default=1400)
    p.add_argument("--height", type=int, default=1100)
    p.add_argument("--settle", type=float, default=6.0, help="seconds before polling")
    p.add_argument("--port", type=int, default=9333)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as profile:
        proc = subprocess.Popen(
            [
                args.browser,
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--no-first-run",
                "--no-default-browser-check",
                f"--user-data-dir={profile}",
                f"--remote-debugging-port={args.port}",
                "about:blank",
            ]
        )
        try:
            for _ in range(50):
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{args.port}/json/version")
                    break
                except OSError:
                    time.sleep(0.2)
            asyncio.run(
                capture(args.port, args.base_url, args.out, args.width, args.height, args.settle)
            )
        finally:
            proc.terminate()
            proc.wait(timeout=30)


if __name__ == "__main__":
    main()
