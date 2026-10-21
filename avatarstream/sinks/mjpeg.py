"""A small local web server: live preview for OBS and a status page.

    /            full-window picture, add it in OBS as a Browser Source
    /status      latency, level meter and the OBS sync offset to use
    /stream.mjpg the raw MJPEG stream
    /stats       the same numbers as JSON
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Callable, Optional

import cv2
from aiohttp import web

log = logging.getLogger(__name__)

BOUNDARY = "frame"

VIEW_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>AIAvatarStream</title>
<style>html,body{margin:0;height:100%;background:#000;overflow:hidden}
img{width:100%;height:100%;object-fit:contain;display:block}</style></head>
<body><img src="/stream.mjpg" alt=""></body></html>"""

STATUS_HTML = """<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>AIAvatarStream 状态</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{margin:0;font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif;background:#111518;color:#e6ecee}
main{max-width:980px;margin:0 auto;padding:20px 16px;display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:20px}
@media(max-width:760px){main{grid-template-columns:minmax(0,1fr)}}
img{width:100%;border-radius:8px;background:#000;display:block}
h1{font-size:18px;margin:0 0 12px}
dl{display:grid;grid-template-columns:auto 1fr;gap:6px 14px;margin:0}
dt{color:#8ea1a8}dd{margin:0;font-variant-numeric:tabular-nums;font-weight:600}
.meter{height:10px;background:#263035;border-radius:5px;overflow:hidden;margin:4px 0 14px}
.meter i{display:block;height:100%;background:#4cc38a;width:0;transition:width .1s}
.speak{display:inline-block;padding:0 10px;border-radius:99px;background:#263035}
.speak.on{background:#2d6a4f}
.tip{margin-top:16px;padding:12px;border-radius:8px;background:#1b2327;color:#b8c6cb;font-size:14px}
</style></head><body><main>
<img src="/stream.mjpg" alt="数字人画面预览">
<section><h1>直播状态</h1>
<div>输入音量 <span id="db">-</span> dB</div><div class="meter"><i id="bar"></i></div>
<dl>
<dt>说话</dt><dd><span id="spk" class="speak">否</span></dd>
<dt>输出帧率</dt><dd id="fps">-</dd>
<dt>平均延迟</dt><dd id="lat">-</dd>
<dt>延迟 P95</dt><dd id="p95">-</dd>
<dt>口型推理</dt><dd id="ifps">-</dd>
<dt>卡顿次数</dt><dd id="stalls">-</dd>
</dl>
<div class="tip">OBS 里如果直接采集变声器的声音，把那个音频源的“同步偏移”设成 <b id="off">-</b> 毫秒，声音就会和口型对上。</div>
</section></main>
<script>
async function tick(){try{const s=await (await fetch('/stats')).json();
const ms=v=>v==null?'-':v+' ms';
document.getElementById('db').textContent=s.level_db;
document.getElementById('bar').style.width=Math.max(0,Math.min(100,(s.level_db+70)/70*100))+'%';
const k=document.getElementById('spk');k.textContent=s.speaking?'是':'否';k.className='speak'+(s.speaking?' on':'');
document.getElementById('fps').textContent=s.fps;
document.getElementById('lat').textContent=ms(s.latency_ms);
document.getElementById('p95').textContent=ms(s.latency_p95_ms);
document.getElementById('ifps').textContent=s.infer_fps?s.infer_fps+' 帧/秒':'-';
document.getElementById('stalls').textContent=s.stalls;
document.getElementById('off').textContent=s.obs_sync_offset_ms==null?'-':s.obs_sync_offset_ms;
}catch(e){}}
setInterval(tick,500);tick();
</script></body></html>"""


class MJPEGSink:
    def __init__(self, host: str = "127.0.0.1", port: int = 8010, quality: int = 85,
                 stats: Optional[Callable[[], dict]] = None):
        self.host, self.port, self.quality = host, port, quality
        self.stats = stats or (lambda: {})
        self._jpeg: Optional[bytes] = None
        self._seq = 0
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._cond: Optional[asyncio.Condition] = None
        self._runner: Optional[web.AppRunner] = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._serve, name="mjpeg", daemon=True)

    # --- player side -------------------------------------------------------
    def send(self, frame) -> None:
        ok, buf = cv2.imencode(".jpg", frame.image, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            return
        self._jpeg = buf.tobytes()
        self._seq += 1
        if self._loop is not None:
            self._loop.call_soon_threadsafe(lambda: asyncio.ensure_future(self._notify()))

    async def _notify(self) -> None:
        async with self._cond:
            self._cond.notify_all()

    # --- server side -------------------------------------------------------
    async def _stream(self, request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(headers={
            "Content-Type": f"multipart/x-mixed-replace; boundary={BOUNDARY}",
            "Cache-Control": "no-cache, no-store",
        })
        await resp.prepare(request)
        last = -1
        try:
            while True:
                async with self._cond:
                    await self._cond.wait_for(lambda: self._seq != last)
                last, jpeg = self._seq, self._jpeg
                await resp.write(
                    f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n".encode()
                    + jpeg + b"\r\n"
                )
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        return resp

    async def _view(self, request: web.Request) -> web.Response:
        return web.Response(text=VIEW_HTML, content_type="text/html")

    async def _status(self, request: web.Request) -> web.Response:
        return web.Response(text=STATUS_HTML, content_type="text/html")

    async def _snapshot(self, request: web.Request) -> web.Response:
        if self._jpeg is None:
            return web.Response(status=503, text="还没有画面")
        return web.Response(body=self._jpeg, content_type="image/jpeg")

    async def _stats(self, request: web.Request) -> web.Response:
        return web.Response(text=json.dumps(self.stats(), ensure_ascii=False), content_type="application/json")

    def _serve(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._cond = asyncio.Condition()
        app = web.Application()
        app.router.add_get("/", self._view)
        app.router.add_get("/status", self._status)
        app.router.add_get("/stream.mjpg", self._stream)
        app.router.add_get("/snapshot.jpg", self._snapshot)
        app.router.add_get("/stats", self._stats)
        self._runner = web.AppRunner(app)
        self._loop.run_until_complete(self._runner.setup())
        self._loop.run_until_complete(web.TCPSite(self._runner, self.host, self.port).start())
        self._ready.set()
        self._loop.run_forever()

    def start(self) -> None:
        self._thread.start()
        if not self._ready.wait(timeout=10):
            raise RuntimeError(f"预览服务没能在 {self.host}:{self.port} 启动")
        log.info("预览：http://%s:%d/   状态：http://%s:%d/status", self.host, self.port, self.host, self.port)

    def close(self) -> None:
        if self._loop is None:
            return
        fut = asyncio.run_coroutine_threadsafe(self._runner.cleanup(), self._loop)
        try:
            fut.result(timeout=3)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=3)
