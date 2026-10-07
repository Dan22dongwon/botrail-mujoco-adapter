"""botrail 스튜디오(botrail 자체 뷰어) 화면을 영상으로 녹화한다 — 갤러리의 'botrail' 모드는 이 영상을 재생.

  env -u PYTHONPATH <botrail venv python> tools/botrail_studio_record.py EXAMPLE.py OUT.mp4 [--view-from OUT/mj]

- 예제를 어댑터와 같은 방식(capture)으로 실행해 셀·시퀀스를 얻고, 일괄 베이크한 타임라인을 스튜디오에 올린다.
  시퀀스가 없는 예제는 셀 화면만 (궤도 회전) 녹화.
- 헤드리스 Chrome 으로 스튜디오를 열고 사이드 패널·안내 카드·레인을 숨긴 뒤 0초부터 실시간 재생을 스크린캐스트.
- 카메라는 botrail_mujoco.camera 의 공통 구도 (Isaac RTX · MuJoCo 영상과 같은 시점).
"""
import argparse
import asyncio
import base64
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# NVIDIA GPU 의 Vulkan 으로 WebGL (소프트웨어 렌더 ~7 fps → 60 fps). BOTRAIL_REC_SOFTWARE=1 이면 SwiftShader
GPU_FLAGS = (["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] if os.environ.get("BOTRAIL_REC_SOFTWARE")
             else ["--use-angle=vulkan", "--enable-features=Vulkan", "--ignore-gpu-blocklist"])
HIDE_CSS = (".panel,.focus-card{display:none!important}"
            ".viewport{width:100%!important;flex:1 1 auto!important}")


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def find_chrome():
    for c in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        if shutil.which(c):
            return shutil.which(c)
    raise SystemExit("Chrome/Chromium 이 필요합니다")


async def record(url, out, max_seconds, size, fps, load_wait=30):
    """반환: 녹화 길이(s). 타임라인이 있으면 스튜디오의 'cycle N s' 길이만큼, 없으면 정지 화면 6초."""
    import websockets
    W, H = size
    port = free_port()
    prof = tempfile.mkdtemp(prefix="studio-rec-")
    chrome = subprocess.Popen([find_chrome(), "--headless=new", f"--remote-debugging-port={port}", f"--user-data-dir={prof}",
                               *GPU_FLAGS, f"--window-size={W},{H}",
                               "--hide-scrollbars", "--mute-audio", "about:blank"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                tgt = json.load(urllib.request.urlopen(urllib.request.Request(
                    f"http://127.0.0.1:{port}/json/new?about:blank", method="PUT")))
                break
            except OSError:
                time.sleep(0.2)
        async with websockets.connect(tgt["webSocketDebuggerUrl"], max_size=200_000_000) as ws:
            i = 0
            frames = []

            async def cmd(m, p=None):
                nonlocal i
                i += 1
                my = i
                await ws.send(json.dumps({"id": my, "method": m, "params": p or {}}))
                while True:
                    x = json.loads(await ws.recv())
                    if x.get("method") == "Page.screencastFrame":
                        prm = x["params"]
                        frames.append((prm["metadata"]["timestamp"], base64.b64decode(prm["data"])))
                        await ws.send(json.dumps({"id": 10 ** 9 + prm["sessionId"], "method": "Page.screencastFrameAck",
                                                  "params": {"sessionId": prm["sessionId"]}}))
                    if x.get("id") == my:
                        return x.get("result", x)

            async def ev(e):
                r = await cmd("Runtime.evaluate", {"expression": e, "returnByValue": True, "awaitPromise": True})
                return r.get("result", {}).get("value")

            # 스크린캐스트는 창의 실제 보이는 영역을 찍는다 → 창 크기를 맞춰 안쪽이 정확히 W×H 가 되게
            win = (await cmd("Browser.getWindowForTarget"))["windowId"]
            for _ in range(3):
                iw, ih = await ev("[innerWidth, innerHeight]")
                if (iw, ih) == (W, H):
                    break
                b = (await cmd("Browser.getWindowBounds", {"windowId": win}))["bounds"]
                await cmd("Browser.setWindowBounds", {"windowId": win, "bounds": {
                    "width": b["width"] + W - iw, "height": b["height"] + H - ih}})
                await asyncio.sleep(0.3)
            await cmd("Page.enable")
            await cmd("Page.navigate", {"url": url})
            for _ in range(120):  # 장면·베이크 로드 대기
                await asyncio.sleep(0.5)
                if await ev("!!document.querySelector('canvas') && /Connected/.test(document.body.innerText)"):
                    break
            for _ in range(int(load_wait)):  # 무거운 셀은 타임라인(베이크)이 늦게 붙는다
                await asyncio.sleep(1)
                if await ev("/(cycle|physics)\\s*[0-9.]+\\s*s/.test(document.querySelector('.timeline-dock')?.innerText||'')"):
                    break
            await asyncio.sleep(1)
            cyc = await ev("(()=>{const m=/(?:cycle|physics)\\s*([0-9.]+)\\s*s/.exec(document.querySelector('.timeline-dock')?.innerText||'');return m?+m[1]:0})()")
            has_timeline = bool(cyc)
            duration = min(cyc, max_seconds) if has_timeline else 6.0
            await ev("(()=>{const s=document.createElement('style');s.textContent=%s;document.head.appendChild(s);"
                     "const l=[...document.querySelectorAll('.timeline-button')].find(b=>b.classList.contains('timeline-button-on')"
                     "&&/lanes/.test(b.textContent));if(l)l.click();dispatchEvent(new Event('resize'));})()" % json.dumps(HIDE_CSS))
            await asyncio.sleep(1.5)
            if has_timeline:  # 0초로 (첫 구간 클릭 = 그 시점으로 이동·일시정지) → 재생
                # 구간 막대가 있으면 첫 구간, 없으면(USD 녹화 재생 등) 도크 아래 막대의 왼쪽 끝을 눌러 0초로
                await ev("(()=>{let b=document.querySelector('.timeline-band'),x,y;"
                         "if(b){const r=b.getBoundingClientRect();x=r.x+1;y=r.y+r.height/2}"
                         "else{const d=document.querySelector('.timeline-dock').getBoundingClientRect();x=d.x+14;y=d.bottom-14;"
                         "b=document.elementFromPoint(x,y)}"
                         "for(const ty of ['pointerdown','mousedown','pointerup','mouseup','click'])"
                         "b.dispatchEvent(new MouseEvent(ty,{bubbles:true,clientX:x,clientY:y}));})()")
                await asyncio.sleep(1.0)
            await cmd("Page.startScreencast", {"format": "jpeg", "quality": 82, "maxWidth": W, "maxHeight": H, "everyNthFrame": 1})
            await asyncio.sleep(0.5)
            if has_timeline:
                await ev("(()=>{const p=document.querySelector('.timeline-button');if(/▶/.test(p.textContent))p.click();})()")
            t0 = time.time()  # 재생 시작 = 영상 0초
            while time.time() - t0 < duration + 0.5:
                await cmd("Runtime.evaluate", {"expression": "1"})  # 프레임 이벤트 펌프
                await asyncio.sleep(0.05)
            await cmd("Page.stopScreencast")
    finally:
        chrome.terminate()
        shutil.rmtree(prof, ignore_errors=True)
    if os.environ.get("BOTRAIL_REC_DEBUG"):
        Path("/tmp/claude-1000/-home-dan/df70f8c9-be13-4fcd-9407-3878fd68b1bb/scratchpad/rawframe.jpg").write_bytes(frames[len(frames) // 2][1])
    if len(frames) < 2:
        raise SystemExit("스크린캐스트 프레임이 없습니다")
    # 가변 간격 프레임 → 고정 fps 영상 (각 프레임을 실제 표시 시간만큼)
    tmp = Path(tempfile.mkdtemp(prefix="studio-frames-"))
    frames = [f for f in frames if f[0] <= t0][-1:] + [f for f in frames if f[0] > t0]  # 재생 시작 직전 프레임부터
    frames[0] = (t0, frames[0][1])
    ts0 = t0
    lst = []
    for k, (ts, jpg) in enumerate(frames):
        f = tmp / f"{k:06d}.jpg"
        f.write_bytes(jpg)
        nxt = frames[k + 1][0] if k + 1 < len(frames) else ts + 1 / fps
        lst.append(f"file '{f}'\nduration {max(nxt - ts, 1e-3):.4f}")
    lst.append(f"file '{tmp / f'{len(frames) - 1:06d}.jpg'}'")
    (tmp / "list.txt").write_text("\n".join(lst))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(tmp / "list.txt"),
                    "-t", f"{duration:.3f}", "-vf", f"fps={fps},scale={W}:{H}:force_original_aspect_ratio=decrease,"
                    f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "28",
                    "-preset", "veryfast", "-movflags", "+faststart", str(out)], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"wrote {out} · {len(frames)} frames over {frames[-1][0] - ts0:.1f}s", flush=True)
    return duration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("example")
    ap.add_argument("out")
    ap.add_argument("--view-from", default=None, help="결과 폴더 (view.json / cell_physics.usda) — 다른 엔진 영상과 같은 구도")
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--max-seconds", type=float, default=600.0)
    ap.add_argument("--load-wait", type=float, default=30.0, help="타임라인이 붙기를 기다리는 최대 초")
    ap.add_argument("--bake", action="store_true", help="예제의 studio 호출 대신 시퀀스를 일괄 베이크한 타임라인으로")
    a = ap.parse_args()
    import botrail as bt
    from botrail_mujoco.capture import studio_scene_from_script

    a.out, a.example = str(Path(a.out).resolve()), str(Path(a.example).resolve())
    a.view_from = a.view_from and str(Path(a.view_from).resolve())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    os.chdir(Path(a.out).parent)  # 예제가 남기는 산출물은 출력 폴더로
    cap = studio_scene_from_script(a.example, prefer_sequences=a.bake)
    scene = cap.scene
    view = ""
    if a.view_from and (Path(a.view_from) / "cell_physics.usda").exists():
        from botrail_mujoco.camera import view_for
        eye, target = view_for(a.view_from)
        view = "?view=" + ",".join(str(v) for v in list(eye) + list(target))
    elif cap.view:
        view = "?view=" + ",".join(str(float(v)) for p in cap.view for v in p)
    srv = bt.studio(scene, port=free_port(), open_browser=False, block=False)
    try:
        duration = asyncio.run(record(srv.url + view, Path(a.out).resolve(), a.max_seconds,
                                      tuple(map(int, a.size.split("x"))), a.fps, a.load_wait))
    finally:
        srv.stop()
    Path(a.out).with_suffix(".json").write_text(json.dumps(dict(duration=duration, source=cap.source, fps=a.fps)))


if __name__ == "__main__":
    main()
