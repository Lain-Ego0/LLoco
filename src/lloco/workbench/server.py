"""Local HTTP boundary; training dependencies are loaded only in workers."""

import argparse
import functools
import json
import mimetypes
import signal
import socket
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .catalog import ALGORITHMS, TASK_ASSETS
from .models import Models
from .services import Jobs, command, inside

MODELS = Models()

STATIC = Path(__file__).with_name("static")


def artifacts(root):
  paths = [
    p
    for p in (root / "logs").rglob("*")
    if p.suffix in (".pt", ".onnx") and p.is_file() and p.resolve().is_relative_to(root)
  ]
  return [
    dict(
      path=str(p.relative_to(root)),
      kind=p.suffix[1:],
      size=p.stat().st_size,
      modified=p.stat().st_mtime,
    )
    for p in sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)[:300]
  ]


class Handler(SimpleHTTPRequestHandler):
  def __init__(self, *args, root: Path, jobs: Jobs, **kwargs):
    self.root, self.jobs = root, jobs
    super().__init__(*args, directory=str(STATIC), **kwargs)

  def end_headers(self):
    self.send_header("Cache-Control", "no-store")
    super().end_headers()

  def log_message(self, format, *args):
    pass

  def _json(self, value, status=200):
    data = json.dumps(value).encode()
    self.send_response(status)
    self.send_header("Content-Type", "application/json; charset=utf-8")
    self.send_header("Content-Length", str(len(data)))
    self.end_headers()
    self.wfile.write(data)

  def _file(self, file: Path, download=False):
    if not file.is_file():
      raise ValueError("文件不存在")
    self.send_response(200)
    self.send_header(
      "Content-Type", mimetypes.guess_type(file.name)[0] or "application/octet-stream"
    )
    self.send_header("Content-Length", str(file.stat().st_size))
    if download:
      self.send_header("Content-Disposition", "attachment")
    self.end_headers()
    with file.open("rb") as stream:
      import shutil

      shutil.copyfileobj(stream, self.wfile)

  def do_GET(self):  # noqa: N802
    url = urlparse(self.path)
    query = parse_qs(url.query)
    try:
      if url.path == "/api/state":
        from .motion_library import catalog

        library = catalog(self.root)
        motions = [entry["path"] for entry in library if entry["format"] == "npz"]
        self._json(
          dict(
            algorithms=ALGORITHMS,
            task_assets=TASK_ASSETS,
            motions=motions,
            motion_library=library,
            artifacts=artifacts(self.root),
            jobs=self.jobs.list(),
          )
        )
      elif url.path == "/api/motion-preview":
        from .motion_preview import read_preview

        identifier = query.get("id", [""])[0]
        job = next((j for j in self.jobs.list() if j["id"] == identifier), None)
        if not job or job["action"] not in (
          "gmr-retarget",
          "gmr-convert",
          "csv-convert",
        ):
          raise ValueError("未找到重定向任务")
        preview = read_preview(
          self.jobs.directory / "previews" / identifier,
          int(query.get("cursor", ["0"])[0]),
        )
        preview["job"] = job
        self._json(preview)
      elif url.path == "/api/file":
        path = inside(self.root, query.get("path", [""])[0])
        allowed = {
          ".xml",
          ".urdf",
          ".stl",
          ".obj",
          ".dae",
          ".mtl",
          ".png",
          ".jpg",
          ".jpeg",
          ".pt",
          ".onnx",
        }
        if path.suffix.lower() not in allowed:
          raise ValueError("不支持的文件类型")
        self._file(path, path.suffix in (".pt", ".onnx"))
      elif url.path.startswith("/api/log/"):
        self._json(dict(log=self.jobs.log(url.path.rsplit("/", 1)[1])))
      elif url.path.startswith("/api/"):
        self._json(dict(error="未知接口"), 404)
      else:
        target = (STATIC / unquote(url.path).lstrip("/")).resolve()
        if not target.is_relative_to(STATIC.resolve()):
          raise ValueError("无效路径")
        super().do_GET()
    except (ValueError, OSError) as error:
      self._json(dict(error=str(error)), 400)

  def do_POST(self):  # noqa: N802
    try:
      origin = self.headers.get("Origin")
      if origin and urlparse(origin).netloc != self.headers.get("Host"):
        self._json(dict(error="仅接受同源请求"), 403)
        return
      if self.headers.get_content_type() != "application/json":
        raise ValueError("需要 application/json 请求")
      size = int(self.headers.get("Content-Length", "0"))
      if (
        not 0
        < size
        <= (
          128 * 1024 * 1024
          if urlparse(self.path).path in ("/api/model-load", "/api/motion-import")
          else 1024 * 1024
        )
      ):
        raise ValueError("请求大小无效")
      body = json.loads(self.rfile.read(size))
      if not isinstance(body, dict):
        raise ValueError("请求必须是 JSON 对象")
      action = urlparse(self.path).path.removeprefix("/api/")
      if action == "motion-import":
        from .motion_library import import_file

        self._json(import_file(self.root, body), 201)
        return
      if action == "motion-convert":
        from .motion_library import conversion

        action, body = conversion(self.root, body)
      if action == "npz-preview":
        from .npz_preview import load_motion
        self._json(load_motion(self.root, body.get("path", "")))
        return
      if action == "model-load":
        self._json(MODELS.load(self.root, body))
        return
      if action == "model-pose":
        self._json(MODELS.pose(body))
        return
      if action == "stop":
        self.jobs.stop(body.get("id", ""))
        self._json(dict(ok=True))
        return
      args = command(self.root, action, body)
      if action in ("viser", "tensorboard"):
        port = 8080 if action == "viser" else 6006
        with socket.socket() as probe:
          try:
            probe.bind(("127.0.0.1", port))
          except OSError as error:
            raise ValueError(f"端口 {port} 已占用，请先停止对应服务") from error
      record = self.jobs.start(action, args, body)
      if action in ("viser", "tensorboard"):
        record["port"] = 8080 if action == "viser" else 6006
      self._json(record, 202)
    except (KeyError, TypeError, ValueError, OSError) as error:
      self._json(dict(error=str(error)), 400)


def main():
  parser = argparse.ArgumentParser(description="LLoco independent five-step workbench")
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--port", type=int, default=7860)
  parser.add_argument("--project", type=Path, default=Path.cwd())
  parser.add_argument("--no-browser", action="store_true")
  args = parser.parse_args()
  root = args.project.expanduser().resolve()
  if not (root / "src/lloco").is_dir():
    parser.error("--project 必须指向 LLoco 项目根目录")

  def interrupt(signum, frame):
    raise KeyboardInterrupt

  signal.signal(signal.SIGTERM, interrupt)
  jobs = Jobs(root)
  server = ThreadingHTTPServer(
    (args.host, args.port), functools.partial(Handler, root=root, jobs=jobs)
  )
  print(f"LLoco Workbench: http://{args.host}:{args.port}", flush=True)
  if not args.no_browser:
    webbrowser.open(f"http://{args.host}:{args.port}")
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    pass
  finally:
    jobs.close()
    server.server_close()


if __name__ == "__main__":
  main()
