import * as THREE from "three";
import {
  model,
  api,
  show,
  update,
  scene,
  camera,
  orbit,
  layers,
  status,
} from "./main.js";

export async function start() {
  const identifier = new URLSearchParams(location.search).get("motion");
  const asset = new URLSearchParams(location.search).get("asset");
  document.body.classList.add("motion-mode");
  const controls = document.createElement("div");
  controls.className = "motion-controls";
  controls.innerHTML = `<div class="motion-summary"><span id="motionStage">等待重定向数据</span><span id="motionProgress"></span></div>
    <progress id="motionBar" value="0" max="1"></progress>
    <div class="motion-transport"><button id="motionPlay" disabled>播放</button><input id="motionTime" aria-label="动作时间轴" type="range" min="0" max="0" step="1" value="0"><output id="motionClock">0.00 s</output><select id="motionSpeed" aria-label="播放速度"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select><label><input id="motionLive" type="checkbox" checked>跟随进度</label><label><input id="motionRoot" type="checkbox" checked>固定根位置</label></div>`;
  document.querySelector(".workspace").after(controls);
  const legend = document.createElement("div");
  legend.className = "motion-legend";
  legend.textContent = "橙色：输入人体骨架　／　右侧：G1 重定向姿态";
  document.getElementById("canvas").append(legend);
  const $ = (id) => document.getElementById(id);
  const human = new THREE.Group();
  scene.add(human);
  const lineGeometry = new THREE.BufferGeometry();
  const line = new THREE.LineSegments(
    lineGeometry,
    new THREE.LineBasicMaterial({ color: 0xb76a32 }),
  );
  human.add(line);
  const nodes = new THREE.Points(
    new THREE.BufferGeometry(),
    new THREE.PointsMaterial({ color: 0xb76a32, size: 0.045 }),
  );
  human.add(nodes);
  const frames = [];
  let cursor = 0,
    info = {},
    playing = false,
    current = 0,
    busy = false,
    pending = null,
    follow = true,
    epoch = 0,
    startTime = 0,
    visible = true;
  new IntersectionObserver((entries) => {
    visible = entries[0].isIntersecting;
  }).observe(document.body);
  function position(frame) {
    const locked = $("motionRoot").checked;
    for (const group of Object.values(layers))
      group.position.set(
        0.9 - (locked ? frame.qpos[0] : 0),
        -(locked ? frame.qpos[1] : 0),
        0,
      );
    if (frame.human?.length) {
      human.visible = true;
      const positions = frame.human;
      const root = positions[0];
      human.position.set(
        -0.9 - (locked ? root[0] : 0),
        -(locked ? root[1] : 0),
        0,
      );
      const segments = [];
      for (let i = 0; i < positions.length; i++) {
        const parent = info.parents?.[i];
        if (parent >= 0) segments.push(...positions[parent], ...positions[i]);
      }
      lineGeometry.setAttribute(
        "position",
        new THREE.Float32BufferAttribute(segments, 3),
      );
      lineGeometry.computeBoundingSphere();
      nodes.geometry.setAttribute(
        "position",
        new THREE.Float32BufferAttribute(positions.flat(), 3),
      );
      nodes.geometry.computeBoundingSphere();
    } else {
      human.visible = false;
      legend.textContent = asset
        ? "训练 NPZ 动作回放"
        : "G1 重定向姿态（PKL 不含原始人体骨架）";
      for (const group of Object.values(layers)) group.position.x -= 0.9;
    }
  }
  async function draw(index) {
    if (!model || !frames.length) return;
    if (busy) {
      pending = index;
      return;
    }
    busy = true;
    current = Math.min(frames.length - 1, Math.max(0, index));
    const frame = frames[current];
    try {
      update(await api("model-pose", { id: model.id, qpos: frame.qpos }));
      position(frame);
      $("motionTime").value = current;
      $("motionClock").value =
        `${frame.time.toFixed(2)} s · 帧 ${frame.index + 1}`;
    } catch (error) {
      status(error.message, true);
      playing = false;
    } finally {
      busy = false;
      if (pending !== null) {
        const next = pending;
        pending = null;
        draw(next);
      }
    }
  }
  function setPlaying(value) {
    playing = value;
    $("motionPlay").textContent = value ? "暂停" : "播放";
    epoch = performance.now();
    startTime = frames[current]?.time || 0;
  }
  $("motionPlay").onclick = () => {
    follow = false;
    $("motionLive").checked = false;
    if (current === frames.length - 1) current = 0;
    setPlaying(!playing);
  };
  $("motionTime").oninput = () => {
    follow = false;
    $("motionLive").checked = false;
    setPlaying(false);
  };
  $("motionTime").onchange = () => draw(Number($("motionTime").value));
  $("motionRoot").onchange = () => draw(current);
  $("motionSpeed").onchange = () => setPlaying(playing);
  $("motionLive").onchange = () => {
    follow = $("motionLive").checked;
    if (follow) setPlaying(false);
  };
  try {
    const saved = asset ? await api("npz-preview", { path: asset }) : null;
    show(
      await api("model-load", {
        path:
          saved?.model || "src/lloco/workbench/gmr/assets/g1_mocap_29dof.xml",
      }),
    );
    if (saved) {
      info = saved.info;
      frames.push(...saved.frames);
      follow = false;
      $("motionLive").checked = false;
      $("motionLive").parentElement.hidden = true;
      $("motionTime").max = frames.length - 1;
      $("motionPlay").disabled = false;
      $("motionStage").textContent = "训练动作 · " + info.source;
      $("motionProgress").textContent = `${info.total} 帧 · ${info.fps} Hz`;
      $("motionBar").hidden = true;
      await draw(0);
    }
    orbit.target.set(0, 0, 0.85);
    camera.position.set(1.5, -3.1, 1.8);
    orbit.update();
    $("empty").hidden = true;
    status(
      asset
        ? "读取实际 NPZ 的根姿态与关节角；点击播放或拖动时间轴查看。"
        : "逐帧检查 GMR 求解姿态；源动作时间与右侧机器人同步。",
    );
  } catch (error) {
    status(error.message, true);
    return;
  }
  async function poll() {
    try {
      const response = await fetch(
        `/api/motion-preview?id=${encodeURIComponent(identifier)}&cursor=${cursor}`,
      );
      const data = await response.json();
      if (!response.ok) throw Error(data.error);
      info = data.info;
      if (!data.job.running && !info.total) {
        status("此历史任务没有预览数据，请重新运行重定向。", true);
      }
      cursor = data.cursor;
      frames.push(...data.frames);
      $("motionTime").max = Math.max(0, frames.length - 1);
      $("motionPlay").disabled = !frames.length;
      const stage =
        !data.job.running && data.job.returncode !== 0
          ? data.job.stop_requested
            ? "已停止"
            : "失败"
          : {
              loading: "读取动作",
              retargeting: "求解重定向",
              converting: "生成训练 NPZ",
              complete: "已完成",
              failed: "失败",
            }[info.stage] || info.stage;
      $("motionStage").textContent =
        stage + (info.source ? " · " + info.source : "");
      $("motionProgress").textContent =
        `${info.processed || 0} / ${info.total || 0} 帧`;
      $("motionBar").max = info.total || 1;
      $("motionBar").value = info.processed || 0;
      if (info.error) status(info.error, true);
      if (follow && frames.length && visible) await draw(frames.length - 1);
      window.parent.postMessage(
        {
          type: "lloco:motion-state",
          job: identifier,
          stage,
          processed: info.processed || 0,
          total: info.total || 0,
        },
        location.origin,
      );
      if (!data.job.running && data.frames.length === 0) {
        follow = false;
        $("motionLive").checked = false;
        return;
      }
    } catch (error) {
      status(error.message, true);
    }
    setTimeout(poll, 700);
  }
  if (!asset) poll();
  async function tick() {
    if (playing && visible && !busy && frames.length) {
      const time =
        startTime +
        ((performance.now() - epoch) / 1000) * Number($("motionSpeed").value);
      let index = current;
      while (index < frames.length - 1 && frames[index + 1].time <= time)
        index++;
      if (index !== current) await draw(index);
      if (time >= frames.at(-1).time) setPlaying(false);
    }
    setTimeout(tick, 40);
  }
  tick();
}
