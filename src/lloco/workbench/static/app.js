"use strict";
const $ = (id) => document.getElementById(id);
let state,
  step = 1,
  activeJob = null,
  viewerReady = false,
  initialized = false;
const fieldIds = [
  "motion",
  "motionSource",
  "motionFormat",
  "motionRobot",
  "iterations",
  "saveInterval",
  "numEnvs",
  "checkpoint",
  "modelPath",
];
const saved = (() => {
  try {
    return JSON.parse(localStorage.getItem("lloco.workbench") || "{}");
  } catch {
    return {};
  }
})();
function notice(text, error = false) {
  $("notice").textContent = text;
  $("notice").classList.toggle("error", error);
}
async function api(path, body) {
  const response = await fetch(
    "/api/" + path,
    body === undefined
      ? {}
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  const value = await response.json();
  if (!response.ok) throw Error(value.error || response.statusText);
  return value;
}
function kind() {
  return document.querySelector("[name=kind]:checked").value;
}
function selection() {
  return {
    kind: kind(),
    group: $("group").value,
    algorithm: $("algorithm").value,
    task: $("task").value,
    motion: $("motion").value,
  };
}
function persist() {
  localStorage.setItem(
    "lloco.workbench",
    JSON.stringify({
      ...selection(),
      ...Object.fromEntries(fieldIds.map((id) => [id, $(id).value])),
    }),
  );
}
function options(id, values, preferred) {
  const el = $(id),
    old = preferred ?? el.value;
  el.replaceChildren(...values.map((v) => new Option(v, v)));
  if (values.includes(old)) el.value = old;
}
function summary() {
  const selected = $("group").value === "go2_skill" ? $("algorithm").value : "";
  const placeholder = new Option("请选择 Go2 Skill", "");
  placeholder.disabled = true;
  $("chooseSkills").replaceChildren(placeholder, ...Object.keys(state.algorithms.go2_skill || {}).map(name => new Option(name === "TS" ? "TS Teacher" : name, name)));
  $("chooseSkills").value = selected;

  $("selection").textContent = `${kind()} / ${$("algorithm").value}`;
  $("trainSummary").textContent =
    `${$("task").value} · ${$("numEnvs").value} 个环境 · ${$("iterations").value} 轮`;
  $("algorithmHint").textContent =
    $("group").value === "go2_skill"
      ? "四足算法 / 模板使用各自独立配置。TS 当前入口为 Teacher。"
      : `当前 ${kind()} 分组使用已注册 PPO 配置。`;
  persist();
}
function tasks(preferred) {
  options(
    "task",
    state.algorithms[$("group").value][$("algorithm").value],
    preferred,
  );
  summary();
}
function algorithms(preferred, task) {
  options(
    "algorithm",
    Object.keys(state.algorithms[$("group").value]),
    preferred,
  );
  tasks(task);
}
function branches(restore = false) {
  const tracking = kind() === "tracking";
  $("trackingBranch").open = tracking;
  $("velocityBranch").open = !tracking;
  $("chooseSkills").disabled = tracking;
  options(
    "group",
    tracking ? ["tracking"] : ["velocity", "go2_skill"],
    restore ? saved.group : kind(),
  );
  algorithms(
    restore ? saved.algorithm : undefined,
    restore ? saved.task : undefined,
  );
}
function navigate(value) {
  step = Math.max(1, Math.min(5, value));
  document
    .querySelectorAll("[data-panel]")
    .forEach((el) => (el.hidden = Number(el.dataset.panel) !== step));
  document
    .querySelectorAll("[data-step]")
    .forEach((el) =>
      el.classList.toggle("active", Number(el.dataset.step) === step),
    );
  $("title").textContent = [
    "机器人查看器",
    "策略选择",
    "算法与模板",
    "训练与监控",
    "导出与验证",
  ][step - 1];
  $("prev").disabled = step === 1;
  $("next").disabled = step === 5;
}
function artifactRows() {
  $("artifacts").replaceChildren(
    ...state.artifacts.map((item) => {
      const row = document.createElement("tr");
      for (const text of [
        item.path,
        item.kind.toUpperCase(),
        (item.size / 1048576).toFixed(2) + " MB",
      ]) {
        const td = document.createElement("td");
        td.textContent = text;
        row.append(td);
      }
      const td = document.createElement("td");
      if (item.kind === "pt") {
        const button = document.createElement("button");
        button.textContent = "选择";
        button.onclick = () => {
          $("checkpoint").value = item.path;
          persist();
        };
        td.append(button);
      }
      const a = document.createElement("a");
      a.textContent = "下载";
      a.href = "/api/file?path=" + encodeURIComponent(item.path);
      a.download = "";
      td.append(a);
      row.append(td);
      return row;
    }),
  );
  $("checkpoints").replaceChildren(
    ...state.artifacts
      .filter((a) => a.kind === "pt")
      .map((a) => new Option(a.path, a.path)),
  );
  motionAssets();
}
function jobs() {
  $("jobCount").textContent = state.jobs.length;
  $("jobs").replaceChildren(
    ...state.jobs.map((job) => {
      const row = document.createElement("div");
      row.className = "job";
      const label = document.createElement("span");
      label.textContent = `${job.action} · ${job.selection?.task || ""} · ${job.running ? "运行中" : job.stop_requested ? "已停止" : job.interrupted ? "服务重启后已中断" : job.returncode === 0 ? "已完成" : "失败 (" + job.returncode + ")"}`;
      row.append(label);
      const logs = document.createElement("button");
      logs.textContent = "查看日志";
      logs.onclick = () => {
        activeJob = job.id;
        refreshLog();
      };
      row.append(logs);
      if (["gmr-retarget", "gmr-convert"].includes(job.action)) {
        const preview = document.createElement("button");
        preview.textContent = "动作预览";
        preview.onclick = () => {
          navigate(2);
          document.querySelector("[name=kind][value=tracking]").checked = true;
          branches();
          openMotion(job.id);
        };
        row.append(preview);
      }
      if (job.running && ["viser", "tensorboard"].includes(job.action)) {
        const open = document.createElement("button");
        open.textContent = "打开";
        open.onclick = () => {
          navigate(job.action === "viser" ? 5 : 4);
          const frame = $(
            job.action === "viser" ? "viserFrame" : "tensorboardFrame",
          );
          frame.hidden = false;
          frame.src = `http://${location.hostname}:${job.action === "viser" ? 8080 : 6006}`;
        };
        row.append(open);
      }
      if (job.running) {
        const stop = document.createElement("button");
        stop.textContent = "停止";
        stop.onclick = () => run("stop", { id: job.id });
        row.append(stop);
      }
      return row;
    }),
  );
}
async function refreshLog() {
  if (activeJob) {
    try {
      $("log").textContent =
        (await api("log/" + activeJob)).log || "进程启动中…";
    } catch (error) {
      notice(error.message, true);
    }
  }
}
async function refresh() {
  const data = await api("state");
  state = {
    ...data,
    motion_library: Array.isArray(data.motion_library)
      ? data.motion_library
      : [],
    libraryAvailable: Array.isArray(data.motion_library),
  };
  artifactRows();
  jobs();
  await refreshLog();
}
async function run(action, body) {
  try {
    const result = await api(action, body);
    if (result.id) activeJob = result.id;
    notice(
      action === "stop" ? "已发送停止请求。" : "任务已启动，可在下方查看日志。",
    );
    if (result.port) {
      const frame = $(action === "viser" ? "viserFrame" : "tensorboardFrame");
      frame.hidden = false;
      frame.src = "about:blank";
      const url = `http://${location.hostname}:${result.port}`;
      let attempts = 0;
      const timer = setInterval(async () => {
        try {
          const latest = await api("state");
          const job = latest.jobs.find((j) => j.id === result.id);
          if (!job?.running) {
            clearInterval(timer);
            notice("服务启动失败，请查看日志。", true);
            return;
          }
          await fetch(url, { mode: "no-cors" });
          frame.src = url;
          clearInterval(timer);
        } catch {
          if (++attempts >= 120) {
            clearInterval(timer);
            notice("服务尚未就绪，请查看日志。", true);
          }
        }
      }, 1000);
    }
    await refresh();
    return result;
  } catch (error) {
    notice(error.message, true);
  }
}
$("robotFrame").addEventListener("load", () => {
  viewerReady = true;
});
window.addEventListener("message", (event) => {
  if (
    event.origin !== location.origin ||
    event.source !== $("robotFrame").contentWindow
  )
    return;
  if (event.data.type === "lloco:model-loaded")
    notice("模型已加载：" + event.data.path);
  if (event.data.type === "lloco:model-error") notice(event.data.message, true);
});
document
  .querySelectorAll("[data-step]")
  .forEach((el) => (el.onclick = () => navigate(Number(el.dataset.step))));
$("prev").onclick = () => navigate(step - 1);
$("next").onclick = () => navigate(step + 1);
$("loadModel").onclick = () => {
  if (!viewerReady) {
    notice("查看器正在加载，请稍后重试。");
    return;
  }
  $("robotFrame").contentWindow.postMessage(
    { type: "lloco:model", path: $("modelPath").value },
    location.origin,
  );
  notice("正在加载模型和网格…");
  persist();
};
$("robot").onchange = () => {
  const task = $("robot").value;
  $("modelPath").value = state.task_assets[task];
  for (const [group, algorithms] of Object.entries(state.algorithms))
    for (const [algorithm, tasks] of Object.entries(algorithms))
      if (tasks.includes(task)) {
        document.querySelector(
          `[name=kind][value=${group === "tracking" ? "tracking" : "velocity"}]`,
        ).checked = true;
        branches();
        $("group").value = group;
        options("algorithm", Object.keys(algorithms), algorithm);
        options("task", tasks, task);
        summary();
        return;
      }
};
document
  .querySelectorAll("[name=kind]")
  .forEach((el) => (el.onchange = () => branches()));
for (const [id, branch] of [
  ["trackingBranch", "tracking"],
  ["velocityBranch", "velocity"],
])
  $(id).querySelector("summary").onclick = (event) => {
    event.preventDefault();
    document.querySelector(`[name=kind][value=${branch}]`).checked = true;
    branches();
  };
$("group").onchange = () => algorithms();
$("algorithm").onchange = () => tasks();
$("task").onchange = summary;
$("chooseSkills").onchange = () => {
  const selected = $("chooseSkills").value;
  if (!selected || kind() !== "velocity") return;
  $("group").value = "go2_skill";
  algorithms(selected);
};
fieldIds.forEach((id) => ($(id).onchange = summary));
$("train").onclick = () =>
  run("train", {
    ...selection(),
    iterations: $("iterations").value,
    save_interval: $("saveInterval").value,
    num_envs: $("numEnvs").value,
  });
$("convert").onclick = async () => {
  if (kind() !== "tracking") return;
  const result = await run("motion-convert", {
    source: $("motionSource").value,
    robot: $("motionRobot").value,
  });
  if (result) {
    localStorage.setItem(
      "lloco.pendingMotion",
      JSON.stringify({ id: result.id, output: result.selection.output }),
    );
    notice("任务已启动，结果完成后会自动选为训练动作。");
    if (result.action === "csv-convert") trackCsv(result.id);
    else {
      $("csvProgressPanel").hidden = true;
      openMotion(result.id);
    }
  }
};
$("export").onclick = () =>
  run("export", { ...selection(), checkpoint: $("checkpoint").value });
$("viser").onclick = () =>
  run("viser", { ...selection(), checkpoint: $("checkpoint").value });
$("tensorboard").onclick = () => run("tensorboard", {});
$("refresh").onclick = () =>
  refresh().catch((error) => notice(error.message, true));
(async () => {
  try {
    await refresh();
    if (saved.motionFormat) {
      $("motionFormat").value = saved.motionFormat;
      motionInputs();
    }
    fieldIds.forEach((id) => {
      if (saved[id]) $(id).value = saved[id];
    });
    if (saved.kind === "tracking")
      document.querySelector("[name=kind][value=tracking]").checked = true;
    options("robot", Object.keys(state.task_assets));
    if (!$("modelPath").value)
      $("modelPath").value = state.task_assets[$("robot").value];
    motionInputs();
    branches(true);
    initialized = true;
    motionAssets();
    navigate(1);
    notice("工作台已就绪。按左侧顺序完成模型检查、配置和验证。");
    setInterval(
      () => refresh().catch((error) => notice(error.message, true)),
      3500,
    );
  } catch (error) {
    notice(error.message, true);
  }
})();

function openMotion(id) {
  if (!id) {
    notice("还没有可预览的重定向任务。");
    return;
  }
  localStorage.setItem("lloco.motionPreview", id);
  $("motionFrame").hidden = false;
  $("motionFrame").src = "/viewer/?motion=" + encodeURIComponent(id);
}
$("openMotion").onclick = () =>
  openMotion(localStorage.getItem("lloco.motionPreview"));
window.addEventListener("message", (event) => {
  if (
    event.origin !== location.origin ||
    event.source !== $("motionFrame").contentWindow ||
    event.data?.type !== "lloco:motion-state"
  )
    return;
  const m = event.data;
  $("retargetState").textContent =
    `${m.stage} · ${m.processed} / ${m.total} 帧`;
});

$("motionFormat").onchange = () => {
  motionInputs();
  persist();
};
function assetSelect(id, entries, empty) {
  const previous = $(id).value;
  $(id).replaceChildren(
    new Option(empty, ""),
    ...entries.map((a) => new Option(a.name, a.path)),
  );
  if (entries.some((a) => a.path === previous)) $(id).value = previous;
}
function motionInputs() {
  if (!state) return;
  const format = {
    "gmr-retarget": "bvh",
    "gmr-convert": "pkl",
    "csv-convert": "csv",
  }[$("motionFormat").value];
  assetSelect(
    "motionSource",
    state.motion_library.filter((a) => a.format === format),
    "请选择资产库中的 " + format.toUpperCase() + " 动作",
  );
  $("motionRobot").disabled = format !== "csv";
  if (format !== "csv") $("motionRobot").value = "g1";
  $("convert").textContent = format === "bvh" ? "开始重定向" : "转换为 NPZ";
  $("convert").disabled = !$("motionSource").value;
  $("libraryHint").textContent = !state.libraryAvailable
    ? "资产库接口尚未就绪，请重启工作台后刷新。"
    : format === "csv"
      ? "CSV 按 30 Hz 输入转换为 50 Hz NPZ；结果自动保存到资产库。"
      : "结果自动保存到动作资产库，无需填写输出路径。";
}
function motionAssets() {
  const entries = state.motion_library || [];
  assetSelect(
    "motion",
    entries.filter((a) => a.format === "npz"),
    "请选择用于训练的 NPZ 动作",
  );
  motionInputs();
  $("motionCount").textContent = entries.length + " 个文件";
  $("motionAssets").replaceChildren(
    ...entries.map((asset) => {
      const row = document.createElement("tr");
      for (const text of [
        asset.name,
        asset.format.toUpperCase(),
        (asset.size / 1048576).toFixed(2) + " MB",
      ]) {
        const td = document.createElement("td");
        td.textContent = text;
        row.append(td);
      }
      const action = document.createElement("td");
      if (asset.format === "txt") action.textContent = "Go2 算法内置动作";
      else {
        const button = document.createElement("button");
        button.textContent = asset.format === "npz" ? "用于训练" : "用于转换";
        button.onclick = () => selectMotion(asset);
        action.append(button);
      }
      row.append(action);
      return row;
    }),
  );
  try {
    const pending = JSON.parse(
      localStorage.getItem("lloco.pendingMotion") || "null",
    );
    if (pending && initialized) {
      const job = state.jobs.find((j) => j.id === pending.id);
      if (job?.running && job.action === "csv-convert")
        $("retargetState").textContent = "正在转换 CSV…";
      if (job && !job.running) {
        if (
          job.returncode === 0 &&
          entries.some((a) => a.path === pending.output)
        ) {
          $("motion").value = pending.output;
          persist();
          notice("动作已入库，并选为当前训练动作。");
        }
        localStorage.removeItem("lloco.pendingMotion");
      }
    }
  } catch {
    localStorage.removeItem("lloco.pendingMotion");
  }
}
function selectMotion(asset) {
  if (asset.format === "npz") $("motion").value = asset.path;
  else {
    $("motionFormat").value = {
      bvh: "gmr-retarget",
      pkl: "gmr-convert",
      csv: "csv-convert",
    }[asset.format];
    motionInputs();
    $("motionSource").value = asset.path;
    $("motionRobot").value = asset.path.includes("g1_23dof/")
      ? "g1_23dof"
      : "g1";
    $("convert").disabled = false;
  }
  persist();
}
$("motionSource").onchange = () => {
  $("convert").disabled = !$("motionSource").value;
  if ($("motionFormat").value === "csv-convert")
    $("motionRobot").value = $("motionSource").value.includes("g1_23dof/")
      ? "g1_23dof"
      : "g1";
  persist();
};
$("importMotion").onclick = () => $("motionUpload").click();
$("refreshMotions").onclick = () =>
  refresh().catch((error) => notice(error.message, true));
$("motionUpload").onchange = async (event) => {
  $("importMotion").disabled = true;
  try {
    for (const file of event.target.files) {
      if (file.size > 90 * 1024 * 1024) throw Error("单个文件请小于 90 MB");
      const data = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result.split(",")[1]);
        reader.onerror = reject;
        reader.readAsDataURL(file);
      });
      const result = await api("motion-import", { name: file.name, data });
      await refresh();
      const asset = state.motion_library.find((a) => a.path === result.path);
      if (asset && asset.format !== "txt") selectMotion(asset);
    }
    notice("导入完成，可从资产库选择动作。");
  } catch (error) {
    notice(error.message, true);
  } finally {
    $("importMotion").disabled = false;
    event.target.value = "";
  }
};

let csvProgressRun = 0;
function trackCsv(id) {
  localStorage.setItem("lloco.csvProgress", id);
  const run = ++csvProgressRun;
  $("csvProgressPanel").hidden = false;
  $("playCsvResult").disabled = true;
  $("csvProgress").removeAttribute("value");
  $("csvProgressLabel").textContent = "CSV 转换 · 准备中";
  $("csvProgressCount").textContent = "等待帧数";
  async function poll() {
    try {
      const data = await api("motion-preview?id=" + encodeURIComponent(id));
      if (run !== csvProgressRun) return;
      const { info, job } = data;
      const finished = !job.running;
      const stage = finished
        ? job.stop_requested
          ? "已停止"
          : job.returncode === 0
            ? "已完成"
            : "失败"
        : {
            loading: "读取与初始化",
            converting: "逐帧转换",
            saving: "保存 NPZ",
            complete: "收尾",
            failed: "失败",
          }[info.stage] || "准备中";
      $("csvProgressLabel").textContent =
        "CSV 转换 · " + stage + (info.source ? " · " + info.source : "");
      const total = info.total || 0,
        done = info.processed || 0;
      if (total) {
        $("csvProgress").max = total;
        $("csvProgress").value = done;
        $("csvProgressCount").textContent =
          `${done} / ${total} 帧 · ${Math.floor((done / total) * 100)}%`;
      } else if (finished) {
        $("csvProgress").value = 0;
        $("csvProgressCount").textContent = "未生成帧数据";
      }
      if (info.error) $("csvProgressLabel").textContent += "：" + info.error;
      $("playCsvResult").disabled = !(
        finished &&
        job.returncode === 0 &&
        job.selection?.output
      );
      if (!$("playCsvResult").disabled)
        $("playCsvResult").onclick = () => playNpz(job.selection.output);
      if (finished) return;
    } catch (error) {
      if (run !== csvProgressRun) return;
      $("csvProgressLabel").textContent = "CSV 进度暂不可用：" + error.message;
    }
    if (run === csvProgressRun) setTimeout(poll, 500);
  }
  poll();
}
if (localStorage.getItem("lloco.csvProgress"))
  trackCsv(localStorage.getItem("lloco.csvProgress"));

function playNpz(path) {
  if (!path) {
    notice("请先选择已转换的 NPZ 动作。");
    return;
  }
  $("motionFrame").hidden = false;
  $("motionFrame").src =
    "/viewer/?motion=npz&asset=" + encodeURIComponent(path);
  $("motionFrame").scrollIntoView({ block: "center", behavior: "smooth" });
}
$("playSelectedMotion").onclick = () => playNpz($("motion").value);
