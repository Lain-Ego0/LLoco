import "./style.css";
import loadMujoco from "@mujoco/mujoco";
import mujocoWasmUrl from "@mujoco/mujoco/mujoco.wasm?url";
import * as ort from "onnxruntime-web";
import ortWasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url";
import ortMjsUrl from "onnxruntime-web/ort-wasm-simd-threaded.mjs?url";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";

// Keep ONNX Runtime's sidecar binary out of Vite's dependency-cache URL too.
// The demo is intentionally single-threaded so it remains usable without
// cross-origin-isolation response headers.
ort.env.wasm.wasmPaths = {
  mjs: ortMjsUrl,
  wasm: ortWasmUrl,
};
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;

const policies = {
  handstand: {
    name: "前倒立",
    file: "/policies/go2-handstand.onnx",
    inputSize: 48,
    mode: "stand",
    defaultPose: [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5],
    commandLimit: [-0.4, 0.4],
    note: "静止起步；可用前进和转向控制调整指令。",
  },
  rearStand: {
    name: "后立",
    file: "/policies/go2-rear-stand.onnx",
    inputSize: 45,
    mode: "stand",
    defaultPose: [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5],
    commandLimit: [-0.2, 0.6],
    note: "前脚离地、后脚支撑的策略展示。",
  },
  trot: {
    name: "小跑",
    file: "/policies/go2-trot.onnx",
    inputSize: 470,
    mode: "gait",
    cycleTime: 0.5,
    defaultPose: [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5],
    commandLimit: [-1, 1],
    note: "已通过迁移回放验证的小跑策略；从低速指令开始体验。",
  },
  jump: {
    name: "原地跳跃",
    file: "/policies/go2-jump.onnx",
    inputSize: 470,
    mode: "gait",
    cycleTime: 1.5,
    defaultPose: [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5],
    commandLimit: [-1, 1],
    note: "已通过接触修正验证的跳跃策略；启动后会按自身节律执行。",
  },
  springJump: {
    name: "弹簧跳跃",
    file: "/policies/go2-spring-jump.onnx",
    inputSize: 470,
    mode: "spring",
    defaultPose: [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5],
    commandLimit: [-1, 1],
    note: "来自 Gym 工程的弹簧跳跃策略；前进指令会作为起跳输入。",
  },
  arenaWalk: {
    name: "Arena 平地行走",
    file: "/policies/go2-arena-flat.onnx",
    inputSize: 270,
    mode: "arenaHistory",
    defaultPose: [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5],
    commandLimit: [-1, 1],
    note: "ArenaX 参考工程的 6 帧历史平地行走策略。",
  },
};

const jointNames = [
  "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
  "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
  "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
  "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
];
const actuatorNames = [
  "FL_hip", "FL_thigh", "FL_calf", "FR_hip", "FR_thigh", "FR_calf",
  "RL_hip", "RL_thigh", "RL_calf", "RR_hip", "RR_thigh", "RR_calf",
];
const meshFiles = [
  "base_0.obj", "base_1.obj", "base_2.obj", "base_3.obj", "base_4.obj",
  "hip_0.obj", "hip_1.obj", "thigh_0.obj", "thigh_1.obj", "thigh_mirror_0.obj",
  "thigh_mirror_1.obj", "calf_0.obj", "calf_1.obj", "calf_mirror_0.obj",
  "calf_mirror_1.obj", "foot.obj",
];

const $ = (id) => document.getElementById(id);
const engineState = $("engineState");
const notice = $("notice");
const simulation = { running: false, elapsed: 0, action: new Float32Array(12), command: [0, 0, 0], history: [] };
const terrainState = { kind: "flat", seed: 7, height: 0.28, tool: "platform", elements: [], selected: null };
let mujoco, model, data, policySession, activePolicy, jointAddresses, actuatorAddresses, baseSceneXml, robotAssets;

function setStatus(text, kind = "") {
  engineState.textContent = text;
  engineState.className = `pill ${kind}`;
}

function setNotice(text, error = false) {
  notice.textContent = text;
  notice.classList.toggle("error", error);
}

function setBootStage(text, progress) {
  $("bootStage").textContent = text;
  $("bootProgress").style.width = `${progress}%`;
  $("bootPercent").textContent = `${progress}%`;
}

function revealScene() {
  $("bootScreen").classList.add("reveal");
}

function createScene() {
  const mount = $("scene");
  const scene = new THREE.Scene();
  scene.background = new THREE.Color("#edf0f3");
  const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 100);
  camera.up.set(0, 0, 1);
  camera.position.set(1.6, -2.7, 1.25);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  const canvasMount = document.createElement("div");
  canvasMount.className = "scene-canvas";
  canvasMount.append(renderer.domElement);
  mount.prepend(canvasMount);
  const orbit = new OrbitControls(camera, renderer.domElement);
  orbit.target.set(0, 0, 0.38);
  orbit.enableDamping = true;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x98a5b1, 2.2));
  const key = new THREE.DirectionalLight(0xffffff, 2.4);
  key.position.set(2, -3, 5);
  key.castShadow = true;
  scene.add(key);
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(12, 12),
    new THREE.MeshStandardMaterial({ color: 0xdfe4e8, roughness: 0.92 }),
  );
  floor.receiveShadow = true;
  floor.position.z = -0.002;
  scene.add(floor);
  const grid = new THREE.GridHelper(8, 16, 0xb7c1ca, 0xd7dde2);
  // GridHelper is horizontal in Three.js' Y-up world. This scene and MuJoCo
  // model use Z-up, so rotate it onto the physical floor plane.
  grid.rotation.x = Math.PI / 2;
  grid.position.z = 0.001;
  scene.add(grid);
  const resize = () => {
    const width = mount.clientWidth;
    const height = mount.clientHeight;
    if (!width || !height) return;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(mount);
  resize();
  return { scene, camera, renderer, orbit, floor, grid };
}

const view = createScene();
const terrainVisual = new THREE.Group();
terrainVisual.name = "playground-terrain";
view.scene.add(terrainVisual);
const robot = new THREE.Group();
view.scene.add(robot);
const bodyNodes = new Map();

function attribute(element, name, fallback = "") {
  return element.getAttribute(name) ?? fallback;
}

function vector(text, size, fallback) {
  const values = text.trim().split(/\s+/).map(Number);
  return values.length === size && values.every(Number.isFinite) ? values : fallback;
}

function localTransform(node, element) {
  const pos = vector(attribute(element, "pos"), 3, [0, 0, 0]);
  const quat = vector(attribute(element, "quat"), 4, [1, 0, 0, 0]);
  node.position.fromArray(pos);
  node.quaternion.set(quat[1], quat[2], quat[3], quat[0]);
}

async function buildRobot() {
  const [xml, ...objects] = await Promise.all([
    fetch("/robot/scene_go2.xml").then((response) => response.text()),
    ...meshFiles.map((file) => new OBJLoader().loadAsync(`/robot/assets/${file}`)),
  ]);
  const meshes = new Map(meshFiles.map((file, index) => [file.replace(".obj", ""), objects[index]]));
  const document = new DOMParser().parseFromString(xml, "text/xml");
  const materialColors = { metal: 0xe0e8e8, black: 0x1c242c, white: 0xf2f4f5, gray: 0xabb1c5 };
  robot.clear();
  bodyNodes.clear();
  function visit(body) {
    const name = attribute(body, "name");
    if (!name) return;
    const group = new THREE.Group();
    bodyNodes.set(name, group);
    robot.add(group);
    for (const geom of body.children) {
      if (geom.tagName !== "geom" || !geom.hasAttribute("mesh")) continue;
      const source = meshes.get(attribute(geom, "mesh"));
      if (!source) continue;
      const mesh = source.clone(true);
      mesh.traverse((child) => {
        if (!child.isMesh) return;
        child.material = new THREE.MeshStandardMaterial({
          color: materialColors[attribute(geom, "material")] ?? 0xb8c0c8,
          roughness: 0.62,
          metalness: 0.12,
        });
        child.castShadow = true;
      });
      localTransform(mesh, geom);
      group.add(mesh);
    }
    for (const child of body.children) if (child.tagName === "body") visit(child);
  }
  for (const body of document.querySelectorAll("worldbody > body")) visit(body);
}

async function initializeMujoco() {
  // Vite pre-bundles JS dependencies during development, but the Emscripten
  // loader resolves its sidecar WASM relative to that transient cache path.
  // Giving it Vite's emitted URL makes dev and production use the same binary.
  mujoco = await loadMujoco({
    locateFile: (file) => (file === "mujoco.wasm" ? mujocoWasmUrl : file),
  });
  const [xml, binaries] = await Promise.all([
    fetch("/robot/scene_go2.xml").then((response) => response.text()),
    Promise.all(meshFiles.map(async (file) => [file, new Uint8Array(await (await fetch(`/robot/assets/${file}`)).arrayBuffer())])),
  ]);
  baseSceneXml = xml;
  robotAssets = binaries;
  createMujocoModel(baseSceneXml);
}

function createMujocoModel(xml) {
  const vfs = new mujoco.MjVFS();
  for (const [file, buffer] of robotAssets) vfs.addBuffer(`assets/${file}`, buffer);
  data?.delete?.();
  model?.delete?.();
  model = mujoco.MjModel.from_xml_string(xml, vfs);
  data = new mujoco.MjData(model);
  jointAddresses = jointNames.map((name) => ({
    qpos: model.jnt(name).qposadr,
    dof: model.jnt(name).dofadr,
  }));
  actuatorAddresses = actuatorNames.map((name) => model.actuator(name).id);
  vfs.delete();
}

function seededRandom(seed) {
  let value = (Number(seed) || 1) >>> 0;
  return () => {
    value += 0x6d2b79f5;
    let result = value;
    result = Math.imul(result ^ (result >>> 15), result | 1);
    result ^= result + Math.imul(result ^ (result >>> 7), result | 61);
    return ((result ^ (result >>> 14)) >>> 0) / 4294967296;
  };
}

function box(x, y, z, sx, sy, sz, ry = 0, label = "障碍") {
  return { x, y, z, sx, sy, sz, ry, label };
}

function profileBoxes() {
  const height = terrainState.height;
  if (terrainState.kind === "slope") return [box(2.2, 0, height / 2, 2.2, 2.5, height / 2, -0.15, "坡道")];
  if (terrainState.kind === "stairs") return Array.from({ length: 7 }, (_, index) => {
    const step = index + 1;
    return box(1.2 + index * 0.42, 0, height * step / 14, 0.21, 1.35, height * step / 14, 0, "阶梯");
  });
  if (terrainState.kind === "obstacle_mix") {
    const random = seededRandom(terrainState.seed);
    const boxes = [];
    while (boxes.length < 10) {
      const x = -3.8 + random() * 7.6;
      const y = -2.8 + random() * 5.6;
      if (Math.hypot(x, y) < 1.05) continue;
      boxes.push(box(x, y, height * (0.35 + random() * 0.45) / 2, 0.18 + random() * 0.24, 0.18 + random() * 0.24, height * (0.35 + random() * 0.45) / 2, 0, "随机障碍"));
    }
    return boxes;
  }
  return [];
}

function elementBoxes(element) {
  const { x, y, kind } = element;
  const height = terrainState.height;
  if (kind === "stairs") return Array.from({ length: 5 }, (_, index) => box(x + (index - 2) * 0.28, y, height * (index + 1) / 10, 0.14, 0.8, height * (index + 1) / 10, element.yaw || 0, "台阶"));
  if (kind === "ramp") return [box(x, y, height / 2, 1.2, 0.9, height / 2, element.yaw || -0.18, "斜坡")];
  if (kind === "stones") return Array.from({ length: 6 }, (_, index) => box(x + (index % 3 - 1) * 0.42, y + (Math.floor(index / 3) - .5) * 0.6, height / 4, .14, .14, height / 4, 0, "梅花桩"));
  if (kind === "wall") return [box(x, y, height / 2, 1.25, .12, height / 2, element.yaw || 0, "矮墙")];
  return [box(x, y, height / 2, 0.9, 0.9, height / 2, element.yaw || 0, "高台")];
}

function terrainBoxes() {
  return [...profileBoxes(), ...terrainState.elements.flatMap(elementBoxes)];
}

function clearTerrainVisual() {
  terrainVisual.clear();
}

function addVisualBox(definition, accent = 0x7f9bb0) {
  const mesh = new THREE.Mesh(
    new THREE.BoxGeometry(definition.sx * 2, definition.sy * 2, definition.sz * 2),
    new THREE.MeshStandardMaterial({ color: accent, roughness: .76, metalness: .03 }),
  );
  mesh.position.set(definition.x, definition.y, definition.z);
  mesh.rotation.y = definition.ry;
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  terrainVisual.add(mesh);
}

function renderTerrain() {
  clearTerrainVisual();
  view.floor.material.color.set(terrainState.kind === "flat" ? 0xdfe4e8 : 0xd7e1e6);
  for (const definition of profileBoxes()) addVisualBox(definition, 0x9aabba);
  for (const element of terrainState.elements) for (const definition of elementBoxes(element)) addVisualBox(definition, 0x6f94ae);
}

function terrainXml() {
  const document = new DOMParser().parseFromString(baseSceneXml, "text/xml");
  const worldbodies = document.querySelectorAll("worldbody");
  const worldbody = worldbodies[worldbodies.length - 1];
  terrainBoxes().forEach((definition, index) => {
    const geom = document.createElement("geom");
    geom.setAttribute("name", `playground_terrain_${index}`);
    geom.setAttribute("type", "box");
    geom.setAttribute("pos", `${definition.x} ${definition.y} ${definition.z}`);
    geom.setAttribute("size", `${definition.sx} ${definition.sy} ${definition.sz}`);
    geom.setAttribute("euler", `0 ${definition.ry} 0`);
    geom.setAttribute("rgba", ".48 .61 .7 1");
    geom.setAttribute("friction", "0.9 0.1 0.1");
    worldbody.append(geom);
  });
  return new XMLSerializer().serializeToString(document);
}

function applyTerrain() {
  if (!mujoco || !baseSceneXml) return;
  simulation.running = false;
  $("start").textContent = "开始";
  createMujocoModel(terrainXml());
  renderTerrain();
  reset();
  setNotice(`已应用${terrainState.kind === "flat" ? "平地" : "自定义"}地形；物理碰撞与场景预览已同步更新。`);
}

async function loadPolicy(key) {
  activePolicy = policies[key];
  simulation.running = false;
  $("start").textContent = "开始";
  $("policyName").textContent = activePolicy.name;
  $("policyMeta").textContent = `${activePolicy.inputSize} 维策略输入 · 12 个关节动作`;
  simulation.command.fill(0);
  updateControls();
  setNotice(`正在加载「${activePolicy.name}」ONNX 策略…`);
  policySession = await ort.InferenceSession.create(activePolicy.file, { executionProviders: ["wasm"] });
  reset();
  setNotice(`${activePolicy.note} 点击“开始试玩”后，物理和策略均在此浏览器内运行。`);
}

function reset() {
  if (!model || !data || !activePolicy) return;
  mujoco.mj_resetData(model, data);
  data.qpos[0] = 0;
  data.qpos[1] = 0;
  data.qpos[2] = 0.42;
  data.qpos[3] = 1;
  data.qpos[4] = data.qpos[5] = data.qpos[6] = 0;
  activePolicy.defaultPose.forEach((value, index) => { data.qpos[jointAddresses[index].qpos] = value; });
  data.qvel.fill(0);
  data.ctrl.fill(0);
  simulation.action.fill(0);
  simulation.history = [];
  simulation.elapsed = 0;
  mujoco.mj_forward(model, data);
  updateRobot();
}

function bodyQuaternion() {
  return new THREE.Quaternion(data.qpos[4], data.qpos[5], data.qpos[6], data.qpos[3]);
}

function observation() {
  if (activePolicy.mode === "gait" || activePolicy.mode === "spring") return gaitObservation();
  if (activePolicy.mode === "arenaHistory") return arenaHistoryObservation();
  const quaternion = bodyQuaternion();
  const inverse = quaternion.clone().invert();
  const angularVelocity = new THREE.Vector3(data.qvel[3], data.qvel[4], data.qvel[5]).applyQuaternion(inverse);
  const gravity = new THREE.Vector3(0, 0, -1).applyQuaternion(inverse);
  const values = activePolicy.inputSize === 48 ? [0, 0, 0] : [];
  values.push(angularVelocity.x * 0.25, angularVelocity.y * 0.25, angularVelocity.z * 0.25);
  values.push(gravity.x, gravity.y, gravity.z);
  values.push(simulation.command[0] * 2, simulation.command[1] * 2, simulation.command[2] * 0.25);
  jointAddresses.forEach((address, index) => values.push(data.qpos[address.qpos] - activePolicy.defaultPose[index]));
  jointAddresses.forEach((address) => values.push(data.qvel[address.dof] * 0.05));
  values.push(...simulation.action);
  return new Float32Array(values);
}

function arenaHistoryObservation() {
  const quaternion = bodyQuaternion();
  const inverse = quaternion.clone().invert();
  const angularVelocity = new THREE.Vector3(data.qvel[3], data.qvel[4], data.qvel[5]).applyQuaternion(inverse);
  const gravity = new THREE.Vector3(0, 0, -1).applyQuaternion(inverse);
  const frame = [
    simulation.command[0] * 2, simulation.command[1] * 2, simulation.command[2] * .25,
    angularVelocity.x * .25, angularVelocity.y * .25, angularVelocity.z * .25,
    gravity.x, gravity.y, gravity.z,
  ];
  jointAddresses.forEach((address, index) => frame.push(data.qpos[address.qpos] - activePolicy.defaultPose[index]));
  jointAddresses.forEach((address) => frame.push(data.qvel[address.dof] * .05));
  frame.push(...simulation.action);
  simulation.history.push(frame);
  if (simulation.history.length > 6) simulation.history.shift();
  const values = Array.from({ length: 6 - simulation.history.length }, () => new Array(45).fill(0));
  values.push(...simulation.history);
  return new Float32Array(values.flat());
}

function gaitObservation() {
  const quaternion = bodyQuaternion();
  const inverse = quaternion.clone().invert();
  const angularVelocity = new THREE.Vector3(data.qvel[3], data.qvel[4], data.qvel[5]).applyQuaternion(inverse);
  const euler = new THREE.Euler().setFromQuaternion(quaternion, "XYZ");
  const frame = activePolicy.mode === "spring"
    ? [0, 0, 0.7, 0, simulation.command[0], angularVelocity.x * 0.25, angularVelocity.y * 0.25, angularVelocity.z * 0.25, euler.x, euler.y, euler.z]
    : (() => {
      const phase = simulation.elapsed / activePolicy.cycleTime;
      return [
        Math.sin(2 * Math.PI * phase), Math.cos(2 * Math.PI * phase),
        simulation.command[0] * 2, simulation.command[1] * 2, simulation.command[2] * 0.25,
        angularVelocity.x * 0.25, angularVelocity.y * 0.25, angularVelocity.z * 0.25,
        euler.x, euler.y, euler.z,
      ];
    })();
  jointAddresses.forEach((address, index) => frame.push(data.qpos[address.qpos] - activePolicy.defaultPose[index]));
  jointAddresses.forEach((address) => frame.push(data.qvel[address.dof] * 0.05));
  frame.push(...simulation.action);
  simulation.history.push(frame);
  if (simulation.history.length > 10) simulation.history.shift();
  const values = Array.from({ length: 10 - simulation.history.length }, () => new Array(47).fill(0));
  values.push(...simulation.history);
  return new Float32Array(values.flat());
}

async function policyStep() {
  const obs = observation();
  if (obs.length !== activePolicy.inputSize) throw new Error(`策略观测维度不匹配：${obs.length} / ${activePolicy.inputSize}`);
  const inputName = policySession.inputNames[0];
  const outputName = policySession.outputNames[0];
  const output = await policySession.run({ [inputName]: new ort.Tensor("float32", obs, [1, obs.length]) });
  simulation.action.set(output[outputName].data);
}

function applyControl() {
  jointAddresses.forEach((address, index) => {
    const target = activePolicy.defaultPose[index] + 0.25 * simulation.action[index];
    const limit = index % 3 === 2 ? 31.995 : 21.33;
    const torque = THREE.MathUtils.clamp(40 * (target - data.qpos[address.qpos]) - data.qvel[address.dof], -limit, limit);
    data.ctrl[actuatorAddresses[index]] = torque;
  });
}

function updateRobot() {
  if (!model || !data) return;
  for (const [name, node] of bodyNodes) {
    const id = model.body(name).id;
    node.position.fromArray(data.xpos, id * 3);
    const matrix = data.xmat;
    const offset = id * 9;
    const rotation = new THREE.Matrix4().set(
      matrix[offset], matrix[offset + 1], matrix[offset + 2], 0,
      matrix[offset + 3], matrix[offset + 4], matrix[offset + 5], 0,
      matrix[offset + 6], matrix[offset + 7], matrix[offset + 8], 0,
      0, 0, 0, 1,
    );
    node.quaternion.setFromRotationMatrix(rotation);
  }
}

function updateControls() {
  $("command").textContent = `vx ${simulation.command[0].toFixed(2)} · vy ${simulation.command[1].toFixed(2)} · ω ${simulation.command[2].toFixed(2)}`;
}

let pendingStep = false;
async function tick() {
  if (simulation.running && !pendingStep) {
    pendingStep = true;
    try {
      await policyStep();
      for (let index = 0; index < 4; index += 1) {
        applyControl();
        mujoco.mj_step(model, data);
      }
      simulation.elapsed += 0.02;
      if (data.qpos[2] < -0.2) {
        reset();
        setNotice("机器人离开场地，已自动重置。");
      }
    } catch (error) {
      simulation.running = false;
      $("start").textContent = "开始";
      setNotice(error.message, true);
    } finally { pendingStep = false; }
  }
  updateRobot();
  $("clock").textContent = `${simulation.elapsed.toFixed(2)} s`;
  view.orbit.update();
  view.renderer.render(view.scene, view.camera);
  requestAnimationFrame(tick);
}

const terrainLabels = { platform: "高台", stairs: "台阶", ramp: "斜坡", stones: "梅花桩", wall: "矮墙" };

function updateTerrainHeight() {
  terrainState.height = Number($("terrainHeight").value);
  $("terrainHeightValue").textContent = `${terrainState.height.toFixed(2)} m`;
  repaintTerrainEditor();
}

function terrainCanvasPosition(event) {
  const canvas = $("terrainMap");
  const bounds = canvas.getBoundingClientRect();
  return {
    x: ((event.clientX - bounds.left) / bounds.width - .5) * 8,
    y: (.5 - (event.clientY - bounds.top) / bounds.height) * 6,
  };
}

function repaintTerrainMap() {
  const canvas = $("terrainMap");
  const context = canvas.getContext("2d");
  const { width, height } = canvas;
  context.clearRect(0, 0, width, height);
  context.fillStyle = terrainState.kind === "flat" ? "#e7eee8" : "#dde8eb";
  context.fillRect(0, 0, width, height);
  context.strokeStyle = "#c1d0d5";
  context.lineWidth = 1;
  for (let x = 0; x <= width; x += width / 8) { context.beginPath(); context.moveTo(x, 0); context.lineTo(x, height); context.stroke(); }
  for (let y = 0; y <= height; y += height / 6) { context.beginPath(); context.moveTo(0, y); context.lineTo(width, y); context.stroke(); }
  if (terrainState.kind === "slope") { context.fillStyle = "#9db6c5"; context.fillRect(width * .56, height * .16, width * .29, height * .68); }
  if (terrainState.kind === "stairs") {
    context.fillStyle = "#9db6c5";
    for (let index = 0; index < 7; index += 1) context.fillRect(width * (.54 + index * .038), height * .18, width * .034, height * .64);
  }
  terrainState.elements.forEach((element, index) => {
    const x = (element.x / 8 + .5) * width;
    const y = (.5 - element.y / 6) * height;
    context.beginPath();
    context.arc(x, y, index === terrainState.selected ? 12 : 9, 0, Math.PI * 2);
    context.fillStyle = index === terrainState.selected ? "#185a91" : "#6f94ae";
    context.fill();
    context.fillStyle = "#fff";
    context.font = "10px system-ui";
    context.textAlign = "center";
    context.fillText(String(index + 1), x, y + 3);
  });
}

function renderTerrainElements() {
  const mount = $("terrainElements");
  mount.replaceChildren(...terrainState.elements.map((element, index) => {
    const row = document.createElement("div");
    row.className = `terrain-element${index === terrainState.selected ? " selected" : ""}`;
    row.innerHTML = `<span>${index + 1}. ${terrainLabels[element.kind] ?? "导入障碍"}</span>`;
    row.onclick = () => { terrainState.selected = index; repaintTerrainEditor(); renderTerrainElements(); };
    const remove = document.createElement("button");
    remove.textContent = "删除";
    remove.onclick = (event) => {
      event.stopPropagation();
      terrainState.elements.splice(index, 1);
      terrainState.selected = null;
      repaintTerrainEditor();
    };
    row.append(remove);
    return row;
  }));
}

function repaintTerrainEditor() {
  repaintTerrainMap();
  renderTerrainElements();
  renderTerrain();
}

function addTerrainElement(position) {
  const nearby = terrainState.elements.findIndex((element) => Math.hypot(element.x - position.x, element.y - position.y) < .26);
  if (nearby >= 0) terrainState.selected = nearby;
  else {
    terrainState.elements.push({ kind: terrainState.tool, x: position.x, y: position.y, yaw: 0 });
    terrainState.selected = terrainState.elements.length - 1;
  }
  repaintTerrainEditor();
}

function importTerrainScene(file) {
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const content = String(reader.result);
      if (file.name.toLowerCase().endsWith(".xml")) {
        const document = new DOMParser().parseFromString(content, "text/xml");
        const imported = [...document.querySelectorAll('geom[type="box"]')].map((geom) => {
          const position = vector(geom.getAttribute("pos") ?? "", 3, [0, 0, 0]);
          const size = vector(geom.getAttribute("size") ?? "", 3, [.8, .8, .2]);
          const euler = vector(geom.getAttribute("euler") ?? "", 3, [0, 0, 0]);
          return { kind: size[0] > 1.1 && size[1] < .3 ? "wall" : "platform", x: position[0], y: position[1], yaw: euler[1] || 0 };
        });
        if (!imported.length) throw new Error("未找到可导入的 box 地形；高度场和外部 mesh 需先导出为 ArenaX JSON。");
        terrainState.elements = imported;
      } else {
        const scene = JSON.parse(content);
        const terrain = scene.terrain ?? scene;
        if (terrain.kind && ["flat", "slope", "stairs", "obstacle_mix"].includes(terrain.kind)) terrainState.kind = terrain.kind;
        terrainState.seed = Number(terrain.seed ?? terrainState.seed);
        terrainState.height = Number(terrain.height ?? terrain.obstacle_height ?? terrainState.height);
        terrainState.elements = (scene.elements ?? []).map((element) => ({
          kind: ["platform", "stairs", "ramp", "stepping_stones", "high_wall"].includes(element.kind)
            ? ({ stepping_stones: "stones", high_wall: "wall" }[element.kind] ?? element.kind)
            : "platform",
          x: Number(element.x) || 0, y: Number(element.y) || 0, yaw: Number(element.yaw) || 0,
        }));
      }
      terrainState.selected = null;
      $("terrainKind").value = terrainState.kind;
      $("terrainSeed").value = terrainState.seed;
      $("terrainHeight").value = terrainState.height;
      updateTerrainHeight();
      setNotice("地形已导入预览；点击“应用到场景”后写入浏览器内 MuJoCo 物理模型。");
    } catch (error) { setNotice(error.message || String(error), true); }
  };
  reader.readAsText(file);
}

function exportTerrainScene() {
  const payload = { version: 1, name: "lloco-playground-terrain", terrain: { kind: terrainState.kind, seed: terrainState.seed, height: terrainState.height }, elements: terrainState.elements };
  const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "lloco-playground-terrain.json";
  anchor.click();
  URL.revokeObjectURL(url);
}

for (const [key, config] of Object.entries(policies)) $("policy").add(new Option(config.name, key));
$("policy").onchange = () => loadPolicy($("policy").value).catch((error) => setNotice(error.message, true));
$("start").onclick = () => {
  simulation.running = !simulation.running;
  $("start").textContent = simulation.running ? "暂停" : "继续";
};
$("reset").onclick = () => reset();
$("terrainToggle").onclick = () => {
  const panel = $("terrainPanel");
  panel.hidden = !panel.hidden;
  if (!panel.hidden) repaintTerrainEditor();
};
$("terrainClose").onclick = () => { $("terrainPanel").hidden = true; };
$("terrainKind").onchange = (event) => { terrainState.kind = event.target.value; repaintTerrainEditor(); };
$("terrainSeed").oninput = (event) => { terrainState.seed = Number(event.target.value) || 0; repaintTerrainEditor(); };
$("terrainHeight").oninput = updateTerrainHeight;
document.querySelectorAll("[data-terrain-tool]").forEach((button) => {
  button.onclick = () => {
    terrainState.tool = button.dataset.terrainTool;
    document.querySelectorAll("[data-terrain-tool]").forEach((item) => item.classList.toggle("active", item === button));
    $("terrainHint").textContent = `当前工具：${terrainLabels[terrainState.tool]}。在场地中点击放置。`;
  };
});
$("terrainMap").onclick = (event) => addTerrainElement(terrainCanvasPosition(event));
$("terrainClear").onclick = () => { terrainState.elements = []; terrainState.selected = null; repaintTerrainEditor(); };
$("terrainApply").onclick = () => {
  try { applyTerrain(); } catch (error) { setNotice(error.message || String(error), true); }
};
$("terrainExport").onclick = exportTerrainScene;
$("terrainImport").onchange = (event) => {
  const [file] = event.target.files;
  if (file) importTerrainScene(file);
  event.target.value = "";
};
const pressed = new Set();
const movementKeys = ["KeyQ", "KeyW", "KeyE", "KeyA", "KeyS", "KeyD"];
window.addEventListener("keydown", (event) => {
  if (!movementKeys.includes(event.code)) return;
  event.preventDefault();
  pressed.add(event.code);
});
window.addEventListener("keyup", (event) => {
  if (!movementKeys.includes(event.code)) return;
  pressed.delete(event.code);
});
setInterval(() => {
  if (!pressed.size) return;
  const forward = (pressed.has("KeyW") ? 0.05 : 0) - (pressed.has("KeyS") ? 0.05 : 0);
  const side = (pressed.has("KeyA") ? 0.05 : 0) - (pressed.has("KeyD") ? 0.05 : 0);
  const turn = (pressed.has("KeyQ") ? 0.05 : 0) - (pressed.has("KeyE") ? 0.05 : 0);
  const [min, max] = activePolicy.commandLimit;
  simulation.command[0] = THREE.MathUtils.clamp(simulation.command[0] + forward, min, max);
  simulation.command[1] = THREE.MathUtils.clamp(simulation.command[1] + side, -1, 1);
  simulation.command[2] = THREE.MathUtils.clamp(simulation.command[2] + turn, -1, 1);
  updateControls();
}, 80);

setBootStage("加载机器人场景…", 8);
const robotTask = buildRobot().then(() => setBootStage("机器人场景已就绪…", 38));
const engineTask = initializeMujoco().then(() => setBootStage("MuJoCo 物理引擎已就绪…", 68));

Promise.all([robotTask, engineTask])
  .then(() => {
    setBootStage("加载首个策略网络…", 82);
    return loadPolicy("handstand");
  })
  .then(() => {
    $("start").disabled = false;
    $("reset").disabled = false;
    $("terrainToggle").disabled = false;
    setStatus("浏览器物理引擎已就绪", "ready");
    setBootStage("场景已就绪", 100);
    tick();
    requestAnimationFrame(revealScene);
  })
  .catch((error) => {
    console.error(error);
    setStatus("试玩区初始化失败", "error");
    setNotice(error.message || String(error), true);
    setBootStage("初始化失败，请检查浏览器控制台", 100);
  });
