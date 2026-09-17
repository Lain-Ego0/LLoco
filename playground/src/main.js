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
let mujoco, model, data, policySession, activePolicy, jointAddresses, actuatorAddresses;

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
  return { scene, camera, renderer, orbit };
}

const view = createScene();
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
  const vfs = new mujoco.MjVFS();
  for (const [file, buffer] of binaries) vfs.addBuffer(`assets/${file}`, buffer);
  model = mujoco.MjModel.from_xml_string(xml, vfs);
  data = new mujoco.MjData(model);
  jointAddresses = jointNames.map((name) => ({
    qpos: model.jnt(name).qposadr,
    dof: model.jnt(name).dofadr,
  }));
  actuatorAddresses = actuatorNames.map((name) => model.actuator(name).id);
  vfs.delete();
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

for (const [key, config] of Object.entries(policies)) $("policy").add(new Option(config.name, key));
$("policy").onchange = () => loadPolicy($("policy").value).catch((error) => setNotice(error.message, true));
$("start").onclick = () => {
  simulation.running = !simulation.running;
  $("start").textContent = simulation.running ? "暂停" : "继续";
};
$("reset").onclick = () => reset();
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
