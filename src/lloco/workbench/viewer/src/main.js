import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const $ = (id) => document.getElementById(id);
const scene = new THREE.Scene();
scene.background = new THREE.Color("#edf0f3");
const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000);
camera.up.set(0, 0, 1);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
$("canvas").append(renderer.domElement);
const orbit = new OrbitControls(camera, renderer.domElement);
orbit.enableDamping = true;
scene.add(new THREE.HemisphereLight(0xdcefff, 0x435265, 2.5));
const light = new THREE.DirectionalLight(0xffffff, 3);
light.position.set(3, -4, 6);
scene.add(light);
const grid = new THREE.GridHelper(10, 100, 0xaab7c3, 0xd5dde4);
grid.rotation.x = Math.PI / 2;
scene.add(grid);
const layers = Object.fromEntries(
  ["visual", "collision", "frames", "axes", "inertia"].map((key) => {
    const group = new THREE.Group();
    scene.add(group);
    return [key, group];
  }),
);
let model,
  imported = [],
  loading = false,
  poseQueue = Promise.resolve();
const objects = new Map(),
  meshes = new Map();
function status(message, error = false) {
  $("status").textContent = message;
  $("status").classList.toggle("error", error);
}
async function api(action, body) {
  const response = await fetch("/api/" + action, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const result = await response.json();
  if (!response.ok) throw Error(result.error);
  return result;
}
function transform(object, position, rotation) {
  object.position.fromArray(position);
  const r = rotation.flat();
  object.quaternion.setFromRotationMatrix(
    new THREE.Matrix4().set(
      r[0],
      r[1],
      r[2],
      0,
      r[3],
      r[4],
      r[5],
      0,
      r[6],
      r[7],
      r[8],
      0,
      0,
      0,
      0,
      1,
    ),
  );
}
function geometry(geom) {
  const [x, y, z] = geom.size;
  switch (geom.type) {
    case 0:
      return new THREE.PlaneGeometry(x > 0 ? 2 * x : 10, y > 0 ? 2 * y : 10);
    case 2:
      return new THREE.SphereGeometry(x, 20, 12);
    case 3: {
      const g = new THREE.CapsuleGeometry(x, 2 * y, 6, 16);
      g.rotateX(Math.PI / 2);
      return g;
    }
    case 4: {
      const g = new THREE.SphereGeometry(1, 20, 12);
      g.scale(x, y, z);
      return g;
    }
    case 5: {
      const g = new THREE.CylinderGeometry(x, x, 2 * y, 24);
      g.rotateX(Math.PI / 2);
      return g;
    }
    case 6:
      return new THREE.BoxGeometry(2 * x, 2 * y, 2 * z);
    case 7:
      return meshes.get(String(geom.mesh))?.clone();
    default:
      return null;
  }
}
function clear() {
  for (const group of Object.values(layers)) {
    group.traverse((obj) => {
      obj.geometry?.dispose();
      if (obj.material) {
        for (const material of [obj.material].flat()) material.dispose();
      }
    });
    group.clear();
  }
  for (const g of meshes.values()) g.dispose();
  meshes.clear();
  objects.clear();
}
function visibility() {
  for (const [key, group] of Object.entries(layers))
    group.visible = $(key).checked;
}
function fit() {
  const box = new THREE.Box3();
  for (const geom of model?.geoms || []) {
    if (geom.body === 0) continue;
    const object = objects.get("g" + geom.id);
    if (object) box.expandByObject(object);
  }
  if (box.isEmpty())
    box.setFromCenterAndSize(
      new THREE.Vector3(0, 0, 0.5),
      new THREE.Vector3(1, 1, 1),
    );
  const center = box.getCenter(new THREE.Vector3()),
    distance = Math.max(box.getSize(new THREE.Vector3()).length(), 0.4) * 1.25;
  orbit.target.copy(center);
  camera.position
    .copy(center)
    .add(new THREE.Vector3(1, -1, 0.65).normalize().multiplyScalar(distance));
  orbit.update();
}
function properties() {
  const body = model?.bodies.find((b) => String(b.id) === $("body").value);
  if (body)
    $("properties").textContent =
      `质量  ${body.mass.toPrecision(5)} kg\n主惯量 [kg·m²]\n${body.inertia.map((v) => v.toPrecision(5)).join("\n")}`;
}
function update(pose) {
  if (!model || pose.id !== model.id) return;
  Object.assign(model, pose);
  for (const geom of pose.geoms) {
    for (const prefix of ["g", "c"]) {
      const obj = objects.get(prefix + geom.id);
      if (obj) transform(obj, geom.position, geom.rotation);
    }
  }
  for (const body of pose.bodies) {
    transform(objects.get("b" + body.id), body.position, body.rotation);
    transform(objects.get("i" + body.id), body.com, body.inertia_rotation);
  }
  for (const axis of pose.axes) {
    const arrow = objects.get("a" + axis.id);
    arrow.position.fromArray(axis.anchor);
    arrow.setDirection(new THREE.Vector3(...axis.axis));
    const input = $("j" + axis.id);
    if (input && document.activeElement !== input) input.value = axis.value;
    const output = $("v" + axis.id);
    if (output) output.value = axis.value.toFixed(3);
  }
  properties();
}
function setPose(body) {
  const id = model?.id;
  poseQueue = poseQueue
    .then(async () => {
      if (id !== model?.id) return;
      update(await api("model-pose", { id, ...body }));
    })
    .catch((error) => status(error.message, true));
}
function show(result) {
  clear();
  model = result;
  const unsupported = new Set();
  for (const [id, mesh] of Object.entries(model.meshes)) {
    const g = new THREE.BufferGeometry();
    g.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(mesh.vertices.flat(), 3),
    );
    g.setIndex(mesh.faces.flat());
    g.computeVertexNormals();
    meshes.set(id, g);
  }
  const visualBodies = new Set(
    model.geoms
      .filter((g) => !g.collision && g.color[3] > 0)
      .map((g) => g.body),
  );
  for (const geom of model.geoms) {
    if (geom.type === 0) continue; // The inspector uses its own ground grid.
    const g = geometry(geom);
    if (!g) {
      unsupported.add(geom.type);
      continue;
    }
    const [r, b, c, a] = geom.color;
    const visual = new THREE.Mesh(
      g,
      new THREE.MeshStandardMaterial({
        color: new THREE.Color(r, b, c),
        roughness: 0.65,
        metalness: 0.1,
        transparent: a < 1,
        opacity: a,
        side: THREE.DoubleSide,
      }),
    );
    visual.visible = !(geom.collision && visualBodies.has(geom.body));
    objects.set("g" + geom.id, visual);
    layers.visual.add(visual);
    if (geom.collision && geom.type !== 0) {
      const collision = new THREE.Mesh(
        g.clone(),
        new THREE.MeshBasicMaterial({
          color: 0x73e0ba,
          wireframe: true,
          transparent: true,
          opacity: 0.65,
        }),
      );
      objects.set("c" + geom.id, collision);
      layers.collision.add(collision);
    }
  }
  for (const body of model.bodies) {
    const axes = new THREE.AxesHelper(0.12);
    layers.frames.add(axes);
    objects.set("b" + body.id, axes);
    const ellipsoid = new THREE.Mesh(
      new THREE.SphereGeometry(1, 16, 12),
      new THREE.MeshBasicMaterial({
        color: 0xffc66f,
        wireframe: true,
        transparent: true,
        opacity: 0.7,
      }),
    );
    ellipsoid.scale.fromArray(body.radii);
    layers.inertia.add(ellipsoid);
    objects.set("i" + body.id, ellipsoid);
  }
  for (const axis of model.axes) {
    const arrow = new THREE.ArrowHelper(
      new THREE.Vector3(0, 0, 1),
      new THREE.Vector3(),
      0.18,
      0xec87b6,
      0.045,
      0.02,
    );
    layers.axes.add(arrow);
    objects.set("a" + axis.id, arrow);
  }
  $("joints").replaceChildren(
    ...model.joints.map((joint) => {
      const label = document.createElement("label");
      label.className = "joint";
      label.textContent = joint.name + (joint.slide ? " [m]" : " [rad]");
      const row = document.createElement("div"),
        input = document.createElement("input"),
        out = document.createElement("output");
      input.type = "range";
      input.id = "j" + joint.id;
      input.min = joint.limits[0];
      input.max = joint.limits[1];
      input.step = 0.001;
      input.value = joint.value;
      out.id = "v" + joint.id;
      out.value = joint.value.toFixed(3);
      input.oninput = () => {
        out.value = Number(input.value).toFixed(3);
      };
      input.onchange = () =>
        setPose({ joint: joint.id, value: Number(input.value) });
      row.append(input, out);
      label.append(row);
      return label;
    }),
  );
  $("body").replaceChildren(
    ...model.bodies.map((b) => new Option(b.name, b.id)),
  );
  $("name").textContent = model.name;
  $("counts").textContent =
    `${model.bodies.length} 连杆 · ${model.joints.length} 可调关节`;
  $("empty").hidden = true;
  update(model);
  visibility();
  fit();
  status(
    unsupported.size
      ? `已加载；暂不绘制几何类型 ${[...unsupported].join(", ")}（如高度场）。`
      : "模型已加载。可调整关节并切换检查图层。",
  );
}
async function load(body, path) {
  if (loading) return;
  loading = true;
  status("正在解析模型…");
  try {
    const result = await api("model-load", body);
    show(result);
    if (path)
      window.parent.postMessage(
        { type: "lloco:model-loaded", path },
        location.origin,
      );
  } catch (error) {
    status(error.message, true);
    if (path)
      window.parent.postMessage(
        { type: "lloco:model-error", message: error.message },
        location.origin,
      );
  } finally {
    loading = false;
  }
}
async function loadImported() {
  try {
    const files = await Promise.all(
      imported.map(
        (file) =>
          new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () =>
              resolve({
                name: file.webkitRelativePath || file.name,
                data: reader.result.split(",")[1],
              });
            reader.onerror = reject;
            reader.readAsDataURL(file);
          }),
      ),
    );
    await load({ selected: $("entry").value, files });
  } catch (error) {
    status(String(error), true);
  }
}
function choose(files) {
  imported = Array.from(files);
  const models = imported.filter((f) => /\.(xml|urdf)$/i.test(f.name));
  if (!models.length) {
    status("目录中没有 URDF / MJCF XML。", true);
    return;
  }
  $("entry").replaceChildren(
    ...models.map(
      (f) =>
        new Option(
          f.webkitRelativePath || f.name,
          f.webkitRelativePath || f.name,
        ),
    ),
  );
  $("entry").hidden = false;
  $("loadEntry").hidden = false;
  status("请选择入口模型，再点击“加载所选模型”。");
  if (models.length === 1) loadImported();
}
$("import").onclick = () => $("folder").click();
$("importFiles").onclick = () => $("files").click();
$("folder").onchange = (e) => choose(e.target.files);
$("files").onchange = (e) => choose(e.target.files);
$("loadEntry").onclick = loadImported;
$("fit").onclick = fit;
$("reset").onclick = () => setPose({ reset: true });
$("body").onchange = properties;
for (const key of Object.keys(layers)) $(key).onchange = visibility;
window.addEventListener("message", (event) => {
  if (
    event.origin === location.origin &&
    event.source === window.parent &&
    event.data?.type === "lloco:model"
  )
    load({ path: event.data.path }, event.data.path);
});
new ResizeObserver(() => {
  const w = $("canvas").clientWidth,
    h = $("canvas").clientHeight;
  if (w && h) {
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
}).observe($("canvas"));
visibility();
fit();
renderer.setAnimationLoop(() => {
  orbit.update();
  renderer.render(scene, camera);
});

export { model, api, show, update, scene, camera, orbit, layers, status };
if (new URLSearchParams(location.search).has("motion")) {
  import("./motion.js").then((module) => module.start());
}
