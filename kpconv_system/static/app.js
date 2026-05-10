import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";

const form = document.querySelector("#segmentForm");
const fileInput = document.querySelector("#plyFile");
const weightSelect = document.querySelector("#weightSelect");
const runButton = document.querySelector("#runButton");
const statusText = document.querySelector("#statusText");
const downloadLink = document.querySelector("#downloadLink");
const summaryText = document.querySelector("#summaryText");
const classStats = document.querySelector("#classStats");
const viewerHint = document.querySelector("#viewerHint");
const viewer = document.querySelector("#viewer");
const pointBadge = document.querySelector("#pointBadge");
const fileLabel = document.querySelector("#fileLabel");
const pipelineSteps = [...document.querySelectorAll(".pipeline-step")];

let scene;
let camera;
let renderer;
let controls;
let activeObject;

initViewer();
loadWeights();

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  fileLabel.textContent = file ? file.name : "选择或拖入点云文件";
  if (file) {
    setPipelineStep("upload");
    setStatus(`已选择 ${file.name}，请选择模型权重后开始分割。`);
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!fileInput.files.length) {
    setStatus("请先选择 .ply 点云文件。", true);
    return;
  }
  if (!weightSelect.value) {
    setStatus("请先选择模型权重。", true);
    return;
  }

  const body = new FormData();
  body.append("file", fileInput.files[0]);
  body.append("weight_path", weightSelect.value);

  runButton.disabled = true;
  downloadLink.classList.add("hidden");
  setPipelineStep("infer");
  setStatus("正在上传并推理，点云较大时可能需要等待一会儿。");

  try {
    const response = await fetch("/api/segment", { method: "POST", body });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "分割失败。");
    }

    renderStats(payload.result);
    pointBadge.textContent = `${payload.result.num_points.toLocaleString()} points`;
    downloadLink.href = payload.colored_ply_url;
    downloadLink.classList.remove("hidden");
    await loadPly(payload.colored_ply_url);
    setPipelineStep("view");
    setStatus("分割完成，彩色点云已生成。");
  } catch (error) {
    setPipelineStep("upload");
    setStatus(error.message, true);
  } finally {
    runButton.disabled = false;
  }
});

async function loadWeights() {
  try {
    const response = await fetch("/api/weights");
    const payload = await response.json();
    weightSelect.innerHTML = "";

    if (!payload.weights.length) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "未找到权重，请放入 kpconv_system/weights";
      weightSelect.append(option);
      setStatus("未扫描到权重文件。请将 .tar/.pth/.pt 放入 weights，或保留 KPConv results 训练日志目录。", true);
      return;
    }

    for (const item of payload.weights) {
      const option = document.createElement("option");
      option.value = item.path;
      option.textContent = item.status === "ready" ? item.name : `${item.name}（缺少配置）`;
      option.disabled = item.status !== "ready";
      weightSelect.append(option);
    }

    const firstReady = payload.weights.find((item) => item.status === "ready");
    if (firstReady) {
      weightSelect.value = firstReady.path;
      setStatus("权重扫描完成，可以上传点云开始分割。");
    } else {
      setStatus("扫描到权重，但没有找到对应 parameters.txt。", true);
    }
  } catch (error) {
    setStatus(`权重扫描失败：${error.message}`, true);
  }
}

function initViewer() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf7f9fc);

  camera = new THREE.PerspectiveCamera(55, viewer.clientWidth / viewer.clientHeight, 0.01, 5000);
  camera.position.set(2.5, 2.5, 2.0);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(viewer.clientWidth, viewer.clientHeight);
  viewer.append(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  const light = new THREE.HemisphereLight(0xffffff, 0x78909c, 1.8);
  scene.add(light);
  scene.add(new THREE.GridHelper(6, 12, 0xcbd5e1, 0xe2e8f0));

  window.addEventListener("resize", resizeViewer);
  animate();
}

async function loadPly(url) {
  const loader = new PLYLoader();
  const geometry = await loader.loadAsync(url);
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();

  if (activeObject) {
    scene.remove(activeObject);
    activeObject.geometry.dispose();
    activeObject.material.dispose();
  }

  const material = new THREE.PointsMaterial({
    size: pointSizeForGeometry(geometry),
    vertexColors: true,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.96,
  });
  activeObject = new THREE.Points(geometry, material);
  scene.add(activeObject);

  frameGeometry(geometry);
  viewerHint.classList.add("hidden");
}

function pointSizeForGeometry(geometry) {
  const radius = geometry.boundingSphere?.radius || 1;
  return Math.max(radius / 450, 0.006);
}

function frameGeometry(geometry) {
  const center = new THREE.Vector3();
  const sphere = geometry.boundingSphere;
  sphere.getCenter(center);
  controls.target.copy(center);

  const distance = Math.max(sphere.radius * 2.8, 1);
  camera.position.set(center.x + distance, center.y + distance, center.z + distance * 0.7);
  camera.near = Math.max(distance / 1000, 0.001);
  camera.far = distance * 20;
  camera.updateProjectionMatrix();
  controls.update();
}

function renderStats(result) {
  summaryText.textContent = `${result.num_points.toLocaleString()} 点，推理点 ${result.num_inference_points.toLocaleString()}，耗时 ${result.elapsed_seconds}s`;
  classStats.innerHTML = "";

  for (const cls of result.classes.filter((item) => item.count > 0)) {
    const row = document.createElement("div");
    row.className = "class-row";
    row.innerHTML = `
      <span class="swatch" style="background: rgb(${cls.color.join(",")})"></span>
      <span class="class-name">${cls.name}</span>
      <span class="class-count">${cls.count.toLocaleString()}</span>
      <span class="class-percent">${cls.percent}%</span>
    `;
    classStats.append(row);
  }

  if (result.messages?.length) {
    setStatus(result.messages.join(" "));
  }
}

function resizeViewer() {
  const width = viewer.clientWidth;
  const height = viewer.clientHeight;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

function setStatus(message, isError = false) {
  statusText.textContent = message;
  statusText.classList.toggle("error", isError);
}

function setPipelineStep(step) {
  const order = ["upload", "infer", "view"];
  const activeIndex = order.indexOf(step);
  pipelineSteps.forEach((item) => {
    const itemIndex = order.indexOf(item.dataset.step);
    item.classList.toggle("active", itemIndex <= activeIndex);
  });
}
