/**
 * Three.js 3D Tiles viewer — main entry point.
 *
 * Z-up coordinate system (matching MicroJSON world coords).
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TileManager } from './TileManager.js';
import { FeatureSelector } from './FeatureSelector.js';
import { InfoPanel } from './InfoPanel.js';
import { PyramidSelector } from './PyramidSelector.js';
import { SlicePlanePanel } from './SlicePlanePanel.js';

// --- Scene setup ---
const canvas = document.getElementById('canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);

// --- Camera (Z-up) ---
const camera = new THREE.PerspectiveCamera(50, 1, 1, 200000);
camera.up.set(0, 0, 1);

// --- Controls ---
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.1;
controls.minDistance = 500;
controls.maxDistance = 100000;

// --- Lighting ---
scene.add(new THREE.AmbientLight(0xffffff, 0.6));

const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(1, -0.5, 1).normalize();
scene.add(dirLight);

const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
dirLight2.position.set(-1, 0.5, -0.5).normalize();
scene.add(dirLight2);

// --- Tile Manager (baseUrl set after pyramid selection) ---
let tileManager = new TileManager(scene, '/tiles/default/');

// --- Slice Plane ---
const sliceContainer = document.getElementById('slice-controls');
const slicePlanePanel = new SlicePlanePanel(sliceContainer, { renderer, scene });

// --- Info Panel ---
const infoPanel = new InfoPanel(camera, scene, canvas);
infoPanel.slicePanel = slicePlanePanel;

// --- Feature Selector ---
const selectorContainer = document.getElementById('feature-selector');
const featureSelector = new FeatureSelector(selectorContainer, (selected) => {
    tileManager.setSelectedFeatures(selected);
});

// --- Stats ---
const statLoaded = document.getElementById('stat-loaded');
const statVisible = document.getElementById('stat-visible');
const statMemory = document.getElementById('stat-memory');
const statFPS = document.getElementById('stat-fps');
const loadingEl = document.getElementById('loading');

let lastTime = performance.now();
let frameCount = 0;
let fps = 0;

// --- GPU Budget Slider ---
const gpuSlider = document.getElementById('gpu-budget-slider');
const gpuLabel = document.getElementById('gpu-budget-label');
gpuSlider.addEventListener('input', () => {
    const mb = parseInt(gpuSlider.value);
    gpuLabel.textContent = mb;
    tileManager.maxGpuMB = mb;
});

// --- LOD Controls ---
const lodRadios = document.querySelectorAll('input[name="lod-mode"]');
const zoomSlider = document.getElementById('zoom-slider');
const zoomLabel = document.getElementById('zoom-label');
const zoomDistEl = document.getElementById('zoom-distribution');

for (const radio of lodRadios) {
    radio.addEventListener('change', () => {
        if (!radio.checked) return;  // ignore the unchecked radio's event
        tileManager.lodMode = radio.value;
        zoomSlider.disabled = (radio.value === 'dynamic');
        if (radio.value === 'forced') {
            tileManager.forcedZoom = parseInt(zoomSlider.value);
        }
        console.log(`[LOD] mode=${radio.value} forcedZoom=${tileManager.forcedZoom}`);
    });
}
zoomSlider.addEventListener('input', () => {
    const z = parseInt(zoomSlider.value);
    zoomLabel.textContent = z;
    tileManager.forcedZoom = z;
});

// --- Resize ---
function onResize() {
    const w = window.innerWidth - 300; // sidebar width
    const h = window.innerHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
}
window.addEventListener('resize', onResize);
onResize();

// --- Render loop ---
let animating = false;
function animate() {
    requestAnimationFrame(animate);
    controls.update();
    tileManager.update(camera);
    renderer.render(scene, camera);

    // Stats + FPS
    frameCount++;
    const now = performance.now();
    if (now - lastTime >= 1000) {
        fps = Math.round(frameCount * 1000 / (now - lastTime));
        frameCount = 0;
        lastTime = now;
        statFPS.textContent = fps;
    }
    statLoaded.textContent = tileManager.loadedCount;
    statVisible.textContent = tileManager.visibleCount;
    statMemory.textContent = tileManager.gpuMB;
    zoomDistEl.textContent = tileManager.zoomDistribution
        ? `LOD: ${tileManager.zoomDistribution}` : '';
}

/**
 * Reset camera to frame the tileset bounding volume.
 */
function resetCamera() {
    if (!tileManager.root?.box3) return;
    const box = tileManager.root.box3;
    const center = new THREE.Vector3();
    box.getCenter(center);
    const size = new THREE.Vector3();
    box.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z);

    camera.position.set(
        center.x + maxDim * 0.6,
        center.y - maxDim * 0.6,
        center.z + maxDim * 0.5,
    );
    camera.lookAt(center);
    controls.target.copy(center);
    controls.update();

    slicePlanePanel.updateBounds(box);
}

/**
 * Sync zoom slider to tile manager state after init/switch.
 */
function syncZoomSlider() {
    zoomSlider.max = tileManager.maxZoom;
    zoomSlider.value = tileManager.maxZoom;
    zoomLabel.textContent = tileManager.maxZoom;
    tileManager.forcedZoom = tileManager.maxZoom;
}

/**
 * Load a pyramid: init tile manager + feature selector, reset camera.
 */
async function loadPyramid(pyramid) {
    loadingEl.style.display = '';
    loadingEl.textContent = `Loading ${pyramid.label}...`;

    const baseUrl = `/tiles/${pyramid.id}/`;
    const featuresUrl = `/tiles/${pyramid.id}/features.json`;

    await tileManager.switchPyramid(baseUrl);
    await featureSelector.init(featuresUrl);
    syncZoomSlider();
    resetCamera();

    loadingEl.style.display = 'none';
    console.log(`Switched to pyramid "${pyramid.id}": ${Object.keys(tileManager.featureIndex).length} features`);
}

// --- Pyramid Selector ---
const pyramidContainer = document.getElementById('pyramid-selector');
const pyramidSelector = new PyramidSelector(pyramidContainer, async (pyramid) => {
    await loadPyramid(pyramid);
});

// --- Init ---
async function init() {
    try {
        const defaultPyramid = await pyramidSelector.init('/tiles/pyramids.json');

        if (defaultPyramid) {
            // Use the first pyramid from manifest
            const baseUrl = `/tiles/${defaultPyramid.id}/`;
            tileManager = new TileManager(scene, baseUrl);
            await tileManager.init();
            await featureSelector.init(`/tiles/${defaultPyramid.id}/features.json`);
        } else {
            // Fallback: no manifest, try legacy single-pyramid path
            tileManager = new TileManager(scene, '/tiles/');
            await tileManager.init();
            await featureSelector.init('/tiles/features.json');
        }

        syncZoomSlider();
        resetCamera();
        loadingEl.style.display = 'none';

        if (!animating) {
            animating = true;
            animate();
        }

        console.log(`Loaded feature index: ${Object.keys(tileManager.featureIndex).length} features`);
    } catch (err) {
        loadingEl.textContent = `Error: ${err.message}`;
        console.error('Init failed:', err);
    }
}

init();
