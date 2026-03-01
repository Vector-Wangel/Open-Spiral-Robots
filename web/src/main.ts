/**
 * OpenSpiRobs Web — MuJoCo Simulator
 * Loads pre-exported XML+STL scenes or user-uploaded files into MuJoCo WASM.
 */

import { MujocoViewer } from './simulation/mujoco-viewer';
import GUI from 'lil-gui';

// Built-in scenes (served from public/assets/)
const SCENES: Record<string, { xml: string; stl: string }> = {
  '2-Cable Spiral Robot': { xml: './assets/2cable/robot.xml', stl: './assets/2cable/baselink.stl' },
  '3-Cable Spiral Robot': { xml: './assets/3cable/robot.xml', stl: './assets/3cable/baselink.stl' },
};

// DOM
const simCanvas = document.getElementById('sim-canvas') as HTMLCanvasElement;
const rightPanel = document.getElementById('right-panel') as HTMLElement;

// MuJoCo viewer
const viewer = new MujocoViewer(simCanvas);

// GUI
const gui = new GUI({ container: rightPanel, title: 'OpenSpiRobs', width: 270 });

const sceneFolder = gui.addFolder('Scene');
const sceneParams = { scene: Object.keys(SCENES)[0] };
sceneFolder.add(sceneParams, 'scene', Object.keys(SCENES)).name('Default Scene').onChange((name: string) => {
  loadBuiltinScene(name);
});

const uploadBtn = sceneFolder.add({ upload: () => uploadScene() }, 'upload').name('Upload XML + STL');
uploadBtn.domElement.closest('.controller')?.classList.add('primary');

// Load the default scene on start
loadBuiltinScene(sceneParams.scene);

/** Fetch a built-in scene from public/assets/ and load into MuJoCo. */
async function loadBuiltinScene(name: string) {
  const scene = SCENES[name];
  if (!scene) return;

  try {
    console.log('[Scene] Loading built-in: %s', name);
    const [xmlResp, stlResp] = await Promise.all([
      fetch(scene.xml),
      fetch(scene.stl),
    ]);
    if (!xmlResp.ok) throw new Error(`Failed to fetch ${scene.xml}: ${xmlResp.status}`);
    if (!stlResp.ok) throw new Error(`Failed to fetch ${scene.stl}: ${stlResp.status}`);

    const xmlString = await xmlResp.text();
    const stlBuffer = await stlResp.arrayBuffer();

    await viewer.load(xmlString, stlBuffer, rightPanel, {
      onReload: () => loadBuiltinScene(name),
    });
    viewer.start();
  } catch (e) {
    console.error('[Scene] Failed to load built-in scene:', e);
  }
}

/** Prompt user to select XML + STL files, then load into MuJoCo. */
function uploadScene() {
  const input = document.createElement('input');
  input.type = 'file';
  input.multiple = true;
  input.accept = '.xml,.stl';
  input.addEventListener('change', async () => {
    const files = input.files;
    if (!files || files.length === 0) return;

    let xmlString: string | null = null;
    let stlBuffer: ArrayBuffer | null = null;

    for (const file of Array.from(files)) {
      const name = file.name.toLowerCase();
      if (name.endsWith('.xml')) {
        xmlString = await file.text();
      } else if (name.endsWith('.stl')) {
        stlBuffer = await file.arrayBuffer();
      }
    }

    if (!xmlString) { console.warn('[Upload] No XML file selected'); return; }
    if (!stlBuffer) { console.warn('[Upload] No STL file selected'); return; }

    console.log('[Upload] Loaded XML (%d bytes) + STL (%d bytes)', xmlString.length, stlBuffer.byteLength);

    try {
      await viewer.load(xmlString, stlBuffer, rightPanel, {
        onReload: () => uploadScene(),
      });
      viewer.start();
    } catch (e) {
      console.error('[Upload] Failed to load uploaded scene:', e);
    }
  });
  input.click();
}

console.log('OpenSpiRobs MuJoCo Simulator loaded.');
