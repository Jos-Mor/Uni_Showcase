import {
  loadShadersFromURLS,
  loadShadersFromScripts,
  setupWebGL,
  buildProgramFromSources,
} from "../libs/utils.js";
import { vec2, flatten, sizeof } from "../libs/MV.js";

/** @type {WebGL2RenderingContext} */
var gl;
/** @type {WebGLProgram} */
var drawProgram;
/** @type {WebGLProgram} */
var iterationProgram;
/** @type {HTMLCanvasElement} */
var canvas;

var aspect;
/** @type {WebGLBuffer} */
var aBuffer, bBuffer;
/** @type {HTMLDivElement} */
var hudElement;

var vertices;
const nPoints = 500000;
const defaultIterationNumber = 50;
var alternateIterationNumber = -1;
var currentIterations;

var isDragging = false;
var offset = { x: 0, y: 0 }; //center of the screen in relation to the one initialized
var dragInit = { x: 0, y: 0 }; //initial coordinates where mouse started dragging
const zoomFactor = 2;
var zoomLevel = 0.0;
const pointSize = 1.3;
var bottomLeft = {x: 0, y:0};
var topRight = {x:0, y:0};

const IFS_BarnsleyFern = {
  matrices: [
    [
      [0.0, 0.0, 0.0],
      [0.0, 0.16, 0.0],
      [0.0, 0.0, 1.0],
    ],
    [
      [0.85, 0.04, 0.0],
      [-0.04, 0.85, 1.6],
      [0.0, 0.0, 1.0],
    ],
    [
      [0.2, -0.26, 0.0],
      [0.23, 0.22, 1.6],
      [0.0, 0.0, 1.0],
    ],
    [
      [-0.15, 0.28, 0.0],
      [0.26, 0.24, 0.44],
      [0.0, 0.0, 1.0],
    ],
  ],
  probabilities: [0.01, 0.85, 0.07, 0.07],
  numFunctions: 4,
};

const IFS_CulcitaFern = {
  matrices: [
    [
      [0, 0, 0],
      [0, 0.25, -0.14],
      [0, 0, 1],
    ],
    [
      [0.85, 0.002, 0],
      [-0.02, 0.83, 1],
      [0, 0, 1],
    ],
    [
      [0.09, -0.28, 0],
      [0.3, 0.11, 0.6],
      [0, 0, 1],
    ],
    [
      [-0.09, 0.28, 0],
      [0.3, 0.09, 0.7],
      [0, 0, 1],
    ],
  ],
  probabilities: [0.02, 0.84, 0.07, 0.07],
  numFunctions: 4,
};

const IFS_CyclosorusFern = {
  matrices: [
    [
      [0, 0, 0],
      [0, 0.25, -0.4],
      [0, 0, 1],
    ],
    [
      [0.95, 0.005, -0.002],
      [-0.005, 0.93, 0.5],
      [0, 0, 1],
    ],
    [
      [0.035, -0.2, -0.09],
      [0.16, 0.04, 0.02],
      [0, 0, 1],
    ],
    [
      [-0.04, 0.2, 0.083],
      [0.16, 0.04, 0.12],
      [0, 0, 1],
    ],
  ],
  probabilities: [0.02, 0.84, 0.07, 0.07],
  numFunctions: 4,
};

const IFS_FishboneFern = {
  matrices: [
    [
      [0, 0, 0],
      [0, 0.25, -0.4],
      [0, 0, 1],
    ],
    [
      [0.95, 0.002, -0.002],
      [-0.002, 0.93, 0.5],
      [0, 0, 1],
    ],
    [
      [0.035, -0.11, -0.05],
      [0.27, 0.01, 0.005],
      [0, 0, 1],
    ],
    [
      [-0.04, 0.11, 0.047],
      [0.27, 0.01, 0.06],
      [0, 0, 1],
    ],
  ],
  probabilities: [0.02, 0.84, 0.07, 0.07],
  numFunctions: 4,
};

const IFS_Spiral = {
  matrices: [
    [
      [0.787879, 0.424242, 1.758647],
      [-0.212346, 0.864198, 1.408065],
      [0, 0, 1],
    ],
    [
      [0.088272, 0.520988, 0.78536],
      [-0.463889, -0.377778, 8.095795],
      [0, 0, 1],
    ],
    [
      [0.181818, -0.136364, 6.086107],
      [0.090909, 0.181818, 1.568035],
      [0, 0, 1],
    ],
  ],
  probabilities: [0.9, 0.05, 0.05],
  numFunctions: 3,
};

const IFS_MandelbrotLike = {
  matrices: [
    [
      [0.202, -0.805, -0.373],
      [-0.689, -0.342, -0.653],
      [0, 0, 1],
    ],
    [
      [0.138, 0.665, 0.66],
      [-0.502, -0.222, -0.277],
      [0, 0, 1],
    ],
  ],
  probabilities: [0.5, 0.5],
  numFunctions: 2,
};

const IFS_Tree1 = {
  matrices: [
    [
      [0.05, 0, -0.06],
      [0, 0.4, -0.47],
      [0, 0, 1],
    ],
    [
      [-0.05, 0, -0.06],
      [0, -0.4, -0.47],
      [0, 0, 1],
    ],
    [
      [0.03, -0.14, -0.16],
      [0, 0.26, -0.01],
      [0, 0, 1],
    ],
    [
      [-0.03, 0.14, -0.16],
      [0, -0.26, -0.01],
      [0, 0, 1],
    ],
    [
      [0.56, 0.44, 0.3],
      [-0.37, 0.51, 0.15],
      [0, 0, 1],
    ],
    [
      [0.19, 0.07, -0.2],
      [-0.1, 0.15, 0.28],
      [0, 0, 1],
    ],
    [
      [-0.33, -0.34, -0.54],
      [-0.33, 0.34, 0.39],
      [0, 0, 1],
    ],
  ],
  probabilities: [1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7],
  numFunctions: 7,
};

const IFS_Tree2 = {
  matrices: [
    [
      [0.01, 0, 0],
      [0, 0.45, 0],
      [0, 0, 1],
    ],
    [
      [-0.01, 0, 0],
      [0, -0.45, 0.4],
      [0, 0, 1],
    ],
    [
      [0.42, -0.42, 0],
      [0.42, 0.42, 0.4],
      [0, 0, 1],
    ],
    [
      [0.42, 0.42, 0],
      [-0.42, 0.42, 0.4],
      [0, 0, 1],
    ],
  ],
  probabilities: [0.25, 0.25, 0.25, 0.25],
  numFunctions: 4,
};

const IFS_Dragon = {
  matrices: [
    [
      [0.824074, 0.281428, -1.88229],
      [-0.212346, 0.864198, -0.110607],
      [0, 0, 1],
    ],
    [
      [0.088272, 0.520988, 0.78536],
      [-0.463889, -0.377778, 8.095795],
      [0, 0, 1],
    ],
  ],
  probabilities: [0.8, 0.2],
  numFunctions: 2,
};

const IFS_MapleLeaf = {
  matrices: [
    [
      [0.14, 0.01, -0.08],
      [0, 0.51, -1.31],
      [0, 0, 1],
    ],
    [
      [0.43, 0.52, 1.49],
      [-0.45, 0.5, -0.75],
      [0, 0, 1],
    ],
    [
      [0.45, -0.49, -1.62],
      [0.47, 0.47, -0.74],
      [0, 0, 1],
    ],
    [
      [0.49, 0, 0.02],
      [0, 0.51, 1.62],
      [0, 0, 1],
    ],
  ],
  probabilities: [0.25, 0.25, 0.25, 0.25],
  numFunctions: 4,
};

var chosenIFS = IFS_BarnsleyFern;
var ifs = 0;

/**
 * Resizes the viewport and redraws the buffer accordingly
 */
function resize() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;

  aspect = canvas.width / canvas.height;

  //Set canvas
  setCanvas();

  // Setup the viewport
  updateViewPort();
  redraw();
}

function startDragging(e) {
  isDragging = true;

  // Calculate the offset from the mouse position to the center
  dragInit.x = e.clientX - offset.x;
  dragInit.y = -e.clientY - offset.y;
}

function stopDragging() {
  isDragging = false;
}

function dragViewport(e) {
  if (!isDragging) return;

  const newX = e.clientX - dragInit.x;
  const newY = -e.clientY - dragInit.y;

  /*
    const mouseX = (e.clientX / canvas.clientWidth) * 2 - 1;
    const mouseY = -((e.clientY / canvas.clientHeight) * 2 - 1);

    const newCenterX = mouseX + mouseOffset.x;
    const newCenterY = mouseY + mouseOffset.y;
    */

  offset.x = newX;
  offset.y = newY;

  // Change the viewport
  updateViewPort();

  redraw();
}

function zoom(e) {
  var direction = detectMouseWheelDirection(e);
  var zoomChange = direction * zoomFactor;

  zoomLevel += zoomChange;

  // Ensure the zoom level stays between 0.1x and 3x
  zoomLevel = Math.max(zoomLevel, -1000);
  zoomLevel = Math.min(zoomLevel, 1000);

  resize();
}

function detectMouseWheelDirection(e) {
  var delta = null;
  var direction = false;
  if (!e) {
    // if the event is not provided, do nothing
    console.log("No event provided for mousewheel");
    return;
  }
  if (e.wheelDelta) {
    // will work in most cases
    delta = e.wheelDelta / 60;
  } else if (e.detail) {
    // fallback for Firefox
    delta = -e.detail / 2;
  }
  if (delta !== null) {
    direction = delta > 0 ? -1 : 1;
  }

  return direction;
}

function updateHUD() {
  hudElement.innerHTML = `Scale: ${zoomLevel}<br>Offset: ${offset.x},${offset.y} <br>Points: ${nPoints}<br>Iterations: ${currentIterations - 1}`;
}

function updateViewPort() {
  var zoom = zoomLevel * 10;
  gl.viewport(
    offset.x + zoom * aspect,
    offset.y + zoom,
    canvas.width - zoom * aspect * 2,
    canvas.height - zoom * 2
  );
}

function setCanvas(){
  switch(ifs){
    case 0:
      bottomLeft.x = -5.0*aspect;
      bottomLeft.y = 0.0;
      topRight.x = 5*aspect;
      topRight.y = 10.0;
      break;  
    case 1:
      bottomLeft.x = -3*aspect;
      bottomLeft.y = 0.0;
      topRight.x = 3*aspect;
      topRight.y = 6.0;      
      break;
    case 2:
      bottomLeft.x = -3.6*aspect;
      bottomLeft.y = 0.0;
      topRight.x = 3.6*aspect;
      topRight.y = 7.2;
      break;
    case 3:
      bottomLeft.x = -3.6*aspect;
      bottomLeft.y = 0.0;
      topRight.x = 3.6*aspect;
      topRight.y = 7.2;
      break;
    case 4:
      bottomLeft.x = -2*aspect;
      bottomLeft.y = -8.0;
      topRight.x = 16*aspect;
      topRight.y = 10.0;
      break;
    case 5:
      bottomLeft.x = -1*aspect;
      bottomLeft.y = -1.5;
      topRight.x = 1*aspect;
      topRight.y = 0.5;
      break;
    case 6:
      bottomLeft.x = -1.5* aspect;
      bottomLeft.y = -1.5;
      topRight.x = 1.5*aspect;
      topRight.y = 1.5;
      break;
    case 7:
      bottomLeft.x = -0.6*aspect;
      bottomLeft.y = -0.1;
      topRight.x = 0.6*aspect;
      topRight.y = 1.1;
      break;
    case 8:
      bottomLeft.x = -9* aspect;
      bottomLeft.y = -5.0;
      topRight.x = 9*aspect;
      topRight.y = 13.0;
      break;
    case 9:
      bottomLeft.x = -5*aspect;
      bottomLeft.y = -5.0;
      topRight.x = 5*aspect;
      topRight.y = 5.0;
      break;
    default:
      bottomLeft.x = -5.0*aspect;
      bottomLeft.y = 0.0;
      topRight.x = 5*aspect;
      topRight.y = 10.0;
      break;
  }

}

/**
 * Self-explanatory. Duplicated code turned into a function
 */
function createPointsAndBuffer() {
  vertices = [];

  for (let i = 0; i < nPoints; i++) {
    vertices.push(vec2(Math.random(), Math.random()));
  }

  aBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, aBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, flatten(vertices), gl.STREAM_DRAW);
  console.log(flatten(vertices));

  bBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, bBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, nPoints * sizeof["vec2"], gl.STREAM_DRAW);

  gl.useProgram(iterationProgram);

  for (let i = 0; i < chosenIFS.numFunctions; i++) {
    const matrixLocation = gl.getUniformLocation(
      iterationProgram,
      "m[" + i + "]"
    );
    gl.uniformMatrix3fv(matrixLocation, true, flatten(chosenIFS.matrices[i]));

    const probabilityLocation = gl.getUniformLocation(
      iterationProgram,
      "p[" + i + "]"
    );
    gl.uniform1f(probabilityLocation, chosenIFS.probabilities[i]);
  }

  const numFunctionsLocation = gl.getUniformLocation(
    iterationProgram,
    "nfuncs"
  );

  gl.uniform1i(numFunctionsLocation, chosenIFS.numFunctions);
}

function setup(shaders) {
  // Setup
  canvas = document.getElementById("gl-canvas");
  gl = setupWebGL(canvas, { alpha: true });
  hudElement = document.getElementById("hud");
  currentIterations = 0;

  drawProgram = buildProgramFromSources(
    gl,
    shaders["shader.vert"],
    shaders["shader.frag"]
  );
  iterationProgram = buildProgramFromSources(
    gl,
    shaders["iteration.vert"],
    shaders["iteration.frag"],
    ["vNewPos"]
  );

  createPointsAndBuffer();

  //Events

  window.addEventListener("resize", resize);
  resize();

  window.addEventListener("keydown", (event) => {
    if (event.defaultPrevented) {
      return; //Should do nothing if the default action has been cancelled
    }

    let handled = false;
    if (event.key != undefined) {
      //Decide what to do when certain keys are pressed
      switch (event.key) {
        case "+":
          window.requestAnimationFrame(function () {
            animate(1);
          });
          break;
        case "-":
          alternateIterationNumber = currentIterations - 1;
          restart(alternateIterationNumber);
          break;
        case "i":
          zoomLevel = 0.0;
          offset.x = 0;
          offset.y = 0;
          updateViewPort();
          alternateIterationNumber = 0;
          restart(0);
          break;
        case '0':
          chosenIFS = IFS_BarnsleyFern;
          ifs = 0;
          zoomLevel = 0.0;
          offset.x = 0;
          offset.y = 0;
          resize();
          alternateIterationNumber = 0;
          restart(defaultIterationNumber+1);
          break;
        case '1':
          chosenIFS = IFS_CulcitaFern;
          ifs = 1;
          zoomLevel = 0.0;
          offset.x = 0;
          offset.y = 0;
          resize();
          alternateIterationNumber = 0;
          restart(defaultIterationNumber+1);
          break;  
        case '2':
          chosenIFS = IFS_CyclosorusFern;
          ifs = 2;
          zoomLevel = 0.0;
          offset.x = 0;
          offset.y = 0;
          resize();
          alternateIterationNumber = 0;
          restart(defaultIterationNumber+1);
          break;  
        case '3':
          chosenIFS = IFS_FishboneFern;
          ifs = 3;
          zoomLevel = 0.0;
          offset.x = 0;
          offset.y = 0;
          resize();
          alternateIterationNumber = 0;
          restart(defaultIterationNumber+1);
          break;
        case '4':
          chosenIFS = IFS_Spiral;
          ifs = 4;
          zoomLevel = 0.0;
          offset.x = 0;
          offset.y = 0;
          resize();
          alternateIterationNumber = 
          restart(defaultIterationNumber+1);0;
          break;
        case '5':
          chosenIFS = IFS_MandelbrotLike;
          ifs = 5;
          zoomLevel = 0.0;
          offset.x = 0;
          offset.y = 0;
          resize();
          alternateIterationNumber = 0;
          restart(defaultIterationNumber+1);
          break;
        case '6':
          chosenIFS = IFS_Tree1;
          ifs = 6;
          zoomLevel = 0.0;
          offset.x = 0;
          offset.y = 0;
          resize();
          alternateIterationNumber =
          restart(defaultIterationNumber+1); 0;
          break;
        case '7':
          chosenIFS = IFS_Tree2;
          ifs = 7;
          zoomLevel = 0.0;
          offset.x = 0;
          offset.y = 0;
          resize();
          alternateIterationNumber =
          restart(defaultIterationNumber+1); 0;
          break;
        case '8':
          chosenIFS = IFS_Dragon;
          ifs = 8;
          zoomLevel = 0.0;
          offset.x = 0;
          offset.y = 0;
          resize();
          alternateIterationNumber = 
          restart(defaultIterationNumber+1);0;
          break;
        case '9':
          chosenIFS = IFS_MapleLeaf;
          ifs = 9;
          zoomLevel = 0.0;
          offset.x = 0;
          offset.y = 0;
          resize();
          alternateIterationNumber = 0;
          restart(defaultIterationNumber+1);
          break;
        default:
          break;
      }
      handled = true;
    }
    if (handled) {
      // Suppress "double action" if event handled
      event.preventDefault();
    }
  });

  window.addEventListener("wheel", zoom);

  canvas.addEventListener("mousedown", startDragging);
  canvas.addEventListener("mousemove", dragViewport);
  canvas.addEventListener("mouseup", stopDragging);

  //Blend with background
  
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);


  // Setup the background color
  gl.clearColor(0.0, 0.0, 0.0, 1.0);

  // Call animate for the first time
  window.requestAnimationFrame(function () {
    if (alternateIterationNumber < 0) {
      animate(defaultIterationNumber + 1);
    } else {
      animate(alternateIterationNumber);
    }
  });
}

function animate(iterations) {
  // Drawing code

  gl.clear(gl.COLOR_BUFFER_BIT);

  gl.useProgram(drawProgram);

  const uBottomLeft = gl.getUniformLocation(drawProgram, "uBottomLeft");
  gl.uniform2f(uBottomLeft, bottomLeft.x, bottomLeft.y);
  const uTopRight = gl.getUniformLocation(drawProgram, "uTopRight");
  gl.uniform2f(uTopRight, topRight.x, topRight.y);
  const pointSizeLocation = gl.getUniformLocation(drawProgram, "u_pointSize");
  gl.uniform1f(pointSizeLocation, pointSize); // or any other suitable size

  gl.bindBuffer(gl.ARRAY_BUFFER, aBuffer);
  const vPosition = gl.getAttribLocation(drawProgram, "vPosition");
  gl.vertexAttribPointer(vPosition, 2, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(vPosition);

  gl.drawArrays(gl.Points, 0, nPoints);

  // Iteration code

  gl.useProgram(iterationProgram);

  gl.bindBuffer(gl.ARRAY_BUFFER, aBuffer);
  const vOldPos = gl.getAttribLocation(iterationProgram, "vOldPos");
  gl.vertexAttribPointer(vOldPos, 2, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(vOldPos);

  const transformFeedback = gl.createTransformFeedback();
  gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, transformFeedback);

  gl.enable(gl.RASTERIZER_DISCARD);

  gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, bBuffer);
  gl.beginTransformFeedback(gl.POINTS);
  gl.drawArrays(gl.POINTS, 0, nPoints);
  gl.endTransformFeedback();
  gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, null);

  gl.deleteTransformFeedback(transformFeedback);

  gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);

  gl.disable(gl.RASTERIZER_DISCARD);

  var temp = aBuffer;
  aBuffer = bBuffer;
  bBuffer = temp;

  currentIterations++;
  updateHUD();

  if (iterations > 1) {
    window.requestAnimationFrame(function () {
      animate(iterations - 1);
    });
  }
}

/**
 * Restarts buffers and counters (created so we don't have to run setup() again)
 * @param {number} iterations
 */
function restart(iterations) {
  gl.deleteBuffer(aBuffer);
  gl.deleteBuffer(bBuffer);
  currentIterations = 0;

  createPointsAndBuffer();

  window.requestAnimationFrame(function () {
    animate(iterations);
  });
}

function redraw() {
  gl.clear(gl.COLOR_BUFFER_BIT);

  gl.useProgram(drawProgram);

  const uBottomLeft = gl.getUniformLocation(drawProgram, "uBottomLeft");
  gl.uniform2f(uBottomLeft, bottomLeft.x, bottomLeft.y);
  const uTopRight = gl.getUniformLocation(drawProgram, "uTopRight");
  gl.uniform2f(uTopRight, topRight.x, topRight.y);

  gl.bindBuffer(gl.ARRAY_BUFFER, bBuffer);
  const vPosition = gl.getAttribLocation(drawProgram, "vPosition");
  gl.vertexAttribPointer(vPosition, 2, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(vPosition);

  gl.drawArrays(gl.Points, 0, nPoints);
  updateHUD();
}

loadShadersFromURLS([
  "shader.vert",
  "shader.frag",
  "iteration.vert",
  "iteration.frag",
]).then((shaders) => setup(shaders));