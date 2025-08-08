# Computer Graphics Course Projects

A collection of WebGL-based computer graphics applications developed for the Computer Graphics and Interfaces (CGI) course. This repository demonstrates fundamental computer graphics concepts including fractal generation, 3D rendering, lighting models, and interactive controls.

## Projects Overview

### 1. Interactive Fractal Generator
Real-time fractal visualization using Iterated Function Systems (IFS) with GPU-accelerated computation.

### 2. 3D Scene Renderer
Interactive 3D scene with multiple objects, Phong lighting model, and camera controls.

---

## Project 1: Interactive Fractal Generator

### Features
- **Multiple Fractal Types**: 10 different IFS implementations including Barnsley Fern, Dragon Curve, and Tree structures
- **GPU Computation**: Uses WebGL 2.0 Transform Feedback for efficient fractal iteration
- **Interactive Controls**: Mouse panning, wheel zooming, and keyboard shortcuts
- **Real-time Rendering**: 500,000 points rendered in real-time
- **Dual Buffer System**: Ping-pong rendering for smooth animation

### Controls
- **Mouse**: Drag to pan, scroll wheel to zoom
- **Keyboard**:
  - `0-9`: Switch between different fractal types
  - `+`: Advance iterations
  - `-`: Step back iterations  
  - `i`: Reset view

### Technical Implementation
- **WebGL 2.0** with Transform Feedback for GPU-based computation
- **Iterated Function Systems** with probability-based function selection
- **Custom vertex/fragment shaders** for point rendering and iteration
- **Viewport manipulation** for smooth zooming and panning

### Fractal Types Available
1. **Barnsley Fern** (0) - Classic fern fractal
2. **Culcita Fern** (1) - Alternative fern variation
3. **Cyclosorus Fern** (2) - Another fern type
4. **Fishbone Fern** (3) - Skeletal fern structure
5. **Spiral** (4) - Spiral pattern
6. **Mandelbrot-like** (5) - Mandelbrot-inspired pattern
7. **Tree 1** (6) - Tree structure with 7 functions
8. **Tree 2** (7) - Simpler tree structure
9. **Dragon** (8) - Dragon curve fractal
10. **Maple Leaf** (9) - Leaf-shaped fractal

---

## Project 2: 3D Scene Renderer

### Features
- **3D Object Rendering**: Cube, sphere, cow, and bunny models
- **Phong Lighting Model**: Ambient, diffuse, and specular lighting
- **Multiple Light Sources**: Three configurable point lights
- **Interactive Camera**: Mouse-controlled orbital camera
- **Material Properties**: Configurable ambient, diffuse, specular, and shininess
- **Wireframe Mode**: Toggle between solid and wireframe rendering
- **GUI Controls**: dat.GUI interface for real-time parameter adjustment

### Controls
- **Mouse**: 
  - Drag to rotate camera around scene
  - Scroll wheel to zoom
- **Keyboard**:
  - `1-4`: Select objects (Cube, Sphere, Cow, Bunny)
- **GUI**: Adjust camera parameters, object transforms, and materials

### Technical Implementation
- **WebGL** with custom vertex/fragment shaders
- **Matrix transformations** for 3D positioning and camera
- **Phong lighting** with multiple light sources
- **Normal matrix calculations** for proper lighting
- **Perspective projection** with configurable parameters

### Objects and Materials
- **Background Board**: Large textured surface
- **Geometric Primitives**: Cube and sphere with different materials
- **Complex Models**: Cow and bunny meshes demonstrating model loading
- **Dynamic Lighting**: Three rotating point lights illuminating the scene

### Running the Projects
```bash
cd *project_directory*
# Serve with any HTTP server, e.g.:
python -m http.server 8000
# Open http://localhost:8000
```
OR

Open with VSCode and utilize "open in live server" option on the html file

---

## Technical Stack

### Core Technologies
- **WebGL 2.0** (Project 1) / **WebGL 1.0** (Project 2)
- **JavaScript ES6** with module imports
- **HTML5 Canvas** for rendering surface
- **GLSL** for vertex and fragment shaders

### External Libraries
- **MV.js**: Matrix and vector mathematics
- **utils.js**: WebGL utilities and shader loading
- **dat.GUI**: Real-time parameter controls (Project 2)
- **Object libraries**: Predefined 3D models (cube, sphere, cow, bunny)

### Key Concepts Demonstrated
- **GPU Programming**: Transform feedback, shader programming
- **Computer Graphics**: 3D transformations, lighting models, projection
- **Mathematical Visualization**: Fractal geometry, iterated function systems
- **Interactive Design**: Mouse/keyboard controls, real-time parameter adjustment
- **Performance Optimization**: Efficient buffer management, GPU computation

---

## Setup and Requirements

### Prerequisites
- Modern web browser with WebGL 2.0 support
- Local HTTP server (required for ES6 modules)
- **WebGL Libraries**: The `libs/` folder with required utilities (included)

### Installation
1. Clone or download this repository
2. **Important**: Ensure the `libs/` folder is present in the Computer Graphics Project directory
3. Navigate to the desired project directory
4. Start a local HTTP server:
   ```bash
   # Python 3
   python -m http.server 8000
   
   # Python 2
   python -m SimpleHTTPServer 8000
   
   # Node.js
   npx http-server
   ```
4. Open `http://localhost:8000` in your browser

### Development Process

This repository includes my raw development notes (`proj3/notas.txt`) - unedited, typos and all. These notes show my actual learning process as I worked through complex graphics concepts like Phong lighting equations, vector mathematics, and debugging strategies. They're kept in their original form to demonstrate authentic problem-solving and learning, not polished post-completion documentation.

### File Structure
```
Computer Graphics Project/
├── proj1(altered)/           # Fractal Generator
│   ├── app.js               # Main application logic
│   ├── index.html           # HTML entry point
│   ├── style.css            # Styling
│   └── shaders/             # GLSL shaders
│       ├── shader.vert      # Vertex shader for rendering
│       ├── shader.frag      # Fragment shader for rendering
│       ├── iteration.vert   # Vertex shader for computation
│       └── iteration.frag   # Fragment shader for computation
├── proj3/                   # 3D Scene Renderer
│   ├── app.js               # Main application logic
│   ├── index.html           # HTML entry point
│   ├── style.css            # Styling
│   ├── notas.txt            # Raw development notes (authentic learning process)
│   └── shaders/             # GLSL shaders
│       ├── shader.vert      # Vertex shader
│       ├── shader.frag      # Fragment shader
│       ├── board.vert       # Board-specific vertex shader
│       └── board.frag       # Board-specific fragment shader
└── README.md                # This file
```

---

## Skills Demonstrated

### Computer Graphics Programming
- WebGL API usage and shader programming
- 3D mathematics and transformations
- Lighting models and material properties
- GPU-based computation techniques

### Software Engineering
- Modular JavaScript development
- Event-driven programming
- Real-time interactive applications
- Code organization and documentation

### Mathematical Implementation
- Fractal geometry and iterated function systems
- Linear algebra for 3D graphics
- Probability distributions for stochastic processes
- Coordinate system transformations

---

## Course Context

These projects were developed as part of the Computer Graphics and Interfaces course, demonstrating practical application of:
- Fundamental computer graphics concepts
- Real-time rendering techniques
- Interactive user interface design
- Mathematical visualization methods

Each project showcases different aspects of computer graphics programming, from mathematical visualization to realistic 3D rendering with lighting.
