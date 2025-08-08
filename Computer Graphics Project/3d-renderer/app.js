import { buildProgramFromSources, loadShadersFromURLS, setupWebGL } from '../libs/utils.js';
import { length, flatten, inverse, mult, normalMatrix, perspective, lookAt, vec4, vec3, vec2, subtract, add, scale, rotate, normalize, translate } from '../libs/MV.js';

import * as dat from '../libs/dat.gui.module.js';

import * as CUBE from '../libs/objects/cube.js';
import * as SPHERE from '../libs/objects/sphere.js';
import * as COW from '../libs/objects/cow.js';
import * as BUNNY from '../libs/objects/bunny.js';

import * as STACK from '../libs/stack.js';

function setup(shaders) {
    const canvas = document.getElementById('gl-canvas');
    const gl = setupWebGL(canvas);

    CUBE.init(gl);
    SPHERE.init(gl);
    COW.init(gl);
    BUNNY.init(gl);

    const program = buildProgramFromSources(gl, shaders['shader.vert'], shaders['shader.frag']);

    // Camera  
    let camera = {
        eye: vec3(0,0,5),
        at: vec3(0,0,0),
        up: vec3(0,1,0),
        fovy: 45,
        aspect: 1, // Updated further down
        near: 0.1,
        far: 20
    }

    let options = {
        wireframe: false,
        normals: true
    }

    const gui = new dat.GUI();

    const optionsGui = gui.addFolder("options");
    optionsGui.add(options, "wireframe");
    optionsGui.add(options, "normals");

    const cameraGui = gui.addFolder("camera");

    cameraGui.add(camera, "fovy").min(1).max(100).step(1).listen();
    cameraGui.add(camera, "aspect").min(0).max(10).step(0.01).listen().domElement.style.pointerEvents = "none";
    
    cameraGui.add(camera, "near").min(0.1).max(20).step(0.01).listen().onChange( function(v) {
        camera.near = Math.min(camera.far-0.5, v);
    });

    cameraGui.add(camera, "far").min(0.1).max(20).step(0.01).listen().onChange( function(v) {
        camera.far = Math.max(camera.near+0.5, v);
    });

    const eye = cameraGui.addFolder("eye");
    eye.add(camera.eye, 0).step(0.05).listen().domElement.style.pointerEvents = "none";;
    eye.add(camera.eye, 1).step(0.05).listen().domElement.style.pointerEvents = "none";;
    eye.add(camera.eye, 2).step(0.05).listen().domElement.style.pointerEvents = "none";;

    const at = cameraGui.addFolder("at");
    at.add(camera.at, 0).step(0.05).listen().domElement.style.pointerEvents = "none";;
    at.add(camera.at, 1).step(0.05).listen().domElement.style.pointerEvents = "none";;
    at.add(camera.at, 2).step(0.05).listen().domElement.style.pointerEvents = "none";;

    const up = cameraGui.addFolder("up");
    up.add(camera.up, 0).step(0.05).listen().domElement.style.pointerEvents = "none";;
    up.add(camera.up, 1).step(0.05).listen().domElement.style.pointerEvents = "none";;
    up.add(camera.up, 2).step(0.05).listen().domElement.style.pointerEvents = "none";;

    //Objects GUI
    let BOARDObject = {
        position: vec3(0.0, 0.0, -3.0),
        rotation: vec3(0.0, 0.0, 0.0),
        scale: vec3(4.0, 4.0, 0.2),
        ambient: vec3(100.0, 90.0, 70.0),
        diffuse: vec3(100.0, 90.0, 70.0),
        specular: vec3(0.0, 0.0, 0.0),
        shininess: 300.0,
    }
    let CUBEObject = {
        position: vec3(-1.0, 1.0, 0.0),
        rotation: vec3(0.0, 0.0, 0.0),
        scale: vec3(1.0, 1.0, 1.0),
        ambient: vec3(150.0, 50.0, 50.0),
        diffuse: vec3(150.0, 50.0, 50.0),
        specular: vec3(200.0, 200.0, 200.0),
        shininess: 100.0,
    }
    let SPHEREObject = {
        position: vec3(1.0, 1.0, 0.0),
        rotation: vec3(0.0, 0.0, 0.0),
        scale: vec3(1.0, 1.0, 1.0),
        ambient: vec3(50.0, 150.0, 50.0),
        diffuse: vec3(50.0, 150.0, 50.0),
        specular: vec3(200.0, 200.0, 200.0),
        shininess: 100.0,
    }
    let COWObject = {
        position: vec3(-1.0, -1.0, 0.0),
        rotation: vec3(90.0, 0.0, 0.0),
        scale: vec3(1.0, 1.0, 1.0),
        ambient: vec3(100.0, 100.0, 100.0),
        diffuse: vec3(100.0, 100.0, 100.0),
        specular: vec3(200.0, 200.0, 200.0),
        shininess: 100.0,
    }
    let BUNNYObject = {
        position: vec3(1.0, -1.0, 0.0),
        rotation: vec3(90.0, 0.0, 0.0),
        scale: vec3(1.0, 1.0, 1.0),
        ambient: vec3(0.0, 100.0, 150.0),
        diffuse: vec3(0.0, 100.0, 150.0),
        specular: vec3(200.0, 200.0, 200.0),
        shininess: 100.0
    }
    let light1 = {
        on: true,
        position: vec3(0.0, 0.0, 12.0),
        scale: vec3(0.2, 0.2, 0.2),
        rotation: vec3(0.5, 0.0, 0.0),
        ambient: vec3(255.0, 255.0, 255.0),
        diffuse: vec3(255.0, 255.0, 255.0),
        specular: vec3(0.0, 0.0, 0.0),
        shininess: 300.0
    }
    let light2 = {
        on: true,
        position: vec3(12.0, 0.0, 0.0),
        scale: vec3(0.2, 0.2, 0.2),
        rotation: vec3(0.0, 0.5, 0.0),
        ambient: vec3(255.0, 255.0, 255.0),
        diffuse: vec3(255.0, 255.0, 255.0),
        specular: vec3(0.0, 0.0, 0.0),
        shininess: 300.0
    }
    let light3 = {
        on: true,
        position: vec3(12.0, 0.0, 0.0),
        scale: vec3(0.2, 0.2, 0.2),
        rotation: vec3(0.0, 0.0, 0.5),
        ambient: vec3(255.0, 255.0, 255.0),
        diffuse: vec3(255.0, 255.0, 255.0),
        specular: vec3(0.0, 0.0, 0.0),
        shininess: 300.0
    }

    const objects = {
        Cube: {
            name: 'Cube',
            object: CUBEObject
        },
        Sphere: {
            name: 'Sphere',
            object: SPHEREObject
        },
        Cow: {
            name: 'Cow',
            object: COWObject
        },
        Bunny: {
            name: 'Bunny',
            object: BUNNYObject
        }
    };
    
    let selectedObject = null;
    const objectsGui = new dat.GUI();
    const dropDownSelectedObject = { name: 'Cube' };
    const objectSelector = objectsGui.add(dropDownSelectedObject, 'name', Object.keys(objects)).name('Name');
    let currentObject = objects[dropDownSelectedObject.name].object;


    const transformFolder = objectsGui.addFolder('Transform');
    let hasFolders = false;
    const updateTransformGUI = () => {
        if (hasFolders){
            transformFolder.removeFolder('position');
            transformFolder.removeFolder('rotation');
            transformFolder.removeFolder('scale');
            //transformFolder.removeFolder('material');
        }
        const positionFolder = transformFolder.addFolder('position');
        positionFolder.add(currentObject.position, 0).min(-1.5).max(1.5).step(0.1).name('x').listen();
        positionFolder.add(currentObject.position, 1).min(-1.5).max(1.5).step(0.1).name('y').listen();
        positionFolder.add(currentObject.position, 2).min(-1.5).max(1.5).step(0.1).name('z').listen();
    
        const rotationFolder = transformFolder.addFolder('rotation');
        rotationFolder.add(currentObject.rotation, 0).min(-360).max(360).step(0.1).name('x').listen();
        rotationFolder.add(currentObject.rotation, 1).min(-360).max(360).step(0.1).name('y').listen();
        rotationFolder.add(currentObject.rotation, 2).min(-360).max(360).step(0.1).name('z').listen();
    
        const scaleFolder = transformFolder.addFolder('scale');
        scaleFolder.add(currentObject.scale, 0).min(0).max(1).step(0.1).name('x').listen();
        scaleFolder.add(currentObject.scale, 1).min(0).max(1).step(0.1).name('y').listen();
        scaleFolder.add(currentObject.scale, 2).min(0).max(1).step(0.1).name('z').listen();

        //const materialFolder = transformFolder.addFolder('material');
        //materialFolder.add(currentObject.ambient, 0).name('Ka');
        //materialFolder.add(currentObject.diffuse).name('Kd');
        //materialFolder.add(currentObject.specular).name('Ks');
        //materialFolder.add(currentObject.shininess).name('shininess').min(0).max(255).step(1).listen();

        hasFolders = true;
    };
    
    objectSelector.onChange(function (value) {
        currentObject = objects[dropDownSelectedObject.name].object;
        selectedObject = selectedObject == currentObject ? null : currentObject;
        updateTransformGUI();
    });
    
    updateTransformGUI();

    //helper function to remove folders from GUI
    dat.GUI.prototype.removeFolder = function(name) {
        var folder = this.__folders[name];
        if (!folder) {
          return;
        }
        folder.close();
        this.__ul.removeChild(folder.domElement.parentNode);
        delete this.__folders[name];
        this.onResize();
      }
    
      document.onkeydown = function (event) {
		switch (event.key) {
			case "1":
                currentObject = CUBEObject;
                dropDownSelectedObject.name = "Cube";
                objectSelector.setValue('Cube');
                updateTransformGUI();
				break;
			case "2": 
                currentObject = SPHEREObject;
                dropDownSelectedObject.name = "Sphere";
                objectSelector.setValue('Sphere');
                updateTransformGUI();
				break;
			case "3": 
                currentObject = COWObject;
                dropDownSelectedObject.name = "Cow";
                objectSelector.setValue('Cow');
                break;
			case "4": 
                currentObject = BUNNYObject;
                dropDownSelectedObject.name = "Bunny";
                objectSelector.setValue('Bunny'); 
				break;
        }
    }

    // matrices
    let mView, mProjection;

    let down = false;
    let lastX, lastY;

    gl.clearColor(0.19, 0.19, 0.19, 1.0);
    gl.enable(gl.DEPTH_TEST);

    resizeCanvasToFullWindow();

    window.addEventListener('resize', resizeCanvasToFullWindow);

    window.addEventListener('wheel', function(event) {
        //mouse wheel zoom
        const factor = 1 - event.deltaY/1000;
        camera.fovy = Math.max(1, Math.min(100, camera.fovy * factor)); 
    });

    function inCameraSpace(m) {
        const mInvView = inverse(mView);

        return mult(mInvView, mult(m, mView));
    }

    canvas.addEventListener('mousemove', function(event) {
        if(down) {
            const dx = event.offsetX - lastX;
            const dy = event.offsetY - lastY;

            if(dx != 0 || dy != 0) {
                // Do something here...

                const d = vec2(dx, dy);
                const axis = vec3(-dy, -dx, 0);

                const rotation = rotate(0.5*length(d), axis);

                let eyeAt = subtract(camera.eye, camera.at);                
                eyeAt = vec4(eyeAt[0], eyeAt[1], eyeAt[2], 0);
                let newUp = vec4(camera.up[0], camera.up[1], camera.up[2], 0);

                eyeAt = mult(inCameraSpace(rotation), eyeAt);
                newUp = mult(inCameraSpace(rotation), newUp);
                
                console.log(eyeAt, newUp);

                camera.eye[0] = camera.at[0] + eyeAt[0];
                camera.eye[1] = camera.at[1] + eyeAt[1];
                camera.eye[2] = camera.at[2] + eyeAt[2];

                camera.up[0] = newUp[0];
                camera.up[1] = newUp[1];
                camera.up[2] = newUp[2];

                lastX = event.offsetX;
                lastY = event.offsetY;
            }

        }
    });

    canvas.addEventListener('mousedown', function(event) {
        down=true;
        lastX = event.offsetX;
        lastY = event.offsetY;
        gl.clearColor(0.2, 0.0, 0.0, 1.0);
    });

    canvas.addEventListener('mouseup', function(event) {
        down = false;
        gl.clearColor(0.19, 0.19, 0.19, 1.0);
    });

    window.requestAnimationFrame(render);

    function resizeCanvasToFullWindow()
    {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;

        camera.aspect = canvas.width / canvas.height;

        gl.viewport(0,0,canvas.width, canvas.height);
    }

    function uploadModelView() {
        gl.uniformMatrix4fv(gl.getUniformLocation(program, "mModelView"), false, flatten(STACK.modelView()));
    }    

    function drawObject(object){
        STACK.pushMatrix();
        STACK.multScale(object.scale);
        STACK.multTranslation(object.position);
        STACK.multRotationX(object.rotation[0]);
        STACK.multRotationY(object.rotation[1]);
        STACK.multRotationZ(object.rotation[2]);
        gl.uniform3fv(gl.getUniformLocation(program, "uAmbient"), object.ambient);
        gl.uniform3fv(gl.getUniformLocation(program, "uDiffuse"), object.diffuse);
        gl.uniform1f(gl.getUniformLocation(program, "uShininess"), object.shininess);
        gl.uniform3fv(gl.getUniformLocation(program, "uSpecular"), object.specular);

        const pos1 = light1.on ? light1.position : vec3(-999.0, -999.0, -999.0);
        const pos2 = light2.on ? light2.position : vec3(-999.0, -999.0, -999.0);
        const pos3 = light3.on ? light3.position : vec3(-999.0, -999.0, -999.0);
        gl.uniform3fv(gl.getUniformLocation(program, "uLight1"), pos1);
        gl.uniform3fv(gl.getUniformLocation(program, "uLight2"), pos2);
        gl.uniform3fv(gl.getUniformLocation(program, "uLight3"), pos3);
        gl.uniform3fv(gl.getUniformLocation(program, "uEye"), camera.eye);
        uploadModelView(program);
        decideObjectDraw(object);
        STACK.popMatrix();
    }

    function decideObjectDraw(object){
        switch(object){
            case BOARDObject:
                CUBE.draw(gl, program, gl.TRIANGLES);
                break;
            case CUBEObject:
                CUBE.draw(gl, program, options.wireframe ? gl.LINES : gl.TRIANGLES);
                break;
            case SPHEREObject:
                SPHERE.draw(gl, program, options.wireframe ? gl.LINES : gl.TRIANGLES);
                break;
            case COWObject:
                COW.draw(gl, program, options.wireframe ? gl.LINES : gl.TRIANGLES);
                break;
            case BUNNYObject:
                BUNNY.draw(gl, program, options.wireframe ? gl.LINES : gl.TRIANGLES);
                break;
            default:
                break;
        }
    }

    function newLightPosition(object, variableCoords, fixedCoord){
        STACK.pushMatrix();
        STACK.multScale(object.scale);        
        const radians = object.rotation[fixedCoord] * (Math.PI / 180);  // Convert degrees to radians
        const coord1 = object.position[variableCoords[0]] * Math.cos(radians) - object.position[variableCoords[1]] * Math.sin(radians);
        const coord2 = object.position[variableCoords[0]] * Math.sin(radians) + object.position[variableCoords[1]] * Math.cos(radians);
        STACK.multTranslation(object.position);

        return [coord1, coord2];
    }

    function updateLightPosition(object){
        gl.uniform3fv(gl.getUniformLocation(program, "uAmbient"), object.ambient);
        gl.uniform3fv(gl.getUniformLocation(program, "uDiffuse"), object.diffuse);
        gl.uniform1f(gl.getUniformLocation(program, "uShininess"), object.shininess);
        gl.uniform3fv(gl.getUniformLocation(program, "uSpecular"), object.specular);
        uploadModelView(program);
        SPHERE.draw(gl, program, gl.TRIANGLES);
        STACK.popMatrix();
    }

    function drawLight(object){
        let variableCoords;
        let fixedCoord;
        let coords;
        switch(object){
            case light1:
                variableCoords = [1, 2];
                fixedCoord = 0;
                coords = newLightPosition(object, variableCoords, fixedCoord);
                light1.position = [object.position[fixedCoord], coords[0], coords[1]];
                break;
            case light2:
                variableCoords = [0, 2];
                fixedCoord = 1;
                coords = newLightPosition(object, variableCoords, fixedCoord);
                light2.position = [coords[0], object.position[fixedCoord], coords[1]];
                break;
            case light3:
                variableCoords = [0, 1];
                fixedCoord = 2;
                coords = newLightPosition(object, variableCoords, fixedCoord);
                light3.position = [coords[0], coords[1], object.position[fixedCoord]];
                break;
            default:
                break;
        }
        updateLightPosition(object);
    }

    function selectObject(object){
        STACK.pushMatrix();
        options.wireframe = true;
        STACK.multTranslation(object.position);
        STACK.multScale(object.scale);
        STACK.multRotationX(object.rotation[0]);
        STACK.multRotationY(object.rotation[1]);
        STACK.multRotationZ(object.rotation[2]);
        gl.uniform3fv(gl.getUniformLocation(program, "uAmbient"), [150, 150, 150]);
        gl.uniform3fv(gl.getUniformLocation(program, "uDiffuse"), [150, 150, 150]);
        gl.uniform1f(gl.getUniformLocation(program, "uShininess"), object.shininess);
        gl.uniform3fv(gl.getUniformLocation(program, "uSpecular"), [250, 250, 250]);

        const pos1 = light1.on ? light1.position : vec3(-999.0, -999.0, -999.0);
        const pos2 = light2.on ? light2.position : vec3(-999.0, -999.0, -999.0);
        const pos3 = light3.on ? light3.position : vec3(-999.0, -999.0, -999.0);
        gl.uniform3fv(gl.getUniformLocation(program, "uLight1"), pos1);
        gl.uniform3fv(gl.getUniformLocation(program, "uLight2"), pos2);
        gl.uniform3fv(gl.getUniformLocation(program, "uLight3"), pos3);
        gl.uniform3fv(gl.getUniformLocation(program, "uEye"), camera.eye);
        uploadModelView(program);
        decideObjectDraw(object);
        options.wireframe = false;
        STACK.popMatrix();
    }
    
    function render(){
        window.requestAnimationFrame(render);

        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        gl.useProgram(program);

        mView = lookAt(camera.eye, camera.at, camera.up);
        STACK.loadMatrix(mView);

        mProjection = perspective(camera.fovy, camera.aspect, camera.near, camera.far);

        gl.useProgram(program);
        gl.uniformMatrix4fv(gl.getUniformLocation(program, "mModelView"), false, flatten(STACK.modelView()));
        gl.uniformMatrix4fv(gl.getUniformLocation(program, "mProjection"), false, flatten(mProjection));
        gl.uniformMatrix4fv(gl.getUniformLocation(program, "mNormals"), false, flatten(normalMatrix(STACK.modelView())));
        gl.uniform1i(gl.getUniformLocation(program, "uUseNormals"), options.normals);

        if (light1.on){
            drawLight(light1);
        }
        if (light2.on){
            drawLight(light2);
        }
        if (light3.on){
            drawLight(light3);
        }
        drawObject(BOARDObject);
        drawObject(CUBEObject);
        drawObject(SPHEREObject);
        drawObject(COWObject);
        drawObject(BUNNYObject);
        if (selectedObject != null)
            selectObject(selectedObject);
    }
}

const urls = ['shader.vert', 'shader.frag'];

loadShadersFromURLS(urls).then( shaders => setup(shaders));
