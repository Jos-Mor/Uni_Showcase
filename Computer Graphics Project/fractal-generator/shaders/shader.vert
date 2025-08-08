uniform vec2 uBottomLeft;
uniform vec2 uTopRight; 

attribute vec2 vPosition;
attribute float vFunction;  // index of IFS function applied

varying float fFunction;

uniform float u_pointSize;

void main(void) {
    fFunction = vFunction;
    float width = uTopRight.x - uBottomLeft.x;
    float height = uTopRight.y - uBottomLeft.y;

    gl_PointSize = u_pointSize;
    gl_Position = vec4((vPosition - uBottomLeft) * vec2(2.0/width, 2.0/height) - vec2(1.0, 1.0), 0.0, 1.0);
}
