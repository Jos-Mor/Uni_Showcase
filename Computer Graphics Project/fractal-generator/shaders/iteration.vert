attribute vec2 vOldPos;

varying vec2 vNewPos;
varying float newFunction;

const int MAX_FUNCS = 7;

uniform mat3 m[MAX_FUNCS]; // array of matrices with linear functions
uniform float p[MAX_FUNCS]; // array with probabilities
uniform int nfuncs; // number of IFS functions

float random(vec2 co){
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

void main() {
    float rand = random(vOldPos);  // Generate a pseudo-random number
    
    int chosenFunc = 0;
    float cumulativeProb = 0.0;
    for (int i = 0; i < MAX_FUNCS; i++) {
        cumulativeProb += p[i];
        if (rand < cumulativeProb) {
            chosenFunc = i;
            break;
        }
    }

    vNewPos = (m[chosenFunc] * vec3(vOldPos, 1.0)).xy;
    newFunction = float(chosenFunc);
}