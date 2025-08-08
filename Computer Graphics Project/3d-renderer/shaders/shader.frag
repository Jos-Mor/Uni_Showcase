precision highp float;

varying vec3 fNormal;

uniform vec3 uAmbient;
uniform vec3 uDiffuse;
uniform vec3 uSpecular;
uniform float uShininess;

uniform vec3 uLight1;
uniform vec3 uLight2;
uniform vec3 uLight3;
uniform vec3 uEye;

void main() {
    vec3 amb = uAmbient / 255.0;
    vec3 diff = uDiffuse / 255.0;
    vec3 spec = uSpecular / 255.0;

    vec3 normal = normalize(fNormal);

    vec3 lightDirection1 = normalize(uLight1 - uEye);
    vec3 lightDirection2 = normalize(uLight2 - uEye);
    vec3 lightDirection3 = normalize(uLight3 - uEye);

    float diffuse1 = max(dot(normal, lightDirection1), 0.0);
    float diffuse2 = max(dot(normal, lightDirection2), 0.0);
    float diffuse3 = max(dot(normal, lightDirection3), 0.0);
    vec3 diffuseColor1 = diff * diffuse1;
    vec3 diffuseColor2 = diff * diffuse2;
    vec3 diffuseColor3 = diff * diffuse3;

    vec3 reflectionDirection1 = reflect(-lightDirection1, normal);
    vec3 reflectionDirection2 = reflect(-lightDirection2, normal);
    vec3 reflectionDirection3 = reflect(-lightDirection3, normal);
    float specular1 = pow(max(dot(reflectionDirection1, normalize(uEye - gl_FragCoord.xyz)), 0.0), uShininess);
    float specular2 = pow(max(dot(reflectionDirection2, normalize(uEye - gl_FragCoord.xyz)), 0.0), uShininess);
    float specular3 = pow(max(dot(reflectionDirection3, normalize(uEye - gl_FragCoord.xyz)), 0.0), uShininess);
    vec3 specularColor1 = spec * specular1;
    vec3 specularColor2 = spec * specular2;
    vec3 specularColor3 = spec * specular3;

    vec3 color1 = diffuseColor1 + specularColor1;
    vec3 color2 = diffuseColor2 + specularColor2;
    vec3 color3 = diffuseColor3 + specularColor3;

    vec3 finalColor = amb;

    if (uLight1.x != -999.0 && uLight1.y != -999.0 && uLight1.z != -999.0) {
        finalColor += color1;
    }
    if (uLight2.x != -999.0 && uLight2.y != -999.0 && uLight2.z != -999.0) {
        finalColor += color2;
    }
    if (uLight3.x != -999.0 && uLight3.y != -999.0 && uLight3.z != -999.0) {
        finalColor += color3;
    }
    
    gl_FragColor = vec4(finalColor, 1.0);
}
