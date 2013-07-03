/*
# Volumetric Ray Casting Shader v1.0
# written (or actually composed) by Martins Upitis (martinsh)
#
#
# Features:
#
# - Single pass Raycasting.
# - Uses 2D texture instead of Volume texture
# - Volume slicing with GLSL clipping plane
# - Sample dithering with noise pattern
#
# To-Do list:
#
# - add lighting and shadowing
# - add color ramp option
#
# References:
#
# Single pass Raycasting - http://prideout.net/blog/?p=64
# Volume data extraction from 2D texture - http://demos.vicomtech.org/volren/)
# Polar to Cartesian coordinate conversion - can`t remember...
*/

uniform float azimuth, elevation, clipPlaneDepth; //clipping plane variables
uniform bool clip, dither;
uniform sampler2D tex;
varying vec3 pos;

const float maxDist = sqrt(2.0);
const int numSamples = 256;
const float stepSize = maxDist/float(numSamples);

float opacityFactor = 128.0;
float lightFactor = 1.2;

const float numberOfSlices = 96.0;
const float slicesOverX = -10.0;
const float slicesOverY = 10.0;

struct Ray
{
    vec3 Origin;
    vec3 Dir;
};

struct AABB
{
    vec3 Min;
    vec3 Max;
};

bool IntersectBox(Ray r, AABB aabb, out float t0, out float t1)
{
    vec3 invR = 1.0 / r.Dir;
    vec3 tbot = invR * (aabb.Min-r.Origin);
    vec3 ttop = invR * (aabb.Max-r.Origin);
    vec3 tmin = min(ttop, tbot);
    vec3 tmax = max(ttop, tbot);
    vec2 t = max(tmin.xx, tmin.yz);
    t0 = max(t.x, t.y);
    t = min(tmax.xx, tmax.yz);
    t1 = min(t.x, t.y);
    return t0 <= t1;
}

vec3 p2cart(float azimuth,float elevation)
{
    float x, y, z, k;
    float ele = -elevation * 3.14159 / 180.0;
    float azi = (azimuth + 90.0) * 3.14159 / 180.0;

    k = cos( ele );
    z = sin( ele );
    y = sin( azi ) * k;
    x = cos( azi ) * k;

    return vec3( x, z, y );
}

float tex3D(sampler2D texture, vec3 volpos)
{
    float s1,s2;
    float dx1,dy1;
    float dx2,dy2;

    vec2 texpos1,texpos2;

    s1 = floor(volpos.z*numberOfSlices);
    s2 = s1+1.0;

    dx1 = fract(s1/slicesOverX);
    dy1 = floor(s1/slicesOverY)/slicesOverY;

    dx2 = fract(s2/slicesOverX);
    dy2 = floor(s2/slicesOverY)/slicesOverY;

    texpos1.x = dx1+(volpos.x/slicesOverX);
    texpos1.y = dy1+(volpos.y/slicesOverY);

    texpos2.x = dx2+(volpos.x/slicesOverX);
    texpos2.y = dy2+(volpos.y/slicesOverY);

    return mix( texture2D(texture,texpos1).x, texture2D(texture,texpos2).x, (volpos.z*numberOfSlices)-s1);
}

void main()
{
    vec3 clipPlane = p2cart(azimuth, elevation);
    vec3 view = normalize(pos - gl_ModelViewMatrixInverse[3].xyz);
    Ray eye = Ray( gl_ModelViewMatrixInverse[3].xyz, normalize(view) );

    AABB aabb = AABB(vec3(-1.0), vec3(+1.0));

    float tnear, tfar;
    IntersectBox(eye, aabb, tnear, tfar);
    if (tnear < 0.0) tnear = 0.0;

    vec3 rayStart = eye.Origin + eye.Dir * tnear;
    vec3 rayStop = eye.Origin + eye.Dir * tfar;
    rayStart = 0.5 * (rayStart + 1.0);
    rayStop = 0.5 * (rayStop + 1.0);

    vec3 pos = rayStart;
    vec3 dir = rayStop - rayStart;
    vec3 step = normalize(dir) * stepSize;
    float travel = distance(rayStop, rayStart);

    float len = length(dir);
    dir = normalize(dir);

    if (clip)
    {
        gl_FragColor.a = 0.0;
        //next, see if clip plane faces viewer
        bool frontface = (dot(dir , clipPlane) > 0.0);
        //next, distance from ray origin to clip plane
        float dis = dot(dir,clipPlane);
        if (dis != 0.0  )  dis = (-clipPlaneDepth - dot(clipPlane, rayStart.xyz-0.5)) / dis;
        if ((!frontface) && (dis < 0.0)) return;
        if ((frontface) && (dis > len)) return;
        if ((dis > 0.0) && (dis < len))
            {
            if (frontface) {
                rayStart = rayStart + dir * dis;
            } else {
                rayStop =  rayStart + dir * dis;
            }

            pos = rayStart;
            step = normalize(rayStop-rayStart) * stepSize;
            travel = distance(rayStop, rayStart);
        }
    }

    vec4 accum = vec4(0.0, 0.0, 0.0, 0.0);
    vec4 sample = vec4(0.0, 0.0, 0.0, 0.0);
    vec4 value = vec4(0.0, 0.0, 0.0, 0.0);

    if (dither) //jaggy artefact dithering
    {
        pos = pos + step * (fract(sin(gl_FragCoord.x * 12.9898 + gl_FragCoord.y * 78.233) * 43758.5453));
    }

    for (int i=0; i < numSamples && travel > 0.0; ++i, pos += step, travel -= stepSize) {

        float tf_pos;

        tf_pos = tex3D(tex, pos);

        value = vec4(tf_pos);

        // Process the volume sample
        sample.a = value.a * opacityFactor * (1.0/float(numSamples));
        sample.rgb = value.rgb * sample.a * lightFactor;

        accum.rgb += (1.0 - accum.a) * sample.rgb;
        accum.a += sample.a;


        if(accum.a>=1.0)
        break;
    }

    gl_FragColor.rgb = accum.rgb;
    gl_FragColor.a = accum.a;
}

