/**
* animated 2D sine pattern.
*/
kernel void sineWave(global float4 * vertex, int size, float time) {

    unsigned int x = get_global_id(0);
    unsigned int y = get_global_id(1);

    // calculate uv coordinates
    float u = x / (float) size;
    float v = y / (float) size;

    u = u*2.0f - 1.0f;
    v = v*2.0f - 1.0f;

    // calculate simple sine wave pattern
    float freq = 4.0f;
    float w = sin(u*freq + time) * cos(v*freq + time) * 0.5f;

    // write output vertex
    vertex[y*size + x] = (float4)(u*10.0f, w*10.0f, v*10.0f, 1.0f);
}

