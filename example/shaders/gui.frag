#version 450
#extension GL_ARB_separate_shader_objects : enable

layout (location = 0) in vec2 uv;

layout (location = 0) out vec4 fragColor;

layout(std140, binding = 0) readonly buffer SourceBuffer {
  vec4 data[];
} sourceBuffer;

layout(std140, binding = 1) readonly buffer DenoiseBuffer {
  vec4 data[];
} denoiseBuffer;

void main() {
  uint offsetX = uint(uv.x * 1281);
  uint offsetY = uint(uv.y * 719);
  vec4 pixel = denoiseBuffer.data[offsetY * 1281 + offsetX];
  fragColor = pixel.rgba / 255;
}
