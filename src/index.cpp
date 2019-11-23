#define NAPI_EXPERIMENTAL
#include <napi.h>

#include <windows.h>

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_win32.h>

#include <cuda_runtime.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <iomanip>
#include <iostream>
#include <string>

#include <aclapi.h>
#include <dxgi1_2.h>

#include "CUDABuffer.h"
#include "WindowsSecurityAttributes.h"

#define ASSERT_OPTIX(call)                                            \
{                                                                     \
  OptixResult res = call;                                             \
  if ( res != OPTIX_SUCCESS )                                         \
    {                                                                 \
      fprintf( stderr, "Optix call (%s) failed with code %d (line %d)\n", #call, res, __LINE__ ); \
      exit( 2 );                                                      \
    }                                                                 \
}

#define ASSERT_CUDA(err) __checkCudaErrors(err, __FILE__, __LINE__)
inline void __checkCudaErrors(cudaError_t err, const char *file, const int line) {
  if (CUDA_SUCCESS != err) {
    fprintf(stderr,
            "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, "
            "line %i.\n",
            err, "", file, line);
    exit(EXIT_FAILURE);
  }
}

static void context_log_cb( unsigned int level, const char* tag, const char* message, void* cbdata) {
  std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: " << message << "\n";
}

static bool lossless = false;

static OptixDeviceContext context = nullptr;
static OptixDenoiser denoiser = nullptr;

static cudaStream_t cudaStream;

static CUDABuffer denoiserState;
static CUDABuffer denoiserScratch;
static CUDABuffer denoiserIntensity;

static OptixImage2D rgbLayer;
static OptixImage2D albedoLayer;
static OptixImage2D normalLayer;
static cudaExternalMemory_t rgbMemory;
static cudaExternalMemory_t albedoMemory;
static cudaExternalMemory_t normalMemory;

static OptixImage2D outputLayer;
static cudaExternalMemory_t outputMemory;

static cudaExternalSemaphore_t waitSemaphore;
static cudaExternalSemaphore_t updateSemaphore;

static unsigned int width = 512;
static unsigned int height = 512;

template<typename T> inline T* UnpackNVKHandle(Napi::Value value) {
  Napi::Env env = value.Env();
  Napi::Object obj = value.As<Napi::Object>();
  Napi::ArrayBuffer buffer = obj.Get("memoryBuffer").As<Napi::ArrayBuffer>();
  T* handle = reinterpret_cast<T*>(buffer.Data());
  return handle;
};

template<typename T> inline T* UnpackNVKStructure(Napi::Value value) {
  Napi::Env env = value.Env();
  Napi::Object obj = value.As<Napi::Object>();
  Napi::ArrayBuffer buffer = obj.Get("memoryBuffer").As<Napi::ArrayBuffer>();
  T* structure = reinterpret_cast<T*>(buffer.Data());
  return structure;
};

std::pair<void*, cudaExternalMemory_t> GetExternalMappedBuffer(void* handle, uint64_t size) {
  cudaExternalMemory_t extMem;

  cudaExternalMemoryHandleDesc externalMemoryHandleDesc;
  memset(&externalMemoryHandleDesc, 0, sizeof(externalMemoryHandleDesc));

  externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
  externalMemoryHandleDesc.size = size;
  externalMemoryHandleDesc.flags = cudaExternalMemoryDedicated;
  externalMemoryHandleDesc.handle.win32.handle = handle;

  ASSERT_CUDA(cudaImportExternalMemory(&extMem, &externalMemoryHandleDesc));

  cudaExternalMemoryBufferDesc externalMemoryBufferDesc;
  memset(&externalMemoryBufferDesc, 0, sizeof(externalMemoryBufferDesc));
  externalMemoryBufferDesc.offset = 0;
  externalMemoryBufferDesc.size = size;
  externalMemoryBufferDesc.flags = 0;

  void* addr = nullptr;
  ASSERT_CUDA(cudaExternalMemoryGetMappedBuffer((void**)&addr, extMem, &externalMemoryBufferDesc));

  return std::make_pair(addr, extMem);
};

static Napi::Value Create(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (!info[0].IsNumber()) {
    Napi::TypeError::New(env, "Argument 1 must be of type 'Number'").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  if (!info[1].IsNumber()) {
    Napi::TypeError::New(env, "Argument 1 must be of type 'Number'").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  unsigned int width = info[0].As<Napi::Number>().Uint32Value();
  unsigned int height = info[1].As<Napi::Number>().Uint32Value();

  OptixDenoiserModelKind imageModel = OPTIX_DENOISER_MODEL_KIND_HDR;

  ::width = width;
  ::height = height;

  cudaFree(0);
  CUcontext cuCtx = 0;
  optixInit();
  OptixDeviceContextOptions options = {};
  options.logCallbackLevel = 4;
  options.logCallbackFunction = &context_log_cb;
  ASSERT_OPTIX(optixDeviceContextCreate(cuCtx, &options, &context));

  OptixDenoiserOptions denoiserOptions;
  denoiserOptions.inputKind = OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL;
  denoiserOptions.pixelFormat = OPTIX_PIXEL_FORMAT_FLOAT4;

  ASSERT_CUDA(cudaStreamCreate(&cudaStream));

  ASSERT_OPTIX(optixDenoiserCreate(context, &denoiserOptions, &denoiser));

  // TODO:
  // ldr or hdr
  ASSERT_OPTIX(optixDenoiserSetModel(denoiser, imageModel, NULL, 0));

  OptixDenoiserSizes denoiserReturnSizes;
  ASSERT_OPTIX(
    optixDenoiserComputeMemoryResources(
      denoiser,
      width, height,
      &denoiserReturnSizes
    )
  );

  denoiserState.resize(denoiserReturnSizes.stateSizeInBytes);
  denoiserScratch.resize(denoiserReturnSizes.recommendedScratchSizeInBytes);
  denoiserIntensity.resize(sizeof(float));

  rgbLayer.width = ::width;
  rgbLayer.height = ::height;
  rgbLayer.rowStrideInBytes = ::width * sizeof(float4);
  rgbLayer.pixelStrideInBytes = sizeof(float4);
  rgbLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

  albedoLayer.width = ::width;
  albedoLayer.height = ::height;
  albedoLayer.rowStrideInBytes = ::width * sizeof(float4);
  albedoLayer.pixelStrideInBytes = sizeof(float4);
  albedoLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

  normalLayer.width = ::width;
  normalLayer.height = ::height;
  normalLayer.rowStrideInBytes = ::width * sizeof(float4);
  normalLayer.pixelStrideInBytes = sizeof(float4);
  normalLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

  outputLayer.width = ::width;
  outputLayer.height = ::height;
  outputLayer.rowStrideInBytes = ::width * sizeof(float4);
  outputLayer.pixelStrideInBytes = sizeof(float4);
  outputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

  ASSERT_OPTIX(
    optixDenoiserSetup(
      denoiser,
      cudaStream,
      width, height,
      denoiserState.d_pointer(),
      denoiserState.size(),
      denoiserScratch.d_pointer(),
      denoiserScratch.size()
    )
  );

  return env.Undefined();
};

static Napi::Value Destroy(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  cudaFree(0);

  ASSERT_CUDA(cudaDestroyExternalMemory(rgbMemory));
  ASSERT_CUDA(cudaDestroyExternalMemory(albedoMemory));
  ASSERT_CUDA(cudaDestroyExternalMemory(normalMemory));

  ASSERT_CUDA(cudaDestroyExternalMemory(outputMemory));

  ASSERT_CUDA(cudaDestroyExternalSemaphore(waitSemaphore));
  ASSERT_CUDA(cudaDestroyExternalSemaphore(updateSemaphore));

  denoiserState.free();
  denoiserScratch.free();
  denoiserIntensity.free();

  ASSERT_CUDA(cudaStreamDestroy(cudaStream));

  if (denoiser != nullptr) {
    ASSERT_OPTIX(optixDenoiserDestroy(denoiser));
  }

  if (context != nullptr) {
    ASSERT_OPTIX(optixDeviceContextDestroy(context));
  }

  ASSERT_CUDA(cudaDeviceReset());

  return env.Undefined();
};

static Napi::Value Denoise(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (!info[0].IsNumber()) {
    Napi::TypeError::New(env, "Argument 1 must be of type 'Number'").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  float blendFactor = info[0].As<Napi::Number>().FloatValue();

  OptixDenoiserParams denoiserParams;
  denoiserParams.denoiseAlpha = 0;
  denoiserParams.hdrIntensity = (CUdeviceptr) denoiserIntensity.d_pointer();
  denoiserParams.blendFactor = blendFactor;

  {
    cudaExternalSemaphoreWaitParams extSemaphoreWaitParams;
    memset(&extSemaphoreWaitParams, 0, sizeof(extSemaphoreWaitParams));
    extSemaphoreWaitParams.params.fence.value = 0;
    extSemaphoreWaitParams.flags = 0;
    ASSERT_CUDA(cudaWaitExternalSemaphoresAsync(&waitSemaphore, &extSemaphoreWaitParams, 1, cudaStream));
  }

  std::vector<OptixImage2D> layers = {
    rgbLayer,
    albedoLayer,
    normalLayer
  };

  ASSERT_OPTIX(
    optixDenoiserComputeIntensity(
      denoiser,
      cudaStream,
      &layers[0],
      (CUdeviceptr) denoiserIntensity.d_pointer(),
      (CUdeviceptr) denoiserScratch.d_pointer(),
      denoiserScratch.size()
    )
  );

  ASSERT_OPTIX(
    optixDenoiserInvoke(
      denoiser,
      cudaStream,
      &denoiserParams,
      denoiserState.d_pointer(),
      denoiserState.size(),
      layers.data(), layers.size(),
      0, 0,
      &outputLayer,
      denoiserScratch.d_pointer(),
      denoiserScratch.size()
    )
  );

  {
    cudaExternalSemaphoreSignalParams extSemaphoreSignalParams;
    memset(&extSemaphoreSignalParams, 0, sizeof(extSemaphoreSignalParams));
    extSemaphoreSignalParams.params.fence.value = 0;
    extSemaphoreSignalParams.flags = 0;
    ASSERT_CUDA(cudaSignalExternalSemaphoresAsync(&updateSemaphore, &extSemaphoreSignalParams, 1, cudaStream));
  }

  ASSERT_CUDA(cudaStreamSynchronize(cudaStream));

  return env.Undefined();
};

static Napi::Value GetWindowSecurityAttributes(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  return Napi::BigInt::New(env, reinterpret_cast<uint64_t>(&winSecurityAttributes));
};

static Napi::Value SetVulkanInputMemory(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  // validate
  if (!info[0].IsBigInt()) {
    Napi::TypeError::New(env, "Argument 1 must be of type 'BigInt'").ThrowAsJavaScriptException();
    return env.Undefined();
  }
  if (!info[1].IsBigInt()) {
    Napi::TypeError::New(env, "Argument 2 must be of type 'BigInt'").ThrowAsJavaScriptException();
    return env.Undefined();
  }
  if (!info[2].IsBigInt()) {
    Napi::TypeError::New(env, "Argument 3 must be of type 'BigInt'").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  // rgb input memory
  {
    void* handle = reinterpret_cast<void*>(info[0].As<Napi::BigInt>().Uint64Value(&lossless));
    auto mappedMemory = GetExternalMappedBuffer(handle, width * height * sizeof(float4));
    rgbMemory = mappedMemory.second;
    rgbLayer.data = (CUdeviceptr) mappedMemory.first;
  }
  // albedo input memory
  {
    void* handle = reinterpret_cast<void*>(info[1].As<Napi::BigInt>().Uint64Value(&lossless));
    auto mappedMemory = GetExternalMappedBuffer(handle, width * height * sizeof(float4));
    albedoMemory = mappedMemory.second;
    albedoLayer.data = (CUdeviceptr) mappedMemory.first;
  }
  // normal input memory
  {
    void* handle = reinterpret_cast<void*>(info[2].As<Napi::BigInt>().Uint64Value(&lossless));
    auto mappedMemory = GetExternalMappedBuffer(handle, width * height * sizeof(float4));
    normalMemory = mappedMemory.second;
    normalLayer.data = (CUdeviceptr) mappedMemory.first;
  }

  return env.Undefined();
};

static Napi::Value SetVulkanOutputMemory(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (!info[0].IsBigInt()) {
    Napi::TypeError::New(env, "Argument 1 must be of type 'BigInt'").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  void* handle = reinterpret_cast<void*>(info[0].As<Napi::BigInt>().Uint64Value(&lossless));
  auto mappedMemory = GetExternalMappedBuffer(handle, width * height * sizeof(float4));
  outputMemory = mappedMemory.second;
  outputLayer.data = (CUdeviceptr) mappedMemory.first;

  return env.Undefined();
};

static Napi::Value SetWaitSemaphore(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (!info[0].IsBigInt()) {
    Napi::TypeError::New(env, "Argument 1 must be of type 'BigInt'").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  void* waitHandle = reinterpret_cast<void*>(info[0].As<Napi::BigInt>().Uint64Value(&lossless));

  cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc;
  memset(&externalSemaphoreHandleDesc, 0, sizeof(externalSemaphoreHandleDesc));
  externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
  externalSemaphoreHandleDesc.handle.win32.handle = waitHandle;
  externalSemaphoreHandleDesc.flags = 0;

  ASSERT_CUDA(
    cudaImportExternalSemaphore(&waitSemaphore, &externalSemaphoreHandleDesc)
  );

  return env.Undefined();
};

static Napi::Value SetUpdateSemaphore(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (!info[0].IsBigInt()) {
    Napi::TypeError::New(env, "Argument 1 must be of type 'BigInt'").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  void* updateHandle = reinterpret_cast<void*>(info[0].As<Napi::BigInt>().Uint64Value(&lossless));

  cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc;
  memset(&externalSemaphoreHandleDesc, 0, sizeof(externalSemaphoreHandleDesc));
  externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
  externalSemaphoreHandleDesc.handle.win32.handle = updateHandle;
  externalSemaphoreHandleDesc.flags = 0;

  ASSERT_CUDA(
    cudaImportExternalSemaphore(&updateSemaphore, &externalSemaphoreHandleDesc)
  );

  return env.Undefined();
};

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  exports["create"] = Napi::Function::New(env, Create);
  exports["destroy"] = Napi::Function::New(env, Destroy);
  exports["denoise"] = Napi::Function::New(env, Denoise);
  exports["setVulkanInputMemory"] = Napi::Function::New(env, SetVulkanInputMemory);
  exports["setVulkanOutputMemory"] = Napi::Function::New(env, SetVulkanOutputMemory);
  exports["setWaitSemaphore"] = Napi::Function::New(env, SetWaitSemaphore);
  exports["setUpdateSemaphore"] = Napi::Function::New(env, SetUpdateSemaphore);
  exports["getWindowSecurityAttributes"] = Napi::Function::New(env, GetWindowSecurityAttributes);
  return exports;
}

NODE_API_MODULE(addon, Init)
