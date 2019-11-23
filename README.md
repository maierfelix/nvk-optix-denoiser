# nvk-optix

Bindings of NVIDIA's Optix Denoiser for [node-vulkan](https://github.com/maierfelix/nvk). Allows to share VkBuffers between Vulkan and the Optix Denoiser for real-time denoising.


# Screenshots

<a><img src="https://i.imgur.com/Jz3Tum5.png" height="282"></a>

<a><img src="https://i.imgur.com/Od0rGJv.png" height="282"></a>

# API

### denoiser.create

| Name | Type | Description |
| :--- | :--- | :--- |
| width | *Number* | Width of the input to operate |
| height | *Number* | Height of the input to operate |

Creates a new instance of the denoiser with the desired with and height.


### denoiser.destroy

Destroyes the denoiser and all imported memory buffers and semaphores.


### denoiser.denoise

| Name | Type | Description |
| :--- | :--- | :--- |
| blendFactor | *Number* | Blending factor (0-1 range) of the input and the denoised output |

### denoiser.setVulkanInputMemory

| Name | Type | Description |
| :--- | :--- | :--- |
| color | *BigInt* | Importable Vulkan handle to the color channel |
| albedo | *BigInt* | Importable Vulkan handle to the albedo channel |
| normal | *BigInt* | Importable Vulkan handle to the normal channel |

 - Color is the final image to denoise
 - Albedo contains the primary hit albedo color
 - Normal contains the primary hit normals of the scene in camera-space

#### Example:

````js
let memoryGetWin32HandleInfoKHRInfo = new VkMemoryGetWin32HandleInfoKHR();
memoryGetWin32HandleInfoKHRInfo.memory = deviceMemory;
memoryGetWin32HandleInfoKHRInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

let buffer = new ArrayBuffer(0x8);
result = vkGetMemoryWin32HandleKHR(device, memoryGetWin32HandleInfoKHRInfo, { $: buffer.getAddress() });
ASSERT_VK_RESULT(result);

let colorHandle = new BigUint64Array(buffer)[0];
...
..

denoiser.setVulkanInputMemory(
  colorHandle,
  albedoHandle,
  normalHandle
);
````

### denoiser.setWaitSemaphore

| Name | Type | Description |
| :--- | :--- | :--- |
| handle | *BigInt* | Importable Vulkan Semaphore |

Vulkan Semaphore which is used to wait for a finished render of a Vulkan ray-traced image, which then can be processed by the denoiser.


#### Example:

````js
let semaphoreGetWin32HandleInfoKHRInfo = new VkSemaphoreGetWin32HandleInfoKHR();
semaphoreGetWin32HandleInfoKHRInfo.semaphore = denoiseWaitSemaphore;
semaphoreGetWin32HandleInfoKHRInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;

let buffer = new ArrayBuffer(0x8);
result = vkGetSemaphoreWin32HandleKHR(device, semaphoreGetWin32HandleInfoKHRInfo, { $: buffer.getAddress() });
ASSERT_VK_RESULT(result);

let waitHandle = new BigUint64Array(buffer)[0];

denoiser.setWaitSemaphore(waitHandle);
````

### denoiser.setUpdateSemaphore

| Name | Type | Description |
| :--- | :--- | :--- |
| handle | *BigInt* | Importable Vulkan Semaphore |

Vulkan Semaphore which is used to indicate when the denoiser finished denoising, so Vulkan can (for example) display the denoised result.

#### Example:

````js
let semaphoreGetWin32HandleInfoKHRInfo = new VkSemaphoreGetWin32HandleInfoKHR();
semaphoreGetWin32HandleInfoKHRInfo.semaphore = denoiseUpdateSemaphore;
semaphoreGetWin32HandleInfoKHRInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;

let buffer = new ArrayBuffer(0x8);
result = vkGetSemaphoreWin32HandleKHR(device, semaphoreGetWin32HandleInfoKHRInfo, { $: buffer.getAddress() });
ASSERT_VK_RESULT(result);

let updateHandle = new BigUint64Array(buffer)[0];

denoiser.setUpdateSemaphore(updateHandle);
````
