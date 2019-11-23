import fs from "fs";
import nvk from "nvk";
import fastPNG from "fast-png";
import denoiser from "./index.js";
import { performance } from "perf_hooks";
import essentials from "nvk-essentials";
const {GLSL} = essentials;

Object.assign(global, nvk);

/*

let input = fastPNG.decode(fs.readFileSync("./input.png"));
input.data = new Float32Array(input.data);

let output = {
  width: input.width,
  height: input.height,
  depth: 8,
  channels: 4,
  data: new Float32Array(input.width * input.height * 4)
};

let blendFactor = 0.125;

{
  let now = performance.now();
  denoiser.denoise(input, output, blendFactor);
  let then = performance.now();
  console.log(`Denoising took ${then - now}ms`);
}

let sharedImage = denoiser.createSharedImage(device);

fs.writeFileSync("./output.png", fastPNG.encode(output));

*/

function ASSERT_VK_RESULT(result) {
  if (result !== VK_SUCCESS) {
    for (let key in VkResult) {
      if (VkResult[key] === result) {
        throw new Error(`Vulkan assertion failed: '${key}'`);
      }
    };
    throw new Error(`Vulkan assertion failed: '${result}'`);
  }
};

function getMemoryTypeIndex(typeFilter, propertyFlag) {
  let memoryProperties = new VkPhysicalDeviceMemoryProperties();
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, memoryProperties);
  for (let ii = 0; ii < memoryProperties.memoryTypeCount; ++ii) {
    if (
      (typeFilter & (1 << ii)) &&
      (memoryProperties.memoryTypes[ii].propertyFlags & propertyFlag) === propertyFlag
    ) {
      return ii;
    }
  };
  return -1;
};

let result = VK_SUCCESS;

let win = new VulkanWindow({
  width: 640,
  height: 480,
  title: "NVK GUI",
  resizable: false
});

let device = new VkDevice();
let instance = new VkInstance();

let queue = new VkQueue();
let commandPool = new VkCommandPool();

let pipeline = new VkPipeline();
let pipelineLayout = new VkPipelineLayout();

let descriptorSet = new VkDescriptorSet();
let descriptorPool = new VkDescriptorPool();
let descriptorSetLayout = new VkDescriptorSetLayout();

let vertShaderModule = new VkShaderModule();
let fragShaderModule = new VkShaderModule();

let surface = new VkSurfaceKHR();
let renderPass = new VkRenderPass();
let swapchain = new VkSwapchainKHR();

let guiImage = new VkImage();
let guiImageView = new VkImageView();
let guiImageMemory = new VkDeviceMemory();
let guiImageSampler = new VkSampler();

let semaphoreImageAvailable = new VkSemaphore();
let semaphoreRenderingAvailable = new VkSemaphore();

let winSecurityAttributes = denoiser.getWindowSecurityAttributes();

let physicalDevice = null;

let imageViews = [];
let framebuffers = [];
let commandBuffers = [];

let amountOfImagesInSwapchain = { $: 0 };

let validationLayers = [
  "VK_LAYER_LUNARG_standard_validation"
];

let deviceExtensions = [
  VK_KHR_SWAPCHAIN_EXTENSION_NAME,
  VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
  VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
  VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
  VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME
];

let instanceExtensions = win.getRequiredInstanceExtensions();

/** Create Instance **/
{
  let appInfo = new VkApplicationInfo();
  appInfo.pApplicationName = "Hello!";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "No Engine";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_1;

  let createInfo = new VkInstanceCreateInfo();
  createInfo.pApplicationInfo = appInfo;
  createInfo.enabledExtensionCount = instanceExtensions.length;
  createInfo.ppEnabledExtensionNames = instanceExtensions;
  createInfo.enabledLayerCount = 0;
  createInfo.enabledLayerCount = validationLayers.length;
  createInfo.ppEnabledLayerNames = validationLayers;

  result = vkCreateInstance(createInfo, null, instance);
  ASSERT_VK_RESULT(result);
}

/** Create Physical Device **/
{
  let deviceCount = { $:0 };
  vkEnumeratePhysicalDevices(instance, deviceCount, null);
  if (deviceCount.$ <= 0) console.error("Error: No render devices available!");
  let devices = [...Array(deviceCount.$)].map(() => new VkPhysicalDevice());
  result = vkEnumeratePhysicalDevices(instance, deviceCount, devices);
  ASSERT_VK_RESULT(result);

  physicalDevice = devices[0];

  let deviceProperties = new VkPhysicalDeviceProperties();
  vkGetPhysicalDeviceProperties(physicalDevice, deviceProperties);

  console.log(`Using device: ${deviceProperties.deviceName}`);

  // get device UUID

  let physicalDeviceIDProperties = new VkPhysicalDeviceIDProperties();
  let physicalDeviceProperties2 = new VkPhysicalDeviceProperties2();
  physicalDeviceProperties2.pNext = physicalDeviceIDProperties;

  vkGetPhysicalDeviceProperties2(physicalDevice, physicalDeviceProperties2);
  let {deviceUUID} = physicalDeviceIDProperties;
}

/** Create Logical Device **/
{
  let deviceQueueInfo = new VkDeviceQueueCreateInfo();
  deviceQueueInfo.queueFamilyIndex = 0;
  deviceQueueInfo.queueCount = 1;
  deviceQueueInfo.pQueuePriorities = new Float32Array([1.0, 1.0, 1.0, 1.0]);

  let enabledFeatures = new VkPhysicalDeviceFeatures();
  enabledFeatures.samplerAnisotropy = true;

  let deviceInfo = new VkDeviceCreateInfo();
  deviceInfo.queueCreateInfoCount = 1;
  deviceInfo.pQueueCreateInfos = [deviceQueueInfo];
  deviceInfo.enabledExtensionCount = deviceExtensions.length;
  deviceInfo.ppEnabledExtensionNames = deviceExtensions;
  deviceInfo.pEnabledFeatures = enabledFeatures;

  result = vkCreateDevice(physicalDevice, deviceInfo, null, device);
  ASSERT_VK_RESULT(result);
}

/** Create Device Queue **/
{
  vkGetDeviceQueue(device, 0, 0, queue);
}

/** Create Command Pool **/
{
  let commandPoolInfo = new VkCommandPoolCreateInfo();
  commandPoolInfo.queueFamilyIndex = 0;

  result = vkCreateCommandPool(device, commandPoolInfo, null, commandPool);
  ASSERT_VK_RESULT(result);
}

let input = fastPNG.decode(fs.readFileSync("./input.png"));

// source buffer - contains our pixels
let srcBuffer = new VkBuffer();
let srcMemory = new VkDeviceMemory();
// destination buffer on device
let dstBuffer = new VkBuffer();
let dstMemory = new VkDeviceMemory();
// denoise buffer on device
let denoiseBuffer = new VkBuffer();
let denoiseMemory = new VkDeviceMemory();
let dstMemorySize = 0;
// for synchronization between cuda<->vulkan
let denoiseWaitSemaphore = new VkSemaphore();
let denoiseUpdateSemaphore = new VkSemaphore();

/** Create GUI Image **/
{
  let byteLength = input.width * input.height * 4 * Float32Array.BYTES_PER_ELEMENT;
  // create source buffer
  {
    let bufferInfo = new VkBufferCreateInfo();
    bufferInfo.size = byteLength;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    result = vkCreateBuffer(device, bufferInfo, null, srcBuffer);
    ASSERT_VK_RESULT(result);

    let memoryRequirements = new VkMemoryRequirements();
    vkGetBufferMemoryRequirements(device, srcBuffer, memoryRequirements);

    let memoryTypeIndex = getMemoryTypeIndex(
      memoryRequirements.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    let memoryAllocateInfo = new VkMemoryAllocateInfo();
    memoryAllocateInfo.allocationSize = memoryRequirements.size;
    memoryAllocateInfo.memoryTypeIndex = memoryTypeIndex;

    result = vkAllocateMemory(device, memoryAllocateInfo, null, srcMemory);
    ASSERT_VK_RESULT(result);

    result = vkBindBufferMemory(device, srcBuffer, srcMemory, 0n);
    ASSERT_VK_RESULT(result);

    dstMemorySize = memoryRequirements.size;

    // map
    let dataPtr = { $: 0n };
    result = vkMapMemory(device, srcMemory, 0n, bufferInfo.size, 0, dataPtr);
    ASSERT_VK_RESULT(result);

    // copy
    let mappedBuffer = ArrayBuffer.fromAddress(dataPtr.$, bufferInfo.size);

    let srcView = new Uint8Array(new Float32Array(input.data).buffer);
    let dstView = new Uint8Array(mappedBuffer);
    dstView.set(srcView, 0x0);
  }

  // create destination buffer
  {
    let bufferInfo = new VkBufferCreateInfo();
    bufferInfo.size = dstMemorySize;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    result = vkCreateBuffer(device, bufferInfo, null, dstBuffer);
    ASSERT_VK_RESULT(result);

    let memoryRequirements = new VkMemoryRequirements();
    vkGetBufferMemoryRequirements(device, dstBuffer, memoryRequirements);

    let vulkanExportMemoryWin32HandleInfoKHR = new VkExportMemoryWin32HandleInfoKHR();
    vulkanExportMemoryWin32HandleInfoKHR.pNext = null;
    vulkanExportMemoryWin32HandleInfoKHR.pAttributes = winSecurityAttributes;
    vulkanExportMemoryWin32HandleInfoKHR.dwAccess = 0x80000000 | 1;

    let vulkanExportMemoryAllocateInfoKHR = new VkExportMemoryAllocateInfoKHR();
    vulkanExportMemoryAllocateInfoKHR.pNext = vulkanExportMemoryWin32HandleInfoKHR;
    vulkanExportMemoryAllocateInfoKHR.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    let memoryAllocateInfo = new VkMemoryAllocateInfo();
    memoryAllocateInfo.pNext = vulkanExportMemoryAllocateInfoKHR;
    memoryAllocateInfo.allocationSize = memoryRequirements.size;
    memoryAllocateInfo.memoryTypeIndex = getMemoryTypeIndex(
      memoryRequirements.memoryTypeBits,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );;

    result = vkAllocateMemory(device, memoryAllocateInfo, null, dstMemory);
    ASSERT_VK_RESULT(result);

    result = vkBindBufferMemory(device, dstBuffer, dstMemory, 0n);
    ASSERT_VK_RESULT(result);
  }

  // create denoise buffer
  {
    let bufferInfo = new VkBufferCreateInfo();
    bufferInfo.size = dstMemorySize;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    result = vkCreateBuffer(device, bufferInfo, null, denoiseBuffer);
    ASSERT_VK_RESULT(result);

    let memoryRequirements = new VkMemoryRequirements();
    vkGetBufferMemoryRequirements(device, denoiseBuffer, memoryRequirements);

    let vulkanExportMemoryWin32HandleInfoKHR = new VkExportMemoryWin32HandleInfoKHR();
    vulkanExportMemoryWin32HandleInfoKHR.pNext = null;
    vulkanExportMemoryWin32HandleInfoKHR.pAttributes = winSecurityAttributes;
    vulkanExportMemoryWin32HandleInfoKHR.dwAccess = 0x80000000 | 1;

    let vulkanExportMemoryAllocateInfoKHR = new VkExportMemoryAllocateInfoKHR();
    vulkanExportMemoryAllocateInfoKHR.pNext = vulkanExportMemoryWin32HandleInfoKHR;
    vulkanExportMemoryAllocateInfoKHR.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    let memoryAllocateInfo = new VkMemoryAllocateInfo();
    memoryAllocateInfo.pNext = vulkanExportMemoryAllocateInfoKHR;
    memoryAllocateInfo.allocationSize = memoryRequirements.size;
    memoryAllocateInfo.memoryTypeIndex = getMemoryTypeIndex(
      memoryRequirements.memoryTypeBits,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );;

    result = vkAllocateMemory(device, memoryAllocateInfo, null, denoiseMemory);
    ASSERT_VK_RESULT(result);

    result = vkBindBufferMemory(device, denoiseBuffer, denoiseMemory, 0n);
    ASSERT_VK_RESULT(result);
  }

  // create wait and update semaphores
  {
    let semaphoreInfo = new VkSemaphoreCreateInfo();

    let exportSemaphoreWin32HandleInfoKHRInfo = new VkExportSemaphoreWin32HandleInfoKHR();
    exportSemaphoreWin32HandleInfoKHRInfo.pNext = null;
    exportSemaphoreWin32HandleInfoKHRInfo.pAttributes = winSecurityAttributes;
    exportSemaphoreWin32HandleInfoKHRInfo.dwAccess = 0x80000000 | 1;

    let exportSemaphoreKHRInfo = new VkExportSemaphoreCreateInfoKHR();
    exportSemaphoreKHRInfo.pNext = exportSemaphoreWin32HandleInfoKHRInfo;
    exportSemaphoreKHRInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    semaphoreInfo.pNext = exportSemaphoreKHRInfo;

    result = vkCreateSemaphore(device, semaphoreInfo, null, denoiseWaitSemaphore);
    ASSERT_VK_RESULT(result);

    result = vkCreateSemaphore(device, semaphoreInfo, null, denoiseUpdateSemaphore);
    ASSERT_VK_RESULT(result);
  }

  // copy source buffer into destination buffer
  {
    let commandBufferAllocateInfo = new VkCommandBufferAllocateInfo();
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandPool = commandPool;
    commandBufferAllocateInfo.commandBufferCount = 1;

    let commandBuffer = new VkCommandBuffer();
    vkAllocateCommandBuffers(device, commandBufferAllocateInfo, [commandBuffer]);

    let commandBufferBeginInfo = new VkCommandBufferBeginInfo();
    commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, commandBufferBeginInfo);

    let bufferCopy = new VkBufferCopy();
    bufferCopy.srcOffset = 0x0;
    bufferCopy.dstOffset = 0x0;
    bufferCopy.size = byteLength;

    vkCmdCopyBuffer(
      commandBuffer,
      srcBuffer,
      dstBuffer,
      1, [bufferCopy]
    );

    vkEndCommandBuffer(commandBuffer);

    let submitInfo = new VkSubmitInfo();
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = [commandBuffer];

    vkQueueSubmit(queue, 1, [submitInfo], null);
    vkQueueWaitIdle(queue);

    vkFreeCommandBuffers(device, commandPool, 1, [commandBuffer]);
  }

  // create handle for the destination buffer
  // so we can use it as a shared buffer for the denoiser
  {
    denoiser.create(input.width, input.height);

    // create input memory handle
    {
      let memoryGetWin32HandleInfoKHRInfo = new VkMemoryGetWin32HandleInfoKHR();
      memoryGetWin32HandleInfoKHRInfo.memory = dstMemory;
      memoryGetWin32HandleInfoKHRInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

      let buffer = new ArrayBuffer(0x8);
      result = vkGetMemoryWin32HandleKHR(device, memoryGetWin32HandleInfoKHRInfo, { $: buffer.getAddress() });
      ASSERT_VK_RESULT(result);

      denoiser.setVulkanInputMemory(new BigUint64Array(buffer)[0]);
    }

    // create output memory handle
    {
      let memoryGetWin32HandleInfoKHRInfo = new VkMemoryGetWin32HandleInfoKHR();
      memoryGetWin32HandleInfoKHRInfo.memory = denoiseMemory;
      memoryGetWin32HandleInfoKHRInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

      let buffer = new ArrayBuffer(0x8);
      result = vkGetMemoryWin32HandleKHR(device, memoryGetWin32HandleInfoKHRInfo, { $: buffer.getAddress() });
      ASSERT_VK_RESULT(result);

      denoiser.setVulkanOutputMemory(new BigUint64Array(buffer)[0]);
    }

    // create wait semaphore handle
    {
      let semaphoreGetWin32HandleInfoKHRInfo = new VkSemaphoreGetWin32HandleInfoKHR();
      semaphoreGetWin32HandleInfoKHRInfo.semaphore = denoiseWaitSemaphore;
      semaphoreGetWin32HandleInfoKHRInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;

      let buffer = new ArrayBuffer(0x8);
      result = vkGetSemaphoreWin32HandleKHR(device, semaphoreGetWin32HandleInfoKHRInfo, { $: buffer.getAddress() });
      ASSERT_VK_RESULT(result);

      denoiser.setWaitSemaphore(new BigUint64Array(buffer)[0]);
    }

    // create update semaphore handle
    {
      let semaphoreGetWin32HandleInfoKHRInfo = new VkSemaphoreGetWin32HandleInfoKHR();
      semaphoreGetWin32HandleInfoKHRInfo.semaphore = denoiseUpdateSemaphore;
      semaphoreGetWin32HandleInfoKHRInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;

      let buffer = new ArrayBuffer(0x8);
      result = vkGetSemaphoreWin32HandleKHR(device, semaphoreGetWin32HandleInfoKHRInfo, { $: buffer.getAddress() });
      ASSERT_VK_RESULT(result);

      denoiser.setUpdateSemaphore(new BigUint64Array(buffer)[0]);
    }

  }

}

/** Create descriptors **/
{
  let descriptorSetLayoutBinding = new VkDescriptorSetLayoutBinding();
  descriptorSetLayoutBinding.binding = 0;
  descriptorSetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  descriptorSetLayoutBinding.descriptorCount = 1;
  descriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  descriptorSetLayoutBinding.pImmutableSamplers = null;

  let descriptorSetLayoutInfo = new VkDescriptorSetLayoutCreateInfo();
  descriptorSetLayoutInfo.bindingCount = 2;
  descriptorSetLayoutInfo.pBindings = [
    new VkDescriptorSetLayoutBinding({
      binding: 0,
      descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      descriptorCount: 1,
      stageFlags: VK_SHADER_STAGE_FRAGMENT_BIT
    }),
    new VkDescriptorSetLayoutBinding({
      binding: 1,
      descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      descriptorCount: 1,
      stageFlags: VK_SHADER_STAGE_FRAGMENT_BIT
    })
  ];

  result = vkCreateDescriptorSetLayout(device, descriptorSetLayoutInfo, null, descriptorSetLayout);
  ASSERT_VK_RESULT(result);

  let descriptorPoolInfo = new VkDescriptorPoolCreateInfo();
  descriptorPoolInfo.maxSets = 1;
  descriptorPoolInfo.poolSizeCount = 1;
  descriptorPoolInfo.pPoolSizes = [
    new VkDescriptorPoolSize({
      type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      descriptorCount: 2
    })
  ];

  result = vkCreateDescriptorPool(device, descriptorPoolInfo, null, descriptorPool);
  ASSERT_VK_RESULT(result);

  let descriptorSetAllocInfo = new VkDescriptorSetAllocateInfo();
  descriptorSetAllocInfo.descriptorPool = descriptorPool;
  descriptorSetAllocInfo.descriptorSetCount = 1;
  descriptorSetAllocInfo.pSetLayouts = [descriptorSetLayout];

  result = vkAllocateDescriptorSets(device, descriptorSetAllocInfo, [descriptorSet]);
  ASSERT_VK_RESULT(result);

  let dstBufferWriteDescriptorSet = new VkWriteDescriptorSet();
  dstBufferWriteDescriptorSet.dstSet = descriptorSet;
  dstBufferWriteDescriptorSet.dstBinding = 0;
  dstBufferWriteDescriptorSet.dstArrayElement = 0;
  dstBufferWriteDescriptorSet.descriptorCount = 1;
  dstBufferWriteDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dstBufferWriteDescriptorSet.pBufferInfo = [
    new VkDescriptorBufferInfo({
      buffer: dstBuffer,
      offset: 0x0,
      range: VK_WHOLE_SIZE
    })
  ];

  let denoiseBufferWriteDescriptorSet = new VkWriteDescriptorSet();
  denoiseBufferWriteDescriptorSet.dstSet = descriptorSet;
  denoiseBufferWriteDescriptorSet.dstBinding = 1;
  denoiseBufferWriteDescriptorSet.dstArrayElement = 0;
  denoiseBufferWriteDescriptorSet.descriptorCount = 1;
  denoiseBufferWriteDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  denoiseBufferWriteDescriptorSet.pBufferInfo = [
    new VkDescriptorBufferInfo({
      buffer: denoiseBuffer,
      offset: 0x0,
      range: VK_WHOLE_SIZE
    })
  ];

  vkUpdateDescriptorSets(
    device,
    2, [dstBufferWriteDescriptorSet, denoiseBufferWriteDescriptorSet],
    0, null
  );
}

/** Create Surface **/
{
  result = win.createSurface(instance, null, surface);
  ASSERT_VK_RESULT(result);

  let surfaceSupport = { $: false };
  vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, 0, surface, surfaceSupport);
  if (!surfaceSupport) console.error(`No surface creation support!`);
}

/** Create Swapchain **/
{
  let swapchainInfo = new VkSwapchainCreateInfoKHR();
  swapchainInfo.surface = surface;
  swapchainInfo.minImageCount = 3;
  swapchainInfo.imageFormat = VK_FORMAT_B8G8R8A8_UNORM;
  swapchainInfo.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
  swapchainInfo.imageExtent.width = win.width;
  swapchainInfo.imageExtent.height = win.height
  swapchainInfo.imageArrayLayers = 1;
  swapchainInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  swapchainInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  swapchainInfo.queueFamilyIndexCount = 0;
  swapchainInfo.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
  swapchainInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  swapchainInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
  swapchainInfo.clipped = true;
  swapchainInfo.oldSwapchain = null;

  result = vkCreateSwapchainKHR(device, swapchainInfo, null, swapchain);
  ASSERT_VK_RESULT(result);

  vkGetSwapchainImagesKHR(device, swapchain, amountOfImagesInSwapchain, null);
  let swapchainImages = [...Array(amountOfImagesInSwapchain.$)].map(() => new VkImage());

  result = vkGetSwapchainImagesKHR(device, swapchain, amountOfImagesInSwapchain, swapchainImages);
  ASSERT_VK_RESULT(result);

  for (let ii = 0; ii < amountOfImagesInSwapchain.$; ++ii) {
    let imageViewInfo = new VkImageViewCreateInfo();
    imageViewInfo.image = swapchainImages[ii];
    imageViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    imageViewInfo.format = VK_FORMAT_B8G8R8A8_UNORM;
    imageViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageViewInfo.subresourceRange.baseMipLevel = 0;
    imageViewInfo.subresourceRange.levelCount = 1;
    imageViewInfo.subresourceRange.baseArrayLayer = 0;
    imageViewInfo.subresourceRange.layerCount = 1;

    imageViews[ii] = new VkImageView();
    result = vkCreateImageView(device, imageViewInfo, null, imageViews[ii])
    ASSERT_VK_RESULT(result);
  };
}

/** Create Shader Modules **/
{
  let vertSrc = GLSL.toSPIRVSync({
    source: fs.readFileSync(`./shaders/gui.vert`),
    extension: `vert`
  }).output;

  let fragSrc = GLSL.toSPIRVSync({
    source: fs.readFileSync(`./shaders/gui.frag`),
    extension: `frag`
  }).output;

  let vertShaderModuleInfo = new VkShaderModuleCreateInfo();
  vertShaderModuleInfo.pCode = vertSrc;
  vertShaderModuleInfo.codeSize = vertSrc.byteLength;
  result = vkCreateShaderModule(device, vertShaderModuleInfo, null, vertShaderModule);
  ASSERT_VK_RESULT(result);

  let fragShaderModuleInfo = new VkShaderModuleCreateInfo();
  fragShaderModuleInfo.pCode = fragSrc;
  fragShaderModuleInfo.codeSize = fragSrc.byteLength;
  result = vkCreateShaderModule(device, fragShaderModuleInfo, null, fragShaderModule);
  ASSERT_VK_RESULT(result);
}

/** Create Pipeline Layout **/
{
  let pipelineLayoutInfo = new VkPipelineLayoutCreateInfo();
  pipelineLayoutInfo.setLayoutCount = 1;
  pipelineLayoutInfo.pSetLayouts = [descriptorSetLayout];

  result = vkCreatePipelineLayout(device, pipelineLayoutInfo, null, pipelineLayout);
  ASSERT_VK_RESULT(result);
}

/** Create Render Pass **/
{
  let attachmentDescription = new VkAttachmentDescription();
  attachmentDescription.flags = 0;
  attachmentDescription.format = VK_FORMAT_B8G8R8A8_UNORM;
  attachmentDescription.samples = VK_SAMPLE_COUNT_1_BIT;
  attachmentDescription.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  attachmentDescription.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  attachmentDescription.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  attachmentDescription.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  attachmentDescription.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  attachmentDescription.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

  let attachmentReference = new VkAttachmentReference();
  attachmentReference.attachment = 0;
  attachmentReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  let subpassDescription = new VkSubpassDescription();
  subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpassDescription.colorAttachmentCount = 1;
  subpassDescription.pColorAttachments = [attachmentReference];

  let subpassDependency = new VkSubpassDependency();
  subpassDependency.srcSubpass = VK_SUBPASS_EXTERNAL;
  subpassDependency.dstSubpass = 0;
  subpassDependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  subpassDependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  subpassDependency.srcAccessMask = 0;
  subpassDependency.dstAccessMask = (
    VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
  );
  subpassDependency.dependencyFlags = 0;

  let renderPassInfo = new VkRenderPassCreateInfo();
  renderPassInfo.attachmentCount = 1;
  renderPassInfo.pAttachments = [attachmentDescription];
  renderPassInfo.subpassCount = 1;
  renderPassInfo.pSubpasses = [subpassDescription];
  renderPassInfo.dependencyCount = 1;
  renderPassInfo.pDependencies = [subpassDependency];

  result = vkCreateRenderPass(device, renderPassInfo, null, renderPass);
  ASSERT_VK_RESULT(result);
}

/** Create Graphics Pipeline **/
{
  let vertShaderStageInfo = new VkPipelineShaderStageCreateInfo();
  vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
  vertShaderStageInfo.module = vertShaderModule;
  vertShaderStageInfo.pName = "main";

  let fragShaderStageInfo = new VkPipelineShaderStageCreateInfo();
  fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  fragShaderStageInfo.module = fragShaderModule;
  fragShaderStageInfo.pName = "main";

  let viewport = new VkViewport();
  viewport.x = 0;
  viewport.y = 0;
  viewport.width = win.width;
  viewport.height = win.height;
  viewport.minDepth = 0.0;
  viewport.maxDepth = 1.0;

  let scissor = new VkRect2D();
  scissor.offset.x = 0;
  scissor.offset.y = 0;
  scissor.extent.width = win.width;
  scissor.extent.height = win.height;

  let viewportStateInfo = new VkPipelineViewportStateCreateInfo();
  viewportStateInfo.viewportCount = 1;
  viewportStateInfo.pViewports = [viewport];
  viewportStateInfo.scissorCount = 1;
  viewportStateInfo.pScissors = [scissor];

  let rasterizationInfo = new VkPipelineRasterizationStateCreateInfo();
  rasterizationInfo.depthClampEnable = false;
  rasterizationInfo.rasterizerDiscardEnable = false;
  rasterizationInfo.polygonMode = VK_POLYGON_MODE_FILL;
  rasterizationInfo.cullMode = VK_CULL_MODE_FRONT_BIT;
  rasterizationInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rasterizationInfo.depthBiasEnable = false;
  rasterizationInfo.depthBiasConstantFactor = 0.0;
  rasterizationInfo.depthBiasClamp = 0.0;
  rasterizationInfo.depthBiasSlopeFactor = 0.0;
  rasterizationInfo.lineWidth = 1.0;

  let multisampleInfo = new VkPipelineMultisampleStateCreateInfo();
  multisampleInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  multisampleInfo.minSampleShading = 1.0;
  multisampleInfo.pSampleMask = null;
  multisampleInfo.alphaToCoverageEnable = false;
  multisampleInfo.alphaToOneEnable = false;

  let colorBlendAttachment = new VkPipelineColorBlendAttachmentState();
  colorBlendAttachment.blendEnable = true;
  colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
  colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
  colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
  colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
  colorBlendAttachment.colorWriteMask = (
    VK_COLOR_COMPONENT_R_BIT |
    VK_COLOR_COMPONENT_G_BIT |
    VK_COLOR_COMPONENT_B_BIT |
    VK_COLOR_COMPONENT_A_BIT
  );

  let colorBlendInfo = new VkPipelineColorBlendStateCreateInfo();
  colorBlendInfo.logicOpEnable = false;
  colorBlendInfo.logicOp = VK_LOGIC_OP_NO_OP;
  colorBlendInfo.attachmentCount = 1;
  colorBlendInfo.pAttachments = [colorBlendAttachment];
  colorBlendInfo.blendConstants = [0.0, 0.0, 0.0, 0.0];

  let graphicsPipelineInfo = new VkGraphicsPipelineCreateInfo();
  graphicsPipelineInfo.stageCount = 2;
  graphicsPipelineInfo.pStages = [vertShaderStageInfo, fragShaderStageInfo];
  graphicsPipelineInfo.pVertexInputState = new VkPipelineVertexInputStateCreateInfo();
  graphicsPipelineInfo.pInputAssemblyState = new VkPipelineInputAssemblyStateCreateInfo({
    topology: VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
  });
  graphicsPipelineInfo.pViewportState = viewportStateInfo;
  graphicsPipelineInfo.pRasterizationState = rasterizationInfo;
  graphicsPipelineInfo.pMultisampleState = multisampleInfo;
  graphicsPipelineInfo.pColorBlendState = colorBlendInfo;
  graphicsPipelineInfo.layout = pipelineLayout;
  graphicsPipelineInfo.renderPass = renderPass;
  graphicsPipelineInfo.subpass = 0;
  graphicsPipelineInfo.basePipelineIndex = -1;

  result = vkCreateGraphicsPipelines(device, null, 1, [graphicsPipelineInfo], null, [pipeline]);
  ASSERT_VK_RESULT(result);

  for (let ii = 0; ii < amountOfImagesInSwapchain.$; ++ii) {
    let framebufferInfo = new VkFramebufferCreateInfo();
    framebufferInfo.renderPass = renderPass;
    framebufferInfo.attachmentCount = 1;
    framebufferInfo.pAttachments = [imageViews[ii]];
    framebufferInfo.width = win.width;
    framebufferInfo.height = win.height;
    framebufferInfo.layers = 1;
    framebuffers[ii] = new VkFramebuffer();
    result = vkCreateFramebuffer(device, framebufferInfo, null, framebuffers[ii]);
    ASSERT_VK_RESULT(result);
  };
}

/** Create Synchronization **/
{
  let semaphoreInfo = new VkSemaphoreCreateInfo();
  result = vkCreateSemaphore(device, semaphoreInfo, null, semaphoreImageAvailable);
  ASSERT_VK_RESULT(result);

  result = vkCreateSemaphore(device, semaphoreInfo, null, semaphoreRenderingAvailable);
  ASSERT_VK_RESULT(result);
}

/** Create Command Buffers **/
{
  let commandBufferAllocateInfo = new VkCommandBufferAllocateInfo();
  commandBufferAllocateInfo.commandPool = commandPool;
  commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  commandBufferAllocateInfo.commandBufferCount = amountOfImagesInSwapchain.$;

  for (let ii = 0; ii < amountOfImagesInSwapchain.$; ++ii) {
    commandBuffers.push(new VkCommandBuffer());
  };

  result = vkAllocateCommandBuffers(device, commandBufferAllocateInfo, commandBuffers);
  ASSERT_VK_RESULT(result);

  /** Record Command Buffers **/
  for (let ii = 0; ii < commandBuffers.length; ++ii) {
    let commandBuffer = commandBuffers[ii];

    let commandBufferBeginInfo = new VkCommandBufferBeginInfo();
    commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    commandBufferBeginInfo.pInheritanceInfo = null;
    result = vkBeginCommandBuffer(commandBuffer, commandBufferBeginInfo);
    ASSERT_VK_RESULT(result);

    let renderPassBeginInfo = new VkRenderPassBeginInfo();
    renderPassBeginInfo.renderPass = renderPass;
    renderPassBeginInfo.framebuffer = framebuffers[ii];
    renderPassBeginInfo.renderArea.offset.x = 0;
    renderPassBeginInfo.renderArea.offset.y = 0;
    renderPassBeginInfo.renderArea.extent.width = win.width;
    renderPassBeginInfo.renderArea.extent.height = win.height;
    renderPassBeginInfo.clearValueCount = 1;
    renderPassBeginInfo.pClearValues = [new VkClearValue()];

    vkCmdBeginRenderPass(commandBuffer, renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, [descriptorSet], 0, null);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    vkCmdEndRenderPass(commandBuffer);

    result = vkEndCommandBuffer(commandBuffer);
    ASSERT_VK_RESULT(result);
  };
}

let initialSubmit = true;
(function drawLoop() {
  win.pollEvents();

  let imageIndex = { $: 0 };
  vkAcquireNextImageKHR(device, swapchain, Number.MAX_SAFE_INTEGER, semaphoreImageAvailable, null, imageIndex);

  // wait for the vulkan draw commands to finish
  let submitInfo = new VkSubmitInfo();
  // wait for the present frame to be ready
  // and the denoiser to be finished with denoising
  submitInfo.waitSemaphoreCount = 2;
  submitInfo.pWaitSemaphores = [semaphoreImageAvailable, denoiseUpdateSemaphore];
  submitInfo.pWaitDstStageMask = new Int32Array([
    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT
  ]);
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = [commandBuffers[imageIndex.$]];
  // present as soon as all vulkan commands are flushed
  // the denoiser can now denoiser our input buffer
  submitInfo.signalSemaphoreCount = 2;
  submitInfo.pSignalSemaphores = [semaphoreRenderingAvailable, denoiseWaitSemaphore];

  result = vkQueueSubmit(queue, 1, [submitInfo], null);
  ASSERT_VK_RESULT(result);

  denoiser.denoise(0.25);

  // present the frame as soon as all vulkan commands are flushed
  let presentInfo = new VkPresentInfoKHR();
  // wait for the present and denoiser semaphores to be done
  presentInfo.waitSemaphoreCount = 1;
  presentInfo.pWaitSemaphores = [semaphoreRenderingAvailable];
  presentInfo.swapchainCount = 1;
  presentInfo.pSwapchains = [swapchain];
  presentInfo.pImageIndices = new Uint32Array([imageIndex.$]);
  presentInfo.pResults = null;

  result = vkQueuePresentKHR(queue, presentInfo);
  if (result === VK_SUBOPTIMAL_KHR || result === VK_ERROR_OUT_OF_DATE_KHR) {
    // end
  } else {
    ASSERT_VK_RESULT(result);
    setImmediate(drawLoop);
  }

})();
