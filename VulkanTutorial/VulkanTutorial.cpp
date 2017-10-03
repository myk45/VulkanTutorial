

#include <stdlib.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <string.h>
#include <array>
#include <fstream>
#include <vulkan/vulkan.h>

// GLFW
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

// Vulkan tutorial code.
// We shall cover the following concepts:-

// Memory in Vulkan:
//   - Show memory available on Device
//   - Show buffer creation, image creation.
//   - Show allocation of actual memory for buffers.
//   - Show concept of staging buffer.

// Descriptor set
//   - Example usage in shader

// Swapchain
// Shaders
// Renderpass


//=====================================================================
// VK resources
static VkImage image = VK_NULL_HANDLE;
static VkInstance vkTutInstance;
static VkPhysicalDevice vkTutPhysDevice;
static VkDevice vkTutDevice;
static VkImageView imageView;
static VkDeviceMemory imageMemory;
static VkCommandPool cmdPool;
static uint32_t queueNodeIndex = UINT32_MAX;
static VkCommandBuffer cmdBuffer;
static VkQueue queue;
static VkFence fence;
static VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties;
static VkBuffer        srcBuffer, dstBuffer;
static VkDeviceMemory  srcMemory, dstMemory;
static VkSurfaceKHR    surface;
static VkSwapchainKHR  swapchain;
static uint32_t        imageCount;
static VkDescriptorSetLayout descriptorSetLayout;
static VkShaderModule shaderModule;
static VkShaderModule vsm;
static VkShaderModule fsm;
static VkRenderPass   renderPass;
static VkFramebuffer  frameBuffer;

int width  = 1920;
int height = 1080;

// The pipeline layout is used by a pipline to access the descriptor sets
// It defines interface (without binding any actual data) between the shader stages used by the pipeline and the shader resources
// A pipeline layout can be shared among multiple pipelines as long as their interfaces match
static VkPipelineLayout pipelineLayout;

// The descriptor set stores the resources bound to the binding points in a shader
// It connects the binding points of the different shaders with the buffers and images used for those bindings
static VkDescriptorSet descriptorSet;

static VkDescriptorPool descriptorPool = VK_NULL_HANDLE;

static VkPipeline pipeline;
static VkPipelineCache pipelineCache;
//=====================================================================

// Vertex layout used in this example
struct Vertex {
	float position[3];
	float color[3];
};

// Vertex buffer and attributes
struct {
	VkDeviceMemory memory;															// Handle to the device memory for this buffer
	VkBuffer buffer;																// Handle to the Vulkan buffer object that the memory is bound to
} vertices;

// Index buffer
struct
{
	VkDeviceMemory memory;
	VkBuffer buffer;
	uint32_t count;
} indices;

// Uniform buffer block object
struct {
	VkDeviceMemory memory;
	VkBuffer buffer;
	VkDescriptorBufferInfo descriptor;
}  uniformBufferVS;

//=====================================================================
typedef struct _SwapChainBuffers {
	VkImage     image;
	VkImageView view;
} SwapChainBuffer;

static std::vector<SwapChainBuffer> buffers;
static std::vector<VkImage>         images;
//=====================================================================


//=====================================================================
static char const vss[] =
"#version 450\n"
"\n"
"#extension GL_ARB_separate_shader_objects : enable\n"
"#extension GL_ARB_shading_language_420pack : enable\n"
"\n"
"layout(location = 0) in vec3 inPos; \n"
"layout(location = 1) in vec3 inColor; \n"
"\n"
"const vec2 data[4] = vec2[]\n"
"(\n"
"	vec2(-1.0, 1.0), \n"
"	vec2(-1.0, -1.0), \n"
"	vec2(1.0, 1.0), \n"
"	vec2(1.0, -1.0)\n"
");\n"

"layout(binding = 0) uniform UBO\n"
"{\n"
"	mat4 projectionMatrix; \n"
"	mat4 modelMatrix; \n"
"	mat4 viewMatrix; \n"
"} ubo; \n"
"\n"
"layout(location = 0) out vec3 outColor; \n"
"\n"
"out gl_PerVertex\n"
"{ \n"
"	vec4 gl_Position; \n"
"}; \n"
"void main()\n"
"{\n"
"	outColor = inColor;\n"
"	gl_Position = ubo.projectionMatrix * ubo.viewMatrix * ubo.modelMatrix * vec4(inPos.xyz, 1.0);\n"
"   gl_Position = vec4( data[ gl_VertexID ], 0.0, 1.0);\n"
"}\n"
;

static char const fss[] =
"#version 450 core\n"
"#extension GL_ARB_separate_shader_objects : enable\n"
"#extension GL_ARB_shading_language_420pack : enable\n"
"\n"
"layout(location = 0) in vec3 inColor; \n"
"\n"
"layout(location = 0) out vec4 outFragColor; \n"
"\n"
"void main()\n"
"{\n"
"	outFragColor = vec4(1.0, 1.0, 0.0, 1.0); \n"
"}\n"
;

//=====================================================================

bool createDevice()
{
	// First figure out how many devices are in the system.
	uint32_t physicalDeviceCount = 0;
	vkEnumeratePhysicalDevices(vkTutInstance, &physicalDeviceCount, nullptr);
	vkEnumeratePhysicalDevices(vkTutInstance, &physicalDeviceCount, &vkTutPhysDevice);

	//std::cout << "Physical devices: " << physicalDeviceCount;

	uint32_t queueFamilyPropertyCount;
	std::vector<VkQueueFamilyProperties> queueFamilyProperties;
	vkGetPhysicalDeviceMemoryProperties(vkTutPhysDevice, &physicalDeviceMemoryProperties);
	vkGetPhysicalDeviceQueueFamilyProperties(vkTutPhysDevice, &queueFamilyPropertyCount, nullptr);
	queueFamilyProperties.resize(queueFamilyPropertyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(vkTutPhysDevice, &queueFamilyPropertyCount, queueFamilyProperties.data());

	// Search for a graphics and a present queue in the array of queue
	// families, try to find one that supports both
	uint32_t graphicsQueueNodeIndex = UINT32_MAX;
	uint32_t presentQueueNodeIndex = UINT32_MAX;
	for (uint32_t i = 0; i < queueFamilyPropertyCount; i++)
	{
		if ((queueFamilyProperties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0)
		{
			if (graphicsQueueNodeIndex == UINT32_MAX)
			{
				graphicsQueueNodeIndex = i;
				break;
			}
		}
	}
	//queueNodeIndex = graphicsQueueNodeIndex;

	auto err = VK_SUCCESS;
	VkPhysicalDeviceFeatures supportedFeatures;
	VkPhysicalDeviceFeatures requiredFeatures = {};
	vkGetPhysicalDeviceFeatures(vkTutPhysDevice, &supportedFeatures);
	requiredFeatures.multiDrawIndirect = supportedFeatures.multiDrawIndirect;
	requiredFeatures.tessellationShader = VK_TRUE;
	requiredFeatures.geometryShader = VK_TRUE;

	const VkDeviceQueueCreateInfo deviceQueueCreateInfo =
	{
		VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
		nullptr,
		0,
		graphicsQueueNodeIndex,
		1,
		nullptr
	};

	std::vector<const char*> extNames;
	extNames.push_back(VK_KHR_DISPLAY_SWAPCHAIN_EXTENSION_NAME);
	const VkDeviceCreateInfo deviceCreateInfo =
	{
		VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		nullptr,
		0,
		1,
		&deviceQueueCreateInfo,
		0,
		nullptr,
		0,//extNames.size(),
		nullptr,//extNames.data(),
		&requiredFeatures
	};

	err = vkCreateDevice(vkTutPhysDevice, &deviceCreateInfo, nullptr, &vkTutDevice);
	assert(err == VK_SUCCESS);

	queueNodeIndex = graphicsQueueNodeIndex;
	vkGetDeviceQueue(vkTutDevice, graphicsQueueNodeIndex, 0, &queue);
	return err == VK_SUCCESS;
}

bool createInstance()
{
	std::vector<const char*> extNames;
	extNames.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
	extNames.push_back(VK_KHR_SURFACE_EXTENSION_NAME);

	auto err = VK_SUCCESS;

	//Create a Vulkan Instance
	VkInstanceCreateInfo InstanceCreateInfo;
	InstanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	InstanceCreateInfo.pNext = NULL;
	InstanceCreateInfo.pApplicationInfo = NULL;
	InstanceCreateInfo.enabledExtensionCount = extNames.size();
	InstanceCreateInfo.ppEnabledExtensionNames = extNames.data();
	InstanceCreateInfo.enabledLayerCount = 0;
	InstanceCreateInfo.ppEnabledLayerNames = nullptr;
	err = vkCreateInstance(&InstanceCreateInfo, nullptr, &vkTutInstance);

	return err == VK_SUCCESS;
}

uint32_t getMemoryType(uint32_t typeBits, VkFlags properties)
{
	for (uint32_t i = 0; i < 32; i++)
	{
		if ((typeBits & 1) == 1)
		{
			if (physicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & properties)
			{
				return i;
			}
		}
		typeBits >>= 1;
	}

	assert(0);
	return 0;
}

bool createBuffer(VkBuffer& buffer, VkDeviceMemory& bufferMemory, unsigned int size)
{
	static const VkBufferCreateInfo bufferCreateInfo =
	{
		VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		nullptr,
		0,
		size,
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VK_SHARING_MODE_EXCLUSIVE,
		0,
		nullptr
	};

	vkCreateBuffer(vkTutDevice, &bufferCreateInfo, nullptr, &buffer);
	assert(buffer != VK_NULL_HANDLE);

	VkMemoryAllocateInfo memAlloc = {};
	VkMemoryRequirements memReqs;
	vkGetBufferMemoryRequirements(vkTutDevice, buffer, &memReqs);
	memAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	memAlloc.allocationSize = memReqs.size;
	memAlloc.memoryTypeIndex = getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
	auto result = vkAllocateMemory(vkTutDevice, &memAlloc, nullptr, &bufferMemory);
	assert(result == VK_SUCCESS);

	result = vkBindBufferMemory(vkTutDevice, buffer, bufferMemory, 0);
	assert(result == VK_SUCCESS);

	return result == VK_SUCCESS;
}

bool createImage()
{
	auto result = VK_SUCCESS;

	static const VkImageCreateInfo imageCreateInfo =
	{
		VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		nullptr,                    // pNext
		0,                          // flags
		VK_IMAGE_TYPE_2D,           // imageType
		VK_FORMAT_R8G8B8A8_UNORM,   // format
		{ 1920, 1080, 1 },          // extent
		1,                          // mipLevels
		1,                          // arrayLayers
		VK_SAMPLE_COUNT_1_BIT,      // samples
		VK_IMAGE_TILING_LINEAR,     // tiling         /* TODO: try VK_IMAGE_TILING_OPTIMAL */
		VK_IMAGE_USAGE_SAMPLED_BIT, // usage
		VK_SHARING_MODE_EXCLUSIVE,  // sharingMode
		0,
		nullptr,
		VK_IMAGE_LAYOUT_UNDEFINED
	};

	result = vkCreateImage(vkTutDevice, &imageCreateInfo, nullptr, &image);
	assert(result == VK_SUCCESS);

	VkMemoryAllocateInfo memAlloc = {};
	VkMemoryRequirements memReqs;
	vkGetImageMemoryRequirements(vkTutDevice, image, &memReqs);
	memAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	memAlloc.allocationSize = memReqs.size;
	memAlloc.memoryTypeIndex = getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT/*VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT*/);
	result = vkAllocateMemory(vkTutDevice, &memAlloc, nullptr, &imageMemory);
	assert(result == VK_SUCCESS);

	result = vkBindImageMemory(vkTutDevice, image, imageMemory, 0);
	assert(result == VK_SUCCESS);

	return result == VK_SUCCESS;
}

bool createImageView()
{
	VkImageViewCreateInfo viewInfo = {};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = image;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;     // type(should be compatible with parent).
	viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;  // format(should be compatible with parent).
	viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; // or depth bit.
	viewInfo.subresourceRange.baseMipLevel = 0;                         // where the miplevel begins.
	viewInfo.subresourceRange.levelCount = 10;                        // how many layers do you want.  
	viewInfo.subresourceRange.baseArrayLayer = 0;                         // Image array base layer.  
	viewInfo.subresourceRange.layerCount = 1;                         // Image array number of layers

	auto res = vkCreateImageView(vkTutDevice, &viewInfo, nullptr, &imageView);
	return res == VK_SUCCESS;
}

bool cleanup()
{
	vkDestroyImageView(vkTutDevice, imageView, nullptr);

	vkDestroyBuffer(vkTutDevice, srcBuffer, nullptr);
	vkDestroyBuffer(vkTutDevice, dstBuffer, nullptr);
	vkDestroyImage(vkTutDevice, image, nullptr);

	vkFreeMemory(vkTutDevice, srcMemory, nullptr);
	vkFreeMemory(vkTutDevice, dstMemory, nullptr);
	vkFreeMemory(vkTutDevice, imageMemory, nullptr);

	vkDestroyShaderModule(vkTutDevice, vsm, nullptr);
	vkDestroyShaderModule(vkTutDevice, fsm, nullptr);

	vkDestroyDescriptorSetLayout(vkTutDevice, descriptorSetLayout, nullptr);
	vkDestroyPipelineLayout(vkTutDevice, pipelineLayout, nullptr);
	vkDestroyDescriptorPool(vkTutDevice, descriptorPool, nullptr);

	vkDestroySurfaceKHR(vkTutInstance, surface, nullptr);
	vkDestroySwapchainKHR(vkTutDevice, swapchain, nullptr);

	vkDestroyCommandPool(vkTutDevice, cmdPool, nullptr);
	vkDestroyDevice(vkTutDevice, nullptr);

	vkDestroyInstance(vkTutInstance, nullptr);
	return true;
}

bool createCommandPool()
{
	VkCommandPoolCreateInfo cmdPoolInfo = {};
	cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	cmdPoolInfo.queueFamilyIndex = queueNodeIndex;
	cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	auto res = vkCreateCommandPool(vkTutDevice, &cmdPoolInfo, NULL, &cmdPool);

	return res == VK_SUCCESS;
}

bool createCommandBuffer()
{
	VkCommandBufferAllocateInfo cmdBufAllocateInfo = {};
	cmdBufAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	cmdBufAllocateInfo.commandPool = cmdPool;
	cmdBufAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cmdBufAllocateInfo.commandBufferCount = 1;

	auto res = vkAllocateCommandBuffers(vkTutDevice, &cmdBufAllocateInfo, &cmdBuffer);
	return res == VK_SUCCESS;
}

bool createFence()
{
	static const VkFenceCreateInfo fenceCreateInfo =
	{
		VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
		nullptr,
		VK_FENCE_CREATE_SIGNALED_BIT
	};

	auto res = vkCreateFence(vkTutDevice, &fenceCreateInfo, nullptr, &fence);
	return res == VK_SUCCESS;
}

bool createDescriptorSetLayout()
{
	VkResult err;

	VkDescriptorSetLayoutBinding layoutBinding = {};
	layoutBinding.descriptorType     = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	layoutBinding.descriptorCount    = 1;
	layoutBinding.stageFlags         = VK_SHADER_STAGE_VERTEX_BIT;
	layoutBinding.pImmutableSamplers = nullptr;

	VkDescriptorSetLayoutCreateInfo descriptorLayout = {};
	descriptorLayout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	descriptorLayout.pNext = nullptr;
	descriptorLayout.bindingCount = 1;
	descriptorLayout.pBindings = &layoutBinding;

	auto res = vkCreateDescriptorSetLayout(vkTutDevice, &descriptorLayout, nullptr, &descriptorSetLayout);

	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
	pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutCreateInfo.pNext = nullptr;
	pipelineLayoutCreateInfo.setLayoutCount = 1;
	pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout;

	vkCreatePipelineLayout(vkTutDevice, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout);

	VkPipelineCacheCreateInfo pipelineCacheCreateInfo = {};
	pipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
	vkCreatePipelineCache(vkTutDevice, &pipelineCacheCreateInfo, nullptr, &pipelineCache);


	VkGraphicsPipelineCreateInfo pipelineCreateInfo = {};
	pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineCreateInfo.layout = pipelineLayout;
	pipelineCreateInfo.renderPass = renderPass;

	// Create different stages
	//{
		// Input assembly state describes how primitives are assembled
		// This pipeline will assemble vertex data as a triangle lists (though we only use one triangle)
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyState = {};
		inputAssemblyState.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

		// Rasterization state
		VkPipelineRasterizationStateCreateInfo rasterizationState = {};
		rasterizationState.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizationState.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizationState.cullMode = VK_CULL_MODE_NONE;
		rasterizationState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterizationState.depthClampEnable = VK_FALSE;
		rasterizationState.rasterizerDiscardEnable = VK_FALSE;
		rasterizationState.depthBiasEnable = VK_FALSE;
		rasterizationState.lineWidth = 1.0f;

		// Color blend state describes how blend factors are calculated (if used)
		// We need one blend attachment state per color attachment (even if blending is not used
		VkPipelineColorBlendAttachmentState blendAttachmentState[1] = {};
		blendAttachmentState[0].colorWriteMask = 0xf;
		blendAttachmentState[0].blendEnable = VK_FALSE;
		VkPipelineColorBlendStateCreateInfo colorBlendState = {};
		colorBlendState.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlendState.attachmentCount = 1;
		colorBlendState.pAttachments = blendAttachmentState;

		// Viewport state sets the number of viewports and scissor used in this pipeline
		// Note: This is actually overriden by the dynamic states (see below)
		VkPipelineViewportStateCreateInfo viewportState = {};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.scissorCount = 1;

		// Enable dynamic states
		// Most states are baked into the pipeline, but there are still a few dynamic states that can be changed within a command buffer
		// To be able to change these we need do specify which dynamic states will be changed using this pipeline. Their actual states are set later on in the command buffer.
		// For this example we will set the viewport and scissor using dynamic states
		std::vector<VkDynamicState> dynamicStateEnables;
		dynamicStateEnables.push_back(VK_DYNAMIC_STATE_VIEWPORT);
		dynamicStateEnables.push_back(VK_DYNAMIC_STATE_SCISSOR);
		VkPipelineDynamicStateCreateInfo dynamicState = {};
		dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicState.pDynamicStates = dynamicStateEnables.data();
		dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStateEnables.size());

		// Depth and stencil state containing depth and stencil compare and test operations
		// We only use depth tests and want depth tests and writes to be enabled and compare with less or equal
		VkPipelineDepthStencilStateCreateInfo depthStencilState = {};
		depthStencilState.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencilState.depthTestEnable = VK_FALSE; //MYK: todo: VK_TRUE;
		depthStencilState.depthWriteEnable = VK_FALSE; // MYK: Todo:
		depthStencilState.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthStencilState.depthBoundsTestEnable = VK_FALSE;
		depthStencilState.back.failOp = VK_STENCIL_OP_KEEP;
		depthStencilState.back.passOp = VK_STENCIL_OP_KEEP;
		depthStencilState.back.compareOp = VK_COMPARE_OP_ALWAYS;
		depthStencilState.stencilTestEnable = VK_FALSE;
		depthStencilState.front = depthStencilState.back;

		// Multi sampling state
		// This example does not make use fo multi sampling (for anti-aliasing), the state must still be set and passed to the pipeline
		VkPipelineMultisampleStateCreateInfo multisampleState = {};
		multisampleState.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampleState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisampleState.pSampleMask = nullptr;

		// Vertex input descriptions 
		// Specifies the vertex input parameters for a pipeline

		// Vertex input binding
		// This example uses a single vertex input binding at binding point 0 (see vkCmdBindVertexBuffers)
		VkVertexInputBindingDescription vertexInputBinding = {};
		vertexInputBinding.binding = 0;
		vertexInputBinding.stride = sizeof(Vertex);
		vertexInputBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		// Inpute attribute bindings describe shader attribute locations and memory layouts
		std::array<VkVertexInputAttributeDescription, 2> vertexInputAttributs;
		// These match the following shader layout (see triangle.vert):
		//	layout (location = 0) in vec3 inPos;
		//	layout (location = 1) in vec3 inColor;
		// Attribute location 0: Position
		vertexInputAttributs[0].binding = 0;
		vertexInputAttributs[0].location = 0;
		// Position attribute is three 32 bit signed (SFLOAT) floats (R32 G32 B32)
		vertexInputAttributs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexInputAttributs[0].offset = offsetof(Vertex, position);
		// Attribute location 1: Color
		vertexInputAttributs[1].binding = 0;
		vertexInputAttributs[1].location = 1;
		// Color attribute is three 32 bit signed (SFLOAT) floats (R32 G32 B32)
		vertexInputAttributs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexInputAttributs[1].offset = offsetof(Vertex, color);

		// Vertex input state used for pipeline creation
		VkPipelineVertexInputStateCreateInfo vertexInputState = {};
		vertexInputState.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInputState.vertexBindingDescriptionCount = 1;
		vertexInputState.pVertexBindingDescriptions = &vertexInputBinding;
		vertexInputState.vertexAttributeDescriptionCount = 2;
		vertexInputState.pVertexAttributeDescriptions = vertexInputAttributs.data();

		// Shaders
		std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages{};

		// Vertex shader
		shaderStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		// Set pipeline stage for this shader
		shaderStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		// Load binary SPIR-V shader
		shaderStages[0].module = vsm;
		// Main entry point for the shader
		shaderStages[0].pName = "main";
		assert(shaderStages[0].module != VK_NULL_HANDLE);

		// Fragment shader
		shaderStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		// Set pipeline stage for this shader
		shaderStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		// Load binary SPIR-V shader
		shaderStages[1].module = fsm;
		// Main entry point for the shader
		shaderStages[1].pName = "main";
		assert(shaderStages[1].module != VK_NULL_HANDLE);
	//}

	pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
	pipelineCreateInfo.pStages = shaderStages.data();
	pipelineCreateInfo.pVertexInputState = &vertexInputState;
	pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
	pipelineCreateInfo.pRasterizationState = &rasterizationState;
	pipelineCreateInfo.pColorBlendState = &colorBlendState;
	pipelineCreateInfo.pMultisampleState = &multisampleState;
	pipelineCreateInfo.pViewportState = &viewportState;
	pipelineCreateInfo.pDepthStencilState = &depthStencilState;
	pipelineCreateInfo.renderPass = renderPass;
	pipelineCreateInfo.pDynamicState = &dynamicState;

	res = vkCreateGraphicsPipelines(vkTutDevice, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipeline);

	return res == VK_SUCCESS;
}

bool createShader(VkShaderStageFlagBits shaderStage, size_t size, const char *code, VkShaderModule &shaderModule)
{
	VkShaderModuleCreateInfo shadderModuleInfo = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };

	shadderModuleInfo.codeSize = size;
	shadderModuleInfo.pCode = (const uint32_t *)code;

	auto result = vkCreateShaderModule(vkTutDevice, &shadderModuleInfo, 0, &shaderModule);

	if (result != VK_SUCCESS) {
		return result;
	}

	return result == VK_SUCCESS;
}

bool allocateDescriptor()
{
	VkDescriptorSetAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.descriptorPool = descriptorPool;
	allocInfo.descriptorSetCount = 1;
	allocInfo.pSetLayouts = &descriptorSetLayout;

	auto err = vkAllocateDescriptorSets(vkTutDevice, &allocInfo, &descriptorSet);
	return err == VK_SUCCESS;
}

bool createDescriptorPool()
{
	VkDescriptorPoolSize typeCounts[1];
	typeCounts[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	typeCounts[0].descriptorCount = 1;
	// For additional types you need to add new entries in the type count list
	// E.g. for two combined image samplers :
	// typeCounts[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	// typeCounts[1].descriptorCount = 2;

	VkDescriptorPoolCreateInfo descriptorPoolInfo = {};
	descriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	descriptorPoolInfo.pNext = nullptr;
	descriptorPoolInfo.poolSizeCount = 1;
	descriptorPoolInfo.pPoolSizes = typeCounts;

	// TODO TODO
	// Set the max. number of descriptor sets that can be requested from this pool (requesting beyond this limit will result in an error)
	descriptorPoolInfo.maxSets = 1;

	auto err = vkCreateDescriptorPool(vkTutDevice, &descriptorPoolInfo, nullptr, &descriptorPool);
	return err == VK_SUCCESS;
}

void setupFrameBuffer()
{
	VkImageView attachments[1];

	VkFramebufferCreateInfo frameBufferCreateInfo = {};
	frameBufferCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
	frameBufferCreateInfo.pNext = NULL;
	frameBufferCreateInfo.renderPass = renderPass;
	frameBufferCreateInfo.attachmentCount = 1;
	frameBufferCreateInfo.pAttachments = attachments;
	frameBufferCreateInfo.width = width;
	frameBufferCreateInfo.height = height;
	frameBufferCreateInfo.layers = 1;

	attachments[0] = imageView;
	VkResult res = vkCreateFramebuffer(vkTutDevice, &frameBufferCreateInfo, nullptr, &frameBuffer);
	assert(res == VK_SUCCESS);
}

void setupRenderPass()
{
	VkAttachmentDescription attachments;
	// Color attachment
	attachments.format = VK_FORMAT_R8G8B8A8_UNORM;
	attachments.samples = VK_SAMPLE_COUNT_1_BIT;
	attachments.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	attachments.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	attachments.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	attachments.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	attachments.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
	
	VkAttachmentReference colorReference = {};
	colorReference.attachment = 0;
	colorReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	// Let's have only the color attachment.
	//VkAttachmentReference depthReference = {};
	//depthReference.attachment = 1;
	//depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkSubpassDescription subpassDescription = {};
	subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpassDescription.colorAttachmentCount = 1;
	subpassDescription.pColorAttachments = &colorReference;
	subpassDescription.pDepthStencilAttachment = nullptr;//&depthReference;
	subpassDescription.inputAttachmentCount = 0;
	subpassDescription.pInputAttachments = nullptr;
	subpassDescription.preserveAttachmentCount = 0;
	subpassDescription.pPreserveAttachments = nullptr;
	subpassDescription.pResolveAttachments = nullptr;

	// Subpass dependencies for layout transitions
	VkSubpassDependency dependencies;

	dependencies.srcSubpass = VK_SUBPASS_EXTERNAL;
	dependencies.dstSubpass = 0;
	dependencies.srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
	dependencies.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependencies.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
	dependencies.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	dependencies.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

	VkRenderPassCreateInfo renderPassInfo = {};
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	renderPassInfo.attachmentCount = 1;
	renderPassInfo.pAttachments = &attachments;
	renderPassInfo.subpassCount = 1;
	renderPassInfo.pSubpasses = &subpassDescription;
	renderPassInfo.dependencyCount = 1;
	renderPassInfo.pDependencies = &dependencies;

	VkResult res = vkCreateRenderPass(vkTutDevice, &renderPassInfo, nullptr, &renderPass);
	assert(res == VK_SUCCESS);
}

void render()
{
	VkPipelineStageFlags waitStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

	VkSubmitInfo submitInfo = {};
	submitInfo.sType				= VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.pWaitDstStageMask    = &waitStageMask;									
	submitInfo.pWaitSemaphores      =  nullptr;							
	submitInfo.waitSemaphoreCount   = 0;								
	submitInfo.pSignalSemaphores    = nullptr;						
	submitInfo.signalSemaphoreCount = 0;						
	submitInfo.pCommandBuffers      = &cmdBuffer;					
	submitInfo.commandBufferCount   = 1;							

	vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
	vkQueueWaitIdle(queue);
}

VkCommandBuffer getCommandBuffer(bool begin)
{
	VkCommandBuffer cmdBuffer;

	VkCommandBufferAllocateInfo cmdBufAllocateInfo = {};
	cmdBufAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	cmdBufAllocateInfo.commandPool = cmdPool;
	cmdBufAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cmdBufAllocateInfo.commandBufferCount = 1;

	vkAllocateCommandBuffers(vkTutDevice, &cmdBufAllocateInfo, &cmdBuffer);

	// If requested, also start the new command buffer
	if (begin)
	{
		VkCommandBufferBeginInfo cmdBufInfo = {};
		cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		cmdBufInfo.pNext = nullptr;
		vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo);
	}

	return cmdBuffer;
}

void flushCommandBuffer(VkCommandBuffer commandBuffer)
{
	assert(commandBuffer != VK_NULL_HANDLE);

	vkEndCommandBuffer(commandBuffer);

	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;

	// Create fence to ensure that the command buffer has finished executing
	VkFenceCreateInfo fenceCreateInfo = {};
	fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceCreateInfo.flags = 0;
	VkFence fence;
	vkCreateFence(vkTutDevice, &fenceCreateInfo, nullptr, &fence);

	// Submit to the queue
	vkQueueSubmit(queue, 1, &submitInfo, fence);
	vkQueueWaitIdle(queue);

	vkDestroyFence(vkTutDevice, fence, nullptr);
	vkFreeCommandBuffers(vkTutDevice, cmdPool, 1, &commandBuffer);
}

void prepareVertices(bool useStagingBuffers)
{
	// A note on memory management in Vulkan in general:
	//	This is a very complex topic and while it's fine for an example application to to small individual memory allocations that is not
	//	what should be done a real-world application, where you should allocate large chunkgs of memory at once isntead.

	// Setup vertices
	std::vector<Vertex> vertexBuffer =
	{
		{ { 1.0f,  1.0f, 0.0f },{ 1.0f, 0.0f, 0.0f } },
		{ { -1.0f,  1.0f, 0.0f },{ 0.0f, 1.0f, 0.0f } },
		{ { 0.0f, -1.0f, 0.0f },{ 0.0f, 0.0f, 1.0f } }
	};
	uint32_t vertexBufferSize = static_cast<uint32_t>(vertexBuffer.size()) * sizeof(Vertex);

	// Setup indices
	std::vector<uint32_t> indexBuffer = { 0, 1, 2 };
	indices.count = static_cast<uint32_t>(indexBuffer.size());
	uint32_t indexBufferSize = indices.count * sizeof(uint32_t);

	VkMemoryAllocateInfo memAlloc = {};
	memAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	VkMemoryRequirements memReqs;

	void *data;

	if (useStagingBuffers)
	{
		// Static data like vertex and index buffer should be stored on the device memory 
		// for optimal (and fastest) access by the GPU
		//
		// To achieve this we use so-called "staging buffers" :
		// - Create a buffer that's visible to the host (and can be mapped)
		// - Copy the data to this buffer
		// - Create another buffer that's local on the device (VRAM) with the same size
		// - Copy the data from the host to the device using a command buffer
		// - Delete the host visible (staging) buffer
		// - Use the device local buffers for rendering

		struct StagingBuffer {
			VkDeviceMemory memory;
			VkBuffer buffer;
		};

		struct {
			StagingBuffer vertices;
			StagingBuffer indices;
		} stagingBuffers;

		// Vertex buffer
		VkBufferCreateInfo vertexBufferInfo = {};
		vertexBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		vertexBufferInfo.size = vertexBufferSize;
		// Buffer is used as the copy source
		vertexBufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		// Create a host-visible buffer to copy the vertex data to (staging buffer)
		vkCreateBuffer(vkTutDevice, &vertexBufferInfo, nullptr, &stagingBuffers.vertices.buffer);
		vkGetBufferMemoryRequirements(vkTutDevice, stagingBuffers.vertices.buffer, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		// Request a host visible memory type that can be used to copy our data do
		// Also request it to be coherent, so that writes are visible to the GPU right after unmapping the buffer
		memAlloc.memoryTypeIndex = getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		vkAllocateMemory(vkTutDevice, &memAlloc, nullptr, &stagingBuffers.vertices.memory);
		// Map and copy
		vkMapMemory(vkTutDevice, stagingBuffers.vertices.memory, 0, memAlloc.allocationSize, 0, &data);
		memcpy(data, vertexBuffer.data(), vertexBufferSize);
		vkUnmapMemory(vkTutDevice, stagingBuffers.vertices.memory);
		vkBindBufferMemory(vkTutDevice, stagingBuffers.vertices.buffer, stagingBuffers.vertices.memory, 0);

		// Create a device local buffer to which the (host local) vertex data will be copied and which will be used for rendering
		vertexBufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		vkCreateBuffer(vkTutDevice, &vertexBufferInfo, nullptr, &vertices.buffer);
		vkGetBufferMemoryRequirements(vkTutDevice, vertices.buffer, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		vkAllocateMemory(vkTutDevice, &memAlloc, nullptr, &vertices.memory);
		vkBindBufferMemory(vkTutDevice, vertices.buffer, vertices.memory, 0);

		// Index buffer
		VkBufferCreateInfo indexbufferInfo = {};
		indexbufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		indexbufferInfo.size = indexBufferSize;
		indexbufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		// Copy index data to a buffer visible to the host (staging buffer)
		vkCreateBuffer(vkTutDevice, &indexbufferInfo, nullptr, &stagingBuffers.indices.buffer);
		vkGetBufferMemoryRequirements(vkTutDevice, stagingBuffers.indices.buffer, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		vkAllocateMemory(vkTutDevice, &memAlloc, nullptr, &stagingBuffers.indices.memory);
		vkMapMemory(vkTutDevice, stagingBuffers.indices.memory, 0, indexBufferSize, 0, &data);
		memcpy(data, indexBuffer.data(), indexBufferSize);
		vkUnmapMemory(vkTutDevice, stagingBuffers.indices.memory);
		vkBindBufferMemory(vkTutDevice, stagingBuffers.indices.buffer, stagingBuffers.indices.memory, 0);

		// Create destination buffer with device only visibility
		indexbufferInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		vkCreateBuffer(vkTutDevice, &indexbufferInfo, nullptr, &indices.buffer);
		vkGetBufferMemoryRequirements(vkTutDevice, indices.buffer, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		vkAllocateMemory(vkTutDevice, &memAlloc, nullptr, &indices.memory);
		vkBindBufferMemory(vkTutDevice, indices.buffer, indices.memory, 0);

		VkCommandBufferBeginInfo cmdBufferBeginInfo = {};
		cmdBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		cmdBufferBeginInfo.pNext = nullptr;

		// Buffer copies have to be submitted to a queue, so we need a command buffer for them
		// Note: Some devices offer a dedicated transfer queue (with only the transfer bit set) that may be faster when doing lots of copies
		VkCommandBuffer copyCmd = getCommandBuffer(true);

		// Put buffer region copies into command buffer
		VkBufferCopy copyRegion = {};

		// Vertex buffer
		copyRegion.size = vertexBufferSize;
		vkCmdCopyBuffer(copyCmd, stagingBuffers.vertices.buffer, vertices.buffer, 1, &copyRegion);
		// Index buffer
		copyRegion.size = indexBufferSize;
		vkCmdCopyBuffer(copyCmd, stagingBuffers.indices.buffer, indices.buffer, 1, &copyRegion);

		// Flushing the command buffer will also submit it to the queue and uses a fence to ensure that all commands have been executed before returning
		flushCommandBuffer(copyCmd);

		// Destroy staging buffers
		// Note: Staging buffer must not be deleted before the copies have been submitted and executed
		vkDestroyBuffer(vkTutDevice, stagingBuffers.vertices.buffer, nullptr);
		vkFreeMemory(vkTutDevice, stagingBuffers.vertices.memory, nullptr);
		vkDestroyBuffer(vkTutDevice, stagingBuffers.indices.buffer, nullptr);
		vkFreeMemory(vkTutDevice, stagingBuffers.indices.memory, nullptr);
	}
	else
	{
		// Don't use staging
		// Create host-visible buffers only and use these for rendering. This is not advised and will usually result in lower rendering performance

		// Vertex buffer
		VkBufferCreateInfo vertexBufferInfo = {};
		vertexBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		vertexBufferInfo.size = vertexBufferSize;
		vertexBufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;

		// Copy vertex data to a buffer visible to the host
		vkCreateBuffer(vkTutDevice, &vertexBufferInfo, nullptr, &vertices.buffer);
		vkGetBufferMemoryRequirements(vkTutDevice, vertices.buffer, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		// VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT is host visible memory, and VK_MEMORY_PROPERTY_HOST_COHERENT_BIT makes sure writes are directly visible
		memAlloc.memoryTypeIndex = getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		vkAllocateMemory(vkTutDevice, &memAlloc, nullptr, &vertices.memory);
		vkMapMemory(vkTutDevice, vertices.memory, 0, memAlloc.allocationSize, 0, &data);
		memcpy(data, vertexBuffer.data(), vertexBufferSize);
		vkUnmapMemory(vkTutDevice, vertices.memory);
		vkBindBufferMemory(vkTutDevice, vertices.buffer, vertices.memory, 0);

		// Index buffer
		VkBufferCreateInfo indexbufferInfo = {};
		indexbufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		indexbufferInfo.size = indexBufferSize;
		indexbufferInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;

		// Copy index data to a buffer visible to the host
		vkCreateBuffer(vkTutDevice, &indexbufferInfo, nullptr, &indices.buffer);
		vkGetBufferMemoryRequirements(vkTutDevice, indices.buffer, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		vkAllocateMemory(vkTutDevice, &memAlloc, nullptr, &indices.memory);
		vkMapMemory(vkTutDevice, indices.memory, 0, indexBufferSize, 0, &data);
		memcpy(data, indexBuffer.data(), indexBufferSize);
		vkUnmapMemory(vkTutDevice, indices.memory);
		vkBindBufferMemory(vkTutDevice, indices.buffer, indices.memory, 0);
	}
}

void buildCommandBuffers()
{
	VkCommandBufferBeginInfo cmdBufInfo = {};
	cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	cmdBufInfo.pNext = nullptr;

	// Set clear values for all framebuffer attachments with loadOp set to clear
	// We use two attachments (color and depth) that are cleared at the start of the subpass and as such we need to set clear values for both
	VkClearValue clearValues[1];
	clearValues[0].color = { { 0.0f, 0.0f, 0.2f, 1.0f } };

	VkRenderPassBeginInfo renderPassBeginInfo = {};
	renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	renderPassBeginInfo.pNext = nullptr;
	renderPassBeginInfo.renderPass = renderPass;
	renderPassBeginInfo.renderArea.offset.x = 0;
	renderPassBeginInfo.renderArea.offset.y = 0;
	renderPassBeginInfo.renderArea.extent.width = width;
	renderPassBeginInfo.renderArea.extent.height = height;
	renderPassBeginInfo.clearValueCount = 1;
	renderPassBeginInfo.pClearValues = clearValues;

	// Set target frame buffer
	renderPassBeginInfo.framebuffer = frameBuffer;

	vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo);

	// Start the first sub pass specified in our default render pass setup by the base class
	// This will clear the color and depth attachment
	vkCmdBeginRenderPass(cmdBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

	// Update dynamic viewport state
	VkViewport viewport = {};
	viewport.height = (float)height;
	viewport.width = (float)width;
	viewport.minDepth = (float) 0.0f;
	viewport.maxDepth = (float) 1.0f;
	vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

	// Update dynamic scissor state
	VkRect2D scissor = {};
	scissor.extent.width = width;
	scissor.extent.height = height;
	scissor.offset.x = 0;
	scissor.offset.y = 0;
	vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

	// Bind descriptor sets describing shader binding points
	vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

	// Bind the rendering pipeline
	// The pipeline (state object) contains all states of the rendering pipeline, binding it will set all the states specified at pipeline creation time
	vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

	// Bind triangle vertex buffer (contains position and colors)
	VkDeviceSize offsets[1] = { 0 };
	vkCmdBindVertexBuffers(cmdBuffer, 0, 1, &vertices.buffer, offsets);

	// Bind triangle index buffer
	vkCmdBindIndexBuffer(cmdBuffer, indices.buffer, 0, VK_INDEX_TYPE_UINT32);

	// Draw indexed triangle
	vkCmdDrawIndexed(cmdBuffer, indices.count, 1, 0, 0, 1);

	vkCmdEndRenderPass(cmdBuffer);

	vkEndCommandBuffer(cmdBuffer);
}

void readPixels()
{
	void* data = nullptr;

	auto res = vkMapMemory(vkTutDevice, imageMemory, 0, 1920 * 1080, 0, &data);
	assert(res == VK_SUCCESS);

	FILE* fp = nullptr;
	fopen_s(&fp, "image.raw", "wb");
	assert(fp);

	fwrite(((unsigned char*)data), 1, 1920 * 1080 * 4, fp);
}

int main(int argc, char** argv)
{
	auto ret = createInstance();
	assert(ret);

	ret = createDevice();
	assert(ret);

	ret = createBuffer(srcBuffer, srcMemory, 1024 * 1024);
	assert(ret);
	ret = createBuffer(dstBuffer, dstMemory, 1024 * 1024);
	assert(ret);

	ret = createImage();
	assert(ret);

	ret = createImageView();
	assert(ret);

	//ret = mapBuffer(srcMemory, 1);
	//ret = mapBuffer(dstMemory, 10);
	//assert(ret);

	ret = createCommandPool();
	assert(ret);

	ret = createCommandBuffer();
	assert(ret);

	ret = createFence();
	assert(ret);

	//ret = dumpBuffer(srcMemory, 10);
	//ret = dumpBuffer(dstMemory, 10);
	//printf("\nCopying from buffer:[%p] to buffer: [%p]\n", srcMemory, dstMemory);
	//ret = copyBuffers(srcBuffer, dstBuffer);
	//assert(ret);

	//ret = dumpBuffer(dstMemory, 10);
	//assert(ret);

	//ret = createDirectSurface();
	//assert(ret);

	//ret = createSwapchain();
	//assert(ret);

	ret = createShader(VK_SHADER_STAGE_VERTEX_BIT, (sizeof(vss) / sizeof(vss[0])), vss, vsm);
	assert(ret);

	ret = createShader(VK_SHADER_STAGE_FRAGMENT_BIT, (sizeof(fss) / sizeof(fss[0])), fss, fsm);
	assert(ret);

	ret = createDescriptorSetLayout();
	assert(ret);

	ret = createDescriptorPool();
	assert(ret);

	ret = allocateDescriptor();
	assert(ret);

	prepareVertices(true);
	/*ret = */setupRenderPass();
	setupFrameBuffer();
	buildCommandBuffers();

	render();

	readPixels();

	//ret = cleanup();
	//assert(ret);

	return 0;
}
