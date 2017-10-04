#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string.h>
#include <cassert>
#include <vulkan/vulkan.h>

// GLFW
#define GLFW_INCLUDE_VULKAN
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

// Static declarations 
static VkInstance instance;
static VkPhysicalDevice physDevice;
static VkDevice device;
static VkPhysicalDeviceMemoryProperties deviceMemoryProperties;
static VkQueue queue;
static VkSurfaceKHR surface;
static VkSwapchainKHR swapchain;
static VkFormat colorFormat;
static VkFormat depthFormat;
static VkColorSpaceKHR colorSpace;
static std::vector<VkImage> images;
static VkCommandPool cmdPool;
static VkCommandBuffer setupCmdBuffer = VK_NULL_HANDLE;
static std::vector<VkCommandBuffer> drawCmdBuffers;
static std::vector<VkCommandBuffer> prePresentCmdBuffers = { VK_NULL_HANDLE };
static std::vector<VkCommandBuffer> postPresentCmdBuffers = { VK_NULL_HANDLE };
static std::vector<VkFramebuffer>frameBuffers;
static VkRenderPass renderPass;
static VkPipelineCache pipelineCache;
static VkPipeline pipeline;
static uint32_t currentBuffer = 0;
static uint32_t queueNodeIndex = UINT32_MAX;
static uint32_t imageCount;
static PFN_vkGetPhysicalDeviceSurfaceSupportKHR fpGetPhysicalDeviceSurfaceSupportKHR;
static PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR fpGetPhysicalDeviceSurfaceCapabilitiesKHR;
static PFN_vkGetPhysicalDeviceSurfaceFormatsKHR fpGetPhysicalDeviceSurfaceFormatsKHR;
static PFN_vkGetPhysicalDeviceSurfacePresentModesKHR fpGetPhysicalDeviceSurfacePresentModesKHR;
static PFN_vkCreateSwapchainKHR fpCreateSwapchainKHR;
static PFN_vkDestroySwapchainKHR fpDestroySwapchainKHR;
static PFN_vkGetSwapchainImagesKHR fpGetSwapchainImagesKHR;
static PFN_vkAcquireNextImageKHR fpAcquireNextImageKHR;
static PFN_vkQueuePresentKHR fpQueuePresentKHR;
GLFWwindow*         window;
static uint32_t width = 1920;
static uint32_t height = 1080;
static VkDescriptorSetLayout descriptorSetLayout;
static VkDescriptorSet descriptorSet;
static VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
static VkDescriptorBufferInfo uniformDescriptor;
static VkBuffer uniformBuffer;
static VkDeviceMemory uniformBufferMemory;
static VkPipelineLayout pipelineLayout;


typedef struct _SwapChainBuffers {
	VkImage image;
	VkImageView view;
} SwapChainBuffer;

struct {
	VkImage image;
	VkDeviceMemory mem;
	VkImageView view;
} depthStencil;

struct {
	VkSemaphore presentComplete;
	VkSemaphore renderComplete;
} semaphores;

struct {
	VkBuffer buf;
	VkDeviceMemory mem;
	VkPipelineVertexInputStateCreateInfo inputState;
	std::vector<VkVertexInputBindingDescription> bindingDescriptions;
	std::vector<VkVertexInputAttributeDescription> attributeDescriptions;
} vertices;

struct {
	int count;
	VkBuffer buf;
	VkDeviceMemory mem;
} indices;

static std::vector<SwapChainBuffer> buffers;

// Macro to get a procedure address based on a vulkan instance
#define GET_INSTANCE_PROC_ADDR(inst, entrypoint)                        \
{                                                                       \
    fp##entrypoint = (PFN_vk##entrypoint) vkGetInstanceProcAddr(inst, "vk"#entrypoint); \
    if (fp##entrypoint == NULL)                                         \
    {                                                                   \
        exit(1);                                                        \
    }                                                                   \
}

// Macro to get a procedure address based on a vulkan device
#define GET_DEVICE_PROC_ADDR(dev, entrypoint)                           \
{                                                                       \
    fp##entrypoint = (PFN_vk##entrypoint) vkGetDeviceProcAddr(dev, "vk"#entrypoint);   \
    if (fp##entrypoint == NULL)                                         \
    {                                                                   \
        exit(1);                                                        \
    }                                                                   \
}

#define VK_CHECK_RESULT(errvar) \
{ \
    if (errvar != VK_SUCCESS) \
        return errvar; \
}

VkBool32 getSupportedDepthFormat(VkPhysicalDevice physicalDevice, VkFormat *depthFormat)
{
	// Since all depth formats may be optional, we need to find a suitable depth format to use
	// Start with the highest precision packed format
	std::vector<VkFormat> depthFormats = {
		VK_FORMAT_D32_SFLOAT_S8_UINT,
		VK_FORMAT_D32_SFLOAT,
		VK_FORMAT_D24_UNORM_S8_UINT,
		VK_FORMAT_D16_UNORM_S8_UINT,
		VK_FORMAT_D16_UNORM
	};

	for (auto& format : depthFormats)
	{
		VkFormatProperties formatProps;
		vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &formatProps);
		// Format must support depth stencil attachment for optimal tiling
		if (formatProps.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
		{
			*depthFormat = format;
			return true;
		}
	}

	return false;
}

// Helper method to allocate a command buffer from the command pool
// A bool can be set to start recording commands in the command buffer.
VkCommandBuffer getCommandBuffer(bool begin)
{
	VkCommandBuffer cmdBuffer;

	VkCommandBufferAllocateInfo cmdBufAllocateInfo = {};
	cmdBufAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	cmdBufAllocateInfo.commandPool = cmdPool;
	cmdBufAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cmdBufAllocateInfo.commandBufferCount = 1;

	if (vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, &cmdBuffer) != VK_SUCCESS) {
		return VK_NULL_HANDLE;
	}

	// If requested, also start the new command buffer
	if (begin)
	{
		VkCommandBufferBeginInfo cmdBufInfo = {};
		cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		if (vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo) != VK_SUCCESS) {
			return VK_NULL_HANDLE;
		}
	}

	return cmdBuffer;
}

// Helper method to submit the specified command buffer and wait on the host.
VkResult flushCommandBuffer(VkCommandBuffer commandBuffer)
{
	VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffer));

	VkSubmitInfo submitInfo = {};
	submitInfo.sType			  = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers	  = &commandBuffer;

	vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
	
	// Tutnote: You can have a semaphore here. Since this gets used only at init() time,
	// even the CPU wait should be OK.
	vkQueueWaitIdle(queue);

	vkFreeCommandBuffers(device, cmdPool, 1, &commandBuffer);
	return VK_SUCCESS;
}

static VkResult CreateShader(VkShaderStageFlagBits shaderStage, size_t size, const char *code, VkShaderModule &shaderModule)
{
	VkResult result;

	VkShaderModuleCreateInfo shadderModuleInfo = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
	shadderModuleInfo.codeSize		= size;
	shadderModuleInfo.pCode			= (const uint32_t *)code;
	result = vkCreateShaderModule(device, &shadderModuleInfo, 0, &shaderModule);

	if (result != VK_SUCCESS) {
		return result;
	}

	return VK_SUCCESS;
}

static void DestroyShader(VkShaderModule shaderModule)
{
	vkDestroyShaderModule(device, shaderModule, NULL);
}

// From the list of memory types exposed by the device, pick the 
// right one. There could be different heaps providing the ability
// to allocate the same type. Just pick the first one.
uint32_t getMemoryType(uint32_t typeBits, VkFlags properties)
{
	for (uint32_t i = 0; i < 32; i++)
	{
		if ((typeBits & 1) == 1)
		{
			if ((deviceMemoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
			{
				return i;
			}
		}
		typeBits >>= 1;
	}

	// todo : throw error
	return 0;
}

// Create a Win32 Vulkan surface
VkResult createWin32Surface()
{
	VkWin32SurfaceCreateInfoKHR surfaceCreateInfo = {};
	surfaceCreateInfo.sType			= VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
	surfaceCreateInfo.hinstance		= GetModuleHandle(nullptr);
	surfaceCreateInfo.hwnd			= glfwGetWin32Window(window);
	VkResult err = vkCreateWin32SurfaceKHR(instance, &surfaceCreateInfo, nullptr, &surface);

	return err;
}

VkResult createSwapchain()
{
	VkResult err;

	// Get physical device surface properties and formats
	VkSurfaceCapabilitiesKHR surfCaps;
	err = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physDevice, surface, &surfCaps);

	// Get available present modes
	uint32_t presentModeCount;
	err = vkGetPhysicalDeviceSurfacePresentModesKHR(physDevice, surface, &presentModeCount, NULL);
	std::vector<VkPresentModeKHR> presentModes(presentModeCount);
	err = vkGetPhysicalDeviceSurfacePresentModesKHR(physDevice, surface, &presentModeCount, presentModes.data());

	VkExtent2D swapchainExtent = {};
	// width and height are either both -1, or both not -1.
	if (surfCaps.currentExtent.width == -1)
	{
		// If the surface size is undefined, the size is set to
		// the size of the images requested.
		swapchainExtent.width = width;
		swapchainExtent.height = height;
	}
	else
	{
		// If the surface size is defined, the swap chain size must match
		swapchainExtent = surfCaps.currentExtent;
		width			= surfCaps.currentExtent.width;
		height			= surfCaps.currentExtent.height;
	}

	VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;

	for (size_t i = 0; i < presentModeCount; i++)
	{
		if (presentModes[i] == VK_PRESENT_MODE_MAILBOX_KHR)
		{
			presentMode = VK_PRESENT_MODE_MAILBOX_KHR;
			break;
		}
		if ((presentMode != VK_PRESENT_MODE_MAILBOX_KHR) && (presentModes[i] == VK_PRESENT_MODE_IMMEDIATE_KHR))
		{
			presentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
		}
	}

	uint32_t desiredNumberOfSwapchainImages = surfCaps.minImageCount + 1;
	if ((surfCaps.maxImageCount > 0) && (desiredNumberOfSwapchainImages > surfCaps.maxImageCount))
	{
		desiredNumberOfSwapchainImages = surfCaps.maxImageCount;
	}

	VkSurfaceTransformFlagsKHR preTransform;
	if (surfCaps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
	{
		preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
	}
	else
	{
		preTransform = surfCaps.currentTransform;
	}

	VkSwapchainCreateInfoKHR scinfo;
	scinfo.sType				 = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	scinfo.pNext				 = NULL;
	scinfo.surface				 = surface;
	scinfo.minImageCount		 = desiredNumberOfSwapchainImages;
	scinfo.imageFormat			 = colorFormat;
	scinfo.imageColorSpace		 = colorSpace;
	scinfo.imageExtent			 = { swapchainExtent.width, swapchainExtent.height };
	scinfo.imageUsage			 = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
	scinfo.preTransform			 = (VkSurfaceTransformFlagBitsKHR)preTransform;
	scinfo.imageArrayLayers		 = 1;
	scinfo.imageSharingMode      = VK_SHARING_MODE_EXCLUSIVE;
	scinfo.queueFamilyIndexCount = 0;
	scinfo.pQueueFamilyIndices	 = NULL;
	scinfo.presentMode		     = presentMode;
	scinfo.clipped				 = true;
	scinfo.compositeAlpha		 = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	scinfo.oldSwapchain		   	 = VK_NULL_HANDLE;
	vkCreateSwapchainKHR(device, &scinfo, NULL, &swapchain);

	err = vkGetSwapchainImagesKHR(device, swapchain, &imageCount, NULL);

	// Get the swap chain images
	images.resize(imageCount);
	err = vkGetSwapchainImagesKHR(device, swapchain, &imageCount, images.data());

	buffers.resize(imageCount);
	for (uint32_t i = 0; i < imageCount; i++)
	{
		VkImageViewCreateInfo colorAttachmentView = {};
		colorAttachmentView.sType							= VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		colorAttachmentView.pNext							= NULL;
		colorAttachmentView.format							= colorFormat;
		colorAttachmentView.components					    = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, 
															   VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A };
		colorAttachmentView.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
		colorAttachmentView.subresourceRange.baseMipLevel   = 0;
		colorAttachmentView.subresourceRange.levelCount     = 1;
		colorAttachmentView.subresourceRange.baseArrayLayer = 0;
		colorAttachmentView.subresourceRange.layerCount		= 1;
		colorAttachmentView.viewType						= VK_IMAGE_VIEW_TYPE_2D;
		colorAttachmentView.flags							= 0;

		buffers[i].image								    = images[i];
		colorAttachmentView.image						    = buffers[i].image;

		err = vkCreateImageView(device, &colorAttachmentView, NULL, &buffers[i].view);
		assert(err == VK_SUCCESS);
	}

	return err;
}

VkResult createCommandPool()
{
	VkResult err;
	VkCommandPoolCreateInfo cmdPoolInfo = {};
	cmdPoolInfo.sType				= VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	cmdPoolInfo.queueFamilyIndex	= queueNodeIndex;
	cmdPoolInfo.flags				= VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	err = vkCreateCommandPool(device, &cmdPoolInfo, NULL, &cmdPool);

	return err;
}

VkResult createSetupCommandBuffer()
{
	VkResult err;
	if (setupCmdBuffer != VK_NULL_HANDLE)
	{
		vkFreeCommandBuffers(device, cmdPool, 1, &setupCmdBuffer);
		setupCmdBuffer = VK_NULL_HANDLE;
	}

	VkCommandBufferAllocateInfo cmdBufAllocateInfo;// =
	cmdBufAllocateInfo.sType		= VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	cmdBufAllocateInfo.pNext	    = NULL;
	cmdBufAllocateInfo.commandPool	= cmdPool;
	cmdBufAllocateInfo.level		= VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cmdBufAllocateInfo.commandBufferCount = 1;
	vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, &setupCmdBuffer);

	VkCommandBufferBeginInfo cmdBufInfo = {};
	cmdBufInfo.sType				= VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	err = vkBeginCommandBuffer(setupCmdBuffer, &cmdBufInfo);

	return err;
}

// We will be creating one Command buffer per FrameBuffer Image.
// They all will be allocated from the same pool.
// We would need extra command buffers for issuing layout transitioning
// operations.
VkResult createCommandBuffers()
{
	VkResult err;

	drawCmdBuffers.resize(imageCount);
	prePresentCmdBuffers.resize(imageCount);
	postPresentCmdBuffers.resize(imageCount);

	VkCommandBufferAllocateInfo cmdBufAllocateInfo;
	cmdBufAllocateInfo.sType			  = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	cmdBufAllocateInfo.pNext			  = NULL;
	cmdBufAllocateInfo.commandPool		  = cmdPool;
	cmdBufAllocateInfo.level		      = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cmdBufAllocateInfo.commandBufferCount = (uint32_t)drawCmdBuffers.size();
	err = vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, drawCmdBuffers.data());
	
	if (err != VK_SUCCESS) {
		return err;
	}

	// Tutnote:
	// Command buffers for submitting present barriers
	// One pre and post present buffer per swap chain image
	err = vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, prePresentCmdBuffers.data());
	if (err != VK_SUCCESS) {
		return err;
	}
	err = vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, postPresentCmdBuffers.data());
	return err;
}

VkResult buildPresentCommandBuffers()
{
	VkResult err;
	VkCommandBufferBeginInfo cmdBufInfo = {};
	cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

	for (uint32_t i = 0; i < imageCount; i++)
	{
		// TutNote: After a present was done by using an Image, the image layout again 
		// needs to be set to VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL so that the 
		// RenderPass can write into it. 
		// We ignore what contents were there earlier. We just transition to the new layout
		// with an Image memory barrier.
		err = vkBeginCommandBuffer(postPresentCmdBuffers[i], &cmdBufInfo);
		if (err != VK_SUCCESS) {
			return err;
		}

		VkImageMemoryBarrier postPresentBarrier;
		postPresentBarrier.sType			   = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		postPresentBarrier.srcAccessMask	   = 0;
		postPresentBarrier.dstAccessMask       = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		postPresentBarrier.oldLayout	       = VK_IMAGE_LAYOUT_UNDEFINED;
		postPresentBarrier.newLayout		   = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		postPresentBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		postPresentBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		postPresentBarrier.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		postPresentBarrier.image			   = buffers[i].image;

		vkCmdPipelineBarrier(
			postPresentCmdBuffers[i],
			VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
			VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
			0,
			0, NULL,
			0, NULL,
			1, &postPresentBarrier);

		err = vkEndCommandBuffer(postPresentCmdBuffers[i]);
		if (err != VK_SUCCESS) {
			return err;
		}

		// Tutnote: This is needed before presentation can be done.
		// Transforms the (framebuffer) image layout from color attachment to present(khr) for presenting to the swap chain
		err = vkBeginCommandBuffer(prePresentCmdBuffers[i], &cmdBufInfo);
		if (err != VK_SUCCESS) {
			return err;
		}

		VkImageMemoryBarrier prePresentBarrier;
		prePresentBarrier.sType				   = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		prePresentBarrier.srcAccessMask		   = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		prePresentBarrier.dstAccessMask		   = VK_ACCESS_MEMORY_READ_BIT;
		prePresentBarrier.oldLayout			   = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		prePresentBarrier.newLayout			   = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
		prePresentBarrier.srcQueueFamilyIndex  = VK_QUEUE_FAMILY_IGNORED;
		prePresentBarrier.dstQueueFamilyIndex  = VK_QUEUE_FAMILY_IGNORED;
		prePresentBarrier.subresourceRange	   = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		prePresentBarrier.image				   = buffers[i].image;

		vkCmdPipelineBarrier(
			prePresentCmdBuffers[i],
			VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
			VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
			0,
			0, NULL,
			0, NULL,
			1, &prePresentBarrier);

		err = vkEndCommandBuffer(prePresentCmdBuffers[i]);
		if (err != VK_SUCCESS) {
			return err;
		}
	}
	return err;
}

VkResult setupDepthStencil()
{
	VkResult err;

	if (!getSupportedDepthFormat(physDevice, &depthFormat)) {
		std::cout << "couldn't get depthFormat\n";
	}

	VkImageCreateInfo image = {};
	image.sType										 = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	image.pNext										 = NULL;
	image.imageType									 = VK_IMAGE_TYPE_2D;
	image.format									 = depthFormat;
	image.extent									 = { width, height, 1 };
	image.mipLevels									 = 1;
	image.arrayLayers								 = 1;
	image.samples									 = VK_SAMPLE_COUNT_1_BIT;
	image.tiling									 = VK_IMAGE_TILING_OPTIMAL;
	image.usage										 = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
	image.flags										 = 0;
	image.initialLayout								 = VK_IMAGE_LAYOUT_UNDEFINED; // Must be UNDEFINED OR PREINITIALIZED

	VkMemoryAllocateInfo memAlloc = {};
	memAlloc.sType									 = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	memAlloc.pNext									 = NULL;
	memAlloc.allocationSize							 = 0;
	memAlloc.memoryTypeIndex						 = 0;

	VkImageViewCreateInfo depthStencilView = {};
	depthStencilView.sType							 = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	depthStencilView.pNext							 = NULL;
	depthStencilView.viewType						 = VK_IMAGE_VIEW_TYPE_2D;
	depthStencilView.format							 = depthFormat;
	depthStencilView.flags							 = 0;
	depthStencilView.subresourceRange				 = {};
	depthStencilView.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
	depthStencilView.subresourceRange.baseMipLevel   = 0;
	depthStencilView.subresourceRange.levelCount     = 1;
	depthStencilView.subresourceRange.baseArrayLayer = 0;
	depthStencilView.subresourceRange.layerCount	 = 1;

	VkMemoryRequirements memReqs;

	err = vkCreateImage(device, &image, NULL, &depthStencil.image);
	if (err != VK_SUCCESS) {
		return err;
	}

	// Allocate memory for the image.
	vkGetImageMemoryRequirements(device, depthStencil.image, &memReqs);
	memAlloc.allocationSize							 = memReqs.size;
	memAlloc.memoryTypeIndex						 = getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	
	err = vkAllocateMemory(device, &memAlloc, NULL, &depthStencil.mem);
	if (err != VK_SUCCESS) {
		return err;
	}

	// Bind image to memory.
	err = vkBindImageMemory(device, depthStencil.image, depthStencil.mem, 0);
	if (err != VK_SUCCESS) {
		return err;
	}

	// Tutnote: We issue an Image memory barrier here to convert the layout from
	// VK_IMAGE_LAYOUT_UNDEFINED -> VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
	// so that it can be used in the renderpass.
	VkImageMemoryBarrier imageMemoryBarrier;
	imageMemoryBarrier.sType						= VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	imageMemoryBarrier.srcAccessMask				= 0;
	imageMemoryBarrier.dstAccessMask				= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	imageMemoryBarrier.oldLayout					= VK_IMAGE_LAYOUT_UNDEFINED;
	imageMemoryBarrier.newLayout					= VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	imageMemoryBarrier.subresourceRange				= { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
	imageMemoryBarrier.image						= depthStencil.image;

	// Tutnote: TODO xxx
	vkCmdPipelineBarrier(
		setupCmdBuffer,
		VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
		VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
		0,
		0, NULL,
		0, NULL,
		1, &imageMemoryBarrier);

	depthStencilView.image							= depthStencil.image;
	err = vkCreateImageView(device, &depthStencilView, NULL, &depthStencil.view);

	return err;
}

VkResult setupRenderPass()
{
	VkResult err;
	VkAttachmentDescription attachments[2] = {};

	// Color attachment
	attachments[0].format		    = colorFormat;
	attachments[0].samples		    = VK_SAMPLE_COUNT_1_BIT;
	attachments[0].loadOp		    = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[0].storeOp		    = VK_ATTACHMENT_STORE_OP_STORE;
	attachments[0].stencilLoadOp    = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	attachments[0].stencilStoreOp   = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	attachments[0].initialLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	attachments[0].finalLayout	    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; 

	// Depth attachment
	attachments[1].format		    = depthFormat;
	attachments[1].samples	        = VK_SAMPLE_COUNT_1_BIT;
	attachments[1].loadOp		    = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[1].storeOp		    = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	attachments[1].stencilLoadOp    = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	attachments[1].stencilStoreOp   = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	attachments[1].initialLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	attachments[1].finalLayout      = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkAttachmentReference colorReference = {};
	colorReference.attachment	    = 0;
	colorReference.layout		    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkAttachmentReference depthReference = {};
	depthReference.attachment	    = 1;
	depthReference.layout		    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
 
	VkSubpassDescription subpass    = {};
	subpass.pipelineBindPoint	    = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.flags				    = 0;
	subpass.inputAttachmentCount    = 0;
	subpass.pInputAttachments	    = NULL;
	subpass.colorAttachmentCount    = 1;
	subpass.pColorAttachments       = &colorReference;
	subpass.pResolveAttachments     = NULL;
	subpass.pDepthStencilAttachment = &depthReference;
	subpass.preserveAttachmentCount = 0;
	subpass.pPreserveAttachments    = NULL;

	VkRenderPassCreateInfo renderPassInfo = {};
	renderPassInfo.sType			= VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	renderPassInfo.pNext			= NULL;
	renderPassInfo.attachmentCount  = 2;
	renderPassInfo.pAttachments     = attachments;
	renderPassInfo.subpassCount		= 1;
	renderPassInfo.pSubpasses		= &subpass;
	renderPassInfo.dependencyCount	= 0;
	renderPassInfo.pDependencies	= NULL;

	err = vkCreateRenderPass(device, &renderPassInfo, NULL, &renderPass);
	return err;
}

VkResult createPipelineCache()
{
	VkResult err;
	VkPipelineCacheCreateInfo pipelineCacheCreateInfo = {};
	pipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
	err = vkCreatePipelineCache(device, &pipelineCacheCreateInfo, NULL, &pipelineCache);
	return err;
}

VkResult setupFrameBuffer()
{
	VkResult err;
	VkImageView attachments[2];

	// Tutnote: We have 3 swapchain images, and each of them act as a color attachment
	// point for each of the 3 Framebuffers. However, the depth/stencil attachment
	// is shared across all of them. At any given time there's only one sub-pass active
	// in this example. If you have multiple threads trying to write into multiple
	// Framebuffers, then you would need multiple depth attachment points.
	// However, for color images, one would be written in a sub-pass and one would be presented.
	attachments[1]						= depthStencil.view;

	// Create frame buffers for every swap chain image
	frameBuffers.resize(imageCount);

	VkFramebufferCreateInfo frameBufferCreateInfo = {};
	frameBufferCreateInfo.sType			  = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
	frameBufferCreateInfo.pNext			  = NULL;
	frameBufferCreateInfo.renderPass	  = renderPass;
	frameBufferCreateInfo.attachmentCount = 2;
	frameBufferCreateInfo.pAttachments	  = attachments;
	frameBufferCreateInfo.width			  = width;
	frameBufferCreateInfo.height		  = height;
	frameBufferCreateInfo.layers		  = 1;

	for (uint32_t i = 0; i < frameBuffers.size(); i++)
	{
		attachments[0] = buffers[i].view;
		err = vkCreateFramebuffer(device, &frameBufferCreateInfo, NULL, &frameBuffers[i]);
		if (err != VK_SUCCESS) {
			return err;
		}
	}

	return err;
}

VkResult flushSetupCommandBuffer()
{
	VkResult err;
	if (setupCmdBuffer == VK_NULL_HANDLE)
		return VK_SUCCESS;

	err = vkEndCommandBuffer(setupCmdBuffer);
	if (err != VK_SUCCESS) {
		return err;
	}

	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &setupCmdBuffer;

	err = vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
	if (err != VK_SUCCESS) {
		return err;
	}
	err = vkQueueWaitIdle(queue);
	if (err != VK_SUCCESS) {
		return err;
	}

	vkFreeCommandBuffers(device, cmdPool, 1, &setupCmdBuffer);
	setupCmdBuffer = VK_NULL_HANDLE;
	return err;
}

VkResult prepareSemaphore()
{
	VkResult err;
	VkSemaphoreCreateInfo semaphoreCreateInfo = {};
	semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
	semaphoreCreateInfo.pNext = NULL;

	// This semaphore ensures that the image is complete
	// before starting to submit again
	err = vkCreateSemaphore(device, &semaphoreCreateInfo, NULL, &semaphores.presentComplete);
	if (err != VK_SUCCESS) {
		return err;
	}

	// This semaphore ensures that all commands submitted
	// have been finished before submitting the image to the queue
	err = vkCreateSemaphore(device, &semaphoreCreateInfo, NULL, &semaphores.renderComplete);
	return err;
}

VkResult mapAndCopyBuffer(const VkDeviceMemory &bufferMemory, void* src, int sz)
{
	void* data = nullptr;

	auto res = vkMapMemory(device, bufferMemory, 0, 4, 0, &data);
	assert(res == VK_SUCCESS);

	memcpy(data, src, sz);

	vkUnmapMemory(device, uniformBufferMemory);

	return res;
}

// We will use staging buffers to copy buffers.
VkResult prepareVertices()
{
	struct Vertex {
		float pos[3];
		float col[3];
	};

	// Setup vertices
	std::vector<Vertex> vertexBuffer = {
		{ { 1.0f,  1.0f, 0.0f },  { 1.0f, 0.0f, 0.0f } },
		{ { -1.0f,  1.0f, 0.0f},  { 0.0f, 1.0f, 0.0f } },
		{ { 0.0f, -1.0f, 0.0f },  { 0.0f, 0.0f, 1.0f } }
	};
	uint32_t vertexBufferSize = static_cast<uint32_t>(vertexBuffer.size()) * sizeof(Vertex);

	// Setup indices
	std::vector<uint32_t> indexBuffer = { 0, 1, 2 };
	indices.count = static_cast<uint32_t>(indexBuffer.size());
	uint32_t indexBufferSize = indices.count * sizeof(uint32_t);

	VkMemoryAllocateInfo memAlloc = {};
	memAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	VkMemoryRequirements memReqs;

	struct StagingBuffer {
		VkDeviceMemory memory;
		VkBuffer buffer;
	};

	struct {
		StagingBuffer vertices;
		StagingBuffer indices;
	} stagingBuffers;

	// Vertex buffer
	{
		VkBufferCreateInfo vertexBufferInfo = {};
		vertexBufferInfo.sType   = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		vertexBufferInfo.size    = vertexBufferSize;
		vertexBufferInfo.usage   = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		vkCreateBuffer(device, &vertexBufferInfo, NULL, &stagingBuffers.vertices.buffer);

		// TutNote: As you can see, the belo allocation snippet could be made into
		// a neat inline method and reused. The below code is made more verbose
		// for clarity's sake.
		vkGetBufferMemoryRequirements(device, stagingBuffers.vertices.buffer, &memReqs);
		memAlloc.allocationSize  = memReqs.size;
		memAlloc.memoryTypeIndex = getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
		vkAllocateMemory(device, &memAlloc, NULL, &stagingBuffers.vertices.memory);
		vkBindBufferMemory(device, stagingBuffers.vertices.buffer, stagingBuffers.vertices.memory, 0);

		mapAndCopyBuffer(stagingBuffers.vertices.memory, vertexBuffer.data(), (int)memAlloc.allocationSize);
	
		// Tutnote: Create the destination buffer with device only visibility
		// Buffer will be used as a vertex buffer and is the copy destination
		vertexBufferInfo.usage   = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		vkCreateBuffer(device, &vertexBufferInfo, NULL, &vertices.buf);

		vkGetBufferMemoryRequirements(device, vertices.buf, &memReqs);
		memAlloc.allocationSize  = memReqs.size;
		memAlloc.memoryTypeIndex = getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		vkAllocateMemory(device, &memAlloc, NULL, &vertices.mem);
		vkBindBufferMemory(device, vertices.buf, vertices.mem, 0);
	}

	// Index buffer
	{
		VkBufferCreateInfo indexbufferInfo = {};
		indexbufferInfo.sType   = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		indexbufferInfo.size    = indexBufferSize;
		indexbufferInfo.usage   = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		vkCreateBuffer(device, &indexbufferInfo, NULL, &stagingBuffers.indices.buffer);
		
		vkGetBufferMemoryRequirements(device, stagingBuffers.indices.buffer, &memReqs);
		memAlloc.allocationSize  = memReqs.size;
		memAlloc.memoryTypeIndex = getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
		vkAllocateMemory(device, &memAlloc, NULL, &stagingBuffers.indices.memory);
		vkBindBufferMemory(device, stagingBuffers.indices.buffer, stagingBuffers.indices.memory, 0);

		mapAndCopyBuffer(stagingBuffers.indices.memory, indexBuffer.data(), indexBufferSize);

		// Create destination buffer with device only visibility
		indexbufferInfo.usage    = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		vkCreateBuffer(device, &indexbufferInfo, NULL, &indices.buf);
		
		vkGetBufferMemoryRequirements(device, indices.buf, &memReqs);
		memAlloc.allocationSize  = memReqs.size;
		memAlloc.memoryTypeIndex = getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		vkAllocateMemory(device, &memAlloc, NULL, &indices.mem);
		vkBindBufferMemory(device, indices.buf, indices.mem, 0);
	}
	

	// Tutnote: Buffer copies have to be submitted to a queue, so we need a command buffer for them
	// Note that some devices offer a dedicated transfer queue (with only the transfer bit set)
	// If you do lots of copies (especially at runtime) it's advised to use such a queu instead
	// of a generalized graphics queue (that also supports transfers)
	VkCommandBuffer copyCmd = getCommandBuffer(true);

	// Put buffer region copies into command buffer
	// Note that the staging buffer must not be deleted before the copies have been submitted and executed
	VkBufferCopy copyRegion = {};

	// Vertex buffer
	copyRegion.size = vertexBufferSize;
	vkCmdCopyBuffer(copyCmd, stagingBuffers.vertices.buffer, vertices.buf, 1, &copyRegion);
	
	// Index buffer
	copyRegion.size = indexBufferSize;
	vkCmdCopyBuffer(copyCmd, stagingBuffers.indices.buffer, indices.buf, 1, &copyRegion);

	flushCommandBuffer(copyCmd);

	// Destroy staging buffers
	// TutNote: If there are dynamic uploads the app does, it maybe a good idea
	// to keep these staging buffers around. Actually, you could create a whole
	// pool of these buffers and recycle among them.
	vkDestroyBuffer(device, stagingBuffers.vertices.buffer, NULL);
	vkFreeMemory(device, stagingBuffers.vertices.memory, NULL);
	vkDestroyBuffer(device, stagingBuffers.indices.buffer, NULL);
	vkFreeMemory(device, stagingBuffers.indices.memory, NULL);

	// Binding description
	vertices.bindingDescriptions.resize(1);
	vertices.bindingDescriptions[0].binding    = 0; //VERTEX_BUFFER_BIND_ID;
	vertices.bindingDescriptions[0].stride     = sizeof(Vertex);
	vertices.bindingDescriptions[0].inputRate  = VK_VERTEX_INPUT_RATE_VERTEX;

	// Attribute descriptions
	// Describes memory layout and shader attribute locations
	vertices.attributeDescriptions.resize(2);
	// Location 0 : Position
	vertices.attributeDescriptions[0].binding  = 0;//VERTEX_BUFFER_BIND_ID;
	vertices.attributeDescriptions[0].location = 0;
	vertices.attributeDescriptions[0].format   = VK_FORMAT_R32G32B32_SFLOAT;
	vertices.attributeDescriptions[0].offset   = 0;
	// Location 1 : Color
	vertices.attributeDescriptions[1].binding  = 0;//VERTEX_BUFFER_BIND_ID;
	vertices.attributeDescriptions[1].location = 1;
	vertices.attributeDescriptions[1].format   = VK_FORMAT_R32G32B32_SFLOAT;
	vertices.attributeDescriptions[1].offset   = sizeof(float) * 3;

	// Assign to vertex input state
	vertices.inputState.sType						    = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertices.inputState.pNext						    = NULL;
	vertices.inputState.flags						    = 0;
	vertices.inputState.vertexBindingDescriptionCount   = static_cast<uint32_t>(vertices.bindingDescriptions.size());
	vertices.inputState.pVertexBindingDescriptions	    = vertices.bindingDescriptions.data();
	vertices.inputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertices.attributeDescriptions.size());
	vertices.inputState.pVertexAttributeDescriptions    = vertices.attributeDescriptions.data();

	return VK_SUCCESS;
}

VkResult preparePipelines()
{
	VkGraphicsPipelineCreateInfo pipelineCreateInfo = {};
	pipelineCreateInfo.sType	   = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineCreateInfo.layout	   = pipelineLayout;
	pipelineCreateInfo.renderPass  = renderPass;

	// Describes the topoloy used with this pipeline
	VkPipelineInputAssemblyStateCreateInfo inputAssemblyState = {};
	inputAssemblyState.sType	   = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	inputAssemblyState.topology	   = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

	// Rasterization state
	VkPipelineRasterizationStateCreateInfo rasterizationState = {};
	rasterizationState.sType	   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterizationState.polygonMode = VK_POLYGON_MODE_FILL; // Solid polygon mode
	// No culling
	rasterizationState.cullMode				   = VK_CULL_MODE_NONE;
	rasterizationState.frontFace			   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	rasterizationState.depthClampEnable        = VK_FALSE;
	rasterizationState.rasterizerDiscardEnable = VK_FALSE;
	rasterizationState.depthBiasEnable	       = VK_FALSE;
	rasterizationState.lineWidth			   = 1.0f;

	// Color blend state
	VkPipelineColorBlendStateCreateInfo colorBlendState = {};
	colorBlendState.sType									    = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	VkPipelineColorBlendAttachmentState blendAttachmentState[1] = {};
	blendAttachmentState[0].colorWriteMask						= 0xf;
	blendAttachmentState[0].blendEnable						    = VK_FALSE; // No blending in this example
	colorBlendState.attachmentCount								= 1;
	colorBlendState.pAttachments								= blendAttachmentState;

	// Viewport state
	VkPipelineViewportStateCreateInfo viewportState = {};
	viewportState.sType	= VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.viewportCount = 1;
	viewportState.scissorCount  = 1;

	// Enable dynamic states
	// Describes the dynamic states to be used with this pipeline
	// Dynamic states can be set even after the pipeline has been created
	// So there is no need to create new pipelines just for changing
	// a viewport's dimensions or a scissor box
	VkPipelineDynamicStateCreateInfo dynamicState = {};
	// The dynamic state properties themselves are stored in the command buffer
	std::vector<VkDynamicState> dynamicStateEnables;
	dynamicStateEnables.push_back(VK_DYNAMIC_STATE_VIEWPORT);
	dynamicStateEnables.push_back(VK_DYNAMIC_STATE_SCISSOR);
	dynamicState.sType			   = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamicState.pDynamicStates    = dynamicStateEnables.data();
	dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStateEnables.size());

	// Depth and stencil state
	VkPipelineDepthStencilStateCreateInfo depthStencilState = {};
	depthStencilState.sType					= VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depthStencilState.depthTestEnable		= VK_TRUE;
	depthStencilState.depthWriteEnable		= VK_TRUE;
	depthStencilState.depthCompareOp		= VK_COMPARE_OP_LESS_OR_EQUAL;
	depthStencilState.depthBoundsTestEnable = VK_FALSE;
	depthStencilState.back.failOp			= VK_STENCIL_OP_KEEP;
	depthStencilState.back.passOp			= VK_STENCIL_OP_KEEP;
	depthStencilState.back.compareOp		= VK_COMPARE_OP_ALWAYS;
	depthStencilState.stencilTestEnable		= VK_FALSE;
	depthStencilState.front					= depthStencilState.back;

	// Multi sampling state
	VkPipelineMultisampleStateCreateInfo multisampleState = {};
	multisampleState.sType					= VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisampleState.pSampleMask		    = NULL;
	multisampleState.rasterizationSamples	= VK_SAMPLE_COUNT_1_BIT;

	// Shaders
	static char const vss[] =
		"#version 450 core\n"
		"layout(location = 0) in vec3 aVertex;\n"
		"layout(location = 1) in vec3 aColor;\n"
		"layout(location = 0) out vec3 vColor;\n"
		"layout(binding = 0)  uniform UBO\n"
		"{\n"
		"	float color;\n"
		"} ubo;\n"
		"void main()\n"
		"{\n"
		"    vColor = aColor;\n"
		"	 vColor.b = ubo.color;\n"
		"    gl_Position = vec4(aVertex, 1.0);\n"
		"}\n"
		;
	VkShaderModule vsm;
	VkResult result = CreateShader(VK_SHADER_STAGE_VERTEX_BIT, (sizeof(vss) / sizeof(vss[0])), vss, vsm);
	if (result != VK_SUCCESS) {
		return result;
	}

	static char const fss[] =
		"#version 450 core\n"
		"layout(location = 0) in vec3 vColor;\n"
		"layout(location = 0) out vec4 oFrag;\n"
		"void main()\n"
		"{\n"
		"    oFrag = vec4(vColor, 1.0);\n"
		"}\n"
		;
	VkShaderModule fsm;
	result = CreateShader(VK_SHADER_STAGE_FRAGMENT_BIT, (sizeof(fss) / sizeof(fss[0])), fss, fsm);
	if (result != VK_SUCCESS) {
		return result;
	}
	VkPipelineShaderStageCreateInfo stageInfo[2] = { {},{} };
	stageInfo[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stageInfo[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
	stageInfo[0].module = vsm;
	stageInfo[0].pName  = "main";
	stageInfo[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stageInfo[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
	stageInfo[1].module = fsm;
	stageInfo[1].pName  = "main";

	// Assign pipeline state create information
	pipelineCreateInfo.stageCount		   = 2;
	pipelineCreateInfo.pStages			   = stageInfo;
	pipelineCreateInfo.pVertexInputState   = &vertices.inputState;
	pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
	pipelineCreateInfo.pRasterizationState = &rasterizationState;
	pipelineCreateInfo.pColorBlendState	   = &colorBlendState;
	pipelineCreateInfo.pMultisampleState   = &multisampleState;
	pipelineCreateInfo.pViewportState	   = &viewportState;
	pipelineCreateInfo.pDepthStencilState  = &depthStencilState;
	pipelineCreateInfo.renderPass		   = renderPass;
	pipelineCreateInfo.pDynamicState	   = &dynamicState;

	// Create rendering pipeline
	auto err = vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, NULL, &pipeline);

	// shader modules can be destroyed while pipelines created using its shaders are still in use
	DestroyShader(vsm);
	DestroyShader(fsm);

	return err;
}

VkResult setupDescriptorSetLayout()
{
	// Setup layout of descriptors used in this example
	// Basically connects the different shader stages to descriptors for binding uniform buffers, image samplers, etc.
	// So every shader binding should map to one descriptor set layout binding

	// Binding 0: Uniform buffer (Vertex shader)
	VkDescriptorSetLayoutBinding layoutBinding = {};
	layoutBinding.descriptorType     = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	layoutBinding.descriptorCount    = 1;
	layoutBinding.stageFlags	     = VK_SHADER_STAGE_VERTEX_BIT;
	layoutBinding.pImmutableSamplers = nullptr;

	VkDescriptorSetLayoutCreateInfo descriptorLayout = {};
	descriptorLayout.sType			 = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	descriptorLayout.pNext			 = nullptr;
	descriptorLayout.bindingCount	 = 1;
	descriptorLayout.pBindings		 = &layoutBinding;
	vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &descriptorSetLayout);

	// Tutnote: Create the pipeline layout that is used to generate the rendering pipelines that are based on this descriptor set layout
	VkPipelineLayoutCreateInfo pPipelineLayoutCreateInfo = {};
	pPipelineLayoutCreateInfo.sType			 = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pPipelineLayoutCreateInfo.pNext			 = nullptr;
	pPipelineLayoutCreateInfo.setLayoutCount = 1;
	pPipelineLayoutCreateInfo.pSetLayouts    = &descriptorSetLayout;
	VkResult err = vkCreatePipelineLayout(device, &pPipelineLayoutCreateInfo, nullptr, &pipelineLayout);

	return err;
}

VkResult allocateDescriptor()
{
	VkDescriptorSetAllocateInfo allocInfo = {};
	allocInfo.sType					   = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.descriptorPool		   = descriptorPool;
	allocInfo.descriptorSetCount	   = 1;
	allocInfo.pSetLayouts			   = &descriptorSetLayout;
	auto err = vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet);

	VkWriteDescriptorSet writeDescriptorSet = {};
	writeDescriptorSet.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	writeDescriptorSet.sType		   = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writeDescriptorSet.dstSet		   = descriptorSet;
	writeDescriptorSet.pBufferInfo     = &uniformDescriptor;	
	writeDescriptorSet.dstBinding      = 0; // Binds this uniform buffer to binding point 0
	writeDescriptorSet.descriptorCount = 1; // Let's just allocate one descriptor in this example.

	vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);
	return err;
}

VkResult createDescriptorPool()
{
	VkDescriptorPoolSize typeCounts[1];
	typeCounts[0].type				   = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	typeCounts[0].descriptorCount	   = 1;
	// Tutnote: For additional types you need to add new entries in the type count list
	// E.g. for two combined image samplers :
	// typeCounts[1].type			  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	// typeCounts[1].descriptorCount  = 2;

	VkDescriptorPoolCreateInfo descriptorPoolInfo = {};
	descriptorPoolInfo.sType		  = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	descriptorPoolInfo.pNext		  = nullptr;
	descriptorPoolInfo.poolSizeCount  = 1;
	descriptorPoolInfo.pPoolSizes	  = typeCounts;
	descriptorPoolInfo.maxSets		  = 1;
	auto err = vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool);

	return err;
}

VkResult prepareUniformBuffers()
{
	VkMemoryRequirements memReqs;

	VkBufferCreateInfo bufferInfo = {};
	bufferInfo.sType				  = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size					  = 4; // Just 4 bytes will suffice.
	bufferInfo.usage				  = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
	vkCreateBuffer(device, &bufferInfo, nullptr, &uniformBuffer);

	VkMemoryAllocateInfo allocInfo = {};
	allocInfo.sType					  = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.pNext					  = nullptr;
	allocInfo.allocationSize		  = 0;
	allocInfo.memoryTypeIndex		  = 0;
	vkGetBufferMemoryRequirements(device, uniformBuffer, &memReqs);

	allocInfo.allocationSize		  = memReqs.size;
	allocInfo.memoryTypeIndex		  = getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	vkAllocateMemory(device, &allocInfo, nullptr, &(uniformBufferMemory));
	vkBindBufferMemory(device, uniformBuffer, uniformBufferMemory, 0);

	// Reference the buffer in the descriptor
	uniformDescriptor.buffer		 = uniformBuffer;
	uniformDescriptor.offset		 = 0;

	float color = 1.0f;
	auto err = mapAndCopyBuffer(uniformBufferMemory, &color, sizeof(float));

	return err;
}

VkResult buildRenderCommandBuffers()
{
	VkCommandBufferBeginInfo cmdBufInfo = {};
	cmdBufInfo.sType							 = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	cmdBufInfo.pNext							 = NULL;

	// Set clear values for all framebuffer attachments with loadOp set to clear
	// We use two attachments (color and depth) that are cleared at the 
	// start of the subpass and as such we need to set clear values for both
	VkClearValue clearValues[2];
	clearValues[0].color						 = { 1.0f, 1.0f, 1.0f, 0.99f };
	clearValues[1].depthStencil					 = { 1.0f, 0 };

	VkRenderPassBeginInfo renderPassBeginInfo	 = {};
	renderPassBeginInfo.sType					 = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	renderPassBeginInfo.pNext					 = NULL;
	renderPassBeginInfo.renderPass				 = renderPass;
	renderPassBeginInfo.renderArea.offset.x		 = 0;
	renderPassBeginInfo.renderArea.offset.y	 	 = 0;
	renderPassBeginInfo.renderArea.extent.width  = width;
	renderPassBeginInfo.renderArea.extent.height = height;
	renderPassBeginInfo.clearValueCount			 = 2;
	renderPassBeginInfo.pClearValues			 = clearValues;

	for (int32_t i = 0; i < drawCmdBuffers.size(); ++i)
	{
		// Set target frame buffer
		renderPassBeginInfo.framebuffer = frameBuffers[i];

		vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo);

		// Start the first sub pass specified in our default render pass setup by the base class
		// This will clear the color and depth attachment
		vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

		// Update dynamic viewport state
		VkViewport viewport						= {};
		viewport.height							= (float)height;
		viewport.width							= (float)width;
		viewport.minDepth						= (float) 0.0f;
		viewport.maxDepth						= (float) 1.0f;
		vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);

		// Update dynamic scissor state
		VkRect2D scissor						= {};
		scissor.extent.width					= width;
		scissor.extent.height					= height;
		scissor.offset.x					    = 0;
		scissor.offset.y						= 0;
		vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

		// Bind descriptor sets describing shader binding points
		vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, NULL);

		// Bind the rendering pipeline
		// The pipeline (state object) contains all states of the rendering pipeline
		// So once we bind a pipeline all states that were set upon creation of that
		// pipeline will be set
		vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

		// Bind triangle vertex buffer (contains position and colors)
		VkDeviceSize offsets[1] = { 0 };
		vkCmdBindVertexBuffers(drawCmdBuffers[i], 0, 1, &vertices.buf, offsets);

		// Bind triangle index buffer
		vkCmdBindIndexBuffer(drawCmdBuffers[i], indices.buf, 0, VK_INDEX_TYPE_UINT32);

		// Draw indexed triangle
		vkCmdDrawIndexed(drawCmdBuffers[i], indices.count, 1, 0, 0, 1);

		vkCmdEndRenderPass(drawCmdBuffers[i]);

		// TutNote: Add a present memory barrier to the end of the command buffer
		// This will transform the frame buffer color attachment to a
		// new layout for presenting it to the windowing system integration 
		VkImageMemoryBarrier prePresentBarrier = {};
		prePresentBarrier.sType					= VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		prePresentBarrier.pNext					= NULL;
		prePresentBarrier.srcAccessMask			= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		prePresentBarrier.dstAccessMask			= VK_ACCESS_MEMORY_READ_BIT;
		prePresentBarrier.oldLayout				= VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		// Tutnote: This is where we transition the FB image type via an Image barrier.
		// The older format was in VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
		prePresentBarrier.newLayout				= VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
		prePresentBarrier.srcQueueFamilyIndex   = VK_QUEUE_FAMILY_IGNORED;
		prePresentBarrier.dstQueueFamilyIndex	= VK_QUEUE_FAMILY_IGNORED;
		prePresentBarrier.subresourceRange		= { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		prePresentBarrier.image = buffers[i].image;

		VkImageMemoryBarrier *pMemoryBarrier = &prePresentBarrier;
		vkCmdPipelineBarrier(
			drawCmdBuffers[i],
			VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
			VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
			0,
			0, NULL,
			0, NULL,
			1, &prePresentBarrier);

		vkEndCommandBuffer(drawCmdBuffers[i]);
	}

	for (uint32_t i = 0; i < imageCount; i++)
	{
		// Tutnote: Needed for the next frame's rendering.
		VkImageMemoryBarrier postPresentBarrier = {};
		postPresentBarrier.sType				= VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		postPresentBarrier.pNext				= NULL;
		postPresentBarrier.srcAccessMask		= 0;
		postPresentBarrier.dstAccessMask		= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		postPresentBarrier.oldLayout			= VK_IMAGE_LAYOUT_UNDEFINED;
		postPresentBarrier.newLayout			= VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		postPresentBarrier.srcQueueFamilyIndex  = VK_QUEUE_FAMILY_IGNORED;
		postPresentBarrier.dstQueueFamilyIndex  = VK_QUEUE_FAMILY_IGNORED;
		postPresentBarrier.subresourceRange		= { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		postPresentBarrier.image			    = buffers[i].image;

		VkCommandBufferBeginInfo cmdBufInfo = {};
		cmdBufInfo.sType						= VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		vkBeginCommandBuffer(postPresentCmdBuffers[i], &cmdBufInfo);

		// Put post present barrier into command buffer
		vkCmdPipelineBarrier(
			postPresentCmdBuffers[i],
			VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
			VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
			0,
			0, NULL,
			0, NULL,
			1, &postPresentBarrier);

		vkEndCommandBuffer(postPresentCmdBuffers[i]);
	}

	return VK_SUCCESS;
}

VkResult updateScene()
{
	static float color = 0.0f;

	color += 0.001f;
	float finalColor = sin(color);

	return mapAndCopyBuffer(uniformBufferMemory, &finalColor, sizeof(float));
}

VkResult draw()
{
	// Tutnote: Get next image in the swap chain.
	// The presentComplete will be signalled by the Display once the Image has been released.
	vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, semaphores.presentComplete, NULL, &currentBuffer);

	// Tutnote: Submit the post present image barrier to transform the image back to a color attachment
	// that can be used to write to by our render pass
	VkSubmitInfo submitInfo = {};
	submitInfo.sType					= VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount		= 1;
	submitInfo.pCommandBuffers			= &postPresentCmdBuffers[currentBuffer];
	vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);

	// Make sure that the image barrier command submitted to the queue has finished executing

	// Tutnote: TODO: This also could use a semaphore and avoid the CPU wait completely. This is an example only!
	// This is equivalent to a glFinish(). It should be avoided at all costs ince this causes stalling of the submissions.
	vkQueueWaitIdle(queue);

	// The submit info strcuture contains a list of ommand buffers and semaphores to be submitted to a queue
	// If you want to submit multiple command buffers, pass an array
	VkPipelineStageFlags pipelineStages = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
	submitInfo.sType					= VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.pWaitDstStageMask		= &pipelineStages;
	submitInfo.waitSemaphoreCount		= 1;
	submitInfo.pWaitSemaphores			= &semaphores.presentComplete;
	submitInfo.commandBufferCount		= 1;
	submitInfo.pCommandBuffers			= &drawCmdBuffers[currentBuffer];
	submitInfo.signalSemaphoreCount		= 1;
	submitInfo.pSignalSemaphores		= &semaphores.renderComplete;

	// Submit to the graphics queue
	vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);

	VkSubmitInfo submitInfo2 = {};
	submitInfo2.sType					= VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo2.commandBufferCount		= 1;
	submitInfo2.pCommandBuffers			= &prePresentCmdBuffers[currentBuffer];
	vkQueueSubmit(queue, 1, &submitInfo2, VK_NULL_HANDLE);

	// Tutnote: Present the current buffer to the swap chain. We pass the signal semaphore from the submit info
	// to ensure that the image is not rendered until all commands have been submitted
	VkPresentInfoKHR presentInfo = {};
	presentInfo.sType					= VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	presentInfo.pNext					= NULL;
	presentInfo.swapchainCount			= 1;
	presentInfo.pSwapchains				= &swapchain;
	presentInfo.pImageIndices			= &currentBuffer;
	presentInfo.pWaitSemaphores			= &semaphores.renderComplete;
	presentInfo.waitSemaphoreCount		= 1;
	auto err = vkQueuePresentKHR(queue, &presentInfo);

	return err;
}

bool createWindow()
{
	glfwInit();

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

	window = glfwCreateWindow(width, height, "Vulkan Samples", nullptr, nullptr);
	assert(window);

	return window != nullptr;
}

VkResult createInstance()
{
	VkResult err;
	std::vector<const char*> extNames;
	extNames.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
	extNames.push_back(VK_KHR_SURFACE_EXTENSION_NAME);

	VkInstanceCreateInfo instanceCreateInfo;
	instanceCreateInfo.sType					 = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	instanceCreateInfo.pNext					 = NULL;
	instanceCreateInfo.pApplicationInfo			 = NULL;
	instanceCreateInfo.enabledExtensionCount	 = (uint32_t)extNames.size();
	instanceCreateInfo.ppEnabledExtensionNames	 = extNames.data();
	instanceCreateInfo.enabledLayerCount		 = 0;
	instanceCreateInfo.ppEnabledLayerNames		 = NULL;
	err = vkCreateInstance(&instanceCreateInfo, NULL, &instance);

	return err;
}

VkResult setupDevice()
{
	//Get phyiscal device
	uint32_t physicalDeviceCount = 0;
	std::vector<const char*> extNames2;
	extNames2.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
	auto err = vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, NULL);
	err = vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, &physDevice);

	err = createWin32Surface();
	if (err != VK_SUCCESS) {
		std::cout << "error at createDirectSurface: " << err << std::endl;
		return err;
	}

	// Index of the deteced graphics and presenting device queue
	uint32_t queueCount;
	vkGetPhysicalDeviceQueueFamilyProperties(physDevice, &queueCount, NULL);
	std::vector<VkQueueFamilyProperties> queueProps(queueCount);
	vkGetPhysicalDeviceQueueFamilyProperties(physDevice, &queueCount, queueProps.data());

	// Iterate over each queue to learn whether it supports presenting:
	// Find a queue with present support
	// Will be used to present the swap chain images to the windowing system
	std::vector<VkBool32> supportsPresent(queueCount);
	for (uint32_t i = 0; i < queueCount; i++)
	{
		vkGetPhysicalDeviceSurfaceSupportKHR(physDevice, i, surface, &supportsPresent[i]);
	}

	// Search for a graphics and a present queue in the array of queue
	// families, try to find one that supports both
	uint32_t graphicsQueueNodeIndex = UINT32_MAX;
	uint32_t presentQueueNodeIndex = UINT32_MAX;
	for (uint32_t i = 0; i < queueCount; i++)
	{
		if ((queueProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0)
		{
			if (graphicsQueueNodeIndex == UINT32_MAX)
			{
				graphicsQueueNodeIndex = i;
			}

			if (supportsPresent[i] == VK_TRUE)
			{
				graphicsQueueNodeIndex = i;
				presentQueueNodeIndex = i;
				break;
			}
		}
	}
	if (presentQueueNodeIndex == UINT32_MAX)
	{
		// If there's no queue that supports both present and graphics
		// try to find a separate present queue
		for (uint32_t i = 0; i < queueCount; ++i)
		{
			if (supportsPresent[i] == VK_TRUE)
			{
				presentQueueNodeIndex = i;
				break;
			}
		}
	}

	// Exit if either a graphics or a presenting queue hasn't been found
	if (graphicsQueueNodeIndex == UINT32_MAX || presentQueueNodeIndex == UINT32_MAX)
	{
		std::cout << "no graphics or present queue found\n";
	}
	queueNodeIndex = graphicsQueueNodeIndex;

	// Get list of supported surface formats
	uint32_t formatCount;
	err = vkGetPhysicalDeviceSurfaceFormatsKHR(physDevice, surface, &formatCount, NULL);

	std::vector<VkSurfaceFormatKHR> surfaceFormats(formatCount);
	err = vkGetPhysicalDeviceSurfaceFormatsKHR(physDevice, surface, &formatCount, surfaceFormats.data());

	// If the surface format list only includes one entry with VK_FORMAT_UNDEFINED,
	// there is no preferered format, so we assume VK_FORMAT_B8G8R8A8_UNORM
	if ((formatCount == 1) && (surfaceFormats[0].format == VK_FORMAT_UNDEFINED))
	{
		colorFormat = VK_FORMAT_B8G8R8A8_UNORM;
	}
	else
	{
		// Always select the first available color format
		// If you need a specific format (e.g. SRGB) you'd need to
		// iterate over the list of available surface format and
		// check for it's presence
		colorFormat = surfaceFormats[0].format;
	}
	colorSpace = surfaceFormats[0].colorSpace;

	//Create device
	VkDeviceCreateInfo devCreateInfo;
	std::vector<float> queuePriorities = { 0.0f };
	VkDeviceQueueCreateInfo queueCreateInfo = {};
	queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	queueCreateInfo.queueFamilyIndex = graphicsQueueNodeIndex;
	queueCreateInfo.queueCount = 1;
	queueCreateInfo.pQueuePriorities = queuePriorities.data();
	devCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	devCreateInfo.pNext = NULL;
	devCreateInfo.queueCreateInfoCount = 1;
	devCreateInfo.pQueueCreateInfos = &queueCreateInfo;
	devCreateInfo.flags = 0;
	devCreateInfo.enabledLayerCount = 0;
	devCreateInfo.ppEnabledLayerNames = NULL;
	devCreateInfo.enabledExtensionCount = 1;
	devCreateInfo.ppEnabledExtensionNames = extNames2.data();
	devCreateInfo.pEnabledFeatures = NULL;
	err = vkCreateDevice(physDevice, &devCreateInfo, NULL, &device);

	if (err) {
		std::cout << "failed to create device, err: " << err << " !!!\n\n";
		return err;
	}

	vkGetPhysicalDeviceMemoryProperties(physDevice, &deviceMemoryProperties);
	vkGetDeviceQueue(device, graphicsQueueNodeIndex, 0, &queue);

	return err;
}

int main(int argc, char** argv)
{
	if (!createWindow()) {
		std::cout << "failed to create window" << "!!!\n\n";
		return 1;
	}

	auto err = createInstance();
	if (err) {
		std::cout << "failed to create instance, err: " << err << " !!!\n\n";
		return 1;
	}

	err = setupDevice();
	if (err) {
		std::cout << "failed to setup device, err: " << err << " !!!\n\n";
		return 1;
	}

	err = createSwapchain();
	if (err != VK_SUCCESS) {
		std::cout << "error at createSwapchain: " << err << std::endl;
		return 1;
	}

	err = createCommandPool();
	if (err != VK_SUCCESS) {
		std::cout << "error at createCommandPool: " << err << std::endl;
		return 1;
	}	

	err = createDescriptorPool();
	if (err != VK_SUCCESS) {
		std::cout << "error at createDescriptorPool: " << err << std::endl;
		return 1;
	}


	// Setup initial stuff.
	{
		// Tutnote: It is a common practice in Vulkan to use a temporary command buffer initially
		// to issue some simple command buffers for image transitions, buffer copies, etc.
		err = createSetupCommandBuffer();
		if (err != VK_SUCCESS) {
			std::cout << "error at createSetupCommandBuffer: " << err << std::endl;
			return 1;
		}

		err = setupDepthStencil();
		if (err != VK_SUCCESS) {
			std::cout << "error at setupDepthStencil: " << err << std::endl;
			return 1;
		}

		// Flush the setup command buffer.
		err = flushSetupCommandBuffer();
		if (err != VK_SUCCESS) {
			std::cout << "error at flushSetupCommandBuffer: " << err << std::endl;
			return 1;
		}
	}

	err = createPipelineCache();
	if (err != VK_SUCCESS) {
		std::cout << "error at createPipelineCache: " << err << std::endl;
		return 1;
	}

	err = createCommandBuffers();
	if (err != VK_SUCCESS) {
		std::cout << "error at createCommandBuffers: " << err << std::endl;
		return 1;
	}
	err = buildPresentCommandBuffers();
	if (err != VK_SUCCESS) {
		std::cout << "error at buildPresentCommandBuffers: " << err << std::endl;
		return 1;
	}

	err = setupRenderPass();
	if (err != VK_SUCCESS) {
		std::cout << "error at setupRenderPass: " << err << std::endl;
		return 1;
	}

	err = setupFrameBuffer();
	if (err != VK_SUCCESS) {
		std::cout << "error at setupFrameBuffer: " << err << std::endl;
		return 1;
	}

	err = prepareSemaphore();
	if (err != VK_SUCCESS) {
		std::cout << "error at prepareSemaphore: " << err << std::endl;
		return 1;
	}
	err = prepareVertices();
	if (err != VK_SUCCESS) {
		std::cout << "error at prepauboreVertices: " << err << std::endl;
		return 1;
	}

	err = prepareUniformBuffers();
	if (err != VK_SUCCESS) {
		std::cout << "error at prepareUniformBuffers: " << err << std::endl;
		return 1;
	}
	
	err = setupDescriptorSetLayout();
	if (err != VK_SUCCESS) {
		std::cout << "error at setupDescriptorSetLayout: " << err << std::endl;
		return 1;
	}
	
	err = allocateDescriptor();
	if (err != VK_SUCCESS) {
		std::cout << "error at allocateDescriptor: " << err << std::endl;
		return 1;
	}

	err = preparePipelines();
	if (err != VK_SUCCESS) {
		std::cout << "error at preparePipelines: " << err << std::endl;
		return 1;
	}

	err = buildRenderCommandBuffers();
	if (err != VK_SUCCESS) {
		std::cout << "error at buildRenderCommandBuffers: " << err << std::endl;
		return 1;
	}

	while (1) {
		err = draw();

		if (err != VK_SUCCESS) {
			std::cout << "error at draw: " << err << std::endl;
			return 1;
		}

		err = updateScene();
		if (err != VK_SUCCESS) {
			std::cout << "error at updateScene: " << err << std::endl;
			return 1;
		}
	}

	// Clean up
	for (uint32_t i = 0; i < imageCount; i++)
	{
		vkDestroyFramebuffer(device, frameBuffers[i], NULL);
		vkDestroyImageView(device, buffers[i].view, NULL);
	}

	vkDestroySwapchainKHR(device, swapchain, NULL);
	vkDestroySurfaceKHR(instance, surface, NULL);
	vkFreeCommandBuffers(device, cmdPool, imageCount, drawCmdBuffers.data());
	vkFreeCommandBuffers(device, cmdPool, imageCount, prePresentCmdBuffers.data());
	vkFreeCommandBuffers(device, cmdPool, imageCount, postPresentCmdBuffers.data());
	vkDestroyCommandPool(device, cmdPool, NULL);
	vkDestroyImage(device, depthStencil.image, NULL);
	vkDestroyImageView(device, depthStencil.view, NULL);
	vkFreeMemory(device, depthStencil.mem, NULL);
	vkDestroyRenderPass(device, renderPass, NULL);
	vkDestroyPipelineCache(device, pipelineCache, NULL);
	vkDestroySemaphore(device, semaphores.presentComplete, NULL);
	vkDestroySemaphore(device, semaphores.renderComplete, NULL);
	vkFreeMemory(device, vertices.mem, NULL);
	vkFreeMemory(device, indices.mem, NULL);
	vkDestroyBuffer(device, vertices.buf, NULL);
	vkDestroyBuffer(device, indices.buf, NULL);
	vkDestroyPipeline(device, pipeline, NULL);
	vkDestroyDevice(device, NULL);
	vkDestroyInstance(instance, NULL);
}
