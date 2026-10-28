// NVIDIA NGX / DLSS Super Resolution integration.
// The SDK itself is optional and remains outside the Savage Wind repository.
#include "dlss.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../paths.h"

#if defined(_WIN32) && DLSS_SDK_PRESENT

#include <direct.h>
#include <math.h>
#include <wchar.h>
#include <nvsdk_ngx_helpers_vk.h>
#include <nvsdk_ngx_helpers_dlssg_vk.h>

#include "../../imageload.h"

static bool g_initialized;
static bool g_available;
static NVSDK_NGX_Parameter *g_params;
static NVSDK_NGX_Parameter *g_capability_params;
static NVSDK_NGX_Handle *g_feature;
static unsigned g_in_w, g_in_h, g_out_w, g_out_h, g_quality;
static VkFormat g_color_format, g_output_format;
static const char **g_instance_exts;
static const char **g_device_exts;
static unsigned g_instance_ext_count, g_device_ext_count;
static bool g_exts_queried;

static const char *result_name(NVSDK_NGX_Result r) {
    switch (r) {
    case NVSDK_NGX_Result_Success: return "success";
    case NVSDK_NGX_Result_FAIL_FeatureNotSupported: return "feature-not-supported";
    case NVSDK_NGX_Result_FAIL_UnableToInitializeFeature: return "unable-to-initialize-feature";
    case NVSDK_NGX_Result_FAIL_OutOfDate: return "out-of-date";
    case NVSDK_NGX_Result_FAIL_UnsupportedInputFormat: return "unsupported-input-format";
    case NVSDK_NGX_Result_FAIL_MissingInput: return "missing-input";
    default: return "failure";
    }
}

static bool ok(NVSDK_NGX_Result r, const char *what) {
    if (r == NVSDK_NGX_Result_Success) return true;
    LOG("[dlss] %s: %s (%#x)\n", what, result_name(r), (unsigned)r);
    return false;
}

static NVSDK_NGX_Resource_VK image_resource(TexBackend *b, bool rw) {
    static const VkComponentMapping id = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                                          VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
    VkImageView v = vk_image_view(b, VK_IMAGE_VIEW_TYPE_2D, b->format, 0, 1, 0, 1, id,
                                   b->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT);
    VkImageSubresourceRange range = {b->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT,
                                     0, 1, 0, 1};
    return NVSDK_NGX_Create_ImageView_Resource_VK(v, b->image, range, b->format, b->width, b->height, rw);
}

static void release_feature(void) {
    if (g_feature) {
        NVSDK_NGX_VULKAN_ReleaseFeature(g_feature);
        g_feature = NULL;
    }
    if (g_params) {
        NVSDK_NGX_VULKAN_DestroyParameters(g_params);
        g_params = NULL;
    }
    g_in_w = g_in_h = g_out_w = g_out_h = g_quality = 0;
    g_color_format = g_output_format = VK_FORMAT_UNDEFINED;
}

bool vk_dlss_compiled(void) { return true; }

static void query_extensions(void) {
    if (g_exts_queried) return;
    g_exts_queried = true;
    unsigned ni = 0, nd = 0;
    const char **i = NULL, **d = NULL;
    NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_RequiredExtensions(&ni, &i, &nd, &d);
    if (r == NVSDK_NGX_Result_Success) {
        g_instance_exts = i; g_instance_ext_count = ni;
        g_device_exts = d; g_device_ext_count = nd;
        LOG("[dlss] required extensions: %u instance, %u device\n", ni, nd);
    } else {
        LOG("[dlss] required extension query: %s (%#x)\n", result_name(r), (unsigned)r);
    }
}

void vk_dlss_required_instance_extensions(const char ***names, unsigned *count) {
    query_extensions();
    if (names) *names = g_instance_exts;
    if (count) *count = g_instance_ext_count;
}

void vk_dlss_required_device_extensions(const char ***names, unsigned *count) {
    query_extensions();
    if (names) *names = g_device_exts;
    if (count) *count = g_device_ext_count;
}

// ================================================================== DLSS Frame Generation probe
// SAVAGEWIND_DLSSG_PROBE=1: after NGX starts, what DLSS Frame Generation (NGX's own, without Streamline) needs and
// does here: whether it is available (driver, GPU, how many frames it can make), the Vulkan extensions it wants, and
// one 2x feature (3840x2160 out, 2560x1440 depth and motion) run on made-up frames: a square moving 240 px a frame
// over a checkerboard, a white "HUD" bar the HUD-less image lacks. The interpolated and real outputs ->
// build/frames/dlssg_probe_{interp,real}.png (the square should be halfway; a non-production DLL draws a watermark
// on them), and the evaluation's GPU time.
#define PROJECT_ID "6f4bb6ce-20d8-4a8d-9c3d-0b7d5df1c3a1"
#define ENGINE_VERSION "SavageWind-1.0"
static NVSDK_NGX_FeatureCommonInfo g_common;
static wchar_t g_app_data[1024];

static NVSDK_NGX_FeatureDiscoveryInfo discovery(NVSDK_NGX_Feature f) {
    NVSDK_NGX_FeatureDiscoveryInfo d = {};
    d.SDKVersion = NVSDK_NGX_Version_API;
    d.FeatureID = f;
    d.Identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Project_Id;
    d.Identifier.v.ProjectDesc = {PROJECT_ID, NVSDK_NGX_ENGINE_TYPE_CUSTOM, ENGINE_VERSION};
    d.ApplicationDataPath = g_app_data;
    d.FeatureInfo = &g_common;
    return d;
}

namespace {
struct PImg {
    VkImage img;
    VmaAllocation mem;
    VkImageView view;
    VkFormat fmt;
    uint32_t w, h;
    VkImageAspectFlags aspect;
};
struct PBuf {
    VkBuffer buf;
    VmaAllocation mem;
    uint8_t *map;
};
}

static bool pimg(PImg *p, uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags usage) {
    *p = {};
    p->fmt = fmt, p->w = w, p->h = h;
    p->aspect = fmt == VK_FORMAT_D32_SFLOAT ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D, ci.format = fmt, ci.extent = {w, h, 1}, ci.mipLevels = 1, ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT, ci.tiling = VK_IMAGE_TILING_OPTIMAL, ci.usage = usage;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo ai = {};
    ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (vmaCreateImage(g_vk.vma, &ci, &ai, &p->img, &p->mem, NULL) != VK_SUCCESS) return false;
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = p->img, vi.viewType = VK_IMAGE_VIEW_TYPE_2D, vi.format = fmt;
    vi.subresourceRange = {p->aspect, 0, 1, 0, 1};
    return vkCreateImageView(g_vk.dev, &vi, NULL, &p->view) == VK_SUCCESS;
}

static void pimg_free(PImg *p) {
    if (p->view) vkDestroyImageView(g_vk.dev, p->view, NULL);
    if (p->img) vmaDestroyImage(g_vk.vma, p->img, p->mem);
    *p = {};
}

static NVSDK_NGX_Resource_VK pres(PImg *p, bool rw) {
    VkImageSubresourceRange range = {p->aspect, 0, 1, 0, 1};
    return NVSDK_NGX_Create_ImageView_Resource_VK(p->view, p->img, range, p->fmt, p->w, p->h, rw);
}

static bool pbuf(PBuf *b, VkDeviceSize size, VkBufferUsageFlags usage) {
    *b = {};
    VkBufferCreateInfo ci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size = size, ci.usage = usage;
    VmaAllocationCreateInfo ai = {};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo info;
    if (vmaCreateBuffer(g_vk.vma, &ci, &ai, &b->buf, &b->mem, &info) != VK_SUCCESS) return false;
    b->map = (uint8_t *)info.pMappedData;
    return true;
}

static void pbuf_free(PBuf *b) {
    if (b->buf) vmaDestroyBuffer(g_vk.vma, b->buf, b->mem);
    *b = {};
}

static void pbarrier(VkCommandBuffer cb) {
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, NULL,
                         0, NULL);
}

static void pgeneral(VkCommandBuffer cb, PImg *p) {
    VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = 0, b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = p->img, b.subresourceRange = {p->aspect, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL,
                         1, &b);
}

static void pcopy(VkCommandBuffer cb, PBuf *buf, PImg *p, bool to_image) {
    VkBufferImageCopy r = {};
    r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    r.imageExtent = {p->w, p->h, 1};
    if (to_image) vkCmdCopyBufferToImage(cb, buf->buf, p->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
    else vkCmdCopyImageToBuffer(cb, p->img, VK_IMAGE_LAYOUT_GENERAL, buf->buf, 1, &r);
}

// Frame k of the made-up scene (RGBA8): a checkerboard, a red square moving 240 px a frame, and (hud) a white bar
static void pframe(uint8_t *px, uint32_t w, uint32_t h, int k, bool hud) {
    int sx = 800 + 240 * k, sy = 900;
    for (uint32_t y = 0; y < h; y++)
        for (uint32_t x = 0; x < w; x++) {
            uint8_t *p = px + ((size_t)y * w + x) * 4;
            uint8_t c = ((x / 128) + (y / 128)) & 1 ? 150 : 90;
            p[0] = p[1] = p[2] = c, p[3] = 255;
            if ((int)x >= sx && (int)x < sx + 256 && (int)y >= sy && (int)y < sy + 256) p[0] = 230, p[1] = 40, p[2] = 30;
            if (hud && x >= 100 && x < 700 && y >= 100 && y < 180) p[0] = p[1] = p[2] = 255;
        }
}

static VkCommandBuffer pbegin(void) {
    VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = g_vk.pool, ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, ai.commandBufferCount = 1;
    VkCommandBuffer cb;
    vkAllocateCommandBuffers(g_vk.dev, &ai, &cb);
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);
    return cb;
}

static bool psubmit(VkCommandBuffer cb) {
    vkEndCommandBuffer(cb);
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1, si.pCommandBuffers = &cb;
    VkResult r = vkQueueSubmit(g_vk.queue, 1, &si, VK_NULL_HANDLE);
    if (r == VK_SUCCESS) r = vkQueueWaitIdle(g_vk.queue);
    vkFreeCommandBuffers(g_vk.dev, g_vk.pool, 1, &cb);
    if (r != VK_SUCCESS) LOG("[dlssg] probe: submission failed: %d\n", r);
    return r == VK_SUCCESS;
}

static void dlssg_probe(void) {
    // Availability
    int avail = -1, needs_driver = -1, init_result = -1;
    unsigned min_major = 0, min_minor = 0, mfg_max = 0;
    NVSDK_NGX_Parameter_GetI(g_capability_params, NVSDK_NGX_Parameter_FrameGeneration_Available, &avail);
    NVSDK_NGX_Parameter_GetI(g_capability_params, NVSDK_NGX_Parameter_FrameGeneration_NeedsUpdatedDriver, &needs_driver);
    NVSDK_NGX_Parameter_GetI(g_capability_params, NVSDK_NGX_Parameter_FrameGeneration_FeatureInitResult, &init_result);
    NVSDK_NGX_Parameter_GetUI(g_capability_params, NVSDK_NGX_Parameter_FrameGeneration_MinDriverVersionMajor, &min_major);
    NVSDK_NGX_Parameter_GetUI(g_capability_params, NVSDK_NGX_Parameter_FrameGeneration_MinDriverVersionMinor, &min_minor);
    NVSDK_NGX_Result mr = NVSDK_NGX_Parameter_GetUI(g_capability_params, NVSDK_NGX_DLSSG_Parameter_MultiFrameCountMax, &mfg_max);
    LOG("[dlssg] probe: available %d, needs a newer driver %d (min %u.%u), init result %#x, MultiFrameCountMax %u%s\n",
        avail, needs_driver, min_major, min_minor, (unsigned)init_result, mfg_max,
        NVSDK_NGX_SUCCEED(mr) ? "" : " (not set)");

    // Requirements, and the Vulkan extensions frame generation asks for beyond NGX's general list (which device.c
    // enables)
    NVSDK_NGX_FeatureDiscoveryInfo di = discovery(NVSDK_NGX_Feature_FrameGeneration);
    NVSDK_NGX_FeatureRequirement req = {};
    NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_GetFeatureRequirements(g_vk.inst, g_vk.phys, &di, &req);
    LOG("[dlssg] probe: requirements %s: unsupported bits %#x, min architecture %#x, min OS %s\n", result_name(r),
        (unsigned)req.FeatureSupported, req.MinHWArchitecture, req.MinOSVersion);
    uint32_t n = 0;
    VkExtensionProperties *ep = NULL;
    r = NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(&di, &n, &ep);
    LOG("[dlssg] probe: instance extensions %s: %u\n", result_name(r), n);
    for (uint32_t i = 0; r == NVSDK_NGX_Result_Success && i < n; i++) {
        bool have = false;
        for (unsigned j = 0; j < g_instance_ext_count; j++) have |= !strcmp(g_instance_exts[j], ep[i].extensionName);
        LOG("[dlssg] probe:   %s%s\n", ep[i].extensionName, have ? "" : "  <- not in NGX's general list");
    }
    n = 0, ep = NULL;
    r = NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(g_vk.inst, g_vk.phys, &di, &n, &ep);
    LOG("[dlssg] probe: device extensions %s: %u\n", result_name(r), n);
    for (uint32_t i = 0; r == NVSDK_NGX_Result_Success && i < n; i++) {
        bool have = !strcmp(ep[i].extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME);
        for (unsigned j = 0; j < g_device_ext_count; j++) have |= !strcmp(g_device_exts[j], ep[i].extensionName);
        LOG("[dlssg] probe:   %s%s\n", ep[i].extensionName, have ? "" : "  <- not in NGX's general list");
    }
    for (unsigned j = 0; j < g_device_ext_count; j++) LOG("[dlssg] probe:   (general list) %s\n", g_device_exts[j]);
    size_t vram = 0;
    r = NGX_VK_ESTIMATE_VRAM_DLSSG(g_capability_params, 2560, 1440, 3840, 2160, VK_FORMAT_R8G8B8A8_UNORM,
                                   VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_D32_SFLOAT, VK_FORMAT_R8G8B8A8_UNORM, 0, &vram);
    LOG("[dlssg] probe: VRAM estimate %s: %.0f MB\n", result_name(r), vram / 1048576.0);
    if (avail != 1) return;

    // One 2x feature on made-up frames
    const uint32_t W = 3840, H = 2160, RW = 2560, RH = 1440;
    const VkImageUsageFlags color_use = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                                        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    PImg back = {}, hudless = {}, depth = {}, mv = {}, interp = {}, real = {};
    PBuf up_back = {}, up_hudless = {}, down_interp = {}, down_real = {};
    NVSDK_NGX_Parameter *params = NULL;
    NVSDK_NGX_Handle *h = NULL;
    VkQueryPool qp = VK_NULL_HANDLE;
    const int FRAMES = 12;
    double ms[FRAMES] = {0};
    int timed = 0;
    bool good = pimg(&back, W, H, VK_FORMAT_R8G8B8A8_UNORM, color_use) &&
                pimg(&hudless, W, H, VK_FORMAT_R8G8B8A8_UNORM, color_use) &&
                pimg(&depth, RW, RH, VK_FORMAT_D32_SFLOAT,
                     VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                         VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) &&
                pimg(&mv, RW, RH, VK_FORMAT_R16G16_SFLOAT, color_use) &&
                pimg(&interp, W, H, VK_FORMAT_R8G8B8A8_UNORM, color_use) &&
                pimg(&real, W, H, VK_FORMAT_R8G8B8A8_UNORM, color_use) &&
                pbuf(&up_back, (VkDeviceSize)W * H * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT) &&
                pbuf(&up_hudless, (VkDeviceSize)W * H * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT) &&
                pbuf(&down_interp, (VkDeviceSize)W * H * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT) &&
                pbuf(&down_real, (VkDeviceSize)W * H * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    if (!good) LOG("[dlssg] probe: couldn't make its images\n");
    if (good) {
        VkQueryPoolCreateInfo qi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qi.queryType = VK_QUERY_TYPE_TIMESTAMP, qi.queryCount = 2;
        vkCreateQueryPool(g_vk.dev, &qi, NULL, &qp);
        good = ok(NVSDK_NGX_VULKAN_AllocateParameters(&params), "dlssg: allocate parameters");
    }
    if (good) {
        VkCommandBuffer cb = pbegin();
        PImg *all[] = {&back, &hudless, &depth, &mv, &interp, &real};
        for (PImg *p : all) pgeneral(cb, p);
        NVSDK_NGX_DLSSG_Create_Params cp = {};
        cp.Width = W, cp.Height = H, cp.NativeBackbufferFormat = VK_FORMAT_R8G8B8A8_UNORM;
        cp.RenderWidth = RW, cp.RenderHeight = RH;
        r = NGX_VK_CREATE_DLSSG(cb, 1, 1, &h, params, &cp);
        LOG("[dlssg] probe: create %ux%u (render %ux%u): %s (%#x)\n", W, H, RW, RH, result_name(r), (unsigned)r);
        good = psubmit(cb) && r == NVSDK_NGX_Result_Success;
    }
    // A camera that doesn't move (row-major, row vectors): OpenGL-style projection, near 0.1, far 25000, 50 degrees
    const float nr = 0.1f, fr = 25000.0f, fov = 0.8727f, aspect = (float)W / H;
    const float sy_ = 1.0f / tanf(fov / 2), sx_ = sy_ / aspect, A = (fr + nr) / (nr - fr), B = 2 * fr * nr / (nr - fr);
    for (int k = 0; good && k < FRAMES; k++) {
        pframe(up_back.map, W, H, k, true);
        pframe(up_hudless.map, W, H, k, false);
        VkCommandBuffer cb = pbegin();
        pcopy(cb, &up_back, &back, true);
        pcopy(cb, &up_hudless, &hudless, true);
        VkClearDepthStencilValue dz = {0.5f, 0};
        VkImageSubresourceRange dr = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        vkCmdClearDepthStencilImage(cb, depth.img, VK_IMAGE_LAYOUT_GENERAL, &dz, 1, &dr);
        VkClearColorValue zero = {{0, 0, 0, 0}};
        VkImageSubresourceRange cr = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cb, mv.img, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &cr);
        pbarrier(cb);
        NVSDK_NGX_Resource_VK rb = pres(&back, false), rh = pres(&hudless, false), rd = pres(&depth, false),
                              rm = pres(&mv, false), ri = pres(&interp, true), rr = pres(&real, true);
        NVSDK_NGX_VK_DLSSG_Eval_Params ev = {};
        ev.pBackbuffer = &rb, ev.pDepth = &rd, ev.pMVecs = &rm, ev.pHudless = &rh;
        ev.pOutputInterpFrame = &ri, ev.pOutputRealFrame = &rr;
        NVSDK_NGX_DLSSG_Opt_Eval_Params op = {};
        op.multiFrameCount = 1, op.multiFrameIndex = 1;
        float I[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
        float P[4][4] = {{sx_, 0, 0, 0}, {0, sy_, 0, 0}, {0, 0, A, -1}, {0, 0, B, 0}};
        float Pi[4][4] = {{1 / sx_, 0, 0, 0}, {0, 1 / sy_, 0, 0}, {0, 0, 0, 1 / B}, {0, 0, -1, A / B}};
        memcpy(op.cameraViewToClip, P, sizeof P), memcpy(op.clipToCameraView, Pi, sizeof Pi);
        memcpy(op.clipToLensClip, I, sizeof I), memcpy(op.clipToPrevClip, I, sizeof I);
        memcpy(op.prevClipToClip, I, sizeof I);
        op.mvecScale[0] = (float)RW, op.mvecScale[1] = (float)RH;
        op.cameraUp[1] = 1, op.cameraRight[0] = 1, op.cameraFwd[2] = -1;
        op.cameraNear = nr, op.cameraFar = fr, op.cameraFOV = fov, op.cameraAspectRatio = aspect;
        op.cameraMotionIncluded = true;
        op.reset = k == 0;
        op.motionVectorsInvalidValue = 3.0e38f;
        vkCmdResetQueryPool(cb, qp, 0, 2);
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
        r = NGX_VK_EVALUATE_DLSSG(cb, h, params, &ev, &op);
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
        if (k == FRAMES - 1) {
            pbarrier(cb);
            pcopy(cb, &down_interp, &interp, false);
            pcopy(cb, &down_real, &real, false);
        }
        if (r != NVSDK_NGX_Result_Success) {
            LOG("[dlssg] probe: evaluate %d: %s (%#x)\n", k, result_name(r), (unsigned)r);
            good = false;
        }
        good = psubmit(cb) && good;
        uint64_t ts[2];
        if (good && vkGetQueryPoolResults(g_vk.dev, qp, 0, 2, sizeof ts, ts, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS)
            ms[timed++] = (ts[1] - ts[0]) * (double)g_vk.props.limits.timestampPeriod / 1e6;
    }
    if (good) {
        char line[512];
        int at = 0;
        for (int i = 0; i < timed; i++) at += snprintf(line + at, sizeof line - at, " %.2f", ms[i]);
        LOG("[dlssg] probe: evaluate ok, GPU ms per frame:%s\n", line);
        _mkdir("build/frames");
        bool s1 = rt_save_png("build/frames/dlssg_probe_interp.png", down_interp.map, W, H);
        bool s2 = rt_save_png("build/frames/dlssg_probe_real.png", down_real.map, W, H);
        LOG("[dlssg] probe: saved build/frames/dlssg_probe_{interp,real}.png: %d %d\n", s1, s2);
    }
    vkDeviceWaitIdle(g_vk.dev);
    if (h) NVSDK_NGX_VULKAN_ReleaseFeature(h);
    if (params) NVSDK_NGX_VULKAN_DestroyParameters(params);
    if (qp) vkDestroyQueryPool(g_vk.dev, qp, NULL);
    PImg *imgs[] = {&back, &hudless, &depth, &mv, &interp, &real};
    for (PImg *p : imgs) pimg_free(p);
    PBuf *bufs[] = {&up_back, &up_hudless, &down_interp, &down_real};
    for (PBuf *b : bufs) pbuf_free(b);
}

bool vk_dlss_init(void) {
    if (g_initialized) return g_available;
    g_initialized = true;
    if (!g_vk.phys || !g_vk.dev) return false;

    char root[1024];
    const char *data = rt_data_root();
    wchar_t app_data[1024];
    if (!data || mbstowcs(app_data, data, sizeof(app_data) / sizeof(*app_data) - 1) == (size_t)-1) {
        LOG("[dlss] invalid application data path\n");
        return false;
    }
    app_data[(sizeof(app_data) / sizeof(*app_data)) - 1] = 0;

    // Prefer the redistributable next to the executable, then try the sibling
    // SDK checkout used by source builds. Supplying both paths avoids relying
    // on the process working directory (which differs between the launcher and
    // a direct executable launch).
    static const wchar_t *paths[2];
    static wchar_t dll_path[1024], sdk_path[1024];
    unsigned npaths = 0;
    if (rt_resource("nvngx_dlss.dll", NULL, root, sizeof(root))) {
        char *slash = strrchr(root, '\\');
        if (!slash) slash = strrchr(root, '/');
        if (slash) {
            *slash = 0;
            if (mbstowcs(dll_path, root, sizeof(dll_path) / sizeof(*dll_path) - 1) != (size_t)-1) {
                dll_path[(sizeof(dll_path) / sizeof(*dll_path)) - 1] = 0;
                paths[npaths++] = dll_path;
                LOG("[dlss] runtime path: %s\n", root);
            }
        }
    }
    if (rt_resource(NULL, "../DLSS/lib/Windows_x86_64/rel", root, sizeof(root)) &&
        mbstowcs(sdk_path, root, sizeof(sdk_path) / sizeof(*sdk_path) - 1) != (size_t)-1) {
        sdk_path[(sizeof(sdk_path) / sizeof(*sdk_path)) - 1] = 0;
        if (npaths < 2) paths[npaths++] = sdk_path;
        LOG("[dlss] SDK path: %s\n", root);
    }
    NVSDK_NGX_FeatureCommonInfo &common = g_common;
    common.PathListInfo.Path = paths;
    common.PathListInfo.Length = npaths;
    bool probe = getenv("SAVAGEWIND_DLSSG_PROBE") != NULL;
    // (the probe: NGX's own log, nvngx.log in the data folder. A LoggingCallback printing to stderr hung NGX's init)
    if (probe) common.LoggingInfo.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_ON;
    wcscpy(g_app_data, app_data);

    // Project IDs must be GUID-like for custom engines. This ID is stable for
    // Savage Wind and does not require an NVIDIA-issued application ID.
    NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_Init_with_ProjectID(
        PROJECT_ID, NVSDK_NGX_ENGINE_TYPE_CUSTOM, ENGINE_VERSION,
        app_data, g_vk.inst, g_vk.phys, g_vk.dev, nullptr, nullptr, &common);
    if (!ok(r, "init")) return false;
    if (!ok(NVSDK_NGX_VULKAN_GetCapabilityParameters(&g_capability_params), "capabilities")) {
        return false;
    }
    if (probe) dlssg_probe();
    int available = 0;
    r = NVSDK_NGX_Parameter_GetI(g_capability_params, NVSDK_NGX_Parameter_SuperSampling_Available, &available);
    g_available = ok(r, "SuperSampling.Available") && available != 0;
    LOG("[dlss] Super Resolution %s on %s\n", g_available ? "available" : "unavailable", g_vk.props.deviceName);
    if (!g_available) {
        g_capability_params = NULL;
    }
    return g_available;
}

void vk_dlss_shutdown(void) {
    if (!g_initialized) return;
    release_feature();
    if (g_available) NVSDK_NGX_VULKAN_Shutdown1(g_vk.dev);
    g_available = false;
    g_capability_params = NULL;
    g_initialized = false;
}

bool vk_dlss_available(void) { return g_available; }

bool vk_dlss_apply(VkCommandBuffer cb, TexBackend *color, TexBackend *depth, TexBackend *motion,
                   TexBackend *output, unsigned quality, float jitter_x, float jitter_y, bool reset) {
    if (!g_available || !cb || !color || !depth || !motion || !output ||
        !color->image || !depth->image || !motion->image || !output->image ||
        color->width == 0 || color->height == 0 || output->width == 0 || output->height == 0)
        return false;

    if (!g_feature || g_in_w != color->width || g_in_h != color->height || g_out_w != output->width ||
        g_out_h != output->height || g_quality != quality || g_color_format != color->format ||
        g_output_format != output->format) {
        release_feature();
        NVSDK_NGX_DLSS_Create_Params create = {};
        create.Feature.InWidth = color->width;
        create.Feature.InHeight = color->height;
        create.Feature.InTargetWidth = output->width;
        create.Feature.InTargetHeight = output->height;
        static const NVSDK_NGX_PerfQuality_Value q[] = {
            NVSDK_NGX_PerfQuality_Value_UltraQuality, NVSDK_NGX_PerfQuality_Value_MaxQuality,
            NVSDK_NGX_PerfQuality_Value_Balanced, NVSDK_NGX_PerfQuality_Value_MaxPerf};
        unsigned requested = quality < 4 ? quality : 1;
        create.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_IsHDR | NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;
        // Some driver/NGX combinations reject Ultra Quality at a particular
        // render size. Retry the next conservative mode before disabling DLSS.
        static const unsigned fallback[] = {0, 1, 2, 3};
        bool created = false;
        for (unsigned attempt = 0; attempt < 4 && !created; attempt++) {
            unsigned mode = attempt == 0 ? requested : fallback[attempt];
            create.Feature.InPerfQualityValue = q[mode];
            if (!ok(NVSDK_NGX_VULKAN_AllocateParameters(&g_params), "allocate parameters")) return false;
            NVSDK_NGX_Result r = NGX_VULKAN_CREATE_DLSS_EXT1(g_vk.dev, cb, 1, 1, &g_feature, g_params, &create);
            if (r == NVSDK_NGX_Result_Success) {
                created = true;
                if (attempt) LOG("[dlss] quality %u rejected; using fallback mode %u\n", requested, mode);
            } else {
                LOG("[dlss] quality mode %u rejected: %s (%#x)\n", mode, result_name(r), (unsigned)r);
                release_feature();
            }
        }
        if (!created) {
            g_available = false;
            return false;
        }
        g_in_w = color->width; g_in_h = color->height; g_out_w = output->width; g_out_h = output->height;
        g_quality = quality; g_color_format = color->format; g_output_format = output->format;
    }

    NVSDK_NGX_Resource_VK in_color = image_resource(color, false);
    NVSDK_NGX_Resource_VK in_depth = image_resource(depth, false);
    NVSDK_NGX_Resource_VK in_motion = image_resource(motion, false);
    NVSDK_NGX_Resource_VK out = image_resource(output, true);
    NVSDK_NGX_VK_DLSS_Eval_Params eval = {};
    eval.Feature.pInColor = &in_color;
    eval.Feature.pInOutput = &out;
    eval.pInDepth = &in_depth;
    eval.pInMotionVectors = &in_motion;
    eval.InJitterOffsetX = jitter_x;
    eval.InJitterOffsetY = jitter_y;
    eval.InRenderSubrectDimensions.Width = color->width;
    eval.InRenderSubrectDimensions.Height = color->height;
    eval.InReset = reset ? 1 : 0;
    // (the vectors are uv in the previous frame - uv in this one, exec.c cam_motion; NGX takes them in render pixels)
    eval.InMVScaleX = (float)motion->width;
    eval.InMVScaleY = (float)motion->height;
    NVSDK_NGX_Result r = NGX_VULKAN_EVALUATE_DLSS_EXT(cb, g_feature, g_params, &eval);
    if (!ok(r, "evaluate")) {
        release_feature();
        return false;
    }
    return true;
}

#else

bool vk_dlss_compiled(void) { return false; }
void vk_dlss_required_instance_extensions(const char ***names, unsigned *count) {
    if (names) *names = NULL;
    if (count) *count = 0;
}
void vk_dlss_required_device_extensions(const char ***names, unsigned *count) {
    if (names) *names = NULL;
    if (count) *count = 0;
}
bool vk_dlss_init(void) { return false; }
void vk_dlss_shutdown(void) {}
bool vk_dlss_available(void) { return false; }
bool vk_dlss_apply(VkCommandBuffer, TexBackend *, TexBackend *, TexBackend *, TexBackend *, unsigned, float, float, bool) {
    return false;
}

#endif
