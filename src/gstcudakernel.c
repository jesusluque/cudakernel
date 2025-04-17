/*
 * GStreamer CUDA Kernel Plugin
 * 
 * This plugin loads and executes CUDA kernels at runtime within a GStreamer pipeline.
 * It supports loading PTX or CUBIN files and executing arbitrary kernel functions.
 *
 * Author: Claude
 * License: LGPL-2.1
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>
/* Add CUDA runtime for dim3 type */
#include <cuda_runtime.h>
#include <json-glib/json-glib.h>

GST_DEBUG_CATEGORY_STATIC (gst_cuda_kernel_debug);
#define GST_CAT_DEFAULT gst_cuda_kernel_debug

#define MAX_KERNEL_PARAMS 16

typedef struct _GstCudaKernelParam {
  const char *name;
  GType type;
  union {
    gint int_val;
    gfloat float_val;
    gdouble double_val;
    gboolean bool_val;
    gpointer ptr_val;
  } data;
  gboolean is_pointer;
} GstCudaKernelParam;

/* Properties */
enum
{
  PROP_0,
  PROP_KERNEL_PATH,
  PROP_KERNEL_FUNCTION,
  PROP_KERNEL_PARAMETERS,
  PROP_DEVICE_ID,
  PROP_BLOCK_SIZE_X,
  PROP_BLOCK_SIZE_Y
};

/* Supported formats */
#define VIDEO_CAPS GST_VIDEO_CAPS_MAKE("{ RGBA, BGRA, RGB, BGR, I420, YV12, NV12, NV21 }")

#define GST_TYPE_CUDA_KERNEL \
  (gst_cuda_kernel_get_type())
#define GST_CUDA_KERNEL(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_CUDA_KERNEL,GstCudaKernel))
#define GST_CUDA_KERNEL_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_CUDA_KERNEL,GstCudaKernelClass))
#define GST_IS_CUDA_KERNEL(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_CUDA_KERNEL))
#define GST_IS_CUDA_KERNEL_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_CUDA_KERNEL))

typedef struct _GstCudaKernel GstCudaKernel;
typedef struct _GstCudaKernelClass GstCudaKernelClass;

/* Plugin structure */
struct _GstCudaKernel
{
  GstBaseTransform base_transform;
  
  /* Properties */
  gchar *kernel_path;
  gchar *kernel_function;
  gchar *kernel_parameters_json;
  gint device_id;
  gint block_size_x;
  gint block_size_y;
  
  /* CUDA resources */
  CUcontext cu_ctx;
  CUmodule cu_module;
  CUfunction cu_function;
  gboolean cuda_initialized;
  
  /* Format info */
  GstVideoInfo in_info;
  GstVideoInfo out_info;
  
  /* Kernel parameters */
  GstCudaKernelParam params[MAX_KERNEL_PARAMS];
  gint num_params;
  
  /* Buffer for async operations */
  CUstream cu_stream;
  
  /* For zero-copy */
  gboolean has_cuda_memory;
  
  /* Last modification time of the kernel file */
  GDateTime *last_mtime;
};

/* Plugin class */
struct _GstCudaKernelClass
{
  GstBaseTransformClass parent_class;
};

/* Standard GObject boilerplate */
static GType gst_cuda_kernel_get_type (void);

static void gst_cuda_kernel_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_cuda_kernel_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static void gst_cuda_kernel_dispose (GObject * object);
static void gst_cuda_kernel_finalize (GObject * object);

/* GstBaseTransform method implementations */
static GstCaps *gst_cuda_kernel_transform_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * filter);
static gboolean gst_cuda_kernel_set_caps (GstBaseTransform * trans,
    GstCaps * incaps, GstCaps * outcaps);
static gboolean gst_cuda_kernel_start (GstBaseTransform * trans);
static gboolean gst_cuda_kernel_stop (GstBaseTransform * trans);
static GstFlowReturn gst_cuda_kernel_transform (GstBaseTransform * trans,
    GstBuffer * inbuf, GstBuffer * outbuf);
static gboolean gst_cuda_kernel_propose_allocation (GstBaseTransform * trans,
    GstQuery * decide_query, GstQuery * query);
static gboolean gst_cuda_kernel_decide_allocation (GstBaseTransform * trans,
    GstQuery * query);

/* Helper functions */
static gboolean gst_cuda_kernel_init_cuda (GstCudaKernel * filter);
static gboolean gst_cuda_kernel_load_module (GstCudaKernel * filter);
static void gst_cuda_kernel_unload_module (GstCudaKernel * filter);
static gboolean gst_cuda_kernel_parse_parameters (GstCudaKernel * filter);
static gboolean gst_cuda_kernel_reload_if_needed (GstCudaKernel * filter);

/* Plugin registration */
static gboolean plugin_init (GstPlugin * plugin);

/* Define plugin */
#define PLUGIN_NAME "cudakernel"
#define PLUGIN_DESC "CUDA Kernel Element"
#define PLUGIN_VERSION "1.0"

/* Implementation starts here */

G_DEFINE_TYPE (GstCudaKernel, gst_cuda_kernel, GST_TYPE_BASE_TRANSFORM);

/* Initialize class */
static void
gst_cuda_kernel_class_init (GstCudaKernelClass * klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstElementClass *element_class = GST_ELEMENT_CLASS (klass);
  GstBaseTransformClass *base_transform_class = GST_BASE_TRANSFORM_CLASS (klass);
  
  GST_DEBUG_CATEGORY_INIT (gst_cuda_kernel_debug, "cudakernel", 0, "CUDA Kernel Element");
  
  /* Set up plugin properties */
  gobject_class->set_property = gst_cuda_kernel_set_property;
  gobject_class->get_property = gst_cuda_kernel_get_property;
  gobject_class->dispose = gst_cuda_kernel_dispose;
  gobject_class->finalize = gst_cuda_kernel_finalize;
  
  g_object_class_install_property (gobject_class, PROP_KERNEL_PATH,
      g_param_spec_string ("kernel-path", "Kernel Path",
          "Path to the CUDA kernel file (PTX or CUBIN)",
          NULL, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
          
  g_object_class_install_property (gobject_class, PROP_KERNEL_FUNCTION,
      g_param_spec_string ("kernel-function", "Kernel Function",
          "Name of the kernel function to call",
          "process", G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
          
  g_object_class_install_property (gobject_class, PROP_KERNEL_PARAMETERS,
      g_param_spec_string ("kernel-parameters", "Kernel Parameters",
          "JSON-formatted parameters to pass to the kernel",
          "{}", G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
          
  g_object_class_install_property (gobject_class, PROP_DEVICE_ID,
      g_param_spec_int ("device-id", "Device ID",
          "CUDA device ID to use",
          0, G_MAXINT, 0, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
          
  g_object_class_install_property (gobject_class, PROP_BLOCK_SIZE_X,
      g_param_spec_int ("block-size-x", "Block Size X",
          "CUDA thread block size X dimension",
          1, 1024, 16, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
          
  g_object_class_install_property (gobject_class, PROP_BLOCK_SIZE_Y,
      g_param_spec_int ("block-size-y", "Block Size Y",
          "CUDA thread block size Y dimension",
          1, 1024, 16, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
          
  /* Configure element details */
  gst_element_class_set_static_metadata (element_class,
      "CUDA Kernel Processor",
      "Filter/Effect/Video",
      "Applies custom CUDA kernels to video frames",
      "Claude <claude@anthropic.com>");
  
  /* Configure pad templates */
  gst_element_class_add_pad_template (element_class,
      gst_pad_template_new ("src", GST_PAD_SRC, GST_PAD_ALWAYS,
          gst_caps_from_string (VIDEO_CAPS)));
  gst_element_class_add_pad_template (element_class,
      gst_pad_template_new ("sink", GST_PAD_SINK, GST_PAD_ALWAYS,
          gst_caps_from_string (VIDEO_CAPS)));
  
  /* Set transform virtual methods */
  base_transform_class->transform_caps = GST_DEBUG_FUNCPTR (gst_cuda_kernel_transform_caps);
  base_transform_class->set_caps = GST_DEBUG_FUNCPTR (gst_cuda_kernel_set_caps);
  base_transform_class->start = GST_DEBUG_FUNCPTR (gst_cuda_kernel_start);
  base_transform_class->stop = GST_DEBUG_FUNCPTR (gst_cuda_kernel_stop);
  base_transform_class->transform = GST_DEBUG_FUNCPTR (gst_cuda_kernel_transform);
  base_transform_class->propose_allocation = GST_DEBUG_FUNCPTR (gst_cuda_kernel_propose_allocation);
  base_transform_class->decide_allocation = GST_DEBUG_FUNCPTR (gst_cuda_kernel_decide_allocation);
  
  /* By default, operate in place to avoid unnecessary copies */
  base_transform_class->transform_ip_on_passthrough = FALSE;
}

/* Initialize instance */
static void
gst_cuda_kernel_init (GstCudaKernel * filter)
{
  filter->kernel_path = NULL;
  filter->kernel_function = g_strdup ("process");
  filter->kernel_parameters_json = g_strdup ("{}");
  filter->device_id = 0;
  filter->block_size_x = 16;
  filter->block_size_y = 16;
  
  filter->cuda_initialized = FALSE;
  filter->cu_ctx = NULL;
  filter->cu_module = NULL;
  filter->cu_function = NULL;
  filter->cu_stream = NULL;
  filter->has_cuda_memory = FALSE;
  
  filter->num_params = 0;
  filter->last_mtime = NULL;
  
  /* Set transform properties */
  gst_base_transform_set_in_place (GST_BASE_TRANSFORM (filter), TRUE);
  gst_base_transform_set_passthrough (GST_BASE_TRANSFORM (filter), FALSE);
}

static void
gst_cuda_kernel_dispose (GObject * object)
{
  GstCudaKernel *filter = GST_CUDA_KERNEL (object);
  
  gst_cuda_kernel_unload_module (filter);
  
  if (filter->cu_stream) {
    cuStreamDestroy(filter->cu_stream);
    filter->cu_stream = NULL;
  }
  
  if (filter->cu_ctx) {
    cuCtxDestroy(filter->cu_ctx);
    filter->cu_ctx = NULL;
  }

  G_OBJECT_CLASS (gst_cuda_kernel_parent_class)->dispose (object);
}

static void
gst_cuda_kernel_finalize (GObject * object)
{
  GstCudaKernel *filter = GST_CUDA_KERNEL (object);
  
  g_free (filter->kernel_path);
  g_free (filter->kernel_function);
  g_free (filter->kernel_parameters_json);
  
  if (filter->last_mtime) {
    g_date_time_unref(filter->last_mtime);
  }
  
  G_OBJECT_CLASS (gst_cuda_kernel_parent_class)->finalize (object);
}

static void
gst_cuda_kernel_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstCudaKernel *filter = GST_CUDA_KERNEL (object);

  switch (prop_id) {
    case PROP_KERNEL_PATH:
      if (filter->kernel_path)
        g_free (filter->kernel_path);
      filter->kernel_path = g_value_dup_string (value);
      break;
    case PROP_KERNEL_FUNCTION:
      if (filter->kernel_function)
        g_free (filter->kernel_function);
      filter->kernel_function = g_value_dup_string (value);
      break;
    case PROP_KERNEL_PARAMETERS:
      if (filter->kernel_parameters_json)
        g_free (filter->kernel_parameters_json);
      filter->kernel_parameters_json = g_value_dup_string (value);
      if (filter->cuda_initialized) {
        gst_cuda_kernel_parse_parameters (filter);
      }
      break;
    case PROP_DEVICE_ID:
      filter->device_id = g_value_get_int (value);
      break;
    case PROP_BLOCK_SIZE_X:
      filter->block_size_x = g_value_get_int (value);
      break;
    case PROP_BLOCK_SIZE_Y:
      filter->block_size_y = g_value_get_int (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_cuda_kernel_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstCudaKernel *filter = GST_CUDA_KERNEL (object);

  switch (prop_id) {
    case PROP_KERNEL_PATH:
      g_value_set_string (value, filter->kernel_path);
      break;
    case PROP_KERNEL_FUNCTION:
      g_value_set_string (value, filter->kernel_function);
      break;
    case PROP_KERNEL_PARAMETERS:
      g_value_set_string (value, filter->kernel_parameters_json);
      break;
    case PROP_DEVICE_ID:
      g_value_set_int (value, filter->device_id);
      break;
    case PROP_BLOCK_SIZE_X:
      g_value_set_int (value, filter->block_size_x);
      break;
    case PROP_BLOCK_SIZE_Y:
      g_value_set_int (value, filter->block_size_y);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static GstCaps *
gst_cuda_kernel_transform_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * filter)
{
  GstCaps *ret;
  
  /* We currently just pass the same caps - assuming kernel doesn't change format */
  ret = gst_caps_copy (caps);
  
  /* Apply any additional filter */
  if (filter) {
    GstCaps *intersection;
    intersection = gst_caps_intersect_full (filter, ret, GST_CAPS_INTERSECT_FIRST);
    gst_caps_unref (ret);
    ret = intersection;
  }
  
  GST_DEBUG_OBJECT (trans, "transformed caps from %" GST_PTR_FORMAT " to %" GST_PTR_FORMAT,
        caps, ret);
        
  return ret;
}

static gboolean
gst_cuda_kernel_set_caps (GstBaseTransform * trans, GstCaps * incaps,
    GstCaps * outcaps)
{
  GstCudaKernel *filter = GST_CUDA_KERNEL (trans);
  
  /* Store video info for both sink and src pads */
  if (!gst_video_info_from_caps (&filter->in_info, incaps)) {
    GST_ERROR_OBJECT (filter, "Failed to parse input caps");
    return FALSE;
  }
  
  if (!gst_video_info_from_caps (&filter->out_info, outcaps)) {
    GST_ERROR_OBJECT (filter, "Failed to parse output caps");
    return FALSE;
  }
  
  /* Check for CUDA memory */
  GstCapsFeatures *features = gst_caps_get_features (incaps, 0);
  filter->has_cuda_memory = features && 
      gst_caps_features_contains (features, GST_CAPS_FEATURE_MEMORY_CUDA_MEMORY);
  
  if (filter->has_cuda_memory) {
    GST_INFO_OBJECT (filter, "Detected CUDA memory buffers");
  } else {
    GST_INFO_OBJECT (filter, "Using system memory buffers");
  }
  
  return TRUE;
}

static gboolean
gst_cuda_kernel_start (GstBaseTransform * trans)
{
  GstCudaKernel *filter = GST_CUDA_KERNEL (trans);
  
  /* Initialize CUDA */
  if (!gst_cuda_kernel_init_cuda (filter)) {
    GST_ERROR_OBJECT (filter, "Failed to initialize CUDA");
    return FALSE;
  }
  
  /* Load the kernel module if path is provided */
  if (filter->kernel_path) {
    if (!gst_cuda_kernel_load_module (filter)) {
      GST_ERROR_OBJECT (filter, "Failed to load CUDA kernel module");
      return FALSE;
    }
  } else {
    GST_WARNING_OBJECT (filter, "No kernel path provided, plugin will passthrough until set");
  }
  
  /* Parse initial parameters */
  gst_cuda_kernel_parse_parameters (filter);
  
  return TRUE;
}

static gboolean
gst_cuda_kernel_stop (GstBaseTransform * trans)
{
  GstCudaKernel *filter = GST_CUDA_KERNEL (trans);
  
  /* Unload the kernel module */
  gst_cuda_kernel_unload_module (filter);
  
  /* Don't destroy the CUDA context yet - that happens in dispose() */
  
  return TRUE;
}

static GstFlowReturn
gst_cuda_kernel_transform (GstBaseTransform * trans, GstBuffer * inbuf,
    GstBuffer * outbuf)
{
  GstCudaKernel *filter = GST_CUDA_KERNEL (trans);
  GstFlowReturn ret = GST_FLOW_OK;
  GstMapInfo in_map, out_map;
  CUdeviceptr d_in, d_out;
  CUresult result;
  
  /* If no module is loaded or no kernel function set, just passthrough */
  if (!filter->cu_module || !filter->cu_function) {
    GST_LOG_OBJECT (filter, "No CUDA kernel loaded, passing through");
    if (inbuf != outbuf) {
      gst_buffer_copy_into(outbuf, inbuf, GST_BUFFER_COPY_ALL, 0, -1);
    }
    return GST_FLOW_OK;
  }
  
  /* Check if we need to reload the kernel */
  if (!gst_cuda_kernel_reload_if_needed (filter)) {
    GST_ERROR_OBJECT (filter, "Failed to reload CUDA kernel");
    return GST_FLOW_ERROR;
  }
  
  /* First mandatory params */
  int width = GST_VIDEO_INFO_WIDTH(&filter->in_info);
  int height = GST_VIDEO_INFO_HEIGHT(&filter->in_info);
  int params_offset = 0;
  
  /* Map buffers */
  if (filter->has_cuda_memory) {
    /* Get CUDA device pointers directly */
    if (!gst_buffer_map (inbuf, &in_map, GST_MAP_READ | GST_MAP_CUDA)) {
      GST_ERROR_OBJECT (filter, "Failed to map input buffer for CUDA access");
      return GST_FLOW_ERROR;
    }
    d_in = (CUdeviceptr)in_map.data;
    
    if (inbuf != outbuf) {
      if (!gst_buffer_map (outbuf, &out_map, GST_MAP_WRITE | GST_MAP_CUDA)) {
        gst_buffer_unmap (inbuf, &in_map);
        GST_ERROR_OBJECT (filter, "Failed to map output buffer for CUDA access");
        return GST_FLOW_ERROR;
      }
      d_out = (CUdeviceptr)out_map.data;
    } else {
      /* In-place processing */
      d_out = d_in;
    }
  } else {
    /* System memory, need to upload to GPU */
    if (!gst_buffer_map (inbuf, &in_map, GST_MAP_READ)) {
      GST_ERROR_OBJECT (filter, "Failed to map input buffer");
      return GST_FLOW_ERROR;
    }
    
    /* Upload to GPU */
    CUdeviceptr d_temp = 0;
    size_t buffer_size = in_map.size;
    
    result = cuMemAlloc(&d_temp, buffer_size);
    if (result != CUDA_SUCCESS) {
      gst_buffer_unmap (inbuf, &in_map);
      GST_ERROR_OBJECT (filter, "Failed to allocate GPU memory: %d", result);
      return GST_FLOW_ERROR;
    }
    
    result = cuMemcpyHtoDAsync(d_temp, in_map.data, buffer_size, filter->cu_stream);
    if (result != CUDA_SUCCESS) {
      cuMemFree(d_temp);
      gst_buffer_unmap (inbuf, &in_map);
      GST_ERROR_OBJECT (filter, "Failed to upload data to GPU: %d", result);
      return GST_FLOW_ERROR;
    }
    
    d_in = d_temp;
    
    /* For the output buffer */
    if (inbuf != outbuf) {
      if (!gst_buffer_map (outbuf, &out_map, GST_MAP_WRITE)) {
        cuMemFree(d_temp);
        gst_buffer_unmap (inbuf, &in_map);
        GST_ERROR_OBJECT (filter, "Failed to map output buffer");
        return GST_FLOW_ERROR;
      }
      
      /* Allocate GPU memory for output */
      CUdeviceptr d_out_temp = 0;
      result = cuMemAlloc(&d_out_temp, out_map.size);
      if (result != CUDA_SUCCESS) {
        cuMemFree(d_temp);
        gst_buffer_unmap (inbuf, &in_map);
        gst_buffer_unmap (outbuf, &out_map);
        GST_ERROR_OBJECT (filter, "Failed to allocate GPU output memory: %d", result);
        return GST_FLOW_ERROR;
      }
      
      d_out = d_out_temp;
    } else {
      /* In-place processing */
      d_out = d_in;
    }
  }
  
  /* Prepare parameters for kernel launch */
  void *args[MAX_KERNEL_PARAMS + 4];  /* +4 for width, height, in_ptr, out_ptr */
  
  /* Set mandatory parameters first */
  args[0] = &d_in;      /* Input buffer device pointer */
  args[1] = &d_out;     /* Output buffer device pointer */
  args[2] = &width;     /* Frame width */
  args[3] = &height;    /* Frame height */
  params_offset = 4;    /* Start custom params after these */
  
  /* Add custom parameters from JSON */
  for (int i = 0; i < filter->num_params && params_offset < MAX_KERNEL_PARAMS + 4; i++) {
    GstCudaKernelParam *param = &filter->params[i];
    
    switch (param->type) {
      case G_TYPE_INT:
        args[params_offset] = &param->data.int_val;
        break;
      case G_TYPE_FLOAT:
        args[params_offset] = &param->data.float_val;
        break;
      case G_TYPE_DOUBLE:
        args[params_offset] = &param->data.double_val;
        break;
      case G_TYPE_BOOLEAN:
        args[params_offset] = &param->data.bool_val;
        break;
      case G_TYPE_POINTER:
        args[params_offset] = &param->data.ptr_val;
        break;
      default:
        GST_WARNING_OBJECT(filter, "Unsupported parameter type for %s", param->name);
        continue;
    }
    
    params_offset++;
  }
  
  /* Launch the kernel */
  dim3 block_size;
  block_size.x = filter->block_size_x;
  block_size.y = filter->block_size_y;
  block_size.z = 1;
  
  dim3 grid_size;
  grid_size.x = (width + block_size.x - 1) / block_size.x;
  grid_size.y = (height + block_size.y - 1) / block_size.y;
  grid_size.z = 1;
  
  result = cuLaunchKernel(filter->cu_function,
                         grid_size.x, grid_size.y, grid_size.z,
                         block_size.x, block_size.y, block_size.z,
                         0, filter->cu_stream, args, NULL);
                         
  if (result != CUDA_SUCCESS) {
    GST_ERROR_OBJECT(filter, "Failed to launch CUDA kernel: %d", result);
    ret = GST_FLOW_ERROR;
    goto cleanup;
  }
  
  /* If using system memory, need to download results back */
  if (!filter->has_cuda_memory) {
    /* Wait for kernel to finish */
    result = cuStreamSynchronize(filter->cu_stream);
    if (result != CUDA_SUCCESS) {
      GST_ERROR_OBJECT(filter, "Failed to synchronize CUDA stream: %d", result);
      ret = GST_FLOW_ERROR;
      goto cleanup;
    }
    
    /* Download results */
    if (inbuf != outbuf) {
      result = cuMemcpyDtoH(out_map.data, d_out, out_map.size);
      if (result != CUDA_SUCCESS) {
        GST_ERROR_OBJECT(filter, "Failed to download results from GPU: %d", result);
        ret = GST_FLOW_ERROR;
      }
    } else {
      result = cuMemcpyDtoH(in_map.data, d_in, in_map.size);
      if (result != CUDA_SUCCESS) {
        GST_ERROR_OBJECT(filter, "Failed to download results from GPU: %d", result);
        ret = GST_FLOW_ERROR;
      }
    }
  } else {
    /* Just synchronize the stream to ensure kernel completion */
    result = cuStreamSynchronize(filter->cu_stream);
    if (result != CUDA_SUCCESS) {
      GST_ERROR_OBJECT(filter, "Failed to synchronize CUDA stream: %d", result);
      ret = GST_FLOW_ERROR;
    }
  }

cleanup:
  /* Clean up resources */
  if (!filter->has_cuda_memory) {
    if (d_in)
      cuMemFree(d_in);
    
    if (inbuf != outbuf && d_out)
      cuMemFree(d_out);
  }
  
  if (inbuf != outbuf) {
    gst_buffer_unmap(outbuf, &out_map);
  }
  
  gst_buffer_unmap(inbuf, &in_map);
  
  return ret;
}

static gboolean
gst_cuda_kernel_propose_allocation (GstBaseTransform * trans,
    GstQuery * decide_query, GstQuery * query)
{
  GstCudaKernel *filter = GST_CUDA_KERNEL (trans);
  GstStructure *config;
  GstCaps *caps;
  guint size;
  gboolean need_pool;

  gst_query_parse_allocation (query, &caps, &need_pool);
  
  /* Check if upstream supports CUDA memory */
  if (need_pool) {
    GstVideoInfo info;
    
    if (!gst_video_info_from_caps (&info, caps)) {
      GST_ERROR_OBJECT (filter, "Invalid caps for allocation query");
      return FALSE;
    }
    
    size = GST_VIDEO_INFO_SIZE (&info);
    
    /* We don't actually create a CUDA buffer pool here,
     * we'll let upstream elements handle that or use system memory */
    GstBufferPool *pool = gst_video_buffer_pool_new ();
    config = gst_buffer_pool_get_config (pool);
    gst_buffer_pool_config_set_params (config, caps, size, 2, 0);
    
    if (!gst_buffer_pool_set_config (pool, config)) {
      GST_ERROR_OBJECT (filter, "Failed to configure buffer pool");
      gst_object_unref (pool);
      return FALSE;
    }
    
    gst_query_add_allocation_pool (query, pool, size, 2, 0);
    gst_object_unref (pool);
  }
  
  /* Inform that we can handle CUDA memory if available */
  GstCapsFeatures *features = gst_caps_features_new (GST_CAPS_FEATURE_MEMORY_CUDA_MEMORY, NULL);
  gst_query_add_allocation_meta (query, GST_VIDEO_META_API_TYPE, NULL);
  gst_query_add_allocation_meta (query, GST_CUDA_META_API_TYPE, NULL);
  gst_caps_features_free (features);
  
  return TRUE;
}

static gboolean
gst_cuda_kernel_decide_allocation (GstBaseTransform * trans, GstQuery * query)
{
  /* We just accept whatever allocation is proposed */
  return TRUE;
}

static gboolean
gst_cuda_kernel_init_cuda (GstCudaKernel * filter)
{
  CUresult result;
  CUdevice device;
  int device_count;
  
  /* Initialize CUDA driver API */
  result = cuInit(0);
  if (result != CUDA_SUCCESS) {
    GST_ERROR_OBJECT (filter, "Failed to initialize CUDA: %d", result);
    return FALSE;
  }
  
  /* Check how many devices are available */
  result = cuDeviceGetCount(&device_count);
  if (result != CUDA_SUCCESS || device_count == 0) {
    GST_ERROR_OBJECT (filter, "No CUDA devices found");
    return FALSE;
  }
  
  /* Validate device ID */
  if (filter->device_id >= device_count) {
    GST_ERROR_OBJECT (filter, "Requested device ID %d exceeds available devices (%d)",
        filter->device_id, device_count);
    return FALSE;
  }
  
  /* Get device handle */
  result = cuDeviceGet(&device, filter->device_id);
  if (result != CUDA_SUCCESS) {
    GST_ERROR_OBJECT (filter, "Failed to get CUDA device: %d", result);
    return FALSE;
  }
  
  /* Create context */
  result = cuCtxCreate(&filter->cu_ctx, CU_CTX_SCHED_AUTO, device);
  if (result != CUDA_SUCCESS) {
    GST_ERROR_OBJECT (filter, "Failed to create CUDA context: %d", result);
    return FALSE;
  }
  
  /* Create stream for async operations */
  result = cuStreamCreate(&filter->cu_stream, CU_STREAM_DEFAULT);
  if (result != CUDA_SUCCESS) {
    GST_ERROR_OBJECT (filter, "Failed to create CUDA stream: %d", result);
    
    /* Clean up context */
    cuCtxDestroy(filter->cu_ctx);
    filter->cu_ctx = NULL;
    return FALSE;
  }
  
  /* Log success */
  char device_name[256];
  cuDeviceGetName(device_name, sizeof(device_name), device);
  GST_INFO_OBJECT (filter, "CUDA initialized on device %d: %s", filter->device_id, device_name);
  
  filter->cuda_initialized = TRUE;
  return TRUE;
}

static gboolean
gst_cuda_kernel_load_module (GstCudaKernel * filter)
{
  CUresult result;
  GError *error = NULL;
  
  /* Check parameters */
  if (!filter->cuda_initialized) {
    GST_ERROR_OBJECT (filter, "CUDA not initialized");
    return FALSE;
  }
  
  if (!filter->kernel_path) {
    GST_ERROR_OBJECT (filter, "No kernel path specified");
    return FALSE;
  }
  
  /* First unload any existing module */
  gst_cuda_kernel_unload_module (filter);
  
  /* Check if file exists and is readable */
  if (!g_file_test (filter->kernel_path, G_FILE_TEST_EXISTS | G_FILE_TEST_IS_REGULAR)) {
    GST_ERROR_OBJECT (filter, "Kernel file '%s' does not exist or is not a regular file", 
        filter->kernel_path);
    return FALSE;
  }
  
  /* Get file modification time for reload detection */
  GFile *file = g_file_new_for_path (filter->kernel_path);
  GFileInfo *file_info = g_file_query_info (file, 
                                          G_FILE_ATTRIBUTE_TIME_MODIFIED,
                                          G_FILE_QUERY_INFO_NONE, 
                                          NULL, 
                                          &error);
  if (error) {
    GST_WARNING_OBJECT (filter, "Could not get file info: %s", error->message);
    g_error_free (error);
  } else {
    if (filter->last_mtime) {
      g_date_time_unref(filter->last_mtime);
    }
    filter->last_mtime = g_file_info_get_modification_date_time(file_info);
    g_object_unref (file_info);
  }
  g_object_unref (file);
  
  /* Load the file */
  gchar *ptx_data = NULL;
  gsize ptx_size = 0;
  
  if (!g_file_get_contents (filter->kernel_path, &ptx_data, &ptx_size, &error)) {
    GST_ERROR_OBJECT (filter, "Failed to read kernel file: %s", error->message);
    g_error_free (error);
    return FALSE;
  }
  
  /* Create module */
  result = cuModuleLoadData(&filter->cu_module, ptx_data);
  g_free (ptx_data);
  
  if (result != CUDA_SUCCESS) {
    GST_ERROR_OBJECT (filter, "Failed to load CUDA module: %d", result);
    return FALSE;
  }
  
  /* Get function */
  result = cuModuleGetFunction(&filter->cu_function, filter->cu_module, filter->kernel_function);
  if (result != CUDA_SUCCESS) {
    GST_ERROR_OBJECT (filter, "Failed to get kernel function '%s': %d", 
        filter->kernel_function, result);
    
    /* Unload module */
    cuModuleUnload(filter->cu_module);
    filter->cu_module = NULL;
    return FALSE;
  }
  
  GST_INFO_OBJECT (filter, "Successfully loaded CUDA kernel '%s' from '%s'", 
      filter->kernel_function, filter->kernel_path);
  
  return TRUE;
}

static void
gst_cuda_kernel_unload_module (GstCudaKernel * filter)
{
  /* Unload any existing module */
  if (filter->cu_module) {
    cuModuleUnload(filter->cu_module);
    filter->cu_module = NULL;
    filter->cu_function = NULL;
  }
}

static gboolean
gst_cuda_kernel_parse_parameters (GstCudaKernel * filter)
{
  JsonParser *parser;
  JsonNode *root;
  JsonObject *object;
  GError *error = NULL;
  GList *members, *l;
  
  /* Reset current parameters */
  filter->num_params = 0;
  
  /* No parameters specified */
  if (!filter->kernel_parameters_json || strlen(filter->kernel_parameters_json) == 0) {
    return TRUE;
  }
  
  /* Parse JSON */
  parser = json_parser_new ();
  if (!json_parser_load_from_data (parser, filter->kernel_parameters_json, 
                                  -1, &error)) {
    GST_ERROR_OBJECT (filter, "Failed to parse parameters JSON: %s", error->message);
    g_error_free (error);
    g_object_unref (parser);
    return FALSE;
  }
  
  /* Get root object */
  root = json_parser_get_root (parser);
  if (!root || !JSON_NODE_HOLDS_OBJECT(root)) {
    GST_ERROR_OBJECT (filter, "Parameters must be a JSON object");
    g_object_unref (parser);
    return FALSE;
  }
  
  object = json_node_get_object (root);
  members = json_object_get_members (object);
  
  /* Process each member */
  for (l = members; l != NULL && filter->num_params < MAX_KERNEL_PARAMS; l = l->next) {
    const gchar *name = (const gchar *) l->data;
    JsonNode *node = json_object_get_member (object, name);
    GstCudaKernelParam *param = &filter->params[filter->num_params];
    
    param->name = name;  /* This memory is managed by JsonObject */
    param->is_pointer = FALSE;
    
    /* Handle different types */
    switch (json_node_get_value_type (node)) {
      case G_TYPE_INT64:
        param->type = G_TYPE_INT;
        param->data.int_val = (gint) json_node_get_int (node);
        break;
      case G_TYPE_DOUBLE:
        {
          double val = json_node_get_double (node);
          /* Check if we should convert to float or keep as double */
          if (val == (float)val) {
            param->type = G_TYPE_FLOAT;
            param->data.float_val = (gfloat) val;
          } else {
            param->type = G_TYPE_DOUBLE;
            param->data.double_val = val;
          }
        }
        break;
      case G_TYPE_BOOLEAN:
        param->type = G_TYPE_BOOLEAN;
        param->data.bool_val = json_node_get_boolean (node);
        break;
      case G_TYPE_STRING:
        /* For now we don't support string parameters - could add later */
        GST_WARNING_OBJECT (filter, "String parameters not supported for '%s'", name);
        continue;
      case G_TYPE_INVALID:
      default:
        if (JSON_NODE_HOLDS_NULL(node)) {
          /* Handle null value as zero */
          param->type = G_TYPE_INT;
          param->data.int_val = 0;
        } else if (JSON_NODE_HOLDS_ARRAY(node)) {
          /* Arrays not supported yet */
          GST_WARNING_OBJECT (filter, "Array parameters not supported for '%s'", name);
          continue;
        } else {
          GST_WARNING_OBJECT (filter, "Unsupported parameter type for '%s'", name);
          continue;
        }
        break;
    }
    
    filter->num_params++;
  }
  
  g_list_free (members);
  g_object_unref (parser);
  
  GST_DEBUG_OBJECT (filter, "Parsed %d parameters", filter->num_params);
  return TRUE;
}

static gboolean
gst_cuda_kernel_reload_if_needed (GstCudaKernel * filter)
{
  GError *error = NULL;
  
  /* Skip if no kernel path */
  if (!filter->kernel_path) {
    return TRUE;
  }
  
  /* Get file modification time */
  GFile *file = g_file_new_for_path (filter->kernel_path);
  GFileInfo *file_info = g_file_query_info (file, 
                                          G_FILE_ATTRIBUTE_TIME_MODIFIED,
                                          G_FILE_QUERY_INFO_NONE, 
                                          NULL, 
                                          &error);
  if (error) {
    GST_WARNING_OBJECT (filter, "Could not get file info: %s", error->message);
    g_error_free (error);
    g_object_unref (file);
    return TRUE;  /* Continue even if we can't check */
  }
  
  GDateTime *mtime = g_file_info_get_modification_date_time(file_info);
  g_object_unref (file_info);
  g_object_unref (file);
  
  /* Check if file was modified */
  if (!filter->last_mtime || g_date_time_compare(mtime, filter->last_mtime) != 0) {
    GST_INFO_OBJECT (filter, "Kernel file modified, reloading");
    
    gboolean result = gst_cuda_kernel_load_module (filter);
    
    if (mtime) {
      g_date_time_unref(mtime);
    }
    
    return result;
  }
  
  if (mtime) {
    g_date_time_unref(mtime);
  }
  
  return TRUE;
}

static gboolean
plugin_init (GstPlugin * plugin)
{
  return gst_element_register (plugin, PLUGIN_NAME, GST_RANK_NONE,
      GST_TYPE_CUDA_KERNEL);
}

/* Define the plugin descriptors */
GST_PLUGIN_DEFINE (
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    cudakernel,
    PLUGIN_DESC,
    plugin_init,
    PLUGIN_VERSION,
    "LGPL",
    "GStreamer",
    "https://gstreamer.freedesktop.org/"
)
