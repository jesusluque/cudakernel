/*
 * GStreamer CUDA Kernel Plugin - UPDATED VERSION
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

/* Windows compatibility */
#ifdef _WIN32
#include <windows.h>
#define G_DIR_SEPARATOR '\\'
#define G_DIR_SEPARATOR_S "\\"
#else
#define G_DIR_SEPARATOR '/'
#define G_DIR_SEPARATOR_S "/"
#endif

GST_DEBUG_CATEGORY_STATIC (gst_cuda_kernel_debug);
#define GST_CAT_DEFAULT gst_cuda_kernel_debug

#define MAX_KERNEL_PARAMS 16
#define CUDA_CHECK(stmt) \
  do { \
    CUresult res = (stmt); \
    if (res != CUDA_SUCCESS) { \
      const char *error_name; \
      cuGetErrorName(res, &error_name); \
      GST_ERROR_OBJECT(filter, "%s failed with error %s (%d) at %s:%d", \
        #stmt, error_name, res, __FILE__, __LINE__); \
      return FALSE; \
    } \
  } while (0)

#define CUDA_CHECK_GOTO(stmt, label) \
  do { \
    CUresult res = (stmt); \
    if (res != CUDA_SUCCESS) { \
      const char *error_name; \
      cuGetErrorName(res, &error_name); \
      GST_ERROR_OBJECT(filter, "%s failed with error %s (%d) at %s:%d", \
        #stmt, error_name, res, __FILE__, __LINE__); \
      goto label; \
    } \
  } while (0)

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
  PROP_BLOCK_SIZE_Y,
  PROP_USE_PINNED_MEMORY  /* Option to use pinned memory */
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
  gboolean use_pinned_memory;  /* Option to use pinned memory */
  
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

  /* CUDA device properties */
  size_t available_memory;
  size_t total_memory;
  
  /* Pinned memory buffers for reuse */
  void *pinned_input;
  void *pinned_output;
  size_t pinned_input_size;
  size_t pinned_output_size;
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

/* Helper functions for memory management */
static void gst_cuda_kernel_cleanup_pinned_memory (GstCudaKernel * filter);
static gboolean gst_cuda_kernel_ensure_pinned_buffer (GstCudaKernel * filter, 
    size_t size, void **pinned_ptr, size_t *current_size);
static gboolean gst_cuda_kernel_check_memory_availability (GstCudaKernel * filter, size_t required_size);

/* Plugin registration */
static gboolean plugin_init (GstPlugin * plugin);

/* Define plugin */
#define PLUGIN_NAME "cudakernel"
#define PLUGIN_DESC "CUDA Kernel Element"
#define PLUGIN_VERSION "1.0.2"  /* Bumped version number */

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

  /* Add pinned memory property */
  g_object_class_install_property (gobject_class, PROP_USE_PINNED_MEMORY,
      g_param_spec_boolean ("use-pinned-memory", "Use Pinned Memory",
          "Use pinned host memory for faster transfers (for system memory only)",
          TRUE, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
          
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
  filter->use_pinned_memory = TRUE;  /* Default to using pinned memory */
  
  filter->cuda_initialized = FALSE;
  filter->cu_ctx = NULL;
  filter->cu_module = NULL;
  filter->cu_function = NULL;
  filter->cu_stream = NULL;
  filter->has_cuda_memory = FALSE;
  
  filter->num_params = 0;
  filter->last_mtime = NULL;
  
  /* Initialize memory tracking */
  filter->available_memory = 0;
  filter->total_memory = 0;
  
  /* Initialize pinned memory pointers */
  filter->pinned_input = NULL;
  filter->pinned_output = NULL;
  filter->pinned_input_size = 0;
  filter->pinned_output_size = 0;
  
  /* Set transform properties */
  gst_base_transform_set_in_place (GST_BASE_TRANSFORM (filter), TRUE);
  gst_base_transform_set_passthrough (GST_BASE_TRANSFORM (filter), FALSE);
}

static void
gst_cuda_kernel_dispose (GObject * object)
{
  GstCudaKernel *filter = GST_CUDA_KERNEL (object);
  
  gst_cuda_kernel_unload_module (filter);
  
  /* Clean up pinned memory */
  gst_cuda_kernel_cleanup_pinned_memory (filter);
  
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
    case PROP_USE_PINNED_MEMORY:
      filter->use_pinned_memory = g_value_get_boolean (value);
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
    case PROP_USE_PINNED_MEMORY:
      g_value_set_boolean (value, filter->use_pinned_memory);
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
  
  /* Clean up pinned memory */
  gst_cuda_kernel_cleanup_pinned_memory (filter);
  
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
  CUdeviceptr d_in = 0, d_out = 0;
  CUresult result;
  gboolean allocated_device_memory = FALSE;
  
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
    /* If using system memory, need to handle differently */
    if (!gst_buffer_map (inbuf, &in_map, GST_MAP_READ)) {
      GST_ERROR_OBJECT (filter, "Failed to map input buffer");
      return GST_FLOW_ERROR;
    }
    
    if (inbuf != outbuf) {
      if (!gst_buffer_map (outbuf, &out_map, GST_MAP_WRITE)) {
        gst_buffer_unmap (inbuf, &in_map);
        GST_ERROR_OBJECT (filter, "Failed to map output buffer");
        return GST_FLOW_ERROR;
      }
    }
    
    /* Check if we have enough GPU memory */
    size_t buffer_size = in_map.size;
    size_t total_required = buffer_size;
    
    if (inbuf != outbuf) {
      total_required += out_map.size;
    }
    
    if (!gst_cuda_kernel_check_memory_availability(filter, total_required)) {
      GST_ERROR_OBJECT (filter, "Not enough GPU memory available. Need %zu bytes", 
          total_required);
      
      /* In low memory situations, fallback to passthrough */
      GST_WARNING_OBJECT (filter, "Falling back to passthrough due to memory constraints");
      if (inbuf != outbuf) {
        gst_buffer_copy_into(outbuf, inbuf, GST_BUFFER_COPY_ALL, 0, -1);
        gst_buffer_unmap (outbuf, &out_map);
      }
      gst_buffer_unmap (inbuf, &in_map);
      return GST_FLOW_OK;
    }
    
    /* Always start by allocating device memory first, before attempting pinned memory */
    /* Allocate GPU memory for input */
    result = cuMemAlloc(&d_in, buffer_size);
    if (result != CUDA_SUCCESS) {
      gst_buffer_unmap (inbuf, &in_map);
      if (inbuf != outbuf) {
        gst_buffer_unmap (outbuf, &out_map);
      }
      
      GST_WARNING_OBJECT (filter, "Failed to allocate GPU memory (%d bytes): %d - falling back to passthrough", 
          (int)buffer_size, result);
      
      /* In low memory situations, fallback to passthrough */
      if (inbuf != outbuf) {
        gst_buffer_copy_into(outbuf, inbuf, GST_BUFFER_COPY_ALL, 0, -1);
      }
      return GST_FLOW_OK;
    }
    allocated_device_memory = TRUE;
    
    /* For the output buffer - allocate before attempting pinned memory */
    if (inbuf != outbuf) {
      /* Allocate GPU memory for output */
      result = cuMemAlloc(&d_out, out_map.size);
      if (result != CUDA_SUCCESS) {
        cuMemFree(d_in);
        gst_buffer_unmap (inbuf, &in_map);
        gst_buffer_unmap (outbuf, &out_map);
        
        GST_WARNING_OBJECT (filter, "Failed to allocate GPU output memory: %d - falling back to passthrough", result);
        
        /* In low memory situations, fallback to passthrough */
        gst_buffer_copy_into(outbuf, inbuf, GST_BUFFER_COPY_ALL, 0, -1);
        return GST_FLOW_OK;
      }
    } else {
      /* In-place processing */
      d_out = d_in;
    }
    
    /* Transfer data using either pinned or regular memory */
    if (filter->use_pinned_memory) {
      /* Try to use pinned memory for transfer */
      gboolean pinned_transfer = TRUE;
      
      /* Get or allocate pinned input buffer */
      if (!gst_cuda_kernel_ensure_pinned_buffer(filter, buffer_size, 
          &filter->pinned_input, &filter->pinned_input_size)) {
        /* Pinned allocation failed - fall back to regular transfer */
        pinned_transfer = FALSE;
      }
      
      if (pinned_transfer) {
        /* If we got pinned memory, use it */
        /* Copy data into pinned memory */
        memcpy(filter->pinned_input, in_map.data, buffer_size);
        
        /* Upload from pinned memory */
        result = cuMemcpyHtoDAsync(d_in, filter->pinned_input, buffer_size, filter->cu_stream);
        if (result != CUDA_SUCCESS) {
          cuMemFree(d_in);
          if (inbuf != outbuf) {
            cuMemFree(d_out);
          }
          gst_buffer_unmap (inbuf, &in_map);
          if (inbuf != outbuf) {
            gst_buffer_unmap (outbuf, &out_map);
          }
          GST_ERROR_OBJECT (filter, "Failed to upload data to GPU: %d", result);
          return GST_FLOW_ERROR;
        }
        
        /* For the output buffer, prepare pinned memory if needed */
        if (inbuf != outbuf) {
          if (!gst_cuda_kernel_ensure_pinned_buffer(filter, out_map.size, 
              &filter->pinned_output, &filter->pinned_output_size)) {
            /* Pinned output allocation failed - we'll fall back for the download */
            GST_DEBUG_OBJECT(filter, "Pinned output allocation failed, will use regular transfer for download");
          }
        }
      } else {
        /* Fallback to regular memory transfer for upload */
        GST_DEBUG_OBJECT(filter, "Using regular memory transfer for upload");
        result = cuMemcpyHtoDAsync(d_in, in_map.data, buffer_size, filter->cu_stream);
        if (result != CUDA_SUCCESS) {
          cuMemFree(d_in);
          if (inbuf != outbuf) {
            cuMemFree(d_out);
          }
          gst_buffer_unmap (inbuf, &in_map);
          if (inbuf != outbuf) {
            gst_buffer_unmap (outbuf, &out_map);
          }
          GST_ERROR_OBJECT (filter, "Failed to upload data to GPU: %d", result);
          return GST_FLOW_ERROR;
        }
      }
    } else {
      /* Use regular memory transfer */
      result = cuMemcpyHtoDAsync(d_in, in_map.data, buffer_size, filter->cu_stream);
      if (result != CUDA_SUCCESS) {
        cuMemFree(d_in);
        if (inbuf != outbuf) {
          cuMemFree(d_out);
        }
        gst_buffer_unmap (inbuf, &in_map);
        if (inbuf != outbuf) {
          gst_buffer_unmap (outbuf, &out_map);
        }
        GST_ERROR_OBJECT (filter, "Failed to upload data to GPU: %d", result);
        return GST_FLOW_ERROR;
      }
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
    if (filter->use_pinned_memory && 
        ((inbuf != outbuf && filter->pinned_output != NULL) || 
         (inbuf == outbuf && filter->pinned_input != NULL))) {
      /* Only use pinned memory if we successfully allocated it earlier */
      if (inbuf != outbuf) {
        /* Copy from GPU to pinned memory */
        result = cuMemcpyDtoHAsync(filter->pinned_output, d_out, out_map.size, filter->cu_stream);
        if (result != CUDA_SUCCESS) {
          GST_WARNING_OBJECT(filter, "Failed to download results to pinned memory: %d - falling back to regular transfer", result);
          /* Fall back to regular transfer */
          result = cuMemcpyDtoH(out_map.data, d_out, out_map.size);
          if (result != CUDA_SUCCESS) {
            GST_ERROR_OBJECT(filter, "Failed to download results from GPU: %d", result);
            ret = GST_FLOW_ERROR;
            goto cleanup;
          }
        } else {
          /* Wait for transfer to complete */
          result = cuStreamSynchronize(filter->cu_stream);
          if (result != CUDA_SUCCESS) {
            GST_ERROR_OBJECT(filter, "Failed to synchronize download stream: %d", result);
            ret = GST_FLOW_ERROR;
            goto cleanup;
          }
          
          /* Copy from pinned memory to output buffer */
          memcpy(out_map.data, filter->pinned_output, out_map.size);
        }
      } else {
        /* In-place processing with pinned memory */
        result = cuMemcpyDtoHAsync(filter->pinned_input, d_in, in_map.size, filter->cu_stream);
        if (result != CUDA_SUCCESS) {
          GST_WARNING_OBJECT(filter, "Failed to download results to pinned memory: %d - falling back to regular transfer", result);
          /* Fall back to regular transfer */
          result = cuMemcpyDtoH(in_map.data, d_in, in_map.size);
          if (result != CUDA_SUCCESS) {
            GST_ERROR_OBJECT(filter, "Failed to download results from GPU: %d", result);
            ret = GST_FLOW_ERROR;
            goto cleanup;
          }
        } else {
          /* Wait for transfer to complete */
          result = cuStreamSynchronize(filter->cu_stream);
          if (result != CUDA_SUCCESS) {
            GST_ERROR_OBJECT(filter, "Failed to synchronize download stream: %d", result);
            ret = GST_FLOW_ERROR;
            goto cleanup;
          }
          
          /* Copy from pinned memory to input buffer */
          memcpy(in_map.data, filter->pinned_input, in_map.size);
        }
      }
    } else {
      /* Regular memory transfer */
      if (inbuf != outbuf) {
        result = cuMemcpyDtoH(out_map.data, d_out, out_map.size);
        if (result != CUDA_SUCCESS) {
          GST_ERROR_OBJECT(filter, "Failed to download results from GPU: %d", result);
          ret = GST_FLOW_ERROR;
          goto cleanup;
        }
      } else {
        result = cuMemcpyDtoH(in_map.data, d_in, in_map.size);
        if (result != CUDA_SUCCESS) {
          GST_ERROR_OBJECT(filter, "Failed to download results from GPU: %d", result);
          ret = GST_FLOW_ERROR;
          goto cleanup;
        }
      }
    }
  } else {
    /* Just synchronize the stream to ensure kernel completion */
    result = cuStreamSynchronize(filter->cu_stream);
    if (result != CUDA_SUCCESS) {
      GST_ERROR_OBJECT(filter, "Failed to synchronize CUDA stream: %d", result);
      ret = GST_FLOW_ERROR;
      goto cleanup;
    }
  }

cleanup:
  /* Clean up resources */
  if (allocated_device_memory) {
    if (d_in) {
      cuMemFree(d_in);
    }
    
    if (inbuf != outbuf && d_out) {
      cuMemFree(d_out);
    }
  }
  
  if (!filter->has_cuda_memory) {
    if (inbuf != outbuf) {
      gst_buffer_unmap(outbuf, &out_map);
    }
    gst_buffer_unmap(inbuf, &in_map);
  } else {
    if (inbuf != outbuf) {
      gst_buffer_unmap(outbuf, &out_map);
    }
    gst_buffer_unmap(inbuf, &in_map);
  }
  
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
  
  /* Get available memory info */
  result = cuMemGetInfo(&filter->available_memory, &filter->total_memory);
  if (result != CUDA_SUCCESS) {
    GST_WARNING_OBJECT (filter, "Failed to get memory information: %d", result);
    /* Not fatal, continue with unknown memory stats */
    filter->available_memory = 0;
    filter->total_memory = 0;
  } else {
    GST_INFO_OBJECT (filter, "GPU Memory: Available %zu bytes of %zu bytes total",
        filter->available_memory, filter->total_memory);
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
  
  /* Normalize path for Windows/Unix compatibility */
  gchar *normalized_path = filter->kernel_path;
#ifdef _WIN32
  /* Replace forward slashes with backslashes on Windows */
  normalized_path = g_strdup(filter->kernel_path);
  for (gchar *p = normalized_path; *p != '\0'; p++) {
    if (*p == '/') *p = '\\';
  }
#endif
  
  /* Check if file exists and is readable */
  if (!g_file_test (normalized_path, G_FILE_TEST_EXISTS | G_FILE_TEST_IS_REGULAR)) {
    GST_ERROR_OBJECT (filter, "Kernel file '%s' does not exist or is not a regular file", 
        normalized_path);
#ifdef _WIN32
    g_free(normalized_path);
#endif
    return FALSE;
  }
  
  /* Get file modification time for reload detection */
  GFile *file = g_file_new_for_path (normalized_path);
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
  
  if (!g_file_get_contents (normalized_path, &ptx_data, &ptx_size, &error)) {
    GST_ERROR_OBJECT (filter, "Failed to read kernel file: %s", error->message);
    g_error_free (error);
#ifdef _WIN32
    g_free(normalized_path);
#endif
    return FALSE;
  }
  
#ifdef _WIN32
  g_free(normalized_path);
#endif
  
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
  
  /* Normalize path for Windows compatibility */
  gchar *normalized_path = filter->kernel_path;
#ifdef _WIN32
  normalized_path = g_strdup(filter->kernel_path);
  for (gchar *p = normalized_path; *p != '\0'; p++) {
    if (*p == '/') *p = '\\';
  }
#endif
  
  /* Get file modification time */
  GFile *file = g_file_new_for_path (normalized_path);
  GFileInfo *file_info = g_file_query_info (file, 
                                          G_FILE_ATTRIBUTE_TIME_MODIFIED,
                                          G_FILE_QUERY_INFO_NONE, 
                                          NULL, 
                                          &error);
  if (error) {
    GST_WARNING_OBJECT (filter, "Could not get file info: %s", error->message);
    g_error_free (error);
    g_object_unref (file);
#ifdef _WIN32
    g_free(normalized_path);
#endif
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
    
#ifdef _WIN32
    g_free(normalized_path);
#endif
    return result;
  }
  
  if (mtime) {
    g_date_time_unref(mtime);
  }
  
#ifdef _WIN32
  g_free(normalized_path);
#endif
  return TRUE;
}

/* Memory management helper functions */

static void
gst_cuda_kernel_cleanup_pinned_memory (GstCudaKernel * filter)
{
  /* Free any allocated pinned memory */
  if (filter->pinned_input) {
    cuMemFreeHost(filter->pinned_input);
    filter->pinned_input = NULL;
    filter->pinned_input_size = 0;
  }
  
  if (filter->pinned_output) {
    cuMemFreeHost(filter->pinned_output);
    filter->pinned_output = NULL;
    filter->pinned_output_size = 0;
  }
}

static gboolean
gst_cuda_kernel_ensure_pinned_buffer (GstCudaKernel * filter, 
    size_t size, void **pinned_ptr, size_t *current_size)
{
  /* If we already have a buffer of sufficient size, reuse it */
  if (*pinned_ptr && *current_size >= size) {
    return TRUE;
  }
  
  /* Free existing buffer if any */
  if (*pinned_ptr) {
    cuMemFreeHost(*pinned_ptr);
    *pinned_ptr = NULL;
    *current_size = 0;
  }
  
  /* Allocate new buffer - add some extra space to avoid frequent reallocs */
  size_t alloc_size = size + (size / 4);  /* Add 25% extra space */
  CUresult result = cuMemHostAlloc(pinned_ptr, alloc_size, CU_MEMHOSTALLOC_PORTABLE);
  
  if (result != CUDA_SUCCESS) {
    GST_WARNING_OBJECT(filter, "Failed to allocate pinned memory (%zu bytes): %d - disabling pinned memory", 
        alloc_size, result);
    
    /* Disable pinned memory for future operations */
    filter->use_pinned_memory = FALSE;
    
    /* Clean up any existing pinned memory */
    gst_cuda_kernel_cleanup_pinned_memory(filter);
    
    return FALSE;
  }
  
  /* Update size tracking */
  *current_size = alloc_size;
  
  GST_DEBUG_OBJECT(filter, "Allocated new pinned memory buffer of %zu bytes", alloc_size);
  return TRUE;
}

static gboolean
gst_cuda_kernel_check_memory_availability (GstCudaKernel * filter, size_t required_size)
{
  /* If we don't have memory info, update it */
  if (filter->available_memory == 0 || filter->total_memory == 0) {
    CUresult result = cuMemGetInfo(&filter->available_memory, &filter->total_memory);
    if (result != CUDA_SUCCESS) {
      /* Can't get memory info, assume we have enough */
      GST_WARNING_OBJECT(filter, "Failed to get memory information: %d", result);
      return TRUE;
    }
  }
  
  /* Check if we have enough memory available (with some margin) */
  size_t required_with_margin = required_size + (1024 * 1024);  /* Add 1MB margin */
  
  if (filter->available_memory < required_with_margin) {
    GST_WARNING_OBJECT(filter, "Not enough GPU memory available. Need %zu bytes "
        "but only have %zu bytes free (total: %zu)",
        required_with_margin, filter->available_memory, filter->total_memory);
    return FALSE;
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