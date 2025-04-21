/* Basic configuration for GStreamer CUDA kernel plugin */
#ifndef __GST_CUDA_KERNEL_CONFIG_H__
#define __GST_CUDA_KERNEL_CONFIG_H__

/* Version information */
#define PACKAGE "gst-cudakernel"
#define PACKAGE_NAME "GStreamer CUDA Kernel Plugin"
#define PACKAGE_VERSION "1.0.0"
#define PACKAGE_STRING "GStreamer CUDA Kernel Plugin 1.0.0"
#define VERSION "1.0.0"

/* Plugin description */
#define GST_PACKAGE_NAME "GStreamer CUDA kernel elements"
#define GST_PACKAGE_ORIGIN "https://overo.es"
#define GST_LICENSE "custom"

/* Define missing CUDA constants for GStreamer */
#ifndef GST_CUDA_META_API_TYPE
#define GST_CUDA_META_API_TYPE 123
#endif

/* Add missing CUDA memory features */
#ifndef GST_CAPS_FEATURE_MEMORY_CUDA_MEMORY
#define GST_CAPS_FEATURE_MEMORY_CUDA_MEMORY "memory:CUDAMemory"
#endif

/* Define missing GST_MAP flags for CUDA if not available */
#ifndef GST_MAP_CUDA
#define GST_MAP_CUDA (GST_MAP_FLAG_LAST << 1)
#endif

#endif /* __GST_CUDA_KERNEL_CONFIG_H__ */

