project('gst-cudakernel', 'c',
  version : '1.0.0',
  default_options : [ 'warning_level=1',
                      'buildtype=release',
                      'c_std=c11' ]
)

gst_req = '>= 1.20.0'
json_glib_dep = dependency('json-glib-1.0')
gst_dep = [
  dependency('gstreamer-1.0', version : gst_req),
  dependency('gstreamer-base-1.0', version : gst_req),
  dependency('gstreamer-video-1.0', version : gst_req),
]

cc = meson.get_compiler('c')
cuda_dep = declare_dependency(
  dependencies: [
    cc.find_library('cuda', required : true),
  ],
  include_directories: include_directories('/usr/local/cuda/include'),
)

plugin_sources = [
  'src/gstcudakernel.c',
]

library('gstcudakernel',
  plugin_sources,
  dependencies : [gst_dep, json_glib_dep, cuda_dep],
  install : true,
  install_dir : join_paths(get_option('libdir'), 'gstreamer-1.0'),
  c_args : [
    '-DHAVE_CONFIG_H',
    '-DGST_CUDA_META_API_TYPE=123',  # Placeholder for actual API type
  ]
)