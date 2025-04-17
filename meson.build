project('gst-cudakernel', 'c',
  version : '1.0.0',
  default_options : [ 'warning_level=1',
                      'buildtype=release',
                      'c_std=c11' ]
)

# Include configuration directory
conf_inc = include_directories('src/config')

gst_req = '>= 1.20.0'
json_glib_dep = dependency('json-glib-1.0')
gst_dep = [
  dependency('gstreamer-1.0', version : gst_req),
  dependency('gstreamer-base-1.0', version : gst_req),
  dependency('gstreamer-video-1.0', version : gst_req),
]

cc = meson.get_compiler('c')

# Find CUDA libraries with explicit paths
cuda_lib_path = '/usr/local/cuda/lib64'
if not cc.has_header('cuda.h', args: '-I/usr/local/cuda/include')
  cuda_lib_path = '/usr/lib/x86_64-linux-gnu'
  if not cc.has_header('cuda.h', args: '-I/usr/include')
    error('CUDA headers not found')
  endif
endif

cuda_dep = declare_dependency(
  dependencies: [
    cc.find_library('cuda', dirs: [cuda_lib_path], required: true),
  ],
  include_directories: include_directories('/usr/local/cuda/include', '/usr/include'),
)

# Add cudart separately - specify the library location explicitly
cudart_lib = cc.find_library('cudart', dirs: [cuda_lib_path, '/usr/lib/x86_64-linux-gnu'], required: false)
if not cudart_lib.found()
  message('Could not find cudart library, trying libcudart.so directly')
  cudart_lib = cc.find_library('libcudart.so', dirs: [cuda_lib_path, '/usr/lib/x86_64-linux-gnu'], required: true)
endif

plugin_sources = [
  'src/gstcudakernel.c',
]

library('gstcudakernel',
  plugin_sources,
  dependencies : [gst_dep, json_glib_dep, cuda_dep, cudart_lib],
  include_directories : [conf_inc],
  install : true,
  install_dir : '/usr/lib/x86_64-linux-gnu/gstreamer-1.0',
  c_args : [
    '-DHAVE_CONFIG_H',
    '-DGST_CUDA_META_API_TYPE=123',
  ]
)
