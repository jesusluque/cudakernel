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

# Platform-specific CUDA paths
cuda_root = ''
host_system = host_machine.system()
if host_system == 'windows'
  # On Windows, check common installation paths
  if run_command('cmd', '/c', 'if exist "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA" echo found', check: false).stdout().strip() == 'found'
    cuda_root = 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA'
  elif run_command('cmd', '/c', 'if exist "C:\\cuda" echo found', check: false).stdout().strip() == 'found'
    cuda_root = 'C:\\cuda'
  else
    # Try to find CUDA from environment variables
    cuda_root = run_command('cmd', '/c', 'echo %CUDA_PATH%', check: false).stdout().strip()
    if cuda_root == '%CUDA_PATH%'
      error('CUDA installation not found. Set CUDA_PATH environment variable.')
    endif
  endif
  
  cuda_include_dirs = [cuda_root + '\\include']
  cuda_lib_dirs = [cuda_root + '\\lib\\x64']
  
  # Windows-specific installation path for GStreamer plugins
  plugin_install_dir = get_option('libdir') != '' ? get_option('libdir') : 'lib'
  plugin_install_dir = plugin_install_dir + '\\gstreamer-1.0'
  
else
  # Linux paths
  if cc.has_header('cuda.h', args: '-I/usr/local/cuda/include')
    cuda_include_dirs = ['/usr/local/cuda/include']
    cuda_lib_dirs = ['/usr/local/cuda/lib64']
  else
    cuda_include_dirs = ['/usr/include']
    cuda_lib_dirs = ['/usr/lib/x86_64-linux-gnu']
  endif
  
  # Linux install path
  plugin_install_dir = get_option('libdir') != '' ? get_option('libdir') : '/usr/lib/x86_64-linux-gnu'
  plugin_install_dir = plugin_install_dir + '/gstreamer-1.0'
endif

message('CUDA include directories: ' + ' '.join(cuda_include_dirs))
message('CUDA library directories: ' + ' '.join(cuda_lib_dirs))
message('Plugin install directory: ' + plugin_install_dir)

# Find CUDA libraries
cuda_dep = declare_dependency(
  dependencies: [
    cc.find_library('cuda', dirs: cuda_lib_dirs, required: true),
  ],
  include_directories: include_directories(cuda_include_dirs),
)

# Add cudart separately
cudart_lib = cc.find_library('cudart', dirs: cuda_lib_dirs, required: false)
if not cudart_lib.found()
  if host_system == 'windows'
    cudart_lib = cc.find_library('cudart', dirs: cuda_lib_dirs, required: true)
  else
    message('Could not find cudart library, trying libcudart.so directly')
    cudart_lib = cc.find_library('libcudart.so', dirs: cuda_lib_dirs, required: true)
  endif
endif

plugin_sources = [
  'src/gstcudakernel.c',
]

shared_library_name = 'gstcudakernel'
if host_system == 'windows'
  shared_library_name = 'libgstcudakernel'
endif

library(shared_library_name,
  plugin_sources,
  dependencies : [gst_dep, json_glib_dep, cuda_dep, cudart_lib],
  include_directories : [conf_inc],
  install : true,
  install_dir : plugin_install_dir,
  c_args : [
    '-DHAVE_CONFIG_H',
    '-DGST_CUDA_META_API_TYPE=123',
  ]
)