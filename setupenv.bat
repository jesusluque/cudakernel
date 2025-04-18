:: Set up Visual Studio environment
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

:: Set up GStreamer environment 
set GSTREAMER_ROOT=C:\gstreamer\1.0\msvc_x86_64
set PATH=%GSTREAMER_ROOT%\bin;%PATH%
set PKG_CONFIG_PATH=%GSTREAMER_ROOT%\lib\pkgconfig

:: Set CUDA environment
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set PATH=%CUDA_PATH%\bin;%PATH%