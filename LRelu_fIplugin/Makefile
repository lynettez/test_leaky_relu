APP := lrelu_followed_IPlugin

CC := g++

CUDA_INSTALL_PATH ?= /usr/local/cuda-9.0
CUDNN_INSTALL_PATH ?= /root/cudnn-7.0
TRT_INSTALL_PATH ?= /root/TensorRT-3.0.0

SRCS := \
	lrelu_followed_IPlugin.cpp

OBJS := $(SRCS:.cpp=.o)

CPPFLAGS := \
	-std=c++11 -g\
	-I"$(TRT_INSTALL_PATH)/include" \
	-I"$(CUDA_INSTALL_PATH)/include" \
	-I"$(CUDNN_INSTALL_PATH)/include"

LDFLAGS := \
	-lnvparsers \
	-lnvinfer \
	-lcudart \
	-lcudnn \
	-lopencv_objdetect \
	-L"$(TRT_INSTALL_PATH)/lib" \
	-L"$(CUDA_INSTALL_PATH)/lib64" \
	-L"$(CUDNN_INSTALL_PATH)/lib64"

all: $(APP)

%.o: %.cpp
	@echo "Compiling: $<"
	@ $(CC) $(CPPFLAGS) -c $<

$(APP): $(OBJS)
	@echo "Linking: $@"
	@ $(CC) -o $@ $(OBJS) $(CPPFLAGS) $(LDFLAGS)

clean:
	rm -rf $(APP) $(OBJS) *.ppm *.txt
