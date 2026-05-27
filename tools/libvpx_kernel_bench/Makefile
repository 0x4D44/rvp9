# Build the libvpx kernel head-to-head harness with mingw64 gcc.
#
# Usage (from repo root, in an MSYS2 mingw64 / Git Bash shell with
# /c/msys64/mingw64/bin on PATH):
#
#     make -C tools/libvpx_kernel_bench
#
# Output: tools/libvpx_kernel_bench/libvpx_kernel_bench.exe
#
# Links against the existing build/libvpx/libvpx.a; rerun
# scripts/build-libvpx.sh first if that archive is absent.

REPO_ROOT  := $(realpath $(CURDIR)/../..)
LIBVPX_DIR := $(REPO_ROOT)/third_party/libvpx
LIBVPX_BLD := $(REPO_ROOT)/build/libvpx

CC      ?= gcc
CFLAGS  := -O3 -fomit-frame-pointer -Wall -Wextra -std=c11
CFLAGS  += -I$(LIBVPX_BLD) -I$(LIBVPX_DIR) -I$(LIBVPX_DIR)/vpx_dsp
LDFLAGS :=
LDLIBS  := $(LIBVPX_BLD)/libvpx.a -lm

TARGET := libvpx_kernel_bench.exe
SRCS   := main.c

all: $(TARGET)

$(TARGET): $(SRCS) $(LIBVPX_BLD)/libvpx.a
	$(CC) $(CFLAGS) $(SRCS) -o $@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
