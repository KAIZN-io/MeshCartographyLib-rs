# Makefile for building MeshCartographyLib

SHELL := /bin/bash

# Paths
PROJECT_DIR := $(shell pwd)
export Meshes_Dir := $(PROJECT_DIR)/test/meshes

# Platform selection
PLATFORM ?= executive
BUILD_DIR = target/release
ifeq ($(PLATFORM), wasm)
	CMAKE_CMD = emcmake cmake
	BUILD_CMD = emmake ninja
	BUILD_DIR = embuild
else
	CMAKE_CMD = cmake
	BUILD_CMD = ninja
endif

# Determine OS
OS := $(shell uname -s)

.PHONY: all
all: check_submodule build_rust

.PHONY: build_rust
build_rust:
	@echo "Building Rust dependencies..."
	cargo build --release

# Run the Rust executable with an optional argument
.PHONY: run
run:
	@echo "Running with file path: $(FILE_PATH)"
	$(eval FILE_PATH ?= ./test/meshes/ellipsoid_x4.obj)
	RUST_LOG=info cargo run --release -- $(FILE_PATH)

.PHONY: doc
doc:
	cargo doc --no-deps --open

.PHONY: test
test:
	cargo test --release  -- --nocapture

.PHONY: wasm
wasm:
	wasm-pack build --target web

.PHONY: check_submodule
check_submodule:
	@if [ ! "$(shell git submodule status | grep pmp-library | cut -c 1)" = "-" ]; then \
		echo "PMP library submodule already initialized and updated."; \
	else \
		echo "PMP library submodule is empty. Initializing and updating..."; \
		git submodule update --init -- pmp-library; \
		$(MAKE) install_pmp; \
	fi

.PHONY: update_pmp
update_pmp:
	@echo "Updating PMP library submodule..."; \
	git submodule update --remote pmp-library;

.PHONY: install_pmp
install_pmp:
	@echo "Installing PMP library..."; \
	mkdir -p $(BUILD_DIR)/pmp-library; \
	cd $(BUILD_DIR)/pmp-library && \
	$(CMAKE_CMD) -G Ninja $(PROJECT_DIR)/pmp-library -DCMAKE_BUILD_TYPE=Release && \
	ninja && \
	sudo ninja install;

# Cleaning
.PHONY: clean
clean:
	rm -rf $(PROJECT_DIR)/target
