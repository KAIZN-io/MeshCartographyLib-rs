# Makefile for building MeshCartographyLib

SHELL := /bin/bash

# Paths
PROJECT_DIR := $(shell pwd)
export Meshes_Dir := $(PROJECT_DIR)/test/meshes

# Determine OS
OS := $(shell uname -s)

.PHONY: all
all: build_rust

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

# Cleaning
.PHONY: clean
clean:
	rm -rf $(PROJECT_DIR)/target
