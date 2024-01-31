# MeshCartographyLib

**MeshCartographyLib** is a comprehensive geometry processing library designed to seamlessly bridge the realms of 2D and 3D spaces. Rooted in the principles of cartography, it offers tools to transform 3D meshes into 2D representations, akin to creating maps. Whether you're aiming to generate Escher-like tessellations or free-border 2D meshes from closed 3D structures, or harness the power of particle interactions on these meshes, this library facilitates it all.


## How to use

Unfold your closed surface mesh OBJ file over the terminal by passing its file path to the program:  
`make run FILE_PATH=./meshes/ellipsoid_x4.obj`


## Profiling on macOS using Instruments

To profile the `mesh_cartography` executable using Instruments from XCode on macOS:

1. **Build the Executable**: Compile your project in release mode with `make`. Ensure debug symbols are enabled in `Cargo.toml` with `[profile.release] debug = true`.

2. **Open Instruments**: Launch Instruments from Xcode (Xcode -> Open Developer Tool -> Instruments).

3. **Choose Profiling Template**: Select the "Time Profiler" template.

4. **Select Your Executable**: In Instruments, click on "All Processes," then "Choose Target," and navigate to your executable (`target/release/mesh_cartography`).

5. **Set Command-Line Arguments**: If your executable requires arguments (e.g., a file path), set them in the "Arguments" field.

6. **Enable Viewing Log Lines**: Click on the plus sign in the upper right corner and add "os_log".

7. **Start Profiling**: Click the record button. Perform the necessary actions in your application or wait for it to complete.

8. **Analyze Results**: After completion, use Instruments' analysis tools to review the performance data.
