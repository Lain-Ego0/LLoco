# Native deployment

This directory contains only LLoco's C++ deployment source and configuration. Install
the following dependencies on the target system instead of vendoring them here:

- Unitree SDK2 and CycloneDDS
- ONNX Runtime 1.22 or a compatible release
- zlib and CMake

Export a policy with LLoco/mjlab, place it under the matching robot's
`config/policy/<task>/<version>/exported/` directory, then configure and build that
robot directory with CMake. Generated policies and build directories are git-ignored.
