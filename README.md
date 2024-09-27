# GEMM on AIR platform

This repository demonstrates the MLIR-AIR workflow using GEMM as the application.

## Prerequisite

Please follow the guidance at [xilinx.github.io/mlir-air](https://xilinx.github.io/mlir-air/) to install the AIR compilers.

## Run

```
# please activate the AIR Python environment first
make clean
make
```

The intermediate MLIR modules of each stage in the MLIR-AIR flow will be output in the same directory.
