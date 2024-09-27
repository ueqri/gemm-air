from air.ir import *
from air.passmanager import PassManager
from air.dialects.air import module_builder
from air.dialects import func, linalg, arith
from air.compiler.util import run_transform


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


@run
def gemm_module():

    @module_builder
    def my_module():
        M = 256
        N = 256
        K = 256
        dtype = F32Type.get()

        @func.FuncOp.from_py_func(
            MemRefType.get((M, K), dtype),
            MemRefType.get((K, N), dtype),
            MemRefType.get((M, N), dtype),
        )
        def matmul(lhs, rhs, out):
            zero = arith.ConstantOp(dtype, FloatAttr.get(dtype, 0))
            linalg.fill(zero, outs=[out])
            linalg.matmul(lhs, rhs, outs=[out])
            return out

    module = my_module()
    with open("0_gemm.mlir", "w") as f:
        f.write(str(module))

    ################################################
    ## Tiling
    ################################################

    pipeline = (
        "builtin.module("
        + ",".join(
            [
                "one-shot-bufferize{bufferize-function-boundaries=1 unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map}",
                "canonicalize",
                "cse",
            ]
        )
        + ")"
    )
    PassManager.parse(pipeline, context=module.context).run(module.operation)

    tiled_block_size = 64
    transform_ir_string = f"""
    transform.with_pdl_patterns {{
    ^bb0(%arg0: !pdl.operation):
        transform.sequence %arg0 : !pdl.operation failures(propagate) {{
        ^bb1(%arg1: !pdl.operation):
            %matmul = transform.structured.match ops{{["linalg.matmul"]}} in %arg1  : (!pdl.operation) -> !pdl.operation
            %matmul_1, %loops:3 = transform.air.linalg_tile %matmul [{tiled_block_size}, {tiled_block_size}, {tiled_block_size}]
        }}
    }}
    """
    transform_ir = Module.parse(transform_ir_string, context=module.context)
    run_transform(transform_ir, module)

    with open("1_air_tiled.mlir", "w") as f:
        f.write(str(module))

    ################################################
    ## Bind scf.parallel loops to air hierarchies
    ################################################

    pipeline = (
        "builtin.module("
        + ",".join(
            [
                "air-copy-to-dma",
                "air-par-to-herd{depth=1}",
                "air-par-to-launch{has-air-segment=1}",
                "canonicalize",
                "cse",
            ]
        )
        + ")"
    )
    PassManager.parse(pipeline, context=module.context).run(module.operation)

    with open("2_air_bind_scf_parallel.mlir", "w") as f:
        f.write(str(module))

    ################################################
    ## Extract event dependency and optimize schedule
    ################################################

    pipeline = (
        "builtin.module("
        + ",".join(
            [
                "air-dependency",
                "air-dependency-schedule-opt",
                "air-specialize-dma-broadcast",
                "air-dma-to-channel",
                "canonicalize",
                "cse",
                "air-dependency-canonicalize",
                "canonicalize",
                "cse",
                "air-label-scf-for-to-ping-pong",
                "canonicalize",
                "cse",
            ]
        )
        + ")"
    )
    PassManager.parse(pipeline, context=module.context).run(module.operation)

    with open("3_air_channel.mlir", "w") as f:
        f.write(str(module))

    pipeline = (
        "builtin.module("
        + ",".join(
            [
                "air-ping-pong-transform{keep-memref-dealloc=true}",
                "canonicalize",
                "cse",
                "air-specialize-channel-wrap-and-stride",
                "canonicalize",
                "cse",
            ]
        )
        + ")"
    )
    PassManager.parse(pipeline, context=module.context).run(module.operation)

    with open("4_aircc_input.mlir", "w") as f:
        f.write(str(module))

    ################################################
    ## Place herd to segment
    ################################################

    pipeline = (
        "builtin.module("
        + ",".join(
            [
                "air-place-herds{num-rows=4 num-cols=1 row-anchor=2 col-anchor=0}",
                "canonicalize",
                "cse",
            ]
        )
        + ")"
    )
    PassManager.parse(pipeline, context=module.context).run(module.operation)

    with open("5_air_placed.mlir", "w") as f:
        f.write(str(module))

    ################################################
    ## MLIR-AIR to MLIR-AIE
    ################################################

    pipeline = (
        "builtin.module("
        + ",".join(
            [
                "air-to-aie{row-offset=2 col-offset=0 device=npu1_4col emit-while-loop=true}",
                "canonicalize",
            ]
        )
        + ")"
    )
    PassManager.parse(pipeline, context=module.context).run(module.operation)

    with open("6_air_to_aie_aircc_decomp_aiecc.mlir", "w") as f:
        f.write(str(module))

    ################################################
    ## MLIR-AIR runtime lowering
    ################################################

    pipeline = (
        "builtin.module("
        + ",".join(
            [
                "air-to-std",
                "symbol-dce",
                "airrt-to-npu",
                "canonicalize",
            ]
        )
        + ")"
    )
    PassManager.parse(pipeline, context=module.context).run(module.operation)

    with open("7_aie.mlir", "w") as f:
        f.write(str(module))

    print(module)
