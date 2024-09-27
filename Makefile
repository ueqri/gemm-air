HAS_AIR_COMPILER := $(shell python3 -c 'import air' 2>&1)
ifneq (,$(findstring ModuleNotFoundError,$(HAS_AIR_COMPILER)))
    $(error MLIR-AIR compiler not found)
endif

all:
	python3 gemm-air.py

clean:
	rm *.mlir

.PHONY: all, clean
