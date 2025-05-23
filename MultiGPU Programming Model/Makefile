# Makefile global pour toutes les versions Jacobi
VERSIONS = etape1_cpu etape2_cpu_gpu etape3_mpi_gpus etape4_mpi_overlap etape5_nccl etape6_nccl_overlap etape7_nccl_graphs etape8_nvshmem etape9_nvshmem_lt etape10_vshmem_neighborhood_lto etape11_nvshmem_norm_overlap_neighborhood_sync_lto etape12_nvshmem_norm_overlap_neighborhood_sync_lto_ext1

.PHONY: all clean $(VERSIONS)
all: $(VERSIONS)

$(VERSIONS):
	@$(MAKE) -C $@ all

clean:
	@for d in $(VERSIONS); do 		$(MAKE) -C $$d clean; 	done
