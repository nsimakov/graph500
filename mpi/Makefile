CFLAGS = -openmp -g -Wall -std=c99 -Drestrict=__restrict__ -O3 -xHost -DNDEBUG -DSHOWCPUAFF -DDOONENODEOMPPURE # -g -pg
# CFLAGS = -g -Wall -Drestrict= -DGRAPH_GENERATOR_MPI -ffast-math
LDFLAGS = -openmp -g # -g -pg
MPICC = mpicc

all: graph500_mpi_replicated_csc_seqval
# graph500_mpi_replicated

#all: graph500_mpi_simple graph500_mpi_one_sided graph500_mpi_replicated graph500_mpi_replicated_csc graph500_mpi_custom

GENERATOR_SOURCES = ../generator/graph_generator.c ../generator/make_graph.c ../generator/splittable_mrg.c ../generator/utils.c
SOURCES = oned_csr.c oned_csc.c utils.c validate.c onesided.c onesided_emul.c
HEADERS = common.h oned_csr.h oned_csc.h redistribute.h mpi_workarounds.h onesided.h

graph500_mpi_simple: main.c bfs_simple.c $(SOURCES) $(HEADERS) $(GENERATOR_SOURCES)
	$(MPICC) $(CFLAGS) $(LDFLAGS) -o graph500_mpi_simple main.c bfs_simple.c $(SOURCES) $(GENERATOR_SOURCES) -lm

graph500_mpi_one_sided: main.c bfs_one_sided.c $(SOURCES) $(HEADERS) $(GENERATOR_SOURCES)
	$(MPICC) $(CFLAGS) $(LDFLAGS) -o graph500_mpi_one_sided main.c bfs_one_sided.c $(SOURCES) $(GENERATOR_SOURCES) -lm

graph500_mpi_replicated: main.c bfs_replicated.c $(SOURCES) $(HEADERS) $(GENERATOR_SOURCES)
	$(MPICC) $(CFLAGS) $(LDFLAGS) -o graph500_mpi_replicated main.c bfs_replicated.c $(SOURCES) $(GENERATOR_SOURCES) -lm

graph500_mpi_replicated_csc: main.c bfs_replicated_csc.c $(SOURCES) $(HEADERS) $(GENERATOR_SOURCES)
	$(MPICC) $(CFLAGS) $(LDFLAGS) -o graph500_mpi_replicated_csc main.c bfs_replicated_csc.c $(SOURCES) $(GENERATOR_SOURCES) -lm

graph500_mpi_custom: main.c bfs_custom.c $(SOURCES) $(HEADERS) $(GENERATOR_SOURCES)
	$(MPICC) $(CFLAGS) $(LDFLAGS) -o graph500_mpi_custom  main.c bfs_custom.c $(SOURCES) $(GENERATOR_SOURCES) -lm

graph500_mpi_replicated_seqval: main_seqval.c bfs_replicated.c ../verify.c ../xalloc.c ../omp-csr/omp-csr.c $(SOURCES) $(HEADERS) $(GENERATOR_SOURCES)
	$(MPICC) $(CFLAGS) -DSEQVAL -DVERBOSE $(LDFLAGS) -I../omp-csr -o graph500_mpi_replicated_seqval main_seqval.c bfs_replicated.c ../verify.c ../xalloc.c ../omp-csr/omp-csr.c $(SOURCES) $(GENERATOR_SOURCES) -lm

graph500_mpi_replicated_csc_seqval: main_seqval.c bfs_replicated_csc.c ../verify.c ../xalloc.c ../omp-csr/omp-csr.c $(SOURCES) $(HEADERS) $(GENERATOR_SOURCES)
	$(MPICC) $(CFLAGS) -DSEQVAL -DVERBOSE $(LDFLAGS) -I../omp-csr -o graph500_mpi_replicated_csc_seqval main_seqval.c bfs_replicated_csc.c ../verify.c ../xalloc.c ../omp-csr/omp-csr.c $(SOURCES) $(GENERATOR_SOURCES) -lm

clean:
	-rm -f graph500_mpi_* *.o *.a
