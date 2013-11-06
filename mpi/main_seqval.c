/* Copyright (C) 2010 The Trustees of Indiana University.                  */
/*                                                                         */
/* Use, modification and distribution is subject to the Boost Software     */
/* License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at */
/* http://www.boost.org/LICENSE_1_0.txt)                                   */
/*                                                                         */
/*  Authors: Jeremiah Willcock                                             */
/*           Andrew Lumsdaine                                              */

/* These need to be before any possible inclusions of stdint.h or inttypes.h.
 * */
#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS
#endif
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif

#include "../generator/make_graph.h"
#include "../generator/utils.h"
#include "common.h"
#include <math.h>
#include <mpi.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>
#include <limits.h>
#include <stdint.h>
#include <inttypes.h>

#ifdef SHOWCPUAFF
#include <sys/types.h>
#include <unistd.h>
#endif

static int compare_doubles(const void* a, const void* b) {
  double aa = *(const double*)a;
  double bb = *(const double*)b;
  return (aa < bb) ? -1 : (aa == bb) ? 0 : 1;
}

enum {s_minimum, s_firstquartile, s_median, s_thirdquartile, s_maximum, s_mean, s_std, s_LAST};
static void get_statistics(const double x[], int n, double r[s_LAST]) {
  double temp;
  int i;
  /* Compute mean. */
  temp = 0;
  for (i = 0; i < n; ++i) temp += x[i];
  temp /= n;
  r[s_mean] = temp;
  /* Compute std. dev. */
  temp = 0;
  for (i = 0; i < n; ++i) temp += (x[i] - r[s_mean]) * (x[i] - r[s_mean]);
  temp /= n - 1;
  r[s_std] = sqrt(temp);
  /* Sort x. */
  double* xx = (double*)xmalloc(n * sizeof(double));
  memcpy(xx, x, n * sizeof(double));
  qsort(xx, n, sizeof(double), compare_doubles);
  /* Get order statistics. */
  r[s_minimum] = xx[0];
  r[s_firstquartile] = (xx[(n - 1) / 4] + xx[n / 4]) * .5;
  r[s_median] = (xx[(n - 1) / 2] + xx[n / 2]) * .5;
  r[s_thirdquartile] = (xx[n - 1 - (n - 1) / 4] + xx[n - 1 - n / 4]) * .5;
  r[s_maximum] = xx[n - 1];
  /* Clean up. */
  free(xx);
}

static inline int64_t get_pred_from_pred_entry(int64_t val) {
  return (val << 16) >> 16;
}
/* Returns true if result is valid.  Also, updates high 16 bits of each element
 * of pred to contain the BFS level number (or -1 if not visited) of each
 * vertex; this is based on the predecessor map if the user didn't provide it.
 * */
int validate_bfs_result_seq(const tuple_graph* const tg, const int64_t nglobalverts, const size_t nlocalverts, const int64_t root,
		int64_t* const pred, int64_t* const edge_visit_count_ptr, int64_t const max_used_vertex)
{
	assert (tg->edgememory_size >= 0 && tg->max_edgememory_size >= tg->edgememory_size && tg->max_edgememory_size <= tg->nglobaledges);
	assert (pred);
	*edge_visit_count_ptr = 0; /* Ensure it is a valid pointer */
	int ranges_ok = check_value_ranges(nglobalverts, nlocalverts, pred);
	if (root < 0 || root >= nglobalverts) {
		fprintf(stderr, "%d: Validation error: root vertex %" PRId64 " is invalid.\n", rank, root);
		ranges_ok = 0;
	}
	if (!ranges_ok) return 0; /* Fail */

	int validation_passed = 1;
	int root_owner;
	size_t root_local;
	get_vertex_distribution_for_pred(1, &root, &root_owner, &root_local);
	int root_is_mine = (root_owner == rank);

	/* Get maximum values so loop counts are consistent across ranks. */
	uint64_t maxlocalverts_ui = nlocalverts;
	MPI_Allreduce(MPI_IN_PLACE, &maxlocalverts_ui, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
	size_t maxlocalverts = (size_t)maxlocalverts_ui;

	ptrdiff_t max_bufsize = tuple_graph_max_bufsize(tg);
	ptrdiff_t edge_chunk_size = ptrdiff_min(HALF_CHUNKSIZE, max_bufsize);

	assert (tg->edgememory_size >= 0 && tg->max_edgememory_size >= tg->edgememory_size && tg->max_edgememory_size <= tg->nglobaledges);
	assert (pred);

	/* combine results from all processes */
	int64_t* restrict pred_vtx = NULL;
	{
		int irank;
		uint64_t i;
		int64_t nlocalvertsMax=nlocalverts;
		MPI_Allreduce(MPI_IN_PLACE, &nlocalvertsMax, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
		if(rank==0)
		{

			pred_vtx = (int64_t*)xmalloc(nglobalverts * sizeof(int64_t));
			int64_t* pred_tmp;
			int64_t nlocalvertsRemote;

			pred_tmp=pred;
			nlocalvertsRemote=nlocalverts;
			for(irank=0;irank<size;irank++)
			{
				MPI_Barrier(MPI_COMM_WORLD);


				if(irank!=0)
				{
					MPI_Recv(&nlocalvertsRemote, 1, MPI_UINT64_T, irank, 0, MPI_COMM_WORLD,
								 MPI_STATUS_IGNORE);
					MPI_Recv(pred_tmp, nlocalvertsRemote, MPI_UINT64_T, irank, 1, MPI_COMM_WORLD,
													 MPI_STATUS_IGNORE);
					//printf("%d %" PRId64 " \n",rank,nlocalvertsRemote);
				}

				for(i=0;i<nlocalvertsRemote ;i++)
				{
					pred_vtx[vertex_to_global_for_pred(irank,i)]=get_pred_from_pred_entry(pred_tmp[i]);
				}


				if(irank==0)
					pred_tmp = (int64_t*)xmalloc(nlocalvertsMax * sizeof(int64_t));
			}
			xfree(pred_tmp);
		}
		else
		{
			for(irank=0;irank<size;irank++)
			{
				MPI_Barrier(MPI_COMM_WORLD);
				if(rank==irank)
				{
					MPI_Send(&nlocalverts, 1, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD);
					MPI_Send(pred, nlocalverts, MPI_UINT64_T, 0, 1, MPI_COMM_WORLD);
				}
			}
		}
		{
			int irank;
			uint64_t i;
			for(irank=0;irank<size;irank++)
			{
				MPI_Barrier(MPI_COMM_WORLD);
				//if(rank==irank)
				//	for(i=0;i<nlocalverts ;i++)
				//		fprintf(stderr, "%d %" PRId64 " %" PRId64 " %" PRId64 "\n", rank,i,get_pred_from_pred_entry(pred[i]),vertex_to_global_for_pred(rank,i));
			}
		}
	}
	int64_t nedge_traversed;

	if(rank==0)
	{
		uint64_t i, max_bfsvtx=0;

		/*for(i=0;i<tg->edgememory_size ;i++)
		{
			if(tg->edgememory[i].v0>max_bfsvtx)
				max_bfsvtx=tg->edgememory[i].v0;
			if(tg->edgememory[i].v1>max_bfsvtx)
				max_bfsvtx=tg->edgememory[i].v1;
		}*/

		/*int64_t* restrict pred_vtx = (int64_t*)xmalloc((max_used_vertex+1) * sizeof(int64_t));
		for(i=0;i<=max_used_vertex ;i++)
		{
			pred_vtx[i]=get_pred_from_pred_entry(pred[i]);
		}*/


		nedge_traversed=verify_bfs_tree (pred_vtx, max_used_vertex,
				 root,
				 tg->edgememory, tg->nglobaledges);


		if(nedge_traversed<0)
		{
			fprintf(stderr, "Validation error: code %" PRId64 ".\n", nedge_traversed);
			validation_passed=0;
		}
	}
	if(rank==0)
	{
		xfree(pred_vtx);
	}
	MPI_Allreduce(MPI_IN_PLACE, &nedge_traversed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
	*edge_visit_count_ptr=nedge_traversed;
	/* Collect the global validation result. */
	MPI_Allreduce(MPI_IN_PLACE, &validation_passed, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
	return validation_passed;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  setup_globals();

  /* Parse arguments. */
  int SCALE = 16;
    int edgefactor = 16; /* nedges / nvertices, i.e., 2*avg. degree */
    int num_bfs_roots = 64;
    int bCompareMD5=1;
    int bRunPerf=1;
    int bRunVal=1;
    float timeForPerf=300.0;
    int numberOfCyclesForPerf=300;
    //uint8_t refMD5[16];
    int64_t* refEdgeCounts = NULL;
    int64_t* refBFS_Roots = NULL;


    if ( !(argc == 2 || argc == 3)){
        if (rank == 0)
      	  fprintf(stderr, "Usage: %s input_file [number of threads]\n", argv[0]);
          //fprintf(stderr, "Usage: %s SCALE edgefactor\n  SCALE = log_2(# vertices) [integer, required]\n  edgefactor = (# edges) / (# vertices) = .5 * (average vertex degree) [integer, defaults to 16]\n(Random number seed and Kronecker initiator are in main.c)\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if ( argc == 3){
  	  int threads=atoi(argv[2]);
  #ifdef _OPENMP
  	  omp_set_num_threads(threads);
  #else
  	  if(threads!=1)
  		  fprintf(stderr, "ERROR: %s compiled without OpenMP\n", argv[0]);
  #endif
    }


    {
  	  int iRead=0;
  	  int i;
  	  FILE *input_file;
  	  char cbuf[256];

  	  if (rank == 0)
  		  fprintf(stderr, "Reading input from %s\n",argv[1]);

  	  input_file=fopen(argv[1],"r");

  	  if(input_file==NULL){
  		  if (rank == 0)
  			  fprintf(stderr, "Error : can no open %s file\n",argv[1]);
  		  MPI_Barrier(MPI_COMM_WORLD);
  		  MPI_Abort(MPI_COMM_WORLD, 1);
  	  }

  	  fgets(cbuf,256,input_file);iRead+=sscanf(cbuf,"%d",&SCALE);
  	  fgets(cbuf,256,input_file);iRead+=sscanf(cbuf,"%d",&edgefactor);
  	  fgets(cbuf,256,input_file);iRead+=sscanf(cbuf,"%d",&num_bfs_roots);
  	  fgets(cbuf,256,input_file);iRead+=sscanf(cbuf,"%d",&bCompareMD5);
  	  fgets(cbuf,256,input_file);iRead+=sscanf(cbuf,"%d",&bRunPerf);
  	  fgets(cbuf,256,input_file);iRead+=sscanf(cbuf,"%d",&bRunVal);
  	  fgets(cbuf,256,input_file);iRead+=sscanf(cbuf,"%f",&timeForPerf);
  	  fgets(cbuf,256,input_file);iRead+=sscanf(cbuf,"%d",&numberOfCyclesForPerf);
  	  //fgets(cbuf,256,input_file);
  	  //for (i = 0; i < 16; i++)
  		//  iRead+=sscanf(cbuf+i*2,"%2x",&refMD5[i]);
  		  //refMD5[i]=cbuf[i];

  	  refEdgeCounts = (int64_t*)xmalloc(num_bfs_roots * sizeof(int64_t));
  	  refBFS_Roots = (int64_t*)xmalloc(num_bfs_roots * sizeof(int64_t));

  	  for (i = 0; i < num_bfs_roots; i++){
  		  fgets(cbuf,256,input_file);
  		  iRead+=sscanf(cbuf,"%lu %lu ",refBFS_Roots+i,refEdgeCounts+i);
  	  }


  	  //printf("%d %d\n",rank,iRead);
  	  //printf("%d %d\n",rank,SCALE);
  	  //printf("%d %d\n",rank,edgefactor);
  	  if (rank == 0){

  		  fprintf(stderr, "\tScale: %d\n",SCALE);
  		  fprintf(stderr, "\tEdgefactor %d\n",edgefactor);
  		  fprintf(stderr, "\tNumber of BFS roots: %d\n",num_bfs_roots);
  		  fprintf(stderr, "\tCompare md5 on initial edge list: %d\n",bCompareMD5);
  		  fprintf(stderr, "\tRun performance section: %d\n",bRunPerf);
  		  fprintf(stderr, "\tRun validation: %d\n",bRunVal);
  		  fprintf(stderr, "\tTime for performance section in seconds: %f\n",timeForPerf);
  		  fprintf(stderr, "\tMax number of cycles: %d\n",numberOfCyclesForPerf);
  		  fprintf(stderr, "\tNumber of MPI processes: %d\n",size);
#ifdef _OPENMP
  	      fprintf(stderr, "\tMax number of threads per MPI process: %d\n",omp_get_max_threads());

#else
		  fprintf(stderr, "\tMax number of threads per MPI process: compiled without OpenMP\n");
#endif
  		  //fprintf(stderr, "\tReffrence md5 on initial edge list: ");

  		  //for (i = 0; i < 16; i++)
  			//  fprintf(stderr, "%2.2x", refMD5[i]);
  		  //fprintf(stderr, "\n");

  	  }
  	  fclose(input_file);
#ifdef SHOWCPUAFF
  	  pid_t pid=getpid();
  	  for (i = 0; i < size; i++){
  		  if(i==rank){
  			fprintf(stderr, "MPI Process %d\n",rank);
  			sprintf(cbuf,"grep -i cpus_allowed /proc/%d/status",pid);
  			system(cbuf);
  		  }
  		  MPI_Barrier(MPI_COMM_WORLD);
  	  }
#endif


  	  //MPI_Barrier(MPI_COMM_WORLD);
  	  //MPI_Abort(MPI_COMM_WORLD, 1);
    }
//  int SCALE = 16;
//  int edgefactor = 16; /* nedges / nvertices, i.e., 2*avg. degree */
//  if (argc >= 2) SCALE = atoi(argv[1]);
//  if (argc >= 3) edgefactor = atoi(argv[2]);
//  if (argc <= 1 || argc >= 4 || SCALE == 0 || edgefactor == 0) {
//    if (rank == 0) {
//      fprintf(stderr, "Usage: %s SCALE edgefactor\n  SCALE = log_2(# vertices) [integer, required]\n  edgefactor = (# edges) / (# vertices) = .5 * (average vertex degree) [integer, defaults to 16]\n(Random number seed and Kronecker initiator are in main.c)\n", argv[0]);
//    }
//    MPI_Abort(MPI_COMM_WORLD, 1);
//  }

  uint64_t seed1 = 2, seed2 = 3;

  const char* filename = getenv("TMPFILE");
  const int reuse_file = getenv("REUSEFILE")? 1 : 0;
  /* If filename is NULL, store data in memory */

  tuple_graph tg;
  tg.nglobaledges = (int64_t)(edgefactor) << SCALE;
  int64_t nglobalverts = (int64_t)(1) << SCALE;

  tg.data_in_file = (filename != NULL);
  tg.write_file = 1;

  if (tg.data_in_file) {
    int is_opened = 0;
    int mode = MPI_MODE_RDWR | MPI_MODE_EXCL | MPI_MODE_UNIQUE_OPEN;
    if (!reuse_file) {
      mode |= MPI_MODE_CREATE | MPI_MODE_DELETE_ON_CLOSE;
    } else {
      MPI_File_set_errhandler(MPI_FILE_NULL, MPI_ERRORS_RETURN);
      if (MPI_File_open(MPI_COMM_WORLD, (char*)filename, mode,
			MPI_INFO_NULL, &tg.edgefile)) {
	mode |= MPI_MODE_RDWR | MPI_MODE_CREATE | MPI_MODE_DELETE_ON_CLOSE;
      } else {
	MPI_Offset size;
	MPI_File_get_size(tg.edgefile, &size);
	if (size == tg.nglobaledges * sizeof(packed_edge)) {
	  is_opened = 1;
	  tg.write_file = 0;
	} else /* Size doesn't match, assume different parameters. */
	  MPI_File_close (&tg.edgefile);
      }
    }
    MPI_File_set_errhandler(MPI_FILE_NULL, MPI_ERRORS_ARE_FATAL);
    if (!is_opened) {
      MPI_File_open(MPI_COMM_WORLD, (char*)filename, mode, MPI_INFO_NULL, &tg.edgefile);
      MPI_File_set_size(tg.edgefile, tg.nglobaledges * sizeof(packed_edge));
    }
    MPI_File_set_view(tg.edgefile, 0, packed_edge_mpi_type, packed_edge_mpi_type, "native", MPI_INFO_NULL);
    MPI_File_set_atomicity(tg.edgefile, 0);
  }

  /* Make the raw graph edges. */
  /* Get roots for BFS runs, plus maximum vertex with non-zero degree (used by
   * validator). */
  //int num_bfs_roots = 64;
  int64_t* bfs_roots = (int64_t*)xmalloc(num_bfs_roots * sizeof(int64_t));
  int64_t max_used_vertex = 0;

  double make_graph_start = MPI_Wtime();
  {
    /* Spread the two 64-bit numbers into five nonzero values in the correct
     * range. */
    uint_fast32_t seed[5];
    make_mrg_seed(seed1, seed2, seed);

    /* As the graph is being generated, also keep a bitmap of vertices with
     * incident edges.  We keep a grid of processes, each row of which has a
     * separate copy of the bitmap (distributed among the processes in the
     * row), and then do an allreduce at the end.  This scheme is used to avoid
     * non-local communication and reading the file separately just to find BFS
     * roots. */
    MPI_Offset nchunks_in_file = (tg.nglobaledges + FILE_CHUNKSIZE - 1) / FILE_CHUNKSIZE;
    int64_t bitmap_size_in_bytes = int64_min(BITMAPSIZE, (nglobalverts + CHAR_BIT - 1) / CHAR_BIT);
    if (bitmap_size_in_bytes * size * CHAR_BIT < nglobalverts) {
      bitmap_size_in_bytes = (nglobalverts + size * CHAR_BIT - 1) / (size * CHAR_BIT);
    }
    int ranks_per_row = ((nglobalverts + CHAR_BIT - 1) / CHAR_BIT + bitmap_size_in_bytes - 1) / bitmap_size_in_bytes;
    int nrows = size / ranks_per_row;
    int my_row = -1, my_col = -1;
    unsigned char* restrict has_edge = NULL;
    MPI_Comm cart_comm;
    {
      int dims[2] = {size / ranks_per_row, ranks_per_row};
      int periods[2] = {0, 0};
      MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);
    }
    int in_generating_rectangle = 0;
    if (cart_comm != MPI_COMM_NULL) {
      in_generating_rectangle = 1;
      {
        int dims[2], periods[2], coords[2];
        MPI_Cart_get(cart_comm, 2, dims, periods, coords);
        my_row = coords[0];
        my_col = coords[1];
      }
      MPI_Comm this_col;
      MPI_Comm_split(cart_comm, my_col, my_row, &this_col);
      MPI_Comm_free(&cart_comm);
      has_edge = (unsigned char*)xMPI_Alloc_mem(bitmap_size_in_bytes);
      memset(has_edge, 0, bitmap_size_in_bytes);
      /* Every rank in a given row creates the same vertices (for updating the
       * bitmap); only one writes them to the file (or final memory buffer). */
      packed_edge* buf = (packed_edge*)xmalloc(FILE_CHUNKSIZE * sizeof(packed_edge));
      MPI_Offset block_limit = (nchunks_in_file + nrows - 1) / nrows;
      /* fprintf(stderr, "%d: nchunks_in_file = %" PRId64 ", block_limit = %" PRId64 " in grid of %d rows, %d cols\n", rank, (int64_t)nchunks_in_file, (int64_t)block_limit, nrows, ranks_per_row); */
      if (tg.data_in_file) {
        tg.edgememory_size = 0;
        tg.edgememory = NULL;
      } else {
        int my_pos = my_row + my_col * nrows;
        int last_pos = (tg.nglobaledges % ((int64_t)FILE_CHUNKSIZE * nrows * ranks_per_row) != 0) ?
                       (tg.nglobaledges / FILE_CHUNKSIZE) % (nrows * ranks_per_row) :
                       -1;
        int64_t edges_left = tg.nglobaledges % FILE_CHUNKSIZE;
        int64_t nedges = FILE_CHUNKSIZE * (tg.nglobaledges / ((int64_t)FILE_CHUNKSIZE * nrows * ranks_per_row)) +
                         FILE_CHUNKSIZE * (my_pos < (tg.nglobaledges / FILE_CHUNKSIZE) % (nrows * ranks_per_row)) +
                         (my_pos == last_pos ? edges_left : 0);
        /* fprintf(stderr, "%d: nedges = %" PRId64 " of %" PRId64 "\n", rank, (int64_t)nedges, (int64_t)tg.nglobaledges); */
        tg.edgememory_size = nedges;
        tg.edgememory = (packed_edge*)xmalloc(nedges * sizeof(packed_edge));
      }
      MPI_Offset block_idx;
      for (block_idx = 0; block_idx < block_limit; ++block_idx) {
        /* fprintf(stderr, "%d: On block %d of %d\n", rank, (int)block_idx, (int)block_limit); */
        MPI_Offset start_edge_index = int64_min(FILE_CHUNKSIZE * (block_idx * nrows + my_row), tg.nglobaledges);
        MPI_Offset edge_count = int64_min(tg.nglobaledges - start_edge_index, FILE_CHUNKSIZE);
        packed_edge* actual_buf = (!tg.data_in_file && block_idx % ranks_per_row == my_col) ?
                                  tg.edgememory + FILE_CHUNKSIZE * (block_idx / ranks_per_row) :
                                  buf;
        /* fprintf(stderr, "%d: My range is [%" PRId64 ", %" PRId64 ") %swriting into index %" PRId64 "\n", rank, (int64_t)start_edge_index, (int64_t)(start_edge_index + edge_count), (my_col == (block_idx % ranks_per_row)) ? "" : "not ", (int64_t)(FILE_CHUNKSIZE * (block_idx / ranks_per_row))); */
        if (!tg.data_in_file && block_idx % ranks_per_row == my_col) {
          assert (FILE_CHUNKSIZE * (block_idx / ranks_per_row) + edge_count <= tg.edgememory_size);
        }
	if (tg.write_file) {
	  generate_kronecker_range(seed, SCALE, start_edge_index, start_edge_index + edge_count, actual_buf);
	  if (tg.data_in_file && my_col == (block_idx % ranks_per_row)) { /* Try to spread writes among ranks */
	    MPI_File_write_at(tg.edgefile, start_edge_index, actual_buf, edge_count, packed_edge_mpi_type, MPI_STATUS_IGNORE);
	  }
	} else {
	  /* All read rather than syncing up for a row broadcast. */
	  MPI_File_read_at(tg.edgefile, start_edge_index, actual_buf, edge_count, packed_edge_mpi_type, MPI_STATUS_IGNORE);
	}
        ptrdiff_t i;
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (i = 0; i < edge_count; ++i) {
          int64_t src = get_v0_from_edge(&actual_buf[i]);
          int64_t tgt = get_v1_from_edge(&actual_buf[i]);
          if (src == tgt) continue;
          if (src / bitmap_size_in_bytes / CHAR_BIT == my_col) {
#ifdef _OPENMP
#pragma omp atomic
#endif
            has_edge[(src / CHAR_BIT) % bitmap_size_in_bytes] |= (1 << (src % CHAR_BIT));
          }
          if (tgt / bitmap_size_in_bytes / CHAR_BIT == my_col) {
#ifdef _OPENMP
#pragma omp atomic
#endif
            has_edge[(tgt / CHAR_BIT) % bitmap_size_in_bytes] |= (1 << (tgt % CHAR_BIT));
          }
        }
      }
      free(buf);
#if 0
      /* The allreduce for each root acts like we did this: */
      MPI_Allreduce(MPI_IN_PLACE, has_edge, bitmap_size_in_bytes, MPI_UNSIGNED_CHAR, MPI_BOR, this_col);
#endif
      MPI_Comm_free(&this_col);
    } else {
      tg.edgememory = NULL;
      tg.edgememory_size = 0;
    }
    MPI_Allreduce(&tg.edgememory_size, &tg.max_edgememory_size, 1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);
    /* Find roots and max used vertex */
    {
      uint64_t counter = 0;
      int bfs_root_idx;
      for (bfs_root_idx = 0; bfs_root_idx < num_bfs_roots; ++bfs_root_idx) {
        int64_t root;
        while (1) {
          double d[2];
          make_random_numbers(2, seed1, seed2, counter, d);
          root = (int64_t)((d[0] + d[1]) * nglobalverts) % nglobalverts;
          counter += 2;
          if (counter > 2 * nglobalverts) break;
          int is_duplicate = 0;
          int i;
          for (i = 0; i < bfs_root_idx; ++i) {
            if (root == bfs_roots[i]) {
              is_duplicate = 1;
              break;
            }
          }
          if (is_duplicate) continue; /* Everyone takes the same path here */
          int root_ok = 0;
          if (in_generating_rectangle && (root / CHAR_BIT / bitmap_size_in_bytes) == my_col) {
            root_ok = (has_edge[(root / CHAR_BIT) % bitmap_size_in_bytes] & (1 << (root % CHAR_BIT))) != 0;
          }
          MPI_Allreduce(MPI_IN_PLACE, &root_ok, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
          if (root_ok) break;
        }
        bfs_roots[bfs_root_idx] = root;
        if((refBFS_Roots!=NULL) && (rank==0)){
        	if(refBFS_Roots[bfs_root_idx] != bfs_roots[bfs_root_idx])
        		fprintf(stderr,"ERROR: BFS roots do not match reffrence (Ref: %lu Here: %lu)\n",refBFS_Roots[bfs_root_idx], bfs_roots[bfs_root_idx]);
        }
      }
      num_bfs_roots = bfs_root_idx;

      /* Find maximum non-zero-degree vertex. */
      {
        int64_t i;
        max_used_vertex = 0;
        if (in_generating_rectangle) {
          for (i = bitmap_size_in_bytes * CHAR_BIT; i > 0; --i) {
            if (i > nglobalverts) continue;
            if (has_edge[(i - 1) / CHAR_BIT] & (1 << ((i - 1) % CHAR_BIT))) {
              max_used_vertex = (i - 1) + my_col * CHAR_BIT * bitmap_size_in_bytes;
              break;
            }
          }
        }
        MPI_Allreduce(MPI_IN_PLACE, &max_used_vertex, 1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);
      }
    }
    if (in_generating_rectangle) {
      MPI_Free_mem(has_edge);
    }
    if (tg.data_in_file && tg.write_file) {
      MPI_File_sync(tg.edgefile);
    }
  }
  double make_graph_stop = MPI_Wtime();
  double make_graph_time = make_graph_stop - make_graph_start;
  if (rank == 0) { /* Not an official part of the results */
    fprintf(stderr, "graph_generation:               %f s\n", make_graph_time);
  }

  /* Make user's graph data structure. */
  double data_struct_start = MPI_Wtime();
  make_graph_data_structure(&tg);
  double data_struct_stop = MPI_Wtime();
  double data_struct_time = data_struct_stop - data_struct_start;
  if (rank == 0) { /* Not an official part of the results */
    fprintf(stderr, "construction_time:              %f s\n", data_struct_time);
  }

  /* Number of edges visited in each BFS; a double so get_statistics can be
   * used directly. */
  double* edge_counts = (double*)xmalloc(num_bfs_roots * sizeof(double));
  int64_t* edge_counts_ul = (int64_t*)xmalloc(num_bfs_roots * sizeof(int64_t));

  /* Run BFS. */
  int validation_passed = 1;
  double* bfs_times = (double*)xmalloc(num_bfs_roots * sizeof(double));
  double* validate_times = (double*)xmalloc(num_bfs_roots * sizeof(double));
  uint64_t nlocalverts = get_nlocalverts_for_pred();
  int64_t* pred = (int64_t*)xMPI_Alloc_mem(nlocalverts * sizeof(int64_t));

  int bfs_root_idx;
  int CyclesPassed=0;
  int ValidationStep=0;
  if(bRunPerf==0)
  {
	  ValidationStep=1;
	  numberOfCyclesForPerf=1;
  }
  for (bfs_root_idx = 0; bfs_root_idx < num_bfs_roots; ++bfs_root_idx)
  	  bfs_times[bfs_root_idx]=0.0;

  double performance_start = MPI_Wtime();

  while(1){
	  if (rank == 0)fprintf(stderr, "Starting cycle %d.\n", CyclesPassed);

	  for (bfs_root_idx = 0; bfs_root_idx < num_bfs_roots; ++bfs_root_idx) {
		int64_t root = bfs_roots[bfs_root_idx];

		if ((rank == 0)&&(ValidationStep)) fprintf(stderr, "Running BFS %d\n", bfs_root_idx);

		/* Clear the pred array. */
		memset(pred, 0, nlocalverts * sizeof(int64_t));

		/* Do the actual BFS. */
		double bfs_start = MPI_Wtime();
		run_bfs(root, &pred[0]);
		double bfs_stop = MPI_Wtime();
		bfs_times[bfs_root_idx] += bfs_stop - bfs_start;
		if ((rank == 0)&&(ValidationStep)) fprintf(stderr, "Time for BFS %d is %f\n", bfs_root_idx, bfs_stop - bfs_start);

		/* Validate result. */
		//if (!getenv("SKIP_VALIDATION")) {
		if (ValidationStep) {
		  if (rank == 0) fprintf(stderr, "Validating BFS %d\n", bfs_root_idx);

		  double validate_start = MPI_Wtime();
		  int64_t edge_visit_count;
		  int validation_passed_one = validate_bfs_result_seq(&tg, nglobalverts, nlocalverts, root, pred, &edge_visit_count,max_used_vertex);
		  //int validation_passed_one = validate_bfs_result(&tg, max_used_vertex + 1, nlocalverts, root, pred, &edge_visit_count);
		  double validate_stop = MPI_Wtime();
		  validate_times[bfs_root_idx] = validate_stop - validate_start;
		  if (rank == 0) fprintf(stderr, "Validate time for BFS %d is %f\n", bfs_root_idx, validate_times[bfs_root_idx]);
		  edge_counts[bfs_root_idx] = (double)edge_visit_count;
		  edge_counts_ul[bfs_root_idx] = edge_visit_count;
		  if (rank == 0) fprintf(stderr, "TEPS for BFS %d is %g\n", bfs_root_idx, edge_visit_count / bfs_times[bfs_root_idx]);

		  if((refEdgeCounts!=NULL) && (rank==0)){
			  if(refEdgeCounts[bfs_root_idx]!=edge_counts_ul[bfs_root_idx])
				  fprintf(stderr,"ERROR: Edge count do not match reference (Ref: %lu Here: %lu)\n",refEdgeCounts[bfs_root_idx], edge_counts_ul[bfs_root_idx]);
		  }

		  if (!validation_passed_one) {
			  validation_passed = 0;
			  if (rank == 0) fprintf(stderr, "Validation failed for this BFS root; skipping rest.\n");
			  break;
		  }
		}
	  }
	  CyclesPassed++;
	  if((MPI_Wtime()-performance_start>=timeForPerf)||(CyclesPassed>=numberOfCyclesForPerf)){
		  if(bRunVal){
			  if(ValidationStep==0)
				  ValidationStep=1;
			  else break;
		  }
		  else break;
	  }
	  if (validation_passed==0)
		  break;
  }
  if (rank == 0)
	  fprintf(stderr,"Completed %d cycles\n", CyclesPassed);

  for (bfs_root_idx = 0; bfs_root_idx < num_bfs_roots; ++bfs_root_idx) {
	  bfs_times[bfs_root_idx]/=CyclesPassed;
  }
  /* Print results. */
  if (rank == 0) {
	int i;
	for (i = 0; i < num_bfs_roots; ++i)
		fprintf(stdout, "%lu %lu # [%2d] bfs_roots edge_visit_count\n",bfs_roots[i],edge_counts_ul[i],i);

    if (!validation_passed) {
      fprintf(stdout, "No results printed for invalid run.\n");
    } else {
      int i;
      fprintf(stdout, "SCALE:                          %d\n", SCALE);
      fprintf(stdout, "edgefactor:                     %d\n", edgefactor);
      fprintf(stdout, "NBFS:                           %d\n", num_bfs_roots);
      fprintf(stdout, "graph_generation:               %g\n", make_graph_time);
      fprintf(stdout, "num_mpi_processes:              %d\n", size);
      fprintf(stdout, "construction_time:              %g\n", data_struct_time);
      double stats[s_LAST];
      get_statistics(bfs_times, num_bfs_roots, stats);
      fprintf(stdout, "min_time:                       %g\n", stats[s_minimum]);
      fprintf(stdout, "firstquartile_time:             %g\n", stats[s_firstquartile]);
      fprintf(stdout, "median_time:                    %g\n", stats[s_median]);
      fprintf(stdout, "thirdquartile_time:             %g\n", stats[s_thirdquartile]);
      fprintf(stdout, "max_time:                       %g\n", stats[s_maximum]);
      fprintf(stdout, "mean_time:                      %g\n", stats[s_mean]);
      fprintf(stdout, "stddev_time:                    %g\n", stats[s_std]);
      get_statistics(edge_counts, num_bfs_roots, stats);
      fprintf(stdout, "min_nedge:                      %.11g\n", stats[s_minimum]);
      fprintf(stdout, "firstquartile_nedge:            %.11g\n", stats[s_firstquartile]);
      fprintf(stdout, "median_nedge:                   %.11g\n", stats[s_median]);
      fprintf(stdout, "thirdquartile_nedge:            %.11g\n", stats[s_thirdquartile]);
      fprintf(stdout, "max_nedge:                      %.11g\n", stats[s_maximum]);
      fprintf(stdout, "mean_nedge:                     %.11g\n", stats[s_mean]);
      fprintf(stdout, "stddev_nedge:                   %.11g\n", stats[s_std]);
      double* secs_per_edge = (double*)xmalloc(num_bfs_roots * sizeof(double));
      for (i = 0; i < num_bfs_roots; ++i) secs_per_edge[i] = bfs_times[i] / edge_counts[i];
      get_statistics(secs_per_edge, num_bfs_roots, stats);
      fprintf(stdout, "min_TEPS:                       %g\n", 1. / stats[s_maximum]);
      fprintf(stdout, "firstquartile_TEPS:             %g\n", 1. / stats[s_thirdquartile]);
      fprintf(stdout, "median_TEPS:                    %g\n", 1. / stats[s_median]);
      fprintf(stdout, "thirdquartile_TEPS:             %g\n", 1. / stats[s_firstquartile]);
      fprintf(stdout, "max_TEPS:                       %g\n", 1. / stats[s_minimum]);
      fprintf(stdout, "harmonic_mean_TEPS:             %g\n", 1. / stats[s_mean]);
      /* Formula from:
       * Title: The Standard Errors of the Geometric and Harmonic Means and
       *        Their Application to Index Numbers
       * Author(s): Nilan Norris
       * Source: The Annals of Mathematical Statistics, Vol. 11, No. 4 (Dec., 1940), pp. 445-448
       * Publisher(s): Institute of Mathematical Statistics
       * Stable URL: http://www.jstor.org/stable/2235723
       * (same source as in specification). */
      fprintf(stdout, "harmonic_stddev_TEPS:           %g\n", stats[s_std] / (stats[s_mean] * stats[s_mean] * sqrt(num_bfs_roots - 1)));
      free(secs_per_edge); secs_per_edge = NULL;
      free(edge_counts); edge_counts = NULL;
      get_statistics(validate_times, num_bfs_roots, stats);
      fprintf(stdout, "min_validate:                   %g\n", stats[s_minimum]);
      fprintf(stdout, "firstquartile_validate:         %g\n", stats[s_firstquartile]);
      fprintf(stdout, "median_validate:                %g\n", stats[s_median]);
      fprintf(stdout, "thirdquartile_validate:         %g\n", stats[s_thirdquartile]);
      fprintf(stdout, "max_validate:                   %g\n", stats[s_maximum]);
      fprintf(stdout, "mean_validate:                  %g\n", stats[s_mean]);
      fprintf(stdout, "stddev_validate:                %g\n", stats[s_std]);
#if 0
      for (i = 0; i < num_bfs_roots; ++i) {
        fprintf(stdout, "Run %3d:                        %g s, validation %g s\n", i + 1, bfs_times[i], validate_times[i]);
      }
#endif
    }
  }


  MPI_Free_mem(pred);
  free(bfs_roots);
  free_graph_data_structure();

  if (tg.data_in_file) {
    MPI_File_close(&tg.edgefile);
  } else {
    free(tg.edgememory); tg.edgememory = NULL;
  }

  free(bfs_times);
  free(validate_times);
  free(edge_counts_ul);
  cleanup_globals();
  MPI_Finalize();
  return 0;
}
