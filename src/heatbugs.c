/*
 * This file is part of heatbugs_GPU.
 *
 * heatbugs_GPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * heatbugs_GPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with heatbugs_GPU. If not, see <http://www.gnu.org/licenses/>.
 * */

#include <stdio.h>			/* printf(...)	*/
#include <stdlib.h>			/* exit(...)	*/
#include <string.h>

#include <cf4ocl2.h>		/* glib.h included by cf4ocl2.h to handle GError*, MIN(...), g_random_int(). */

#include "heatbugs.h"


/* The cl kernel file pathname. */
#define CL_KERNEL_SRC_FILE "./heatbugs.cl"

/* The main kernel function names. */
#define KERNEL_NAME__INIT_RANDOM			"init_random"
#define KERNEL_NAME__INIT_MAPS				"init_maps"
#define KERNEL_NAME__INIT_SWARM				"init_swarm"
#define KERNEL_NAME__BUG_STEP				"bug_step"
#define KERNEL_NAME__COMP_WORLD_HEAT		"comp_world_heat"
#define KERNEL_NAME__UNHAPPINESS_S1_REDUCE	"unhappiness_step1_reduce"
#define KERNEL_NAME__UNHAPPINESS_S2_MEAN	"unhappiness_step2_mean"



/* OpenCL options. */
#define DIMS_1		1				/* Kernel dimensions: 1 dimension.  */
#define DIMS_2		2				/* Kernel dimensions: 2 dimensions. */
#define NON_BLOCK	CL_FALSE		/* Non blocked operation, read / write... */




#define OKI_DOKI 0
#define NOT_DOKI -1

#define BUG '@'
#define EMPTY '-'




#define ERROR_MSG_AND_EXIT( msg ) \
	do { fprintf( stderr, "\n%s\n\n", msg ); exit( NOT_DOKI ); } while( 0 )

#define HANDLE_ERROR( err ) \
	if (err != NULL) { ERROR_MSG_AND_EXIT( err->message ); }


/** Holder for all OpenCl objects. */
typedef struct ocl_objects {
	CCLContext *ctx;					/* Context.		*/
	CCLDevice *dev;						/* Device.		*/
	CCLQueue *queue;					/* Queue.		*/
	CCLProgram *prg;					/* Kernel code.	*/
} OCLObjects_t;

/** Holder for all HeatBugs kernels. */
typedef struct hb_kernels {
	CCLKernel *init_random;				/* kernel: Initiate random seeds. */
	CCLKernel *init_maps;				/* kernel: Initiate heat map. */
	CCLKernel *init_swarm;				/* kernel: Initiate the world of bugs. */
	CCLKernel *comp_world_heat;			/* kernel: Compute world heat, diffusion then evaporation. */
	CCLKernel *bug_step;				/* kernel: Perform a new bug movement. */
	CCLKernel *unhappiness;				/* kernel: compute unhappiness. */
	CCLKernel *test;
} HBKernels_t;

/** Global work sizes for all kernels. */
typedef struct hb_global_work_sizes	{
	size_t init_random[ DIMS_1 ];
	size_t init_maps[ DIMS_1 ];
	size_t init_swarm[ DIMS_1 ];
	size_t comp_world_heat[ DIMS_2 ];
	size_t bug_step;
	size_t unhappiness;
	size_t test;
} HBGlobalWorkSizes_t;

/* Local work sizes for all kernels. */
typedef struct hb_local_work_sizes	{
	size_t init_random[ DIMS_1 ];
	size_t init_maps[ DIMS_1 ];
	size_t init_swarm[ DIMS_1 ];
	size_t comp_world_heat[ DIMS_2 ];
	size_t bug_step;
	size_t unhappiness;
	size_t test;
} HBLocalWorkSizes_t;


/** Host and Device Buffers. */
typedef struct hb_buffers {

	/** HOST: Random seeds buffer. */
	cl_uint *hst_rng_state;						/* DEBUG: (to remove). */
	/** HOST: Swarm, the bugs position in the map. */
	cl_uint *hst_swarm[2];
	/** HOST: It's the bugs map. Each cell is: 'ideal-Temperature':8bit 'bug':1bit 'output_heat':7bit. */
	cl_uint *hst_swarm_map;
	/** HOST: Temperature map & the buffer. */
	cl_float *hst_heat_map[2];					/* DEBUG: (to remove). */
	/** HOST: To get the Unhappiness. */
	cl_float *hst_unhappiness;

	/** DEVICE: Random seeds buffer. */
	CCLBuffer *dev_rng_state;
	/** DEVICE: Swarm, the bugs position in the map. */
	CCLBuffer *dev_swarm[2];
	/** DEVICE: It's the bugs map. Each cell is: 'ideal-Temperature':8bit 'bug':1bit 'output_heat':7bit. */
	CCLBuffer *dev_swarm_map;
	/** DEVICE: temperature map & the buffer. */
	CCLBuffer *dev_heat_map[2];
	/** DEVICE: To compute bugs unhappiness. */
	CCLBuffer *dev_unhappiness;

} HBBuffers_t;

/** Buffers sizes. */
typedef struct hb_buffers_size {
	size_t rng_state;
	size_t swarm;
	size_t swarm_map;
	size_t heat_map;
	size_t unhappiness;
} HBBuffersSize_t;




const char version[] = "Heatbugs simulation for GPU (parallel processing) v2.6.";



/**
 *	OpenCL built in compiler parameter's template.
 *	The template will be used to fill a string, that by turn, will be used as
 *	OpenCL compiler options. This way, the constants will be available in the
 *	kernel as if they were defined there.
 *	The '-D' option, is the kernel's compiler directive to make the constants
 *	available to the kernel.
 *	*/
const char *cl_compiler_opts_template = QUOTE(
	-D INIT_SEED=%u
	-D BUGS_TEMPERATURE_MIN_IDEAL=%u
	-D BUGS_TEMPERATURE_MAX_IDEAL=%u
	-D BUGS_HEAT_MIN_OUTPUT=%u
	-D BUGS_HEAT_MAX_OUTPUT=%u
	-D BUGS_RANDOM_MOVE_CHANCE=%f
	-D WORLD_DIFFUSION_RATE=%f
	-D WORLD_EVAPORATION_RATE=%f
	-D BUGS_NUMBER=%zu
	-D WORLD_WIDTH=%zu
	-D WORLD_HEIGHT=%zu
	-D WORLD_SIZE=%zu
);


#define HB_ERROR hb_error_quark()



static GQuark hb_error_quark( void ) {
	return g_quark_from_static_string( "hb-error-quark" );
}


//static GQuark ccl_error_quark( void ) {
//	return g_quark_from_static_string( "ccl-error-quark" );
//}





static void setupParameters( parameters_t *const params, GError **err )
{
	 /* TODO: Get simulation parameters. This ones are defaults. */
	params->bugs_temperature_min_ideal = 20;	/* 10 */
	params->bugs_temperature_max_ideal = 30;	/* 40 */
	params->bugs_heat_min_output = 15;			/*  5 */
	params->bugs_heat_max_output = 25;			/* 25 */
	params->bugs_random_move_chance = 0.0;		/*  0%  Valid:[0 .. 100] */

	params->world_diffusion_rate = 0.9;			/* 90%  */
	params->world_evaporation_rate = 0.01;		/*  1%  */
	params->world_width =  5;					/* 100	*/
	params->world_height = 5;					/* 100	*/

	params->bugs_number = 20;					/* 100 Bugs in the world. */
	params->numIterations =  1000;				/* 0 = NonStop. */


	params->world_size = params->world_height * params->world_width;


	ccl_if_err_create_goto( *err, HB_ERROR, params->bugs_number == 0, HB_BUGS_ZERO, error_handler, "There are no bugs." );

	ccl_if_err_create_goto( *err, HB_ERROR, params->bugs_number >= params->world_size, HB_BUGS_OVERFLOW, error_handler, "Number of bugs exceed available world slots." );

	if (params->bugs_number >= 0.8 * params->world_size)
		fprintf( stderr, "Warning: Bugs number close to available world slots.\n" );


error_handler:
	/* If error handler is reached leave function imediately. */

	return;
}


static void setupOCLObjects( OCLObjects_t *const oclobj, const parameters_t *const params, GError **err )
{
	FILE *uranddev = NULL;
	cl_uint init_seed;

	char cl_compiler_opts[512];			/* OpenCL built in compiler/builder parameters. */

	GError *err_setup = NULL;


	/* Read initial seed from linux /dev/urandom */
	uranddev = fopen( "/dev/urandom", "r" );
	fread( &init_seed, sizeof(cl_uint), 1, uranddev );
	fclose( uranddev );

	/**
	 *	Next are OpenCL build options that are send as defines to kernel,
	 *	preventing the need to send extra arguments to kernel. These function
	 *	as work-items private variables.
	 *	A null terminated string is granted in 'cl_compiler_opts'.
	 *	*/

	sprintf( cl_compiler_opts, cl_compiler_opts_template,
				init_seed,
				params->bugs_temperature_min_ideal,
				params->bugs_temperature_max_ideal,
				params->bugs_heat_min_output,
				params->bugs_heat_max_output,
				params->bugs_random_move_chance,
				params->world_diffusion_rate,
				params->world_evaporation_rate,
				params->bugs_number,
				params->world_width,
				params->world_height,
				params->world_size );

	/* DEBUG: Show OpenCL build options. */
	printf("\n\nbuild Options:\n----------------------\n%s\n\n", cl_compiler_opts);


	/* *** GPU preparation. Initiate OpenCL objects. *** */

	/* Create context wrapper for a GPU device. The first found GPU device will be used. */
	oclobj->ctx = ccl_context_new_gpu( &err_setup );
	ccl_if_err_propagate_goto( err, err_setup, error_handler );

	/* Get the device (index 0) in te context. */
	oclobj->dev = ccl_context_get_device( oclobj->ctx, 0, &err_setup );
	ccl_if_err_propagate_goto( err, err_setup, error_handler );

	/* Create a command queue. */
	oclobj->queue = ccl_queue_new( oclobj->ctx, oclobj->dev, CL_QUEUE_PROFILING_ENABLE, &err_setup );
	ccl_if_err_propagate_goto( err, err_setup, error_handler );


	/* *** Program build. *** */

	/* Create a new program from kernel source. */
	oclobj->prg = ccl_program_new_from_source_file( oclobj->ctx, CL_KERNEL_SRC_FILE, &err_setup );
	ccl_if_err_propagate_goto( err, err_setup, error_handler );

	/* Build CL Program. */
	ccl_program_build( oclobj->prg, cl_compiler_opts, &err_setup );
	ccl_if_err_propagate_goto( err, err_setup, error_handler );

error_handler:
	/* If error handler is reached leave function imediately. */

	return;
}


/**
 * Create the kernel objects, one for each kernel function.
 * */
static void setupHBKernels( HBKernels_t *const krnl, HBGlobalWorkSizes_t *const gws, HBLocalWorkSizes_t *const lws, const OCLObjects_t *const oclobj, const parameters_t *const params, GError **err )
{
	GError *err_setup = NULL;
	size_t world_realdims[2] = {params->world_width, params->world_height};


	/** RANDOM INITIALIZATION Kernel. A random per bug. */

	krnl->init_random = ccl_kernel_new( oclobj->prg, KERNEL_NAME__INIT_RANDOM, &err_setup );
	ccl_if_err_propagate_goto( err, err_setup, error_handler );

	ccl_kernel_suggest_worksizes( krnl->init_random, oclobj->dev, DIMS_1, &params->bugs_number, gws->init_random, lws->init_random, &err_setup );
	ccl_if_err_propagate_goto( err, err_setup, error_handler );

	printf( "[ kernel ]: init_random.\n    '-> bugs_num = %zu; gws = %zu; lws = %zu\n", params->bugs_number, gws->init_random[0], lws->init_random[0] );


	/** SWARM_MAP and HEAT_MAP INITIALIZATION Kernel. */

	krnl->init_maps = ccl_kernel_new( oclobj->prg, KERNEL_NAME__INIT_MAPS, &err_setup );
	ccl_if_err_propagate_goto( err, err_setup, error_handler );

	ccl_kernel_suggest_worksizes( krnl->init_maps, oclobj->dev, DIMS_1, &params->world_size, gws->init_maps, lws->init_maps, &err_setup );
	ccl_if_err_propagate_goto( err, err_setup, error_handler );

	printf( "[ kernel ]: init_maps.\n    '-> world_size = %zu; gws = %zu; lws = %zu\n", params->world_size, gws->init_maps[0], lws->init_maps[0] );


	/** SWARM INITIALIZATION. To put bugs in the world and Reset unhappiness vector. */

	krnl->init_swarm = ccl_kernel_new( oclobj->prg, KERNEL_NAME__INIT_SWARM, &err_setup );
	ccl_if_err_propagate_goto( err, err_setup, error_handler );

	/* Dimensions are relactive to the number of bugs to be dropped in swarm map. */
	ccl_kernel_suggest_worksizes( krnl->init_swarm, oclobj->dev, DIMS_1, &params->bugs_number, gws->init_swarm, lws->init_swarm, &err_setup );
	ccl_if_err_propagate_goto( err, err_setup, error_handler );

	printf( "[ kernel ]: init_swarm.\n    '-> bugs_num = %zu; gws = %zu; lws = %zu\n", params->bugs_number, gws->init_swarm[0], lws->init_swarm[0] );


	/** WORLD HEAT Computation. Compute world diffusion followed by world evapotation. */

	krnl->comp_world_heat = ccl_kernel_new( oclobj->prg, KERNEL_NAME__COMP_WORLD_HEAT, &err_setup );
	ccl_if_err_propagate_goto( err, err_setup, error_handler );

	ccl_kernel_suggest_worksizes( krnl->comp_world_heat, oclobj->dev, DIMS_2, world_realdims, gws->comp_world_heat, lws->comp_world_heat, &err_setup );
	ccl_if_err_propagate_goto( err, err_setup, error_handler );

	printf( "[ kernel ]: comp_world_heat.\n    '-> world_dims = [%zu, %zu]; gws = [%zu, %zu]; lws = [%zu, %zu]\n", world_realdims[0], world_realdims[1], gws->comp_world_heat[0], gws->comp_world_heat[1], lws->comp_world_heat[0], lws->comp_world_heat[1] );


	/** Kernel for bug step. */

//	krnl->bug_step = ccl_kernel_new( oclobj->prg, KERNEL_NAME__BUG_STEP, &err_setup );
//	ccl_if_err_propagate_goto( err, err_setup, error_handler );

//	krnl->unhappiness = ccl_kernel_new( oclobj->prg, KERNEL_NAME__UNHAPPINESS, &err_setup );
//	ccl_if_err_propagate_goto( err, err_setup, error_handler );

//	krnl->test = ccl_kernel_new( oclobj->prg, KERNEL_NAME__TEST, &err_setup );
//	ccl_if_err_propagate_goto( err, err_setup, error_handler );



	/* TODO: Calculate global work sizes and local work sizes. */

	//printf("rws={%ld}, gws={%ld}, lws={%ld}\n\n", params->world_size, gws->init, lws->init);

	printf( "\n" );


error_handler:
		/* If error handler is reached leave function imediately. */

	return;
}


static void setupBuffers( HBBuffers_t *const buf, HBBuffersSize_t *const bufsz, const OCLObjects_t *const oclobj, const parameters_t *const params, GError **err )
{
	GError *err_setbuf = NULL;


	/** RANDOM SEEDS */

	bufsz->rng_state = params->bugs_number * sizeof( cl_uint );

	/* DEBUG: (to remove) Allocate vector for random seeds in host memory. */
	buf->hst_rng_state = (cl_uint *) malloc( bufsz->rng_state );
	ccl_if_err_create_goto( *err, HB_ERROR, buf->hst_rng_state == NULL, HB_MALLOC_FAILURE, error_handler, "Unable to allocate host memory for random seeds/states." );

	/* Allocate vector of random seeds in device memory. */
	buf->dev_rng_state = ccl_buffer_new( oclobj->ctx, CL_MEM_READ_WRITE, bufsz->rng_state, NULL, &err_setbuf );
	ccl_if_err_propagate_goto( err, err_setbuf, error_handler );


	/** SWARM */

	bufsz->swarm = params->bugs_number * sizeof( cl_uint );

	buf->hst_swarm[0] = (cl_uint *) malloc( bufsz->swarm );
	ccl_if_err_create_goto( *err, HB_ERROR, buf->hst_swarm[0] == NULL, HB_MALLOC_FAILURE, error_handler, "Unabble to allocate host memory for swarm 0.");

	buf->hst_swarm[1] = (cl_uint *) malloc( bufsz->swarm );
	ccl_if_err_create_goto( *err, HB_ERROR, buf->hst_swarm[1] == NULL, HB_MALLOC_FAILURE, error_handler, "Unabble to allocate host memory for swarm 1.");

	buf->dev_swarm[0] = ccl_buffer_new( oclobj->ctx, CL_MEM_READ_WRITE, bufsz->swarm, NULL, &err_setbuf );
	ccl_if_err_propagate_goto( err, err_setbuf, error_handler );

	buf->dev_swarm[1] = ccl_buffer_new( oclobj->ctx, CL_MEM_READ_WRITE, bufsz->swarm, NULL, &err_setbuf );
	ccl_if_err_propagate_goto( err, err_setbuf, error_handler );


	/** SWARM MAP */

	bufsz->swarm_map = params->world_size * sizeof( cl_uint );

	/* DEBUG: (to remove) Allocate vector for the swarm map in host memory. */
	buf->hst_swarm_map = (cl_uint *) malloc( bufsz->swarm_map );
	ccl_if_err_create_goto( *err, HB_ERROR, buf->hst_swarm_map == NULL, HB_MALLOC_FAILURE, error_handler, "Unable to allocate host memory for swarm map." );

	/* Allocate vector for the swarm map in device memory. */
	buf->dev_swarm_map = ccl_buffer_new( oclobj->ctx, CL_MEM_READ_WRITE, bufsz->swarm_map, NULL, &err_setbuf );
	ccl_if_err_propagate_goto( err, err_setbuf, error_handler );


	/** HEAT MAP */

	bufsz->heat_map = params->world_size * sizeof( cl_float );

	/* DEBUG: (to remove) Allocate temperature vector in host memory. */
	buf->hst_heat_map[0] = (cl_float *) malloc( bufsz->heat_map );
	ccl_if_err_create_goto( *err, HB_ERROR, buf->hst_heat_map[0] == NULL, HB_MALLOC_FAILURE, error_handler, "Unable to allocate host memory for temperature / heat map 1." );

	/* DEBUG: (to remove) Allocate temperature vector in host memory. */
	buf->hst_heat_map[1] = (cl_float *) malloc( bufsz->heat_map );
	ccl_if_err_create_goto( *err, HB_ERROR, buf->hst_heat_map[1] == NULL, HB_MALLOC_FAILURE, error_handler, "Unable to allocate host memory for temperature / heat map 2." );

	/* Allocate two vectors of temperature maps in device memory, one is used for double buffer. */
	buf->dev_heat_map[0] = ccl_buffer_new( oclobj->ctx, CL_MEM_READ_WRITE, bufsz->heat_map, NULL, &err_setbuf );
	ccl_if_err_propagate_goto( err, err_setbuf, error_handler );

	buf->dev_heat_map[1] = ccl_buffer_new( oclobj->ctx, CL_MEM_READ_WRITE, bufsz->heat_map, NULL, &err_setbuf );
	ccl_if_err_propagate_goto( err, err_setbuf, error_handler );


	/** UNHAPINESS */

	bufsz->unhappiness = params->bugs_number * sizeof( cl_float );

	buf->hst_unhappiness = (cl_float *) malloc( bufsz->unhappiness );
	ccl_if_err_create_goto( *err, HB_ERROR, buf->hst_unhappiness == NULL, HB_MALLOC_FAILURE, error_handler, "Unable to allocate host memory for bugs unhappiness." );

	buf->dev_unhappiness = ccl_buffer_new( oclobj->ctx, CL_MEM_READ_WRITE, bufsz->unhappiness, NULL, &err_setbuf );
	ccl_if_err_propagate_goto( err, err_setbuf, error_handler );



error_handler:
	/* If error handler is reached leave function imediately. */

	return;
}


/**
 *	Run all init kernels.
 *	*/
static inline void initiate( HBKernels_t *const krnl, HBGlobalWorkSizes_t *const gws, HBLocalWorkSizes_t *const lws, OCLObjects_t *const oclobj, HBBuffers_t *const buf, HBBuffersSize_t *const bufsz, parameters_t *const params,  GError **err )
{
	GError *err_init = NULL;

	/** Events. */
	CCLEvent *evt_rdwr = NULL;			/* Event signal for reads and writes. */
	CCLEvent *evt_krnl_exec = NULL;		/* Event signal for kernel execution. */
	CCLEventWaitList ewl = NULL;		/* A list of OpenCL events. */


	/** INIT RANDOM. */

	printf( "Init random:\n\tgws = %zu; lws = %zu\n", gws->init_random[0], lws->init_random[0] );

	ccl_kernel_set_arg( krnl->init_random, 0, buf->dev_rng_state );

	evt_krnl_exec = ccl_kernel_enqueue_ndrange( krnl->init_random, oclobj->queue, DIMS_1, NULL, gws->init_random, lws->init_random, &ewl, &err_init );
	ccl_if_err_propagate_goto( err, err_init, error_handler );

	/* Add kernel termination event to the wait list. */
	ccl_event_wait_list_add( &ewl, evt_krnl_exec, NULL );


	/* Read GPU generated seeds/gen_states, after wait for kernel termination. */

	evt_rdwr = ccl_buffer_enqueue_read( buf->dev_rng_state, oclobj->queue, NON_BLOCK, 0, bufsz->rng_state, buf->hst_rng_state, &ewl, &err_init );
	ccl_if_err_propagate_goto( err, err_init, error_handler );

	/* Add read event to the wait list. */
	ccl_event_wait_list_add( &ewl, evt_rdwr, NULL );


	/* Wait events termination. */
	ccl_event_wait( &ewl, &err_init );
	ccl_if_err_propagate_goto( err, err_init, error_handler );


	/* DEBUG: Show GPU generated seeds. */
	for (size_t i = 0; i < params->bugs_number; ++i) {
		printf( "%u, ", buf->hst_rng_state[i]);
	}
	printf( "\n\n" );



	/** RESET SWARM_MAP and HEAT_MAP. */

	printf( "Init maps:\n\tgws = %zu; lws = %zu\n", gws->init_maps[0], lws->init_maps[0] );

	ccl_kernel_set_arg( krnl->init_maps, 0, buf->dev_swarm_map );
	ccl_kernel_set_arg( krnl->init_maps, 1, buf->dev_heat_map[0] );
	ccl_kernel_set_arg( krnl->init_maps, 2, buf->dev_heat_map[1] );

	evt_krnl_exec = ccl_kernel_enqueue_ndrange( krnl->init_maps, oclobj->queue, DIMS_1, NULL, gws->init_maps, lws->init_maps, &ewl, &err_init );
	ccl_if_err_propagate_goto( err, err_init, error_handler );

	/* Add kernel termination event to the wait list. */
	ccl_event_wait_list_add( &ewl, evt_krnl_exec, NULL );


	/* Wait events termination. */
	ccl_event_wait( &ewl, &err_init );
	ccl_if_err_propagate_goto( err, err_init, error_handler );


	/* Read swarm map, after wait for kernel termination. */

	evt_rdwr = ccl_buffer_enqueue_read( buf->dev_swarm_map, oclobj->queue, NON_BLOCK, 0, bufsz->swarm_map, buf->hst_swarm_map, &ewl, &err_init );
	ccl_if_err_propagate_goto( err, err_init, error_handler );

	ccl_event_wait_list_add( &ewl, evt_rdwr, NULL );


	/* Read heat map. */

	evt_rdwr = ccl_buffer_enqueue_read( buf->dev_heat_map[0], oclobj->queue, NON_BLOCK, 0, bufsz->heat_map, buf->hst_heat_map[0], &ewl, &err_init );
	ccl_if_err_propagate_goto( err, err_init, error_handler );

	ccl_event_wait_list_add( &ewl, evt_rdwr, NULL );

	evt_rdwr = ccl_buffer_enqueue_read( buf->dev_heat_map[1], oclobj->queue, NON_BLOCK, 0, bufsz->heat_map, buf->hst_heat_map[1], &ewl, &err_init );
	ccl_if_err_propagate_goto( err, err_init, error_handler );

	ccl_event_wait_list_add( &ewl, evt_rdwr, NULL );


	/* Wait last event termination. */
	ccl_event_wait( &ewl, &err_init );
	ccl_if_err_propagate_goto( err, err_init, error_handler );


	printf( "\tTesting result from init maps. If no Err pass!\n\n" );
	for (size_t i = 0; i < params->world_size; ++i) {
		if (buf->hst_heat_map[0][i] != 0.0f) {
			printf( "\t\tErr heatmap buf_0 at pos: %zu\n", i );
			exit(1);
		}
		if (buf->hst_heat_map[1][i] != 0.0f) {
			printf( "\t\tErr heatmap buf_1 at pos: %zu\n", i );
			exit(1);
		}
		if (buf->hst_swarm_map[i] != 0) {
			printf( "\t\tErr swarm_map at pos: %zu\n", i );
			exit(1);
		}
	}


	/* INIT SWARM Fill the swarm map with bugs, compute unhapinnes. */

	printf( "Init swarm:\n\tgws = %zu; lws = %zu\n", gws->init_swarm[0], lws->init_swarm[0] );

	ccl_kernel_set_arg( krnl->init_swarm, 0, buf->dev_swarm[0] );
	ccl_kernel_set_arg( krnl->init_swarm, 1, buf->dev_swarm[1] );
	ccl_kernel_set_arg( krnl->init_swarm, 2, buf->dev_swarm_map );
	ccl_kernel_set_arg( krnl->init_swarm, 3, buf->dev_unhappiness );
	ccl_kernel_set_arg( krnl->init_swarm, 4, buf->dev_rng_state );

	evt_krnl_exec = ccl_kernel_enqueue_ndrange( krnl->init_swarm, oclobj->queue, DIMS_1, NULL, gws->init_swarm, lws->init_swarm, &ewl, &err_init );
	ccl_if_err_propagate_goto( err, err_init, error_handler );

	/* Add kernel termination event to the wait list. */
	ccl_event_wait_list_add( &ewl, evt_krnl_exec, NULL );


	/* Wait events termination. */
	ccl_event_wait( &ewl, &err_init );
	ccl_if_err_propagate_goto( err, err_init, error_handler );


	/* Read swarm map, after wait for kernel termination. */

	evt_rdwr = ccl_buffer_enqueue_read( buf->dev_swarm_map, oclobj->queue, NON_BLOCK, 0, bufsz->swarm_map, buf->hst_swarm_map, &ewl, &err_init );
	ccl_if_err_propagate_goto( err, err_init, error_handler );

	ccl_event_wait_list_add( &ewl, evt_rdwr, NULL );

	/* Read swarm. */

	evt_rdwr = ccl_buffer_enqueue_read( buf->dev_swarm[0], oclobj->queue, NON_BLOCK, 0, bufsz->swarm, buf->hst_swarm[0], &ewl, &err_init );
	ccl_if_err_propagate_goto( err, err_init, error_handler );

	ccl_event_wait_list_add( &ewl, evt_rdwr, NULL );

	evt_rdwr = ccl_buffer_enqueue_read( buf->dev_swarm[1], oclobj->queue, NON_BLOCK, 0, bufsz->swarm, buf->hst_swarm[1], &ewl, &err_init );
	ccl_if_err_propagate_goto( err, err_init, error_handler );

	ccl_event_wait_list_add( &ewl, evt_rdwr, NULL );

	/* Read unhappiness. */

	evt_rdwr = ccl_buffer_enqueue_read( buf->dev_unhappiness, oclobj->queue, NON_BLOCK, 0, bufsz->unhappiness, buf->hst_unhappiness, &ewl, &err_init );
	ccl_if_err_propagate_goto( err, err_init, error_handler );

	ccl_event_wait_list_add( &ewl, evt_rdwr, NULL );




	/* Wait last event termination. */
	ccl_event_wait( & ewl, &err_init );
	ccl_if_err_propagate_goto( err, err_init, error_handler );


	printf( "\nShow results for swarm initiation.\n\n" );
	for (size_t i = 0; i < params->world_height; i++ )
	{
		for (size_t j = 0; j < params->world_width; j++ )
		{
			printf( "%10u\t", buf->hst_swarm_map[ i * params->world_width + j ] );
		}
		printf( "\n" );
	}
	printf( "\n" );


	printf( "Show results for swarm and unhappiness.\n\n" );
	for ( size_t i = 0; i < params->bugs_number; i++ )
	{
		size_t lin, col;
		lin = buf->hst_swarm[0][i] / params->world_width;
		col = buf->hst_swarm[0][i] % params->world_width;
		printf( "bug: %4zu  loc: %4u [%zu, %zu] -> intent: %u  unhapp: %f\n", i, buf->hst_swarm[0][i], lin, col, buf->hst_swarm[1][i], buf->hst_unhappiness[i] );
	}
	printf( "\n" );


error_handler:
	/* If error handler is reached leave function imediately. */

	return;
}


/*
 *	Compute the world heat. That is, the two step operation, diffusion
 *	followed by evaporation.
 *	*/
static inline void comp_world_heat( HBKernels_t *const krnl, HBGlobalWorkSizes_t *const gws, HBLocalWorkSizes_t *const lws, OCLObjects_t *const oclobj, HBBuffers_t *const buf, HBBuffersSize_t *const bufsz, parameters_t *const params,  GError **err )
{
	GError *err_comp_world_heat = NULL;

	/** Events. */
	CCLEvent *evt_rdwr = NULL;			/* Event signal for reads and writes. */
	CCLEvent *evt_krnl_exec = NULL;		/* Event signal for kernel execution. */
	CCLEventWaitList ewl = NULL;		/* A list of OpenCL events. */


	printf( "Compute world heat:\n\tgws = [%zu,%zu]; lws = [%zu,%zu]\n\n", gws->comp_world_heat[0], gws->comp_world_heat[1], lws->comp_world_heat[0], lws->comp_world_heat[1] );


// DEBUG: Only to test diffusion and evaporation: Init a heat map.
	for ( cl_uint i = 0; i < params->world_size; ++i )
		buf->hst_heat_map[0][i] = i + 1.0f;



// DEBUG: Only to test diffusion and evaporation: Send buffer to device.
	evt_rdwr = ccl_buffer_enqueue_write( buf->dev_heat_map[0], oclobj->queue, NON_BLOCK, 0, bufsz->heat_map, buf->hst_heat_map[0], &ewl, &err_comp_world_heat );
	ccl_if_err_propagate_goto( err, err_comp_world_heat, error_handler );

	ccl_event_wait_list_add( &ewl, evt_rdwr, NULL );



	/** Compute world heat. */

	ccl_kernel_set_arg( krnl->comp_world_heat, 0, buf->dev_heat_map[0] );
	ccl_kernel_set_arg( krnl->comp_world_heat, 1, buf->dev_heat_map[1] );

	evt_krnl_exec = ccl_kernel_enqueue_ndrange( krnl->comp_world_heat, oclobj->queue, DIMS_2, NULL, gws->comp_world_heat, lws->comp_world_heat, &ewl, &err_comp_world_heat );
	ccl_if_err_propagate_goto( err, err_comp_world_heat, error_handler );

	/* Add kernel termination event to the wait list. */
	ccl_event_wait_list_add( &ewl, evt_krnl_exec, NULL );

	ccl_event_wait( &ewl, &err_comp_world_heat );
	ccl_if_err_propagate_goto( err, err_comp_world_heat, error_handler );


	/* DEBUG: check world map. */
	evt_rdwr = ccl_buffer_enqueue_read( buf->dev_heat_map[0], oclobj->queue, NON_BLOCK, 0, bufsz->heat_map, buf->hst_heat_map[0], &ewl, &err_comp_world_heat );
	ccl_if_err_propagate_goto( err, err_comp_world_heat, error_handler );

	evt_rdwr = ccl_buffer_enqueue_read( buf->dev_heat_map[1], oclobj->queue, NON_BLOCK, 0, bufsz->heat_map, buf->hst_heat_map[1], &ewl, &err_comp_world_heat );
	ccl_if_err_propagate_goto( err, err_comp_world_heat, error_handler );

	ccl_event_wait_list_add( &ewl, evt_rdwr, NULL );

	ccl_event_wait( &ewl, &err_comp_world_heat );
	ccl_if_err_propagate_goto( err, err_comp_world_heat, error_handler );

	/* Show heat map. */
	printf( "Computed result for heatmap:\n" );
	for (cl_uint i = 0; i < params->world_size; ++i) {
		if (i % params->world_width == 0 ) printf ("-----------------\n");
		printf( "%f -> %f\n", buf->hst_heat_map[0][i], buf->hst_heat_map[1][i] );
	}


error_handler:
	/* If error handler is reached leave function imediately. */

	return;
}





int main ( int argc, char *argv[] )
{
	FILE *hbResult = NULL;
	GError *err = NULL;					/* Error reporting object, from GLib. */

	parameters_t params;				/* Host data; simulation parameters. */
//	real_t average_unhapiness;			/* Unhapiness in host (buffer). */

	/** OpenCL related objects: context, device, queue, program. */
	OCLObjects_t oclobj = { NULL, NULL, NULL, NULL };

	/** Kernels. */
	HBKernels_t krnl = { NULL, NULL, NULL, NULL, NULL, NULL, NULL };
	/** Global work sizes for all kernels. */
	HBGlobalWorkSizes_t gws = { {0}, {0}, {0}, {0, 0}, 0, 0, 0 };
	/** Local work sizes for all kernels. */
	HBLocalWorkSizes_t  lws = { {0}, {0}, {0}, {0, 0}, 0, 0, 0 };

	/** Buffers for host and device. */
	HBBuffers_t buf = { NULL, { NULL, NULL }, NULL, { NULL, NULL }, NULL, NULL, { NULL, NULL }, NULL, { NULL, NULL }, NULL };
	/** Buffers sizes. */
	HBBuffersSize_t bufsz;



	setupParameters( &params, &err );
	ccl_if_err_goto( err, error_handler );

	setupOCLObjects( &oclobj, &params, &err );
	ccl_if_err_goto( err, error_handler );

	setupHBKernels( &krnl, &gws, &lws, &oclobj, &params, &err );
	ccl_if_err_goto( err, error_handler );

	setupBuffers( &buf, &bufsz, &oclobj, &params, &err );
	ccl_if_err_goto( err, error_handler );



	/* Open output file for results. */
	hbResult = fopen( "../results/heatbugsC_00.csv", "w+" );	/* Open file overwrite. */

	if (hbResult == NULL)
		ERROR_MSG_AND_EXIT( "Error: Could not open output file." );


	/** Run all init kernels. */
	initiate( &krnl, &gws, &lws, &oclobj, &buf, &bufsz, &params, &err );
	ccl_if_err_goto( err, error_handler );

	comp_world_heat( &krnl, &gws, &lws, &oclobj, &buf, &bufsz, &params, &err );
	ccl_if_err_goto( err, error_handler );



	goto clean_all;


error_handler:

	/* Handle error. */
	fprintf( stderr, "Error: %s\n", err->message );
	g_error_free( err );


clean_all:

	/* Clean / Destroy all allocated items. */

	/* Destroy host buffers. */
	if (buf.hst_unhappiness) free( buf.hst_unhappiness );
	if (buf.hst_heat_map[1]) free( buf.hst_heat_map[1] );
	if (buf.hst_heat_map[0]) free( buf.hst_heat_map[0] );
	if (buf.hst_swarm_map) free( buf.hst_swarm_map );
	if (buf.hst_swarm[1]) free( buf.hst_swarm[1] );
	if (buf.hst_swarm[0]) free( buf.hst_swarm[0] );
	if (buf.hst_rng_state) free( buf.hst_rng_state );

	/* Destroy Device buffers. */
	if (buf.dev_unhappiness) ccl_buffer_destroy( buf.dev_unhappiness );
	if (buf.dev_heat_map[1]) ccl_buffer_destroy( buf.dev_heat_map[1] );
	if (buf.dev_heat_map[0]) ccl_buffer_destroy( buf.dev_heat_map[0] );
	if (buf.dev_swarm_map) ccl_buffer_destroy( buf.dev_swarm_map );
	if (buf.dev_swarm[1]) ccl_buffer_destroy( buf.dev_swarm[1] );
	if (buf.dev_swarm[0]) ccl_buffer_destroy( buf.dev_swarm[0] );
	if (buf.dev_rng_state) ccl_buffer_destroy( buf.dev_rng_state );

	/** Destroy kernel wrappers. */
//	if (krnl.test) ccl_kernel_destroy( krnl.test );
//	if (krnl.unhappiness) ccl_kernel_destroy( krnl.unhappiness );
//	if (krnl.bug_step) ccl_kernel_destroy( krnl.bug_step );
	if (krnl.comp_world_heat) ccl_kernel_destroy( krnl.comp_world_heat );
	if (krnl.init_swarm) ccl_kernel_destroy( krnl.init_swarm );
	if (krnl.init_maps) ccl_kernel_destroy( krnl.init_maps );
	if (krnl.init_random) ccl_kernel_destroy( krnl.init_random );

	/** Free remaining OpenCL wrappers. */
	if (oclobj.prg) ccl_program_destroy( oclobj.prg );
	if (oclobj.queue) ccl_queue_destroy( oclobj.queue );
	if (oclobj.ctx) ccl_context_destroy( oclobj.ctx );

	/** Close output file. */
	if (hbResult) fclose( hbResult );

	/* Confirm that memory allocated by wrappers has been properly freed. */
	g_assert( ccl_wrapper_memcheck() );


	return OKI_DOKI;
}