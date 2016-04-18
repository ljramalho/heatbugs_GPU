#include <stdio.h>			/* printf(...)	*/
#include <stdlib.h>			/* exit(...)	*/
#include <string.h>

#include <cf4ocl2.h>		/* glib.h included by cf4ocl2.h to handle GError*, MIN(...), g_random_int(). */

#include "heatbugs.h"


/* The cl kernel file pathname. */
#define CL_KERNEL_SRC_FILE "./heatbugs.cl"

/* The main kernel function names. */
#define KERNEL_NAME__INIT_RANDOM		"init_random"
#define KERNEL_NAME__INIT_HEATMAP		"init_heatmap"
#define KERNEL_NAME__INIT_BUGS			"init_bugs"
//#define KERNEL_NAME__WORLD_DIFFUSE		"world_diffuse"
//#define KERNEL_NAME__WORLD_EVAPORATE	"world_evaporate"
#define KERNEL_NAME__COMP_WORLD_HEAT	"compute_WorldHeat"
#define KERNEL_NAME__BUG_STEP			"bug_step"
#define KERNEL_NAME__UNHAPPINESS		"unhappiness"

#define KERNEL_NAME__TEST				"test"


/* OpenCL options. */
#define DIMS_1	1				/* Kernel dimensions: 1 dimension.  */
#define DIMS_2	2				/* Kernel dimensions: 2 dimensions. */
#define NON_BLOCK	CL_FALSE	/* Non blocked operation, read / write... */




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
	CCLKernel *init_random;				/* kernel: Initiate random seeds.	*/
	CCLKernel *init_heatmap;			/* kernel: Initiate heat map.		*/
	CCLKernel *init_bugs;
	CCLKernel *world_diffuse;			/* kernel: World Diffuse.			*/
	CCLKernel *world_evaporate;			/* kernel: World Evaporate.			*/
	CCLKernel *bug_step;				/* kernel: Bug Step.				*/
	CCLKernel *unhappiness;				/* kernel: compute unhappiness.		*/
	CCLKernel *test;
} HBKernels_t;

/** Global work sizes for all kernels. */
typedef struct hb_global_work_sizes	{
	size_t init_random;
	size_t init_heatmap;
	size_t init_bugs;
	size_t world_diffuse;
	size_t world_evaporate;
	size_t bug_step;
	size_t unhappiness;
	size_t test;
} HBGlobalWorkSizes_t;

/* Local work sizes for all kernels. */
typedef struct hb_local_work_sizes	{
	size_t init_random;
	size_t init_heatmap;
	size_t init_bugs;
	size_t world_diffuse;
	size_t world_evaporate;
	size_t bug_step;
	size_t unhappiness;
	size_t test;
} HBLocalWorkSizes_t;

/** Host and Device Buffers. */
typedef struct hb_buffers {
	/** Host random seeds buffer. */
	cl_uint *hst_rng_seeds;						/* DEBUG: (to remove). */
	/** Host temperature map & buffer. */
	cl_double *hst_heat_map[2];					/* DEBUG: (to remove). */
	/** Device random seeds buffer. */
	CCLBuffer *dev_rng_seeds;
	/** Device temperature map & buffer. */
	CCLBuffer *dev_heat_map[2];
	/** Device bugs Ideal Temperature and Output Heat (double2). */
	CCLBuffer *dev_swarm_heatspecs;
	/** Device bugs world location (uint2). */
	CCLBuffer *dev_swarm_locus;
} HBBuffers_t;

/** Buffers sizes. */


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
	-D BUGS_TEMPERATURE_MIN_IDEAL=%f
	-D BUGS_TEMPERATURE_MAX_IDEAL=%f
	-D BUGS_HEAT_MIN_OUTPUT=%f
	-D BUGS_HEAT_MAX_OUTPUT=%f
	-D BUGS_RANDOM_MOVE_CHANCE=%f
	-D WORLD_DIFFUSION_RATE=%f
	-D WORLD_EVAPORATION_RATE=%f
	-D BUGS_NUMBER=%zu
	-D WORLD_HEIGHT=%zu
	-D WORLD_WIDTH=%zu
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
	 /* TODO: Get simulation parameters. */
	params->bugs_temperature_min_ideal = 20.0;	/* 10.0 */
	params->bugs_temperature_max_ideal = 30.0;	/* 40.0 */
	params->bugs_heat_min_output = 15.0;		/*  5.0 */
	params->bugs_heat_max_output = 25.0;		/* 25.0 */
	params->bugs_random_move_chance = 0.0;		/*  0%  Valid:[0 .. 100] */

	params->world_diffusion_rate = 0.4;			/* 90%  */
	params->world_evaporation_rate = 0.01;		/*  1%  */
	params->world_height = 100;
	params->world_width =  100;

	params->bugs_number = 100;					/* 100 Bugs in the world. */
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

	printf ( "ranseed=%u\n", init_seed );

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
				params->world_height,
				params->world_width,
				params->world_size );

	/* DEBUG: Show OpenCL build options. */
	printf("\n\nbuild Options:\n%s\n\n", cl_compiler_opts);


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


static void setupHBKernels( HBKernels_t *const krnl, HBGlobalWorkSizes_t *const gws, HBLocalWorkSizes_t *const lws, const OCLObjects_t *const oclobj, const parameters_t *const params, GError **err )
{
	GError *err_setup = NULL;


	/* Create the kernel objects, one for each kernel function. */

	/** Kernel for random initialization. A random per bug. */
	krnl->init_random = ccl_kernel_new( oclobj->prg, KERNEL_NAME__INIT_RANDOM, &err_setup );
	ccl_if_err_propagate_goto( err, err_setup, error_handler );

	ccl_kernel_suggest_worksizes( krnl->init_random, oclobj->dev, DIMS_1, &params->bugs_number, &gws->init_random, &lws->init_random, &err_setup );
	ccl_if_err_propagate_goto( err, err_setup, error_handler );

	// DEBUG: printf( "-> bugs_num = %zu, gws = %zu, lws = %zu\n\n", params->bugs_number, gws->init_random, lws->init_random );

	/** Kernel for heatbug's heat map (world) initialization. */
	krnl->init_heatmap = ccl_kernel_new( oclobj->prg, KERNEL_NAME__INIT_HEATMAP, &err_setup );
	ccl_if_err_propagate_goto( err, err_setup, error_handler );

	ccl_kernel_suggest_worksizes( krnl->init_heatmap, oclobj->dev, DIMS_1, &params->world_size, &gws->init_heatmap, &lws->init_heatmap, &err_setup );
	ccl_if_err_propagate_goto( err, err_setup, error_handler );

	// DEBUG: printf( "-> world_size = %zu, gws = %zu, lws = %zu\n\n", params->world_size, gws->init_heatmap, lws->init_heatmap );

	/** Kernel for heatbug's initiation. */
//	krnl->init_bugs = ccl_kernel_new( oclobj->prg, KERNEL_NAME__INIT_BUGS, &err_setup );
//	ccl_if_err_propagate_goto( err, err_setup, error_handler );

//	ccl_kernel_suggest_worksizes( krnl->init_bugs, oclobj->dev, 1, (const size_t *) &params->world_size, &gws->init_bugs, &lws->init_bugs, &err_setup );
//	ccl_if_err_propagate_goto( err, err_setup, error_handler );


	/** Kernel for world diffusion. */

	krnl->world_diffuse = ccl_kernel_new( oclobj->prg, KERNEL_NAME__WORLD_DIFFUSE, &err_setup );
	ccl_if_err_propagate_goto( err, err_setup, error_handler );

	ccl_kernel_suggest_worksizes( krnl->world_diffuse, oclobj->dev, DIMS_1, &params->world_size, &gws->world_diffuse, &lws->world_diffuse, &err_setup );
	ccl_if_err_propagate_goto( err, err_setup, error_handler );

	printf( "rws = %zu, gws = %zu, lws = %zu\n\n", params->world_size, gws->world_diffuse, gws->world_diffuse );

	/** Kernel for world evaporation. */
//	krnl->world_evaporate = ccl_kernel_new( oclobj->prg, KERNEL_NAME__WORLD_EVAPORATE, &err_setup );
//	ccl_if_err_propagate_goto( err, err_setup, error_handler );

//	ccl_kernel_suggest_worksizes( krnl->world_evaporate, oclobj->dev, 1, (const size_t *) &params->world_size, &gws->world_evaporate, &lws->world_evaporate, &err_setup );
//	ccl_if_err_propagate_goto( err, err_setup, error_handler );


	/** Kernel for bug step. */
//	krnl->bug_step = ccl_kernel_new( oclobj->prg, KERNEL_NAME__BUG_STEP, &err_setup );
//	ccl_if_err_propagate_goto( err, err_setup, error_handler );

//	krnl->unhappiness = ccl_kernel_new( oclobj->prg, KERNEL_NAME__UNHAPPINESS, &err_setup );
//	ccl_if_err_propagate_goto( err, err_setup, error_handler );

//	krnl->test = ccl_kernel_new( oclobj->prg, KERNEL_NAME__TEST, &err_setup );
//	ccl_if_err_propagate_goto( err, err_setup, error_handler );



	/* TODO: Calculate global work sizes and local work sizes. */

	//printf("rws={%ld}, gws={%ld}, lws={%ld}\n\n", params->world_size, gws->init, lws->init);




error_handler:
		/* If error handler is reached leave function imediately. */

	return;
}


static void setupBuffers( HBBuffers_t *const buf, const OCLObjects_t *const oclobj, const parameters_t *const params, GError **err )
{
	GError *err_setup = NULL;

// REMOVE: 	CCLEvent *evt_write = NULL;			/* Event signal for seeds buffer write. */
// REMOVE:	CCLEventWaitList ewl = NULL;		/* A list of OpenCL events. */


	/* DEBUG: (to remove) Allocate vector for random seeds in host memory. */
	buf->hst_rng_seeds = (cl_uint *) malloc( params->bugs_number * sizeof(cl_uint) );
	ccl_if_err_create_goto( *err, HB_ERROR, buf->hst_rng_seeds == NULL, HB_MALLOC_FAILURE, error_handler, "Unable to allocate host memory for random seeds." );

	/* Allocate vector of random seeds in device memory. */
	buf->dev_rng_seeds = ccl_buffer_new( oclobj->ctx, CL_MEM_READ_WRITE, params->bugs_number * sizeof(cl_uint), NULL, &err_setup );
	ccl_if_err_propagate_goto( err, err_setup, error_handler );


	/* DEBUG: (to remove) Allocate temperature vector in host memory. */
	buf->hst_heat_map[0] = (cl_double *) malloc( params->world_size * sizeof(cl_double) );
	ccl_if_err_create_goto( *err, HB_ERROR, buf->hst_heat_map[0] == NULL, HB_MALLOC_FAILURE, error_handler, "Unable to allocate host memory for temperature / heat map 1." );

	/* DEBUG: (to remove) Allocate temperature vector in host memory. */
	buf->hst_heat_map[1] = (cl_double *) malloc( params->world_size * sizeof(cl_double) );
	ccl_if_err_create_goto( *err, HB_ERROR, buf->hst_heat_map[1] == NULL, HB_MALLOC_FAILURE, error_handler, "Unable to allocate host memory for temperature / heat map 2." );

	/* Alocate two vectors of temperature maps in device memory, one is using dor double buffer. */
	buf->dev_heat_map[0] = ccl_buffer_new( oclobj->ctx, CL_MEM_READ_WRITE, params->world_size * sizeof(cl_double), NULL, &err_setup );
	ccl_if_err_propagate_goto( err, err_setup, error_handler );

	buf->dev_heat_map[1] = ccl_buffer_new( oclobj->ctx, CL_MEM_READ_WRITE, params->world_size * sizeof(cl_double), NULL, &err_setup );
	ccl_if_err_propagate_goto( err, err_setup, error_handler );



error_handler:
	/* If error handler is reached leave function imediately. */

	return;
}


/**
	Run all init kernels.
  */
static inline void initiate( HBKernels_t *const krnl, HBGlobalWorkSizes_t *const gws, HBLocalWorkSizes_t *const lws, OCLObjects_t *const oclobj, HBBuffers_t *const buf, parameters_t *const params,  GError **err )
{
	GError *err_init = NULL;

	/** Events */
	CCLEvent *evt_rdwr = NULL;			/* Event signal for reads and writes. */
	CCLEvent *evt_krnl_exec = NULL;		/* Event signal for kernel execution. */
	CCLEventWaitList ewl = NULL;		/* A list of OpenCL events. */


	printf( "Init random:\n\tgws = %zu; lws = %zu\n", gws->init_random, lws->init_random );

	/** Init random. */

	ccl_kernel_set_arg( krnl->init_random, 0, buf->dev_rng_seeds );

	evt_krnl_exec = ccl_kernel_enqueue_ndrange( krnl->init_random, oclobj->queue, DIMS_1, NULL, &gws->init_random, &lws->init_random, &ewl, &err_init );
	ccl_if_err_propagate_goto( err, err_init, error_handler );

	/* add kernel termination event to the wait list. */
	ccl_event_wait_list_add( &ewl, evt_krnl_exec, NULL );

	/* DEBUG: (to remove) read seeds. */
	evt_rdwr = ccl_buffer_enqueue_read( buf->dev_rng_seeds, oclobj->queue, NON_BLOCK, 0, params->bugs_number * sizeof(cl_uint), buf->hst_rng_seeds, &ewl, &err_init );
	ccl_if_err_propagate_goto( err, err_init, error_handler );

	/* Add read event to the wait list. */
	ccl_event_wait_list_add( &ewl, evt_rdwr, NULL );

	/* DEBUG: Show GPU generated seeds. */
	for (size_t i = 0; i < params->bugs_number; ++i) {
		printf( "%u, ", buf->hst_rng_seeds[i]);
	}
	printf( "\n\n" );



	printf( "Init temperature map:\n\tgws=%zu; lws=%zu\n", gws->init_heatmap, lws->init_heatmap );

	/** Reset temperature map buffers. */

	ccl_kernel_set_arg( krnl->init_heatmap, 0, buf->dev_heat_map[0] );
	ccl_kernel_set_arg( krnl->init_heatmap, 1, buf->dev_heat_map[1] );

	evt_krnl_exec = ccl_kernel_enqueue_ndrange( krnl->init_heatmap, oclobj->queue, DIMS_1, NULL, &gws->init_heatmap, &lws->init_heatmap, &ewl, &err_init );
	ccl_if_err_propagate_goto( err, err_init, error_handler );

	/* Add kernel termination event to the wait list. */
	ccl_event_wait_list_add( &ewl, evt_krnl_exec, NULL );

	/* DEBUG: (to remove) read temperature map. */
	evt_rdwr = ccl_buffer_enqueue_read( buf->dev_heat_map[0], oclobj->queue, NON_BLOCK, 0, params->world_size * sizeof(cl_double), buf->hst_heat_map[0], &ewl, &err_init );
	ccl_if_err_propagate_goto( err, err_init, error_handler );

	evt_rdwr = ccl_buffer_enqueue_read( buf->dev_heat_map[1], oclobj->queue, NON_BLOCK, 0, params->world_size * sizeof(cl_double), buf->hst_heat_map[1], &ewl, &err_init );
	ccl_if_err_propagate_goto( err, err_init, error_handler );

	printf( "Test:\n" );
	for (size_t i = 0; i < params->world_size; ++i) {
		if (buf->hst_heat_map[0][i] != 0.0f) printf( "Err buf_0: %zu\n", i );
		if (buf->hst_heat_map[1][i] != 0.0f) printf( "Err buf_1: %zu\n", i );
	}
	printf( "\n\n" );


error_handler:
	/* If error handler is reached leave function imediately. */

	return;
}


static inline void world_diffuse()
{
}


static inline void world_evaporate()
{
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
	HBKernels_t krnl = { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
	/** Global work sizes for all kernels. */
	HBGlobalWorkSizes_t gws = { 0, 0, 0, 0, 0, 0, 0, 0 };
	/** Local work sizes for all kernels. */
	HBLocalWorkSizes_t  lws = { 0, 0, 0, 0, 0, 0, 0, 0 };

	/** Buffers for host and device. */
	HBBuffers_t buf = { NULL, { NULL, NULL }, NULL, { NULL, NULL }, NULL, NULL };



	setupParameters( &params, &err );
	ccl_if_err_goto( err, error_handler );

	setupOCLObjects( &oclobj, &params, &err );
	ccl_if_err_goto( err, error_handler );

	setupHBKernels( &krnl, &gws, &lws, &oclobj, &params, &err );
	ccl_if_err_goto( err, error_handler );

	setupBuffers( &buf, &oclobj, &params, &err );
	ccl_if_err_goto( err, error_handler );



	/* Open output file for results. */
	hbResult = fopen( "../results/heatbugsC_00.csv", "w+" );	/* Open file overwrite. */

	if (hbResult == NULL)
		ERROR_MSG_AND_EXIT( "Error: Could not open output file." );


	/** Run all init kernels. */
	initiate( &krnl, &gws, &lws, &oclobj, &buf, &params, &err );



	goto clean_all;


error_handler:

	/* Handle error. */
	fprintf( stderr, "Error: %s\n", err->message );
	g_error_free( err );


clean_all:

	/* Clean / Destroy all allocated items. */

	/* Destroy host buffers. */
	if (buf.hst_rng_seeds) free( buf.hst_rng_seeds );

	/* Destroy Device buffers. */
	//if (dev_swarm_heatspecs) ccl_buffer_destroy( dev_swarm_heatspecs );
	//if (dev_swarm_locus) ccl_buffer_destroy( dev_swarm_locus );
	if (buf.dev_heat_map[1]) ccl_buffer_destroy( buf.dev_heat_map[1] );
	if (buf.dev_heat_map[0]) ccl_buffer_destroy( buf.dev_heat_map[0] );
	if (buf.dev_rng_seeds) ccl_buffer_destroy( buf.dev_rng_seeds );

	/** Destroy kernel wrappers. */
//	if (krnl.test) ccl_kernel_destroy( krnl.test );
//	if (krnl.unhappiness) ccl_kernel_destroy( krnl.unhappiness );
//	if (krnl.bug_step) ccl_kernel_destroy( krnl.bug_step );
//	if (krnl.world_evaporate) ccl_kernel_destroy( krnl.world_evaporate );
	if (krnl.world_diffuse) ccl_kernel_destroy( krnl.world_diffuse );
//	if (krnl.init_bugs) ccl_kernel_destroy( krnl.init_bugs );
	if (krnl.init_heatmap) ccl_kernel_destroy( krnl.init_heatmap );
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