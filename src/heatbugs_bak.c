#include <stdio.h>			/* printf(...)	*/
#include <stdlib.h>			/* exit(...)	*/
#include <string.h>

#include <cf4ocl2.h>		/* glib.h included by cf4ocl2.h to handle GError*, MIN(...), g_random_int(). */

#include "heatbugs.h"



/* The cl kernel file pathname. */
#define CL_KERNEL_SRC_FILE "./heatbugs.cl"

/* The main kernel function names. */
#define KERNEL_NAME__INITIATE			"initiate"
#define KERNEL_NAME__WORLD_DIFFUSE		"world_diffuse"
#define KERNEL_NAME__WORLD_EVAPORATE	"world_evaporate"
#define KERNEL_NAME__BUG_STEP			"bug_step"
#define KERNEL_NAME__UNHAPPINESS		"mean_unhappiness"

#define KERNEL_NAME__TEST				"test"




#define OKI_DOKI 0
#define NOT_DOKI -1

#define BUG '@'
#define EMPTY '-'



/* Auxiliary host functions. */

#define ERROR_MSG_AND_EXIT(msg) \
	do { fprintf(stderr, "\n%s\n\n", msg); exit(NOT_DOKI); } while(0)

#define HANDLE_ERROR(err) \
	if (err != NULL) { ERROR_MSG_AND_EXIT(err->message); }



const char version[] = "Heatbugs simulation for GPU (parallel processing) v1.6.";



/* OpenCL built in compiler parameter's template. */
const char *cl_compiler_opts_template = QUOTE(
	-D BUGS_NUM=%u
	-D B_TEMP_MIN_IDEAL=%f
	-D B_TEMP_MAX_IDEAL=%f
	-D B_HEAT_MIN_OUTPUT=%f
	-D B_HEAT_MAX_OUTPUT=%f
	-D RAND_MOV_CHANCE=%f
	-D WRL_HEIGHT=%u
	-D WRL_WIDTH=%u
	-D WRL_VSIZE=%u
	-D WRL_DIFF_RATE=%f
	-D WRL_EVAP_RATE=%f
);



void setup( simulatio_t * const simul, char *cl_compiler_opts )
{
	simul->world_vsize = simul->world_height * simul->world_width;

	if (simul->number_of_bugs == 0)
		ERROR_MSG_AND_EXIT( "error: There are no bugs." );

	if (simul->number_of_bugs >= simul->world_vsize)
		ERROR_MSG_AND_EXIT( "Error: Bugs number exceed available world slots." );

	if (simul->number_of_bugs >= 0.8 * simul->world_vsize)
		ERROR_MSG_AND_EXIT( "Warning: Bugs number close to available world slots." );

	/*
		Next are OpenCL build options that are send as defines to kernel, and
		prevent the need to send extra arguments to kernel, and work as work-items
		private variables.
		A null terminated string is granted in "cl_compiler_opts".
	*/
	sprintf( cl_compiler_opts, cl_compiler_opts_template, simul->number_of_bugs,
				simul->bugs_min_ideal_temperature, simul->bugs_max_ideal_temperature,
				simul->bugs_min_output_heat, simul->bugs_max_output_heat,
				simul->bugs_random_move_chance,
				simul->world_height, simul->world_width, simul->world_vsize,
				simul->wrl_diffusion_rate, simul->wrl_evaporation_rate );
}



int main( int argc, char *argv[] )
{
	FILE *hbResult;

	simulatio_t simul;					/* Host data, simulation parameters. */
//	real_t average_unhapiness;			/* Unhapiness in host. */


	/* ****** OpenCL related variables ****** */

	/* OpenCL built in compiler/builder parameters. */
	char cl_compiler_opts[512];

	GError* err = NULL;					/* Error reporting object, from GLib. */

	/* Wrappers to OpenCL objects. */
	CCLContext *ctx = NULL;				/* Context. */
	CCLDevice *dev = NULL;				/* Device.  */
	CCLQueue *queue = NULL;				/* Queue.   */
	CCLProgram *prg = NULL;				/* Kernel source code. */

//	CCLKernel *krnl_init = NULL;		/* Object for kernel function "Initiate". */
//	CCLKernel *krnl_wlddiff = NULL;		/* Object for kernel function "World_diffuse". */
//	CCLKernel *krnl_wldevap = NULL;		/* Object for kernel function "world_evaporate". */
//	CCLKernel *krnl_bugstep = NULL;		/* Object for kernel function "bug_step". */
//	CCLKernel *krnl_unhapp = NULL;      /* Object for kernel function "unhappiness". */
	CCLKernel *krnl_test = NULL;		/* Object for kernel function "test". */

	CCLEvent *evt_write_seeds;			/* Event signal for seeds buffer write. */
	CCLEvent *evt_exec_kernel;			/* Event signal for kernel execution. */
	CCLEventWaitList ewl = NULL;		/* A list of OpenCL events. */

	/* Local & Global worksizes. */
	size_t lws;
	size_t gws;


	/* ****** BUFFERS ****** */

	/* Buffers: Host...   */
	cl_double *hst_map_temperature;
	cl_uint *hst_rng_seeds = NULL;		/* Seed's vector to be sent to kernel. */

	/* Buffers: device... */
	CCLBuffer *dev_map_temperature;		/* World temperature map (double) */
//	CCLBuffer *dev_swarm_heatspecs;		/* Ideal_Temperature & Output_Heat (double2) */
//	CCLBuffer *dev_swarm_locus;			/* Bug world location (uint2) */
	CCLBuffer *dev_rng_seeds;			/* Random generator seed for every bug. */




	/* TODO: Get simulation parameters. */
	simul.bugs_min_ideal_temperature = 20.0;	/* 10.0 */
	simul.bugs_max_ideal_temperature = 30.0;	/* 40.0 */
	simul.bugs_min_output_heat = 15.0;			/*  5.0 */
	simul.bugs_max_output_heat = 25.0;			/* 25.0 */
	simul.bugs_random_move_chance = 0.0;		/*  0%  Valid:[0 .. 100] */

	simul.wrl_diffusion_rate = 0.4;				/* 90%  */
	simul.wrl_evaporation_rate = 0.01;			/*  1%  */
	simul.world_height = 100;
	simul.world_width =  100;

	simul.number_of_bugs = 100;					/* 100 Bugs in the world. */
	simul.numIterations =  1000;				/* 0 = NonStop. */



	/* Setup: Check arguments and initiate a 'number_of_bugs' sized random seed vector. */
	setup( &simul, cl_compiler_opts );

	/* DEBUG: Show CL build options. */
	printf("\n\nbuild Options: %s\n\n", cl_compiler_opts);

	/* Open output file for results. */
	hbResult = fopen( "../results/heatbugsC_00.csv", "w+" );	/* Open file overwrite. */

	if (hbResult == NULL)
		ERROR_MSG_AND_EXIT( "Error: Could not open output file." );



	/*
		********* GPU preparation **********************************************
	*/

	/* Create context wrapper for a GPU device. The first found GPU device will be used. */
	ctx = ccl_context_new_gpu( &err );
	HANDLE_ERROR( err );

	/* Get the device (index 0) in te context. */
	dev = ccl_context_get_device( ctx, 0, &err );
	HANDLE_ERROR( err );

	/* Create a command queue. */
	queue = ccl_queue_new( ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err );
	HANDLE_ERROR( err );



	/*
		********* Program build / Kernel preparation ***************************
	*/

	/* Create a new program from kernel source. */
	prg = ccl_program_new_from_source_file( ctx, CL_KERNEL_SRC_FILE, &err );
	HANDLE_ERROR( err );

	/* Build CL Program. */
	ccl_program_build( prg, cl_compiler_opts, &err );
	HANDLE_ERROR( err );


	/* Create the kernel objects, one for each kernel function. */
//	krnl_init = ccl_kernel_new( prg, KERNEL_NAME__INITIATE, NULL );
//	krnl_wlddiff = ccl_kernel_new( prg, KERNEL_NAME__WORLD_DIFFUSE, NULL );
//	krnl_wldevap = ccl_kernel_new( prg, KERNEL_NAME__WORLD_EVAPORATE, NULL );
//	krnl_bugstep = ccl_kernel_new( prg, KERNEL_NAME__BUG_STEP, NULL );
//	krnl_unhapp = ccl_kernel_new( prg, KERNEL_NAME__UNHAPPINESS, NULL );
	krnl_test = ccl_kernel_new( prg, KERNEL_NAME__TEST, &err );
	HANDLE_ERROR( err );



	/*
		********* Init HOST buffers ********************************************
	*/

	/* Create buffer for temperature map */
	//hst_map_temperature

	/* Create and initiate host seed vector. One seed per bug. */
	hst_rng_seeds = (cl_uint *) malloc( simul.number_of_bugs * sizeof(cl_uint) );

	for (cl_uint i = 0; i < simul.number_of_bugs; ++i) {
		hst_rng_seeds[i] = g_random_int();
	}

	/* DEBUG: Show vector */
		for (cl_uint i = 0; i < simul.number_of_bugs; i++) {
			printf("%u ", hst_rng_seeds[i]);
		}
		printf("\n\n");



	/*
		********* Send buffers to device ***************************************
	*/

	/* Create a device buffer for the seeds. */
	dev_rng_seeds = ccl_buffer_new( ctx, CL_MEM_READ_WRITE, simul.number_of_bugs * sizeof(cl_uint), NULL, &err );
	HANDLE_ERROR( err );

	/* Copy host data to device buffer, without waiting for transfer to terminate before continuing host program. */
	evt_write_seeds = ccl_buffer_enqueue_write( dev_rng_seeds, queue, CL_FALSE, 0, simul.number_of_bugs * sizeof(cl_uint), hst_rng_seeds, NULL, &err );
	HANDLE_ERROR( err);


	/* initialize the event wait list and add the transfer event. */
	ccl_event_wait_list_add( &ewl, evt_write_seeds, NULL );




	/*
		********* RUN kernels **************************************************
	*/

	/* Get local work size. */
//	lws = ccl_device_get_info_scalar( dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, size_t, &err );
//	HANDLE_ERROR( err );
//	lws = MIN( simul.number_of_bugs, lws );

	/* Get global worksize, make it multiple of local worksize. */
//	gws = lws * ( (simul.number_of_bugs / lws) + ((simul.number_of_bugs % lws) > 0) );

	size_t rws = simul.number_of_bugs;
	lws = 0;

	ccl_kernel_suggest_worksizes( NULL, dev, 1, &rws, &gws, &lws, &err );
	HANDLE_ERROR( err );

	printf( "\n" );
	printf( "real worksize: %ld\n", rws );
	printf( "global worksize: %ld\n", gws );
	printf( "local worksize:  %ld\n", lws );
	printf( "\n" );






	//hst_map_temperature = (cl_double *) malloc( simul.world_vsize );





	/* Start kernel execution. */

	/* Test kernel. */

	ccl_kernel_set_arg( krnl_test, 0, dev_rng_seeds );

	evt_exec_kernel = ccl_kernel_enqueue_ndrange( krnl_test, queue, 1, NULL, &gws, &lws, &ewl, &err );
	HANDLE_ERROR( err);


	/* add kernel termination event to the wait list. */
	ccl_event_wait_list_add( &ewl, evt_exec_kernel, NULL );


	evt_write_seeds = ccl_buffer_enqueue_read( dev_rng_seeds, queue, CL_FALSE, 0, simul.number_of_bugs * sizeof(cl_uint), hst_rng_seeds, &ewl, &err );
	HANDLE_ERROR( err );

	/* add read event to the wait list. */
	ccl_event_wait_list_add( &ewl, evt_write_seeds, NULL );


/* DEBUG: Show new changed seed vector */
for (cl_uint i = 0; i < simul.number_of_bugs; i++) {
	printf("%u ", hst_rng_seeds[i]);
}
printf("\n\n");



	/* Clean / Destroy all allocated items. */

	/* Destroy host buffers. */
	free( hst_rng_seeds );

	/* Destroy Device buffers. */
	//ccl_buffer_destroy( dev_temperature_map );
	//ccl_buffer_destroy( dev_swarm_heatspecs );
	//ccl_buffer_destroy( dev_swarm_locus );
	ccl_buffer_destroy( dev_rng_seeds );

	/* Destroy wrappers. */
	ccl_kernel_destroy( krnl_test );
//	ccl_kernel_destroy( krnl_unhapp );
//	ccl_kernel_destroy( krnl_bugstep );
//	ccl_kernel_destroy( krnl_wldevap );
//	ccl_kernel_destroy( krnl_wlddiff );
//	ccl_kernel_destroy( krnl_init );

	ccl_program_destroy( prg );
	ccl_queue_destroy( queue );
	ccl_context_destroy( ctx );

	fclose( hbResult );

	/* Confirm that memory allocated by wrappers has been properly freed. */
	g_assert( ccl_wrapper_memcheck() );

	return OKI_DOKI;
}
