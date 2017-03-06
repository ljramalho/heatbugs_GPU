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



#define _GNU_SOURCE	/* this allow 'getopt(..)' function in <unistd.h> to  */
			/* compile under -std=c99 compiler option, because    */
			/* 'getopt' is POSIX, not c99.                        */


#include <stdio.h>	/* printf(...)	*/
#include <stdlib.h>	/* exit(...)	*/
#include <unistd.h>
#include <string.h>
#include <ctype.h>	/* isprint(..)	*/

/*!
	Check documentation of cf4ocl2 @
	https://fakenmc.github.io/cf4ocl/docs/latest/index.html
 */
#include <cf4ocl2.h>	/* glib.h included by cf4ocl2.h to handle GError*,    */
			/* MIN(...), g_random_int().                          */

#include "heatbugs.h"



/** Default parameters. */

#define NUM_ITERATIONS		1000	/* 1000 iterations. (0 = non stop).   */
#define BUGS_NUMBER		100	/* 100 Bugs in the world.             */
#define WORLD_WIDTH		100
#define WORLD_HEIGTH		100

#define WORLD_DIFFUSION_RATE	0.90f	/* [0..1], % temperature to   */
					/* neighbour cells.           */

#define WORLD_EVAPORATION_RATE	0.01f	/* [0..1], % temperature loss */
					/* to 'ether'.                */

#define BUGS_RAND_MOVE_CHANCE	0.00f	/* [0..100], Chance a bug will move.  */
#define BUGS_TEMP_MIN_IDEAL	10	/* [0..200] */
#define BUGS_TEMP_MAX_IDEAL	40	/* [0..200] */
#define BUGS_HEAT_MIN_OUTPUT	5	/* [0..100] */
#define BUGS_HEAT_MAX_OUTPUT	25	/* [0..100] */

/* The file to send results. Directory must exist. */
#define OUTPUT_FILENAME		"../results/heatbugsGPU.csv"


/** The cl kernel file pathname. */
#define CL_KERNEL_SRC_FILE		"./heatbugs.cl"

/** The main kernel function names. */
#define KRNL_NAME__INIT_RANDOM		"init_random"
#define KRNL_NAME__INIT_MAPS		"init_maps"
#define KRNL_NAME__INIT_SWARM		"init_swarm"
#define KRNL_NAME__SET_BUG_MOVE_STATE	"set_bug_move_state"
#define KRNL_NAME__BUG_STEP		"bug_step"
#define KRNL_NAME__COMP_WORLD_HEAT	"comp_world_heat"
#define KRNL_NAME__UNHAPP_S1_REDUCE	"unhappiness_step1_reduce"
#define KRNL_NAME__UNHAPP_S2_AVERAGE	"unhappiness_step2_average"


/** OpenCL options. */
#define DIMS_1		 1		/* Dimensions: 1 dimension.     */
#define DIMS_2		 2		/* Dimensions: 2 dimensions.    */
#define NON_BLOCK	 CL_FALSE	/* Non blocked rd/wr operation. */


#define OKI_DOKI	 0
#define NOT_DOKI	-1


/** Input data used for simulation. */
typedef struct parameters {
	/* IN: The seed to be used. */
	size_t seed;
	/* 'reduce_num_workgroups' is used to send information across functions. */
	size_t reduce_num_workgroups;
	/* IN: Num Iterations to stop. (0 = non stop). */
	size_t numIterations;
	/* IN: Number of bugs in the world. */
	size_t bugs_number;
	/* IN: World width size. */
	size_t world_width;
	/* IN: World height size. */
	size_t world_height;
	/* IN: World's vector size = (world_height * world_width). */
	size_t world_size;
	/* IN: [0..1], % temperature to adjacent cells. */
	float world_diffusion_rate;
	/* IN: [0..1], % temperature's loss to 'ether'.  */
	float world_evaporation_rate;
	/* IN: [0..100], Chance a bug will move. */
	float bugs_random_move_chance;
	/* IN: [0 .. 200], bug's minimum prefered temperature. */
	unsigned int bugs_temperature_min_ideal;
	/* IN: [0 .. 200], bug's maximum prefered temperature. */
	unsigned int bugs_temperature_max_ideal;
	/* IN: [0 .. 100], min heat a bug leave in the world in each step. */
	unsigned int bugs_heat_min_output;
	/* IN: [0 .. 100], max heat a bug leave in the world in each step. */
	unsigned int bugs_heat_max_output;
	/* IN: File to send results. */
	char output_filename[256];
} Parameters_t;


/** Holder for all OpenCl objects. */
typedef struct ocl_objects {
	CCLContext *ctx;			/* Context.	*/
	CCLDevice *dev;				/* Device.	*/
	CCLQueue *queue;			/* Queue.	*/
	CCLProgram *prg;			/* Kernel code.	*/
} OCLObjects_t;


/** Holder for all kernels. */
typedef struct hb_kernels {
	/* Initiate random seeds. */
	CCLKernel *init_random;
	/* Initiate heat_map and swarm_map. */
	CCLKernel *init_maps;
	/* Initiate the world of bugs. */
	CCLKernel *init_swarm;
	/* Perform a new bug movement. */
	CCLKernel *bug_step;
	/* Compute world heat, diffusion then evaporation. */
	CCLKernel *comp_world_heat;
	/* Reduce (sum) the unhappiness vector. */
	CCLKernel *unhapp_stp1_reduce;
	/* Further reduce the unhappiness vector and compute average. */
	CCLKernel *unhapp_stp2_average;
} HBKernels_t;


/** Global work sizes for all kernels. */
typedef struct hb_global_work_sizes	{
	size_t init_random[ DIMS_1 ];
	size_t init_maps[ DIMS_1 ];
	size_t init_swarm[ DIMS_1 ];
	size_t bug_step[ DIMS_1 ];
	size_t comp_world_heat[ DIMS_2 ];
	size_t unhapp_stp1_reduce[ DIMS_1 ];
	size_t unhapp_stp2_average[ DIMS_1 ];
} HBGlobalWorkSizes_t;


/* Local work sizes for all kernels. */
typedef struct hb_local_work_sizes	{
	size_t init_random[ DIMS_1 ];
	size_t init_maps[ DIMS_1 ];
	size_t init_swarm[ DIMS_1 ];
	size_t bug_step[ DIMS_1 ];
	size_t comp_world_heat[ DIMS_2 ];
	size_t unhapp_stp1_reduce[ DIMS_1 ];
	size_t unhapp_stp2_average[ DIMS_1 ];
} HBLocalWorkSizes_t;


/** Host Buffers. */
typedef struct hb_host_buffers {
	cl_uint *bug_step_retry;	/* SIZE: 1		- In any iteration if set, it flags for another recall of the bug_step kernel. */
//	cl_uint *rng_state;		/* SIZE: BUGS_NUM	- Random seeds buffer. DEBUG: (to remove). */
//	cl_uint *swarm_bugPosition;	/* SIZE: BUGS_NUM	- Bug's position in the swarm_map, (swarm_bugPosition). */
//	cl_uint *swarm_map;		/* SIZE: WORLD_SIZE	- Bugs map. Each cell is: 'ideal-Temperature':8bit 'bug':1bit 'output_heat':7bit. */
//	cl_float *heat_map[2];		/* SIZE: WORLD_SIZE	- Temperature map (heat_map) & the buffer (heat_buffer). DEBUG: (to remove). */
//	cl_float *unhappiness;		/* SIZE: NUM_BUGS	- The Unhappiness vector. */
//	cl_float *unhapp_reduced;	/* SIZE: REDUCE_NUM_WORKGROUPS - The number of workgroups performing reduction. */
	cl_float *unhapp_average;	/* SIZE: 1		- Unhappiness average. The expected result at the end of each iteration. */
} HBHostBuffers_t;


/** Device buffers. */
typedef struct hb_device_buffers {
	CCLBuffer *bug_step_retry;	/* SIZE: 1		- In any iteration if set, it flags for another recall of the bug_step kernel. */
	CCLBuffer *rng_state;		/* SIZE: BUGS_NUM	- Random seeds buffer. */
	CCLBuffer *swarm_bugPosition;	/* SIZE: BUGS_NUM	- Bug's position in the swarm_map, (swarm_bugPosition). */
	CCLBuffer *swarm_map;		/* SIZE: WORLD_SIZE	- Bugs map. Each cell is: 'ideal-Temperature':8bit 'bug':1bit 'output_heat':7bit. */
	CCLBuffer *heat_map[2];		/* SIZE: WORLD_SIZE	- Temperature map (heat_map) & the buffer (heat_buffer). */
	CCLBuffer *unhappiness;		/* SIZE: NUM_BUGS	- The Unhappiness vector. */
	CCLBuffer *unhapp_reduced;	/* SIZE: REDOX_NUM_WORKGROUPS - The number of workgroups performing reduction. */
	CCLBuffer *unhapp_average;	/* SIZE: 1		- Unhappiness average. The expected result at the end of each iteration. */
} HBDeviceBuffers_t;


/** Buffers sizes. Sizes are common to host and device buffers. */
typedef struct hb_buffers_size {
	size_t bug_step_retry;		/* VAL: 1 * sizeof( cl_uint ) */
	size_t rng_state;		/* VAL: BUGS_NUM * sizeof( cl_uint ) */
	size_t swarm_bugPosition;	/* VAL: BUGS_NUM * sizeof( cl_uint ) */
	size_t swarm_map;		/* VAL: WORLD_SIZE * sizeof( cl_uint ) */
	size_t heat_map;		/* VAL: WORLD_SIZE * sizeof( cl_float ) */
	size_t unhappiness;		/* VAL: BUGS_NUM * sizeof( cl_float ) */
	size_t unhapp_reduced;		/* VAL: REDOX_NUM_WORKGROUPS * sizeof( cl_float ) */
	size_t unhapp_average;		/* VAL: 1 * sizeof( cl_float ) */
} HBBuffersSize_t;


/** Buffer selectors. In each step they swap value to indicate the correct 'heat_map' buffer to be sent to kernel. */
typedef struct hb_buffer_select {
	cl_uint main;			/* Main buffer selector. */
	cl_uint secd;			/* Secondary buffer selector. */
} HBBufferSelect_t;




const char version[] = "Heatbugs simulation for GPU (parallel processing) v2.6.";



/**
 * OpenCL built in compiler parameter's template.
 * The template will be used to fill a string, that by turn, will be
 * used as OpenCL compiler options. This way, the constants will be
 * available in the kernel as if they were defined there.
 * The '-D' option, is the kernel's compiler directive to make the
 * constants available to the kernel.
 * */
const char *cl_compiler_opts_template = QUOTE(
	-D INIT_SEED=%u
	-D REDUCE_NUM_WORKGROUPS=%zu
	-D BUGS_NUMBER=%zu
	-D WORLD_WIDTH=%zu
	-D WORLD_HEIGHT=%zu
	-D WORLD_SIZE=%zu
	-D WORLD_DIFFUSION_RATE=%f
	-D WORLD_EVAPORATION_RATE=%f
	-D BUGS_RANDOM_MOVE_CHANCE=%f
	-D BUGS_TEMPERATURE_MIN_IDEAL=%u
	-D BUGS_TEMPERATURE_MAX_IDEAL=%u
	-D BUGS_HEAT_MIN_OUTPUT=%u
	-D BUGS_HEAT_MAX_OUTPUT=%u
);


#define HB_ERROR hb_error_quark()


static GQuark hb_error_quark( void ) {
	return g_quark_from_static_string( "hb-error-quark" );
}



/**
 * Sets the parameters passed as command line arguments.
 * If there are no parameters, default parameters are used.
 * Default parameter will be used for every omitted parameter.
 *
 * This function uses the GNU 'getopt' command line argument parser,
 * and requires the macro _GNU_SOURCE to be defined. As such, the
 * code using the 'getopt' function is not portable.
 * Consequence of using 'getopt' is that the previous result of 'argv'
 * parameter may change after 'getopt' is used, therefore 'argv'
 * should not be used again.
 * The function 'getopt' is marked as Thread Unsafe.
 *
 * @param[out]	params - Parameters to be filled with default or
 *                     from command line.
 * @param[in]	argc   - Command line argument counter.
 * @param[in]	argv   - Command line arguments.
 * @param[out]	err    - GLib object for error reporting.
 * */
static inline void getSimulParameters( Parameters_t *const params, int argc,
	char *argv[], GError **err )
{
	FILE *uranddev = NULL;

	int c;	/* Parsed command line option */

	/* The string 't:T:h:H:r:n:d:e:w:W:i:f:' is the parameter string to   */
	/* be checked by 'getopt' function.                                   */
	/* The ':' character means that a value is required after the         */
	/* parameter selector character (i.e. -t 50  or  -t50).               */
	const char matches[] = "t:T:h:H:r:n:d:e:w:W:i:s:f:";


	 /* Default / hardcoded parameters. */
	params->numIterations = NUM_ITERATIONS;				/* i */
	params->bugs_number = BUGS_NUMBER;				/* n */
	params->world_width =  WORLD_WIDTH;				/* w */
	params->world_height = WORLD_HEIGTH;				/* W */
	params->world_diffusion_rate = WORLD_DIFFUSION_RATE;		/* d */
	params->world_evaporation_rate = WORLD_EVAPORATION_RATE;	/* e */

	params->bugs_random_move_chance = BUGS_RAND_MOVE_CHANCE;	/* r */
	params->bugs_temperature_min_ideal = BUGS_TEMP_MIN_IDEAL;	/* t */
	params->bugs_temperature_max_ideal = BUGS_TEMP_MAX_IDEAL;	/* T */
	params->bugs_heat_min_output = BUGS_HEAT_MIN_OUTPUT;		/* h */
	params->bugs_heat_max_output = BUGS_HEAT_MAX_OUTPUT;		/* H */

	strcpy( params->output_filename, OUTPUT_FILENAME );		/* f */


	/* Read initial seed from linux /dev/urandom */
	uranddev = fopen( "/dev/urandom", "r" );
	hb_if_err_create_goto( *err, HB_ERROR,
		uranddev == NULL,
		HB_UNABLE_OPEN_FILE, error_handler,
		"Could not open urandom device to get seed." );

	fread( &params->seed, sizeof( size_t ), 1, uranddev );		/* s */
	fclose( uranddev );


	/* Parse command line arguments using GNU's getopt function. */

	while ( (c = getopt( argc, argv, matches )) != -1 )
	{
		switch (c)
		{
			case 't':
				params->bugs_temperature_min_ideal =
					atoi( optarg );
				break;
			case 'T':
				params->bugs_temperature_max_ideal =
					atoi( optarg );
				break;
			case 'h':
				params->bugs_heat_min_output =
					atoi( optarg );
				break;
			case 'H':
				params->bugs_heat_max_output =
					atoi( optarg );
				break;
			case 'r':
				params->bugs_random_move_chance =
					atof( optarg );
				break;
			case 'n':
				params->bugs_number =
					atoi( optarg );
				break;
			case 'd':
				params->world_diffusion_rate =
					atof( optarg );
				break;
			case 'e':
				params->world_evaporation_rate =
					atof( optarg );
				break;
			case 'w':
				params->world_width =
					atoi( optarg );
				break;
			case 'W':
				params->world_height =
					atoi( optarg );
				break;
			case 'i':
				params->numIterations =
					atoi( optarg );
				break;
			case 's':
				params->seed =
					atoi( optarg );
				break;
			case 'f':
				strcpy( params->output_filename, optarg );
				break;
			case '?':
				hb_if_err_create_goto( *err, HB_ERROR,
					(optopt != ':'
					&& strchr( matches, optopt ) != NULL),
					HB_PARAM_ARG_MISSING, error_handler,
					"Option required argument missing." );

				hb_if_err_create_goto( *err, HB_ERROR,
					(isprint( optopt )),
					HB_PARAM_OPTION_UNKNOWN, error_handler,
					"Unknown option." );

				hb_if_err_create_goto( *err, HB_ERROR,
					CL_TRUE,
					HB_PARAM_CHAR_UNKNOWN, error_handler,
					"Unprintable character in command line." );
			default:
				hb_if_err_create_goto( *err, HB_ERROR,
					CL_TRUE,
					HB_PARAM_PARSING, error_handler,
					"Weird error occurred while parsing parameter." );
		}
	}

	/*
	   NOTE: Check here for extra arguments... see:
	   https://www.gnu.org/software/libc/manual/html_node/Example
	   -of-Getopt.html
	 */

	params->world_size = params->world_height * params->world_width;

	/* Check for bug's number related errors. */
	hb_if_err_create_goto( *err, HB_ERROR,
		params->bugs_number == 0,
		HB_BUGS_ZERO, error_handler,
		"There are no bugs." );

	hb_if_err_create_goto( *err, HB_ERROR,
		params->bugs_number >= params->world_size,
		HB_BUGS_OVERFLOW, error_handler,
		"Number of bugs exceed available world slots." );

	/* Check range related erros in bug's ideal temperature. */
	/* Checking order matters!                               */
	hb_if_err_create_goto( *err, HB_ERROR,
		params->bugs_temperature_min_ideal >
					params->bugs_temperature_max_ideal,
		HB_TEMPERATURE_OVERLAP, error_handler,
		"Bug's ideal temperature range overlaps." );

	hb_if_err_create_goto( *err, HB_ERROR,
		params->bugs_temperature_max_ideal >= 200,
		HB_TEMPERATURE_OUT_RANGE, error_handler,
		"Bug's max ideal temperature is out of range." );

	/* Check for range related error in bug's output heat. */
	/* Checking order matters!                             */
	hb_if_err_create_goto( *err, HB_ERROR,
		params->bugs_heat_min_output > params->bugs_heat_max_output,
		HB_OUTPUT_HEAT_OVERLAP, error_handler,
		"Bug's output heat range overlaps.");

	hb_if_err_create_goto( *err, HB_ERROR,
		params->bugs_heat_max_output >= 100,
		HB_OUTPUT_HEAT_OUT_RANGE, error_handler,
		"Bug's max output heat is out of range." );

	/* If numeber of bugs is 80% of the world space issue a warning. */
	if (params->bugs_number >= 0.8 * params->world_size)
		fprintf( stderr,
			"Warning: Bugs number near available world slots.\n" );


error_handler:
	/* If error handler is reached leave function imediately. */

	return;
}



/**
 * Get and / or create all OpenCL objects.
 *	- Create a Context, get the device (GPU) from the context.
 *	- Create a command queue.
 *	- Create and build a program for devices in the context (GPU).
 *
 * @param[out]	oclobj	- A structure holding the pointers for each OpenCL object.
 * @param[out]	gws	- Structure with global work sizes.
 * @param[out]	lws	- Structure with local work sizes.
 * @param[in]	params	- The simulation parameters. Used to create the program's compiler options.
 * @param[out]	err	- GLib object for error reporting.
 * */
static inline void getOCLObjects( OCLObjects_t *const oclobj,
	HBGlobalWorkSizes_t *const gws, HBLocalWorkSizes_t *const lws,
	Parameters_t *const params, GError **err )
{
	char cl_compiler_opts[512];	/* OpenCL built in compiler/builder */
					/* parameters.                      */

	GError *err_get_oclobj = NULL;


	/* *** GPU preparation. Initiate OpenCL objects. *** */

	/* Create context wrapper for a GPU device. */
	/* First found GPU device will be used.          */
	oclobj->ctx = ccl_context_new_gpu( &err_get_oclobj );
	hb_if_err_propagate_goto( err, err_get_oclobj, error_handler );

	/* Get the device (index 0) in te context. */
	oclobj->dev = ccl_context_get_device( oclobj->ctx, 0, &err_get_oclobj );
	hb_if_err_propagate_goto( err, err_get_oclobj, error_handler );

	/* Create a command queue. */
	oclobj->queue = ccl_queue_new( oclobj->ctx, oclobj->dev,
		CL_QUEUE_PROFILING_ENABLE, &err_get_oclobj );
	hb_if_err_propagate_goto( err, err_get_oclobj, error_handler );


	/* *** Program creation. *** */

	/* Create a new program from kernel's source. */
	oclobj->prg = ccl_program_new_from_source_file( oclobj->ctx,
		CL_KERNEL_SRC_FILE, &err_get_oclobj );
	hb_if_err_propagate_goto( err, err_get_oclobj, error_handler );

	/*
	 * Get OpenCL build options to be sent as 'defines' to kernel,
	 * preventing the need to send extra arguments. These parameters work
	 * as 'work-items' private variables. A null terminated string is
	 * garanted in 'cl_compiler_opts'.
	 */

	/*
	 * Query device for importante parameters before program's build, so
	 * those parameters can be send to the kernel as external defines.
	*/
	ccl_kernel_suggest_worksizes( NULL, oclobj->dev, DIMS_1,
		&params->bugs_number, gws->unhapp_stp1_reduce,
		lws->unhapp_stp1_reduce, &err_get_oclobj );
	hb_if_err_propagate_goto( err, err_get_oclobj, error_handler );

	/*
	* MIN(...) included by cf4ocl2 from glib/gmacros.h.
	*
	* Next line bound the size of global work size to the square of the
	* local work size, so reduce step 1 work.
	*/
	gws->unhapp_stp1_reduce[ 0 ] =
		MIN( SQUARE( lws->unhapp_stp1_reduce[ 0 ] ),
			gws->unhapp_stp1_reduce[ 0 ] );

	params->reduce_num_workgroups =
		gws->unhapp_stp1_reduce[ 0 ] / lws->unhapp_stp1_reduce[ 0 ];


	sprintf( cl_compiler_opts, cl_compiler_opts_template,
				params->seed,
				params->reduce_num_workgroups,
				params->bugs_number,
				params->world_width,
				params->world_height,
				params->world_size,
				params->world_diffusion_rate,
				params->world_evaporation_rate,
				params->bugs_random_move_chance,
				params->bugs_temperature_min_ideal,
				params->bugs_temperature_max_ideal,
				params->bugs_heat_min_output,
				params->bugs_heat_max_output );

	/* DEBUG: Show OpenCL build options. */
	hbprintf("\n\nbuild Options:\n----------------------\n%s\n\n", cl_compiler_opts);


	/* Build CL Program. */
	ccl_program_build( oclobj->prg, cl_compiler_opts, &err_get_oclobj );
	hb_if_err_propagate_goto( err, err_get_oclobj, error_handler );


error_handler:
	/* If error handler is reached leave function imediately. */

	return;
}



/**
 * Create all the buffers for both, host and device.
 *
 * @param[out]	hst_buff - The structure with all host buffers to be filled.
 * @param[out]	dev_buff - The structure with all cf4ocl2 wrapper for device buffers to be filled.
 * @param[out]	bufsz    - The structure with all the buffer sizes to be computed from 'params'. Sizes are common to
 *                         host and device buffers.
 * @param[in]	oclObj   - The structure to the previously created OpenCl objects. From this we need the
 *                         'context' in which to create the buffers.
 * @param[in]	params   - The structure with all simulation parameters. Used to compute buffer sizes.
 * @param[out]	err      - GLib object for error reporting.
 *
 * Remember:
 *      CL_MEM_READ_WRITE     - This flag specifies that the memory object will be read and written by a kernel.
 *                              This is the default.
 *      CL_MEM_WRITE_ONLY     - This flags specifies that the memory object will be written but not read by a kernel.
 *                              Reading from a buffer or image object created with CL_MEM_WRITE_ONLY inside a kernel is undefined.
 *      CL_MEM_ALLOC_HOST_PTR - This flag specifies that the application wants the OpenCL implementation to allocate
 *                              memory from host accessible memory.
 * */
static inline void setupBuffers( HBHostBuffers_t *const hst_buff,
	HBDeviceBuffers_t *const dev_buff, HBBuffersSize_t *const bufsz,
	const OCLObjects_t *const oclobj, const Parameters_t *const params,
	GError **err )
{
	GError *err_setbuf = NULL;


	/** STEP_RETRY_FLAG */
	bufsz->bug_step_retry = sizeof( cl_uint );

	/* Allocate step_retry_flag into device's memory. */
	dev_buff->bug_step_retry = ccl_buffer_new( oclobj->ctx, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
		bufsz->bug_step_retry, NULL, &err_setbuf );
	hb_if_err_propagate_goto( err, err_setbuf, error_handler );


	/** RANDOM SEEDS */

	bufsz->rng_state = params->bugs_number * sizeof( cl_uint );

	/* DEBUG: (to remove) Allocate vector for random seeds in host memory. */
/*
	hst_buff->rng_state = (cl_uint *) malloc( bufsz->rng_state );
	hb_if_err_create_goto( *err, HB_ERROR,
		hst_buff->rng_state == NULL,
		HB_MALLOC_FAILURE, error_handler,
		"Unable to allocate host memory for random seeds/states." );
*/

	/* Allocate vector of random seeds in device memory. */
	dev_buff->rng_state = ccl_buffer_new( oclobj->ctx, CL_MEM_READ_WRITE,
		bufsz->rng_state, NULL, &err_setbuf );
	hb_if_err_propagate_goto( err, err_setbuf, error_handler );


	/** SWARM */

	bufsz->swarm_bugPosition = params->bugs_number * sizeof( cl_uint );

/*
	hst_buff->swarm_bugPosition = (cl_uint *) malloc( bufsz->swarm_bugPosition );
	hb_if_err_create_goto( *err, HB_ERROR,
		hst_buff->swarm_bugPosition == NULL,
		HB_MALLOC_FAILURE, error_handler,
		"Unable to allocate host memory for swarm bug position." );
*/

	dev_buff->swarm_bugPosition = ccl_buffer_new( oclobj->ctx, CL_MEM_READ_WRITE,
		bufsz->swarm_bugPosition, NULL, &err_setbuf );
	hb_if_err_propagate_goto( err, err_setbuf, error_handler );


	/** SWARM MAP */

	bufsz->swarm_map = params->world_size * sizeof( cl_uint );

	/* DEBUG: (to remove) Allocate vector for the swarm map in host memory. */
/*
	hst_buff->swarm_map = (cl_uint *) malloc( bufsz->swarm_map );
	hb_if_err_create_goto( *err, HB_ERROR,
		hst_buff->swarm_map == NULL,
		HB_MALLOC_FAILURE, error_handler,
		"Unable to allocate host memory for swarm map." );
*/

	/* Allocate vector for the swarm map in device memory. */
	dev_buff->swarm_map = ccl_buffer_new( oclobj->ctx, CL_MEM_READ_WRITE,
		bufsz->swarm_map, NULL, &err_setbuf );
	hb_if_err_propagate_goto( err, err_setbuf, error_handler );


	/** HEAT MAP */

	bufsz->heat_map = params->world_size * sizeof( cl_float );

	/* DEBUG: (to remove) Allocate temperature vector in host memory. */
/*
	hst_buff->heat_map[0] = (cl_float *) malloc( bufsz->heat_map );
	hb_if_err_create_goto( *err, HB_ERROR,
		hst_buff->heat_map[0] == NULL,
		HB_MALLOC_FAILURE, error_handler,
		"Unable to allocate host memory for heat map 1." );
*/

	/* DEBUG: (to remove) Allocate temperature vector in host memory. */
/*
	hst_buff->heat_map[1] = (cl_float *) malloc( bufsz->heat_map );
	hb_if_err_create_goto( *err, HB_ERROR,
		hst_buff->heat_map[1] == NULL,
		HB_MALLOC_FAILURE, error_handler,
		"Unable to allocate host memory for heat map 2." );
*/

	/* Allocate two vectors of temperature maps in device memory. */
	/* One vector is used for double buffering.                   */
	dev_buff->heat_map[0] = ccl_buffer_new( oclobj->ctx, CL_MEM_READ_WRITE,
		bufsz->heat_map, NULL, &err_setbuf );
	hb_if_err_propagate_goto( err, err_setbuf, error_handler );

	dev_buff->heat_map[1] = ccl_buffer_new( oclobj->ctx, CL_MEM_READ_WRITE,
		bufsz->heat_map, NULL, &err_setbuf );
	hb_if_err_propagate_goto( err, err_setbuf, error_handler );


	/** UNHAPINESS */

	bufsz->unhappiness = params->bugs_number * sizeof( cl_float );

/*
	hst_buff->unhappiness = (cl_float *) malloc( bufsz->unhappiness );
	hb_if_err_create_goto( *err, HB_ERROR,
		hst_buff->unhappiness == NULL,
		HB_MALLOC_FAILURE, error_handler,
		"Unable to allocate host memory for bugs unhappiness." );
*/

	dev_buff->unhappiness = ccl_buffer_new( oclobj->ctx, CL_MEM_READ_WRITE,
		bufsz->unhappiness, NULL, &err_setbuf );
	hb_if_err_propagate_goto( err, err_setbuf, error_handler );


	/** UNHAPPINESS REDUCED - all group sum to global memory. */

	bufsz->unhapp_reduced =
		params->reduce_num_workgroups * sizeof( cl_float );

/*
	hst_buff->unhapp_reduced = ( cl_float *) malloc( bufsz->unhapp_reduced );
	hb_if_err_create_goto( *err, HB_ERROR,
		hst_buff->unhapp_reduced == NULL,
		HB_MALLOC_FAILURE, error_handler,
		"Unable to allocate host memory for bugs unhappiness reduced." );
*/

	dev_buff->unhapp_reduced = ccl_buffer_new( oclobj->ctx,
		CL_MEM_READ_WRITE, bufsz->unhapp_reduced, NULL, &err_setbuf );
	hb_if_err_propagate_goto( err, err_setbuf, error_handler );


	/** UNHAPPINESS AVERAGE - To get the result. */

	bufsz->unhapp_average = sizeof( cl_float );

	hst_buff->unhapp_average = (cl_float *) malloc( bufsz->unhapp_average );
	hb_if_err_create_goto( *err, HB_ERROR,
		hst_buff->unhapp_average == NULL,
		HB_MALLOC_FAILURE, error_handler,
		"Unable to allocate host memory for bugs unhappiness average." );

	dev_buff->unhapp_average = ccl_buffer_new( oclobj->ctx,
		CL_MEM_READ_WRITE, bufsz->unhapp_average, NULL, &err_setbuf );
	hb_if_err_propagate_goto( err, err_setbuf, error_handler );


error_handler:
	/* If error handler is reached leave function imediately. */

	return;
}



/**
 * Get the kernel objects from the program. One for each kernel
 * function.
 *
 * @param[out]	krnl   -
 * @param[out]	gws    -
 * @param[out]	lws    -
 * @param[in]	oclobj -
 * @param[in]	params -
 * @param[out]	err    - GLib object for error reporting.
 * */
static inline void getKernels( HBKernels_t *const krnl,
	HBGlobalWorkSizes_t *const gws, HBLocalWorkSizes_t *const lws,
	const OCLObjects_t *const oclobj, const Parameters_t *const params,
	GError **err )
{
	GError *err_getkernels = NULL;
	size_t world_realdims[2] = {params->world_width, params->world_height};


	/** RANDOM INITIALIZATION Kernel: */
	/** A random per bug. */

	krnl->init_random = ccl_kernel_new( oclobj->prg,
		KRNL_NAME__INIT_RANDOM,	&err_getkernels );
	hb_if_err_propagate_goto( err, err_getkernels, error_handler );

	ccl_kernel_suggest_worksizes( krnl->init_random, oclobj->dev,
		DIMS_1,	&params->bugs_number,
		gws->init_random, lws->init_random, &err_getkernels );
	hb_if_err_propagate_goto( err, err_getkernels, error_handler );

	hbprintf( "[ kernel ]: init_random.\n    '-> bugs_num = %zu; gws = %zu; lws = %zu\n", params->bugs_number, gws->init_random[0], lws->init_random[0] );


	/** INIT_MAPS kernel: */
	/** SWARM_MAP and HEAT_MAP initialization Kernel. */

	krnl->init_maps = ccl_kernel_new( oclobj->prg,
		KRNL_NAME__INIT_MAPS, &err_getkernels );
	hb_if_err_propagate_goto( err, err_getkernels, error_handler );

	ccl_kernel_suggest_worksizes( krnl->init_maps, oclobj->dev,
		DIMS_1, &params->world_size,
		gws->init_maps, lws->init_maps,	&err_getkernels );
	hb_if_err_propagate_goto( err, err_getkernels, error_handler );

	hbprintf( "[ kernel ]: init_maps.\n    '-> world_size = %zu; gws = %zu; lws = %zu\n", params->world_size, gws->init_maps[0], lws->init_maps[0] );


	/** SWARM INITIALIZATION kernel: */
	/** To put bugs in the world and Reset unhappiness vector. */

	krnl->init_swarm = ccl_kernel_new( oclobj->prg,
		KRNL_NAME__INIT_SWARM, &err_getkernels );
	hb_if_err_propagate_goto( err, err_getkernels, error_handler );

	/* Dimensions are relactive to the number of bugs to be dropped in swarm map. */
	ccl_kernel_suggest_worksizes( krnl->init_swarm, oclobj->dev,
		DIMS_1,	&params->bugs_number,
		gws->init_swarm, lws->init_swarm, &err_getkernels );
	hb_if_err_propagate_goto( err, err_getkernels, error_handler );

	hbprintf( "[ kernel ]: init_swarm.\n    '-> bugs_num = %zu; gws = %zu; lws = %zu\n", params->bugs_number, gws->init_swarm[0], lws->init_swarm[0] );


	/** BUG STEP kernel: */
	/** Compute a feasible movement for each bug. */

	krnl->bug_step = ccl_kernel_new( oclobj->prg,
		KRNL_NAME__BUG_STEP, &err_getkernels );
	hb_if_err_propagate_goto( err, err_getkernels, error_handler );

	ccl_kernel_suggest_worksizes( krnl->bug_step, oclobj->dev,
		DIMS_1,	&params->world_size,
		gws->bug_step, lws->bug_step, &err_getkernels );
	hb_if_err_propagate_goto( err, err_getkernels, error_handler );

	hbprintf( "[ kernel ]: bug_step.\n    '-> world_size = %zu; gws = %zu; lws = %zu\n", params->world_size, gws->bug_step[0], lws->bug_step[0] );


	/** WORLD HEAT Computation kernel: */
	/** Compute world diffusion followed by world evapotation. */

	krnl->comp_world_heat = ccl_kernel_new( oclobj->prg,
		KRNL_NAME__COMP_WORLD_HEAT, &err_getkernels );
	hb_if_err_propagate_goto( err, err_getkernels, error_handler );

	ccl_kernel_suggest_worksizes( krnl->comp_world_heat, oclobj->dev,
		DIMS_2, world_realdims,
		gws->comp_world_heat, lws->comp_world_heat, &err_getkernels );
	hb_if_err_propagate_goto( err, err_getkernels, error_handler );

	hbprintf( "[ kernel ]: comp_world_heat.\n    '-> world_dims = [%zu, %zu]; gws = [%zu, %zu]; lws = [%zu, %zu]\n", world_realdims[0], world_realdims[1], gws->comp_world_heat[0], gws->comp_world_heat[1], lws->comp_world_heat[0], lws->comp_world_heat[1] );


	/** UNHAPPINESS STEP 1 - REDUCE kernel: */
	/** First step to compute bug's unhapines average. */

	krnl->unhapp_stp1_reduce = ccl_kernel_new( oclobj->prg,
		KRNL_NAME__UNHAPP_S1_REDUCE, &err_getkernels );
	hb_if_err_propagate_goto( err, err_getkernels, error_handler );

	/* gws->unhapp_stp1_reduce and lws->unhapp_stp1_reduce were */
	/* previously computed in function 'getOCLObjects(...)'     */

	hbprintf( "[ kernel ]: unhapp_stp1_reduce.\n    '-> bugs_num = %zu; gws = %zu; lws = %zu\n", params->bugs_number, gws->unhapp_stp1_reduce[0], lws->unhapp_stp1_reduce[0] );


	/** UNHAPPINESS STEP 2 - AVERAGE kernel: */
	/** Compute final reduction and then unhappiness average. */

	krnl->unhapp_stp2_average = ccl_kernel_new( oclobj->prg,
		KRNL_NAME__UNHAPP_S2_AVERAGE, &err_getkernels );
	hb_if_err_propagate_goto( err, err_getkernels, error_handler );

	/* The following values for gws and lws must be the ones that allow  */
	/* the final reduction. There will be only one workgroup.            */
	gws->unhapp_stp2_average[ 0 ] = lws->unhapp_stp1_reduce[ 0 ];
	lws->unhapp_stp2_average[ 0 ] = lws->unhapp_stp1_reduce[ 0 ];

	hbprintf( "[ kernel ]: unhapp_stp2_average.\n    '-> gws = %zu; lws = %zu\n", gws->unhapp_stp2_average[0], lws->unhapp_stp2_average[0] );


	hbprintf( "\n" );


error_handler:
		/* If error handler is reached leave function imediately. */

	return;
}



/**
 * Set kernels permanent parameters.
 *
 * @param[in]	krnl	  - Structure holding all kernels.
 * @param[in]	dev_buff - Structure holding all buffers required to be
 *                       setted as kernel parameters.
 * @param[in]	lws      - The local Work Size will be used to set
 *                       device's local memory to be passed to kernel.
 * */
static inline void setKernelParameters( const HBKernels_t *const krnl,
	const HBDeviceBuffers_t *const dev_buff,
	const HBLocalWorkSizes_t *const lws )
{
	/** 'init_random' kernel arguments. */
	ccl_kernel_set_arg( krnl->init_random, 0, dev_buff->rng_state );

	/** 'init_maps' kernel arguments. 'heat_map[0]' and 'heat_map[1]' */
	ccl_kernel_set_arg( krnl->init_maps, 0, dev_buff->swarm_map );
	ccl_kernel_set_arg( krnl->init_maps, 1, dev_buff->heat_map[0] );
	ccl_kernel_set_arg( krnl->init_maps, 2, dev_buff->heat_map[1] );

	/** 'init_swarm' kernel arguments. 'swarm[0]' and 'swarm[1]'      */
	ccl_kernel_set_arg( krnl->init_swarm, 0, dev_buff->swarm_bugPosition );
	ccl_kernel_set_arg( krnl->init_swarm, 1, dev_buff->swarm_map );
	ccl_kernel_set_arg( krnl->init_swarm, 2, dev_buff->unhappiness );
	ccl_kernel_set_arg( krnl->init_swarm, 3, dev_buff->rng_state );

	/** 'bug_step' kernel arguments. */
	ccl_kernel_set_arg( krnl->bug_step, 0, dev_buff->swarm_bugPosition );
	ccl_kernel_set_arg( krnl->bug_step, 1, dev_buff->swarm_map );
	/* ccl_kernel_set_arg( krnl->bug_step, 2, dev_buff->heat_map[0] ); */
	ccl_kernel_set_arg( krnl->bug_step, 3, dev_buff->unhappiness );
	ccl_kernel_set_arg( krnl->bug_step, 4, dev_buff->bug_step_retry );
	ccl_kernel_set_arg( krnl->bug_step, 5, dev_buff->rng_state );

	/** 'comp_world_heat' kernel arguments. */
	/* These arguments change in every iteration. It will be set in 'simulate(..) function. */

	//ccl_kernel_set_arg( krnl->comp_world_heat, 0, dev_buff->heat_map[0] );
	//ccl_kernel_set_arg( krnl->comp_world_heat, 1, dev_buff->heat_map[1] );

	/** 'unhappiness_stp1_reduce' kernel arguments. */
	ccl_kernel_set_arg( krnl->unhapp_stp1_reduce, 0, dev_buff->unhappiness );
	ccl_kernel_set_arg( krnl->unhapp_stp1_reduce, 1, ccl_arg_local( lws->unhapp_stp1_reduce[ 0 ], cl_float ) );
	ccl_kernel_set_arg( krnl->unhapp_stp1_reduce, 2, dev_buff->unhapp_reduced );

	/** 'unhappiness_stp2_average' kernel arguments. */
	ccl_kernel_set_arg( krnl->unhapp_stp2_average, 0, dev_buff->unhapp_reduced );
	ccl_kernel_set_arg( krnl->unhapp_stp2_average, 1, ccl_arg_local( lws->unhapp_stp2_average[ 0 ], cl_float ) );
	ccl_kernel_set_arg( krnl->unhapp_stp2_average, 2, dev_buff->unhapp_average );

	return;
}



/**
 * Run all init kernels.
 * */
static inline void initiate( HBKernels_t *const krnl,
	const HBGlobalWorkSizes_t *const gws,
	const HBLocalWorkSizes_t *const lws,
	OCLObjects_t *const oclobj, HBDeviceBuffers_t *const dev_buff,
	HBHostBuffers_t *const hst_buff, const HBBuffersSize_t *const bufsz,
	const Parameters_t *const params,  GError **err )
{
	GError *err_init = NULL;

	/** Events. */
	CCLEvent *evt_krnl_exec = NULL;  /* Event termination signal for  */
	                                 /* kernel execution. */
	CCLEventWaitList ewl = NULL;     /* Event Waiting List. A list of */
	                                 /* OpenCL events for operations  */
	                                 /* to be finished.               */


	/** INIT RANDOM. */

	hbprintf( "Init random:\n\tgws = %zu; lws = %zu\n", gws->init_random[0], lws->init_random[0] );

	evt_krnl_exec = ccl_kernel_enqueue_ndrange( krnl->init_random, oclobj->queue, DIMS_1, NULL, gws->init_random, lws->init_random, &ewl, &err_init );
	hb_if_err_propagate_goto( err, err_init, error_handler );

	/* Add kernel termination event to the wait list. */
	ccl_event_wait_list_add( &ewl, evt_krnl_exec, NULL );


	/* Read GPU generated seeds/gen_states, after wait for kernel termination. */

//	evt_rdwr = ccl_buffer_enqueue_read( dev_buff->rng_state, oclobj->queue, NON_BLOCK, 0, bufsz->rng_state, hst_buff->rng_state, &ewl, &err_init );
//	hb_if_err_propagate_goto( err, err_init, error_handler );

	/* Add read event to the wait list. */
//	ccl_event_wait_list_add( &ewl, evt_rdwr, NULL );


	/* Wait events termination. */
	ccl_event_wait( &ewl, &err_init );
	hb_if_err_propagate_goto( err, err_init, error_handler );


	/* DEBUG: Show GPU generated seeds. */
//	for (size_t i = 0; i < params->bugs_number; ++i) {
//		hbprintf( "%u, ", hst_buff->rng_state[i]);
//	}
//	hbprintf( "\n\n" );



	/** RESET SWARM_MAP and HEAT_MAP. */

	hbprintf( "Init maps:\n\tgws = %zu; lws = %zu\n", gws->init_maps[0], lws->init_maps[0] );

	evt_krnl_exec = ccl_kernel_enqueue_ndrange( krnl->init_maps, oclobj->queue, DIMS_1, NULL, gws->init_maps, lws->init_maps, &ewl, &err_init );
	hb_if_err_propagate_goto( err, err_init, error_handler );

	/* Add kernel termination event to the wait list. */
	ccl_event_wait_list_add( &ewl, evt_krnl_exec, NULL );


	/* Wait events termination. */
	ccl_event_wait( &ewl, &err_init );
	hb_if_err_propagate_goto( err, err_init, error_handler );


	/* Read swarm map, after wait for kernel termination. */

//	evt_rdwr = ccl_buffer_enqueue_read( dev_buff->swarm_map, oclobj->queue, NON_BLOCK, 0, bufsz->swarm_map, hst_buff->swarm_map, &ewl, &err_init );
//	hb_if_err_propagate_goto( err, err_init, error_handler );

//	ccl_event_wait_list_add( &ewl, evt_rdwr, NULL );


	/* Read heat map. */

//	evt_rdwr = ccl_buffer_enqueue_read( dev_buff->heat_map[0], oclobj->queue, NON_BLOCK, 0, bufsz->heat_map, hst_buff->heat_map[0], &ewl, &err_init );
//	hb_if_err_propagate_goto( err, err_init, error_handler );

//	ccl_event_wait_list_add( &ewl, evt_rdwr, NULL );

//	evt_rdwr = ccl_buffer_enqueue_read( dev_buff->heat_map[1], oclobj->queue, NON_BLOCK, 0, bufsz->heat_map, hst_buff->heat_map[1], &ewl, &err_init );
//	hb_if_err_propagate_goto( err, err_init, error_handler );

//	ccl_event_wait_list_add( &ewl, evt_rdwr, NULL );


	/* Wait last event termination. */
//	ccl_event_wait( &ewl, &err_init );
//	hb_if_err_propagate_goto( err, err_init, error_handler );

/*
	hbprintf( "\tTesting result from init maps. If no Err pass!\n\n" );
	for (size_t i = 0; i < params->world_size; ++i) {
		if (hst_buff->heat_map[0][i] != 0.0f) {
			hbprintf( "\t\tErr heatmap buf_0 at pos: %zu\n", i );
			exit(1);
		}
		if (hst_buff->heat_map[1][i] != 0.0f) {
			hbprintf( "\t\tErr heatmap buf_1 at pos: %zu\n", i );
			exit(1);
		}
		if (hst_buff->swarm_map[i] != 0) {
			hbprintf( "\t\tErr swarm_map at pos: %zu\n", i );
			exit(1);
		}
	}
*/

	/* INIT SWARM Fill the swarm map with bugs, compute unhapinnes. */

	hbprintf( "Init swarm:\n\tgws = %zu; lws = %zu\n", gws->init_swarm[0], lws->init_swarm[0] );

	evt_krnl_exec = ccl_kernel_enqueue_ndrange( krnl->init_swarm, oclobj->queue, DIMS_1, NULL, gws->init_swarm, lws->init_swarm, &ewl, &err_init );
	hb_if_err_propagate_goto( err, err_init, error_handler );

	/* Add kernel termination event to the wait list. */
	ccl_event_wait_list_add( &ewl, evt_krnl_exec, NULL );


	/* Wait events termination. */
	ccl_event_wait( &ewl, &err_init );
	hb_if_err_propagate_goto( err, err_init, error_handler );


	/* Read swarm map, after wait for kernel termination. */

//	evt_rdwr = ccl_buffer_enqueue_read( dev_buff->swarm_map, oclobj->queue, NON_BLOCK, 0, bufsz->swarm_map, hst_buff->swarm_map, &ewl, &err_init );
//	hb_if_err_propagate_goto( err, err_init, error_handler );

//	ccl_event_wait_list_add( &ewl, evt_rdwr, NULL );

	/* Read swarm. */

//	evt_rdwr = ccl_buffer_enqueue_read( dev_buff->swarm[0], oclobj->queue, NON_BLOCK, 0, bufsz->swarm, hst_buff->swarm[0], &ewl, &err_init );
//	hb_if_err_propagate_goto( err, err_init, error_handler );

//	ccl_event_wait_list_add( &ewl, evt_rdwr, NULL );

//	evt_rdwr = ccl_buffer_enqueue_read( dev_buff->swarm[1], oclobj->queue, NON_BLOCK, 0, bufsz->swarm, hst_buff->swarm[1], &ewl, &err_init );
//	hb_if_err_propagate_goto( err, err_init, error_handler );

//	ccl_event_wait_list_add( &ewl, evt_rdwr, NULL );

	/* Read unhappiness. */

//	evt_rdwr = ccl_buffer_enqueue_read( dev_buff->unhappiness, oclobj->queue, NON_BLOCK, 0, bufsz->unhappiness, hst_buff->unhappiness, &ewl, &err_init );
//	hb_if_err_propagate_goto( err, err_init, error_handler );

//	ccl_event_wait_list_add( &ewl, evt_rdwr, NULL );




	/* Wait last event termination. */
//	ccl_event_wait( & ewl, &err_init );
//	hb_if_err_propagate_goto( err, err_init, error_handler );

/*
	hbprintf( "\nShow results for swarm initiation.\n\n" );
	for (size_t i = 0; i < params->world_height; i++ )
	{
		for (size_t j = 0; j < params->world_width; j++ )
		{
			hbprintf( "%10u\t", hst_buff->swarm_map[ i * params->world_width + j ] );
		}
		hbprintf( "\n" );
	}
	hbprintf( "\n" );
*/

/*
	hbprintf( "Show results for swarm and unhappiness.\n\n" );
	for ( size_t i = 0; i < params->bugs_number; i++ )
	{
		size_t lin, col;
		lin = hst_buff->swarm[0][i] / params->world_width;
		col = hst_buff->swarm[0][i] % params->world_width;
		hbprintf( "bug: %4zu  loc: %4u [%zu, %zu] -> intent: %u  unhapp: %f\n", i, hst_buff->swarm[0][i], lin, col, hst_buff->swarm[1][i], hst_buff->unhappiness[i] );
	}
	hbprintf( "\n" );
*/

error_handler:
	/* If error handler is reached leave function imediately. */

	return;
}



static inline void simulate( const HBKernels_t *const krnl,
	const HBGlobalWorkSizes_t *const gws,
	const HBLocalWorkSizes_t *const lws,
	OCLObjects_t *const oclobj, HBDeviceBuffers_t *const dev_buff,
	HBHostBuffers_t *const hst_buff, HBBuffersSize_t *const bufsz,
	const Parameters_t *const params, FILE *hbResultFile, GError **err )
{
//	FILE *hbResultFile = NULL;
        GError *err_simul = NULL;

        CCLEvent *evt_rdwr = NULL;	    /* Read/Write termination event.  */
        CCLEvent *evt_krnl_exec = NULL;	    /* Kernel exec termination event. */
        CCLEventWaitList ewl = NULL;	    /* Event wait list. */

        HBBufferSelect_t bufsel;	    /* Buffer selectors. */
        size_t iter_counter;		    /* Iteration counter. */



	/** Get unhappiness. */
	/* Call reduction first, because initial state does already contain the bug's unhappiness. */

	/* Reduce step 1: */
	evt_krnl_exec = ccl_kernel_enqueue_ndrange( krnl->unhapp_stp1_reduce,
			oclobj->queue, DIMS_1, NULL, gws->unhapp_stp1_reduce,
			lws->unhapp_stp1_reduce, &ewl, &err_simul );
	hb_if_err_propagate_goto( err, err_simul, error_handler );

	/* Add 'kernel termination' event to the wait list. */
	ccl_event_wait_list_add( &ewl, evt_krnl_exec, NULL );

	/* Reduce step 2: */
	evt_krnl_exec = ccl_kernel_enqueue_ndrange( krnl->unhapp_stp2_average,
			oclobj->queue, DIMS_1, NULL, gws->unhapp_stp2_average,
			lws->unhapp_stp2_average, &ewl, &err_simul );
	hb_if_err_propagate_goto( err, err_simul, error_handler );

	/* Add 'kernel termination' event to the wait list. */
	ccl_event_wait_list_add( &ewl, evt_krnl_exec, NULL );


	/** read unhappiness. */

	evt_rdwr = ccl_buffer_enqueue_read( dev_buff->unhapp_average,
			oclobj->queue, NON_BLOCK, 0, bufsz->unhapp_average,
			hst_buff->unhapp_average, &ewl, &err_simul );
	hb_if_err_propagate_goto( err, err_simul, error_handler );

	/* Add read termination event to the wait list. */
	ccl_event_wait_list_add( &ewl, evt_rdwr, NULL );


	/* Wait for last event completion. */
	ccl_event_wait( &ewl, &err_simul );
	hb_if_err_propagate_goto( err, err_simul, error_handler );


	iter_counter = 0;

	/* Output result to file. */
	fprintf( hbResultFile, "%.17g\n", *hst_buff->unhapp_average );


	bufsel.main = 0;    /* On first step, main buffer has index 0.      */
	bufsel.secd = 1;    /* On first step, secondary buffer has index 1. */


	/*******************************/
	/**      SIMULATION LOOP      **/
	/*******************************/

	while ( (iter_counter < params->numIterations) || (params->numIterations == 0) )
	{
		/** Compute world heat, diffusion followed by evaporation. */

		/* Set transient arguments, using 'bufsel' to switch over. */
		ccl_kernel_set_arg( krnl->comp_world_heat, 0, dev_buff->heat_map[ bufsel.main ] );
		ccl_kernel_set_arg( krnl->comp_world_heat, 1, dev_buff->heat_map[ bufsel.secd ] );

		evt_krnl_exec =	ccl_kernel_enqueue_ndrange( krnl->comp_world_heat,
					oclobj->queue, DIMS_2, NULL,
					gws->comp_world_heat, lws->comp_world_heat,
					&ewl, &err_simul );
		hb_if_err_propagate_goto( err, err_simul, error_handler );

		/* Add kernel termination event to wait list. */
		ccl_event_wait_list_add( &ewl, evt_krnl_exec, NULL );


		/** Perform bug step. */

		/* Set transient argument, using 'bufsel' to use apropriate buffer. */
		ccl_kernel_set_arg( krnl->bug_step, 2, dev_buff->heat_map[ bufsel.secd ] );


		evt_krnl_exec = ccl_kernel_enqueue_ndrange( krnl->bug_step,
					oclobj->queue, DIMS_1, NULL,
					gws->bug_step, lws->bug_step,
					&ewl, &err_simul );
		hb_if_err_propagate_goto( err, err_simul, error_handler );

		/* Add kernel termination event to wait list. */
		ccl_event_wait_list_add( &ewl, evt_krnl_exec, NULL );


		/** Get unhappiness. */

		/* Reduce step 1: */
		evt_krnl_exec = ccl_kernel_enqueue_ndrange( krnl->unhapp_stp1_reduce,
					oclobj->queue, DIMS_1, NULL,
					gws->unhapp_stp1_reduce, lws->unhapp_stp1_reduce,
					&ewl, &err_simul );
		hb_if_err_propagate_goto( err, err_simul, error_handler );

		/* Add 'kernel termination' event to the wait list. */
		ccl_event_wait_list_add( &ewl, evt_krnl_exec, NULL );

		/* Reduce step 2: */
		evt_krnl_exec = ccl_kernel_enqueue_ndrange( krnl->unhapp_stp2_average,
					oclobj->queue, DIMS_1, NULL,
					gws->unhapp_stp2_average, lws->unhapp_stp2_average,
					&ewl, &err_simul );
		hb_if_err_propagate_goto( err, err_simul, error_handler );

		/* Add 'kernel termination' event to the wait list. */
		ccl_event_wait_list_add( &ewl, evt_krnl_exec, NULL );

		/* Read unhappiness. */

		evt_rdwr = ccl_buffer_enqueue_read( dev_buff->unhapp_average,
					oclobj->queue, NON_BLOCK, 0,
					bufsz->unhapp_average,
					hst_buff->unhapp_average, &ewl,
					&err_simul );
		hb_if_err_propagate_goto( err, err_simul, error_handler );

		/* Add read termination event to the wait list. */
		ccl_event_wait_list_add( &ewl, evt_rdwr, NULL );

		/* Wait for read event completion. */
		ccl_event_wait( &ewl, &err_simul );
		hb_if_err_propagate_goto( err, err_simul, error_handler );


		/* Output result to file. */
		fprintf( hbResultFile, "%.17g\n", *hst_buff->unhapp_average );

		/* Swap buffer's indices. */
		SWAP( cl_uint, bufsel.main, bufsel.secd );

		/* Next iteration. */
		iter_counter++;
	}


error_handler:

	return;
}








int main ( int argc, char *argv[] )
{
	FILE *hbResultFile = NULL;
	GError *err_main = NULL;	/* Error reporting object, from GLib. */

	Parameters_t params;		/* Host data; simulation parameters. */


	/** OpenCL related objects: context, device, queue, program. */
	OCLObjects_t oclobj = { NULL, NULL, NULL, NULL };

	/** Kernels. */
	HBKernels_t krnl = { NULL, NULL, NULL, NULL, NULL, NULL, NULL };
	/** Global work sizes for all kernels. */
	HBGlobalWorkSizes_t gws = { {0}, {0}, {0}, {0}, {0, 0}, {0}, {0} };
	/** Local work sizes for all kernels. */
	HBLocalWorkSizes_t  lws = { {0}, {0}, {0}, {0}, {0, 0}, {0}, {0} };

	/** Buffers for the host. */
	HBHostBuffers_t hst_buff = { NULL, /*NULL, NULL, NULL, { NULL, NULL }, NULL, NULL,*/ NULL };
	/** Buffers for the device. */
	HBDeviceBuffers_t dev_buff = { NULL, NULL, NULL, NULL, { NULL, NULL }, NULL, NULL, NULL };
	/** Buffers sizes. */
	HBBuffersSize_t bufsz;



	getSimulParameters( &params, argc, argv, &err_main );
	hb_if_err_goto( err_main, error_handler );

	getOCLObjects( &oclobj, &gws, &lws, &params, &err_main );
	hb_if_err_goto( err_main, error_handler );

	setupBuffers( &hst_buff, &dev_buff, &bufsz, &oclobj, &params, &err_main );
	hb_if_err_goto( err_main, error_handler );

	getKernels( &krnl, &gws, &lws, &oclobj, &params, &err_main );
	hb_if_err_goto( err_main, error_handler );

	/* Set the permanent kernel parameters (i.e. the buffers.). */
	setKernelParameters( &krnl, &dev_buff, &lws );


	/* Open output file for results. */
	hbResultFile = fopen( params.output_filename, "w+" );	/* Overwrite. */
	hb_if_err_create_goto( err_main, HB_ERROR,
		hbResultFile == NULL, HB_UNABLE_OPEN_FILE, error_handler,
		"Could not open output file." );


	/* Run all init kernels. */
	initiate( &krnl, &gws, &lws, &oclobj, &dev_buff, &hst_buff, &bufsz, &params, &err_main );
	hb_if_err_goto( err_main, error_handler );


	simulate( &krnl, &gws, &lws, &oclobj, &dev_buff, &hst_buff, &bufsz, &params, hbResultFile, &err_main );
	hb_if_err_goto( err_main, error_handler );


	hbprintf( "End...\n\n" );

	/* TODO: Profiling. */


	goto clean_all;


error_handler:

	/* Handle error. */
	fprintf( stderr, "Error: %s\n\n", err_main->message );
	g_error_free( err_main );


clean_all:


	/* Close output file. */
	if (hbResultFile) fclose( hbResultFile );

	/* Clean / Destroy all allocated items. */

	/* Destroy host buffers. */
	if (hst_buff.unhapp_average)	free( hst_buff.unhapp_average );
	/* if (hst_buff.unhapp_reduced)	free( hst_buff.unhapp_reduced ); */
	/* if (hst_buff.unhappiness)	free( hst_buff.unhappiness ); */
	/* if (hst_buff.heat_map[1])	free( hst_buff.heat_map[1] ); */
	/* if (hst_buff.heat_map[0])	free( hst_buff.heat_map[0] ); */
	/* if (hst_buff.swarm_map)	free( hst_buff.swarm_map ); */
	/* if (hst_buff.swarm)		free( hst_buff.swarm ); */
	/* if (hst_buff.rng_state)	free( hst_buff.rng_state ); */
	if (hst_buff.bug_step_retry)	free( hst_buff.bug_step_retry );

	/* Destroy Device buffers. */
	if (dev_buff.unhapp_average)	ccl_buffer_destroy( dev_buff.unhapp_average );
	if (dev_buff.unhapp_reduced)	ccl_buffer_destroy( dev_buff.unhapp_reduced );
	if (dev_buff.unhappiness)	ccl_buffer_destroy( dev_buff.unhappiness );
	if (dev_buff.heat_map[1])	ccl_buffer_destroy( dev_buff.heat_map[1] );
	if (dev_buff.heat_map[0])	ccl_buffer_destroy( dev_buff.heat_map[0] );
	if (dev_buff.swarm_map)		ccl_buffer_destroy( dev_buff.swarm_map );
	if (dev_buff.swarm_bugPosition)	ccl_buffer_destroy( dev_buff.swarm_bugPosition );
	if (dev_buff.rng_state)		ccl_buffer_destroy( dev_buff.rng_state );
	if (dev_buff.bug_step_retry)	ccl_buffer_destroy( dev_buff.bug_step_retry );

	/** Destroy kernel wrappers. */
	if (krnl.unhapp_stp2_average)	ccl_kernel_destroy( krnl.unhapp_stp2_average );
	if (krnl.unhapp_stp1_reduce)	ccl_kernel_destroy( krnl.unhapp_stp1_reduce );
	if (krnl.comp_world_heat)	ccl_kernel_destroy( krnl.comp_world_heat );
	if (krnl.bug_step)		ccl_kernel_destroy( krnl.bug_step );
	if (krnl.init_swarm)		ccl_kernel_destroy( krnl.init_swarm );
	if (krnl.init_maps)		ccl_kernel_destroy( krnl.init_maps );
	if (krnl.init_random)		ccl_kernel_destroy( krnl.init_random );

	/** Free remaining OpenCL wrappers. */
	if (oclobj.prg)		ccl_program_destroy( oclobj.prg );
	if (oclobj.queue)	ccl_queue_destroy( oclobj.queue );
	if (oclobj.ctx)		ccl_context_destroy( oclobj.ctx );


	/* Confirm that memory allocated by wrappers has been properly freed. */
	g_assert( ccl_wrapper_memcheck() );


	return OKI_DOKI;
}