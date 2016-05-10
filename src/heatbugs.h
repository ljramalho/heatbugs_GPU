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


#ifndef __HEATBUGS_GPU_H_
#define __HEATBUGS_GPU_H_


#include <sys/types.h>


/* Comment next to remove debug output. */
//#define USE_DEBUG


/* Evaluate to the number of elements of a staticaly defined vector. */
#define VSIZE( v ) ( sizeof( v ) / sizeof( v[0] ) )

/* Pre-processor for string concatenation. */
#define QUOTE(...) #__VA_ARGS__

/* Evaluate to 1 if 'val' is an odd integer, evaluate to 0 if 'val' is an even integer. */
#define IS_ODD( val ) ((val) & 1)


/** Heatbugs own error manipulation. **/

/* Jump to error handler. */
#define hb_if_err_goto( err, label ) \
	if ((err) != NULL) { \
		g_debug( CCL_STRD ); \
		g_propagate_error( err_dest, err_src ); \
		goto label; \
	}

/* Skip processing and return. */
#define hb_if_err_propagate( err_dest, err_src, ... ) \
	if ((err) != NULL) { \
		g_debug( CCL_STRD ); \
		g_propagate_error( err_dest, err_src ); \
		return ##__VA_ARGS__; \
	}


/* Input data used for simulation. */
typedef struct parameters {
	unsigned int bugs_temperature_min_ideal;	/* [0 .. 200] */
	unsigned int bugs_temperature_max_ideal;	/* [0 .. 200] */
	unsigned int bugs_heat_min_output;			/* [0 .. 100] */
	unsigned int bugs_heat_max_output;			/* [0 .. 100] */
	double bugs_random_move_chance;		/* Chance a bug will move [0..100].       */
	double world_diffusion_rate;		/* % temperature to adjacent cells [0..1] */
	double world_evaporation_rate;		/* % temperature's loss to 'ether' [0..1] */
	size_t bugs_number;					/* The number of bugs in the world.       */
	size_t world_width;
	size_t world_height;
	size_t world_size;					/* It is (world_height * world_width).    */
	size_t numIterations;				/* Iterations to stop. (0 = no stop).     */
	char output_filename[256];			/* The file to send results. */
} parameters_t;


enum hb_error_codes {
	/** Successfull operation. */
	HB_SUCCESS = 0,
	/** Invalid parameters. */
	HB_INVALID_PARAMETERS = -1,
	/** Unable to open file. */
	HB_UNABLE_OPEN_FILE = -2,
	/** Number of bugs is zero. */
	HB_BUGS_ZERO = -3,
	/** Bugs exceed world slots. */
	HB_BUGS_OVERFLOW = -4,
	/** Memory alocation failed. */
	HB_MALLOC_FAILURE = -5
};


#endif
