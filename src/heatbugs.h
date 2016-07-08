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


/* Evaluate to the number of elements of a staticaly defined vector. */
#define VSIZE( v ) ( sizeof( v ) / sizeof( v[0] ) )

/* Pre-processor for string concatenation. */
#define QUOTE(...) #__VA_ARGS__

/* Evaluate to 1 if 'val' is an odd integer, evaluate to 0 if 'val' is an even integer. */
#define IS_ODD( val ) ((val) & 1)

/* Return the value's square. */
#define SQUARE( x ) ((x) * (x))

/* Swap to values of indicated 'type'. */
#define SWAP( type, a, b ) { type t = a; a = b; b = t; }



/* Define NDEBUG for specific debug. Does not compile in windows. */
#ifndef CCL_STRD
	#ifdef NDEBUG
		#define CCL_STRD G_STRFUNC
	#else
		#define CCL_STRD G_STRLOC
	#endif
#endif


/* Error handling macros. Included here because it was removed from later cf4ocl2. */
/* It relies on Glib. */

#define hb_if_err_create_goto(err, quark, error_condition, error_code, label, msg, ...) \
	if (error_condition) { \
		g_debug(CCL_STRD); \
		g_set_error(&(err), (quark), (error_code), (msg), ##__VA_ARGS__); \
		goto label; \
	}

#define hb_if_err_goto(err, label) \
	if ((err) != NULL) { \
		g_debug(CCL_STRD); \
		goto label; \
	}

#define hb_if_err_propagate_goto(err_dest, err_src, label) \
	if ((err_src) != NULL) { \
		g_debug(CCL_STRD); \
		g_propagate_error(err_dest, err_src); \
		goto label; \
	}



/** Heatbugs own error manipulation. **/
enum hb_error_codes {
	/** Successfull operation. */
	HB_SUCCESS = 0,
	/** Invalid parameters. */
	HB_INVALID_PARAMETER = -1,
	/** A command line option with missing argument. */
	HB_PARAM_ARG_MISSING = -2,
	/** Unknown option in the command line. */
	HB_PARAM_OPTION_UNKNOWN = -3,
	/** Unknown option characters in command line. */
	HB_PARAM_CHAR_UNKNOWN = -4,
	/** Weird error occurred while parsing parameter. */
	HB_PARAM_PARSING = -5,
	/** Number of bugs is zero. */
	HB_BUGS_ZERO = -6,
	/** Bugs exceed world slots. */
	HB_BUGS_OVERFLOW = -7,
	/** Bug's ideal temperature range overlaps. */
	HB_TEMPERATURE_OVERLAP = -8,
	/** Bug's max ideal temperature exceeds range. */
	HB_TEMPERATURE_OUT_RANGE = -9,
	/** Bug's output heat range overlap. */
	HB_OUTPUT_HEAT_OVERLAP = -10,
	/** Bug's max output heat exceeds range. */
	HB_OUTPUT_HEAT_OUT_RANGE = -11,
	/** Unable to open file. */
	HB_UNABLE_OPEN_FILE = -12,
	/** Memory alocation failed. */
	HB_MALLOC_FAILURE = -13
};

#endif
