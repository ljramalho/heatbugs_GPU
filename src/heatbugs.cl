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



/**
 * from:
 * http://www.reedbeta.com/blog/2013/01/12/quick-and-easy-gpu-random-numbers-in-d3d11/
 * http://http.developer.nvidia.com/GPUGems3/gpugems3_ch37.html
 * https://math.stackexchange.com/questions/337782/pseudo-random-number-generation-on-the-gpu
 * https://blogs.unity3d.com/2015/01/07/a-primer-on-repeatable-random-numbers/
 * https://gist.github.com/badboy/6267743
 * */

/**
 * The kernel in this file expect the following preprocessor defines, passed
 * to kernel at compile time with -D option:
 *
	INIT_SEED
	REDUCE_NUM_WORKGROUPS
	BUGS_NUMBER
	WORLD_WIDTH
	WORLD_HEIGHT
	WORLD_SIZE
	WORLD_DIFFUSION_RATE
	WORLD_EVAPORATION_RATE
	BUGS_RANDOM_MOVE_CHANCE
	BUGS_TEMPERATURE_MIN_IDEAL
	BUGS_TEMPERATURE_MAX_IDEAL
	BUGS_HEAT_MIN_OUTPUT
	BUGS_HEAT_MAX_OUTPUT
 * */


/*!	Kernel version: 9.0	!*/



/* Enable doubles datatype in OpenCL kernel. */
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

/* Enable extended atomic operation on both, global and local memory. */
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable



/* Used to drive what shall happen to the agent at each step. */
#define GET_ANY_NEIGHBOUR	0x00ffffff
#define GET_MAX_TEMP_NEIGHBOUR	0x00ffff00
#define GET_MIN_TEMP_NEIGHBOUR	0x00ff00ff

/* Number of neighboring cell that surround the agent's position. */
#define NUM_NEIGHBOURS		8


/*
 * A uint 32 bit data type carries:
 *	- 8 bits for bug ideal temperature;
 *	- 8 bits filled with 1's, for bug presence;
 *	- 8 bits for bug output heat;
 *	- 8 bits flagging the bug's intention to move (aah = want to move; 00h = is at rest).
 *
 * If there is no bug, the uint variable for that bug must be zero, that is EMPTY_CELL.
 * */


 /** Do not change the following values and macros. */

/* Bug want to move, is bug's initial state. */
#define BUG		0x00ff00aa
#define EMPTY_CELL	0x00000000


/* Macros for 32 bits / unsigned int. */

#define BUG_NEW( uint_reg ) uint_reg = BUG

#define SET_BUG( uint_reg ) uint_reg = ((uint_reg & 0xff00ff00) | BUG)
#define SET_BUG_IDEAL_TEMPERATURE( uint_reg, value ) uint_reg = ((uint_reg & 0x00ffffff) | ((value) << 24))
#define SET_BUG_OUTPUT_HEAT( uint_reg, value ) uint_reg = ((uint_reg & 0xffff00ff) | ((value) << 8))


#define SET_BUG_TO_REST( uint_reg ) uint_reg = (uint_reg & 0xffffff00)
#define SET_BUG_TO_MOVE( uint_reg ) uint_reg = ((uint_reg & 0xffffff00) | 0x000000aa)


#define HAS_BUG( uint_reg ) ((uint_reg) != EMPTY_CELL)
#define HAS_NO_BUG( uint_reg ) ((uint_reg) == EMPTY_CELL)

#define GET_BUG_IDEAL_TEMPERATURE( uint_reg ) (((uint_reg) & 0xff000000) >> 24)
#define GET_BUG_OUTPUT_HEAT( uint_reg ) (((uint_reg) & 0x0000ff00) >> 8)

#define BUG_HAS_MOVED( uint_reg ) ((uint_reg & 0x00ff00ff) != BUG)



#define RST_BUG_STEP_RETRY_FLAG		0x00000000
#define SET_BUG_STEP_RETRY_FLAG		0xffffffff

#define RESET_REPEAT_STEP( uint_reg ) uint_reg = RST_BUG_STEP_RETRY_FLAG
#define REPORT_REPEAT_STEP( uint_reg ) uint_reg = SET_BUG_STEP_RETRY_FLAG





/**
 * @brief Performs integer division return the ceiling instead of the floor if
 * it is not an exact division.
 *
 * @param a Integer numerator.
 * @param b Integer denominator.
 * */
#define DIV_CEIL( a, b ) (((a) + (b) - 1) / (b))

/**
 * @brief Compute the next multiple of a given divisor which is equal or larger
 * than a given value.
 *
 * Both val and div are assumed to be positive integers.
 * @param val Minimum value
 * @param div The return value must be a multiple of the divisor.
 * */
 #define NEXT_MULTIPLE( val, div ) ( (val) + (div) - (val) % (div) )




enum {SW = 0, S, SE, W, E, NW, N, NE};


/*
 * https://gist.github.com/Marc-B-Reynolds/0b5f1db5ad7a3e453596
 * https://groups.google.com/forum/#!topic/prng/rajh-G5WvG0
 *
 * Xorshift algorithm from George Marsaglia's paper.
 * Value 0 will never be produced.
 * All non-zero values will be produced with period 2^32 - 1.
 *
 * failures from small crush
 *   1  BirthdaySpacings
 * failures from small crush (bit reversed)
 *   1  BirthdaySpacings
 *   6  MaxOft
 *
 * The rng_state is updated.
 * */
inline uint rand_xorShift32( __global uint *rng_state )
{
	__private uint state = *rng_state;

	state ^= (state << 13);
	state ^= (state >> 17);
	state ^= (state << 5);

	*rng_state = state;

	return state * 1597334677;
}



/*
 * Return random float in range [min .. max[.
 * The rng_state is updated.
 * */
inline double randomFloat( uint min, uint max, __global uint *rng_state )
{
	return (max - min) * (double) rand_xorShift32( rng_state ) * ( 1.0 / 4294967296.0 ) + min;
}



/*
 * Return random integer in range [min .. max[.
 * The rng_state is updated.
 * */
inline uint randomInt( uint min, uint max, __global uint *rng_state )
{
	return (max - min) * (double) rand_xorShift32( rng_state ) * ( 1.0 / 4294967296.0 ) + min;
}



/*
 * For any agent it returns either:
 *		1- A location with best (MAX Temperature | MIN Temperature).
 *		2- Any random location.
 *		3- Self location.
 *
 * World is a vector mapped to a matrix growing from SouthWest (0,0) toward NorthEast, that is, first cartesian quadrant.
 * */
inline uint2 best_neighbour( int todo, __global float *heat_map, __private uint2 best_bug_locus, __global uint *rng_state )
{
	/* Bug vector position in the world to 2D position. */
	__private const uint rc = best_bug_locus.s0 / WORLD_WIDTH;    			/* Central row. */
	__private const uint cc = best_bug_locus.s0 % WORLD_WIDTH;			/* Central col. */

	/* Neighbouring rows and columns. */
	__private const uint rn = (rc + 1) % WORLD_HEIGHT;				/* Row at North.   */
	__private const uint rs = (rc + WORLD_HEIGHT - 1) % WORLD_HEIGHT;		/* Row at South.   */
	__private const uint ce = (cc + 1) % WORLD_WIDTH;				/* Column at East. */
	__private const uint cw = (cc + WORLD_WIDTH - 1) % WORLD_WIDTH;			/* Column at West. */

	__private uint2 neighbour[ NUM_NEIGHBOURS ];	/* NOTE: neighbour[..].s0 is position, neighbour[..].s1 is temperature. */

	/* To index neighbours. Randomly picks the first free neighbouring cell. */
	__private uint NEIGHBOUR_IDX[ NUM_NEIGHBOURS ] = {SW, S, SE, W, E, NW, N, NE};


	/*
	   Fisher-Yates shuffle algorithm to shuffle an index vector so we can pick a neighbour in random order
	   checking each by index.
	 * */
	for (uint i = 0; i < NUM_NEIGHBOURS; i++)
	{
		uint rnd_i = randomInt( i, NUM_NEIGHBOURS, rng_state );

		if (rnd_i == i) continue;

		uint tmp = NEIGHBOUR_IDX[ i ];
		NEIGHBOUR_IDX[ i ] = NEIGHBOUR_IDX[ rnd_i ];
		NEIGHBOUR_IDX[ rnd_i ] = tmp;
	}


	/* Compute back the vector positions. Used on both, best location and random location. */
	neighbour[ SW ].s0 = rs * WORLD_WIDTH + cw;				/* SW neighbour position in the vector. */
	neighbour[ S  ].s0 = rs * WORLD_WIDTH + cc;				/* S  neighbour position in the vector. */
	neighbour[ SE ].s0 = rs * WORLD_WIDTH + ce;				/* SE neighbour position in the vector. */
	neighbour[ W  ].s0 = rc * WORLD_WIDTH + cw;				/* W  neighbour position in the vector. */
	neighbour[ E  ].s0 = rc * WORLD_WIDTH + ce;				/* E  neighbour position in the vector. */
	neighbour[ NW ].s0 = rn * WORLD_WIDTH + cw;				/* NW neighbour position in the vector. */
	neighbour[ N  ].s0 = rn * WORLD_WIDTH + cc;				/* N  neighbour position in the vector. */
	neighbour[ NE ].s0 = rn * WORLD_WIDTH + ce;				/* NE neighbour position in the vector. */

	/* Fetch temperature of all neighbouring positions. Store float in uint using OpenCL type reinterpretation. */
	neighbour[ SW ].s1 = as_uint( heat_map[ neighbour[ SW ].s0 ] );		/* Temperature at SW cell. */
	neighbour[ S  ].s1 = as_uint( heat_map[ neighbour[ S  ].s0 ] );		/* Temperature at S  cell. */
	neighbour[ SE ].s1 = as_uint( heat_map[ neighbour[ SE ].s0 ] );		/* Temperature at SE cell. */
	neighbour[ W  ].s1 = as_uint( heat_map[ neighbour[ W  ].s0 ] );		/* Temperature at W  cell. */
	neighbour[ E  ].s1 = as_uint( heat_map[ neighbour[ E  ].s0 ] );		/* Temperature at E  cell. */
	neighbour[ NW ].s1 = as_uint( heat_map[ neighbour[ NW ].s0 ] );		/* Temperature at NW cell. */
	neighbour[ N  ].s1 = as_uint( heat_map[ neighbour[ N  ].s0 ] );		/* Temperature at N  cell. */
	neighbour[ NE ].s1 = as_uint( heat_map[ neighbour[ NE ].s0 ] );		/* Temperature at NE cell. */


	/** Here there is a random moving chance. **/

	/* Because NEIGHBOUR_IDX is a shuffled vector, whatever the index is, it return a random position. */
	if (todo == GET_ANY_NEIGHBOUR)
		return neighbour[ NEIGHBOUR_IDX[ 0 ] ];


	/** Here there is either, GET_MAX_TEMP_NEIGHBOUR or GET_MIN_TEMP_NEIGHBOUR. **/


	/* best_bug_locus variable has the actual bug's location, and local temperature until otherwise! */

	/* Loop unroll. */
	if (todo == GET_MAX_TEMP_NEIGHBOUR)
	{
		if (as_float( neighbour[ NEIGHBOUR_IDX[ 0 ] ].s1 ) > as_float( best_bug_locus.s1 ))
			best_bug_locus = neighbour[ NEIGHBOUR_IDX[ 0 ] ];

		if (as_float( neighbour[ NEIGHBOUR_IDX[ 1 ] ].s1 ) > as_float( best_bug_locus.s1 ))
			best_bug_locus = neighbour[ NEIGHBOUR_IDX[ 1 ] ];

		if (as_float( neighbour[ NEIGHBOUR_IDX[ 2 ] ].s1 ) > as_float( best_bug_locus.s1 ))
			best_bug_locus = neighbour[ NEIGHBOUR_IDX[ 2 ] ];

		if (as_float( neighbour[ NEIGHBOUR_IDX[ 3 ] ].s1 ) > as_float( best_bug_locus.s1 ))
			best_bug_locus = neighbour[ NEIGHBOUR_IDX[ 3 ] ];

		if (as_float( neighbour[ NEIGHBOUR_IDX[ 4 ] ].s1 ) > as_float( best_bug_locus.s1 ))
			best_bug_locus = neighbour[ NEIGHBOUR_IDX[ 4 ] ];

		if (as_float( neighbour[ NEIGHBOUR_IDX[ 5 ] ].s1 ) > as_float( best_bug_locus.s1 ))
			best_bug_locus = neighbour[ NEIGHBOUR_IDX[ 5 ] ];

		if (as_float( neighbour[ NEIGHBOUR_IDX[ 6 ] ].s1 ) > as_float( best_bug_locus.s1 ))
			best_bug_locus = neighbour[ NEIGHBOUR_IDX[ 6 ] ];

		if (as_float( neighbour[ NEIGHBOUR_IDX[ 7 ] ].s1 ) > as_float( best_bug_locus.s1 ))
			best_bug_locus = neighbour[ NEIGHBOUR_IDX[ 7 ] ];
	}
	else	/* todo == GET_MIN_TEMP_NEIGHBOUR */
	{
		if (as_float( neighbour[ NEIGHBOUR_IDX[ 0 ] ].s1 ) < as_float( best_bug_locus.s1 ))
			best_bug_locus = neighbour[ NEIGHBOUR_IDX[ 0 ] ];

		if (as_float( neighbour[ NEIGHBOUR_IDX[ 1 ] ].s1 ) < as_float( best_bug_locus.s1 ))
			best_bug_locus = neighbour[ NEIGHBOUR_IDX[ 1 ] ];

		if (as_float( neighbour[ NEIGHBOUR_IDX[ 2 ] ].s1 ) < as_float( best_bug_locus.s1 ))
			best_bug_locus = neighbour[ NEIGHBOUR_IDX[ 2 ] ];

		if (as_float( neighbour[ NEIGHBOUR_IDX[ 3 ] ].s1 ) < as_float( best_bug_locus.s1 ))
			best_bug_locus = neighbour[ NEIGHBOUR_IDX[ 3 ] ];

		if (as_float( neighbour[ NEIGHBOUR_IDX[ 4 ] ].s1 ) < as_float( best_bug_locus.s1 ))
			best_bug_locus = neighbour[ NEIGHBOUR_IDX[ 4 ] ];

		if (as_float( neighbour[ NEIGHBOUR_IDX[ 5 ] ].s1 ) < as_float( best_bug_locus.s1 ))
			best_bug_locus = neighbour[ NEIGHBOUR_IDX[ 5 ] ];

		if (as_float( neighbour[ NEIGHBOUR_IDX[ 6 ] ].s1 ) < as_float( best_bug_locus.s1 ))
			best_bug_locus = neighbour[ NEIGHBOUR_IDX[ 6 ] ];

		if (as_float( neighbour[ NEIGHBOUR_IDX[ 7 ] ].s1 ) < as_float( best_bug_locus.s1 ))
			best_bug_locus = neighbour[ NEIGHBOUR_IDX[ 7 ] ];
	}

	return best_bug_locus;
}



inline uint2 any_free_neighbour( __global float *heat_map, __global uint *swarm_map, __private uint2 bug_locus,
					__global uint *rng_state )
{
	/* Bug vector position in the world to 2D position. */
	__private const uint rc = bug_locus.s0 / WORLD_WIDTH;				/* Central row. */
	__private const uint cc = bug_locus.s0 % WORLD_WIDTH;				/* Central col. */

	/* Neighbouring rows and columns. */
	__private const uint rn = (rc + 1) % WORLD_HEIGHT;				/* Row at North.   */
	__private const uint rs = (rc + WORLD_HEIGHT - 1) % WORLD_HEIGHT;		/* Row at South.   */
	__private const uint ce = (cc + 1) % WORLD_WIDTH;				/* Column at East. */
	__private const uint cw = (cc + WORLD_WIDTH - 1) % WORLD_WIDTH;			/* Column at West. */

	__private uint2 neighbour[ NUM_NEIGHBOURS ];	/* NOTE: neighbour[..].s0 is position, neighbour[..].s1 is temperature. */

	/* To index neighbours. Randomly picks the first free neighbouring cell. */
	__private uint NEIGHBOUR_IDX[ NUM_NEIGHBOURS ] = {SW, S, SE, W, E, NW, N, NE};


	/*
	   Fisher-Yates shuffle algorithm to shuffle an index vector so we can pick a neighbour in random order
	   checking each by index.
	 * */
	for (uint i = 0; i < NUM_NEIGHBOURS; i++)
	{
		uint rnd_i = randomInt( i, NUM_NEIGHBOURS, rng_state );

		if (rnd_i == i) continue;

		uint tmp = NEIGHBOUR_IDX[ i ];
		NEIGHBOUR_IDX[ i ] = NEIGHBOUR_IDX[ rnd_i ];
		NEIGHBOUR_IDX[ rnd_i ] = tmp;
	}


	/* Compute back the vector positions. Used on both, best location and random location. */
	neighbour[ SW ].s0 = rs * WORLD_WIDTH + cw;				/* SW neighbour position in the vector. */
	neighbour[ S  ].s0 = rs * WORLD_WIDTH + cc;				/* S  neighbour position in the vector. */
	neighbour[ SE ].s0 = rs * WORLD_WIDTH + ce;				/* SE neighbour position in the vector. */
	neighbour[ W  ].s0 = rc * WORLD_WIDTH + cw;				/* W  neighbour position in the vector. */
	neighbour[ E  ].s0 = rc * WORLD_WIDTH + ce;				/* E  neighbour position in the vector. */
	neighbour[ NW ].s0 = rn * WORLD_WIDTH + cw;				/* NW neighbour position in the vector. */
	neighbour[ N  ].s0 = rn * WORLD_WIDTH + cc;				/* N  neighbour position in the vector. */
	neighbour[ NE ].s0 = rn * WORLD_WIDTH + ce;				/* NE neighbour position in the vector. */

	/* Fetch temperature of all neighbouring positions. Store float in uint using OpenCL type reinterpretation. */
	neighbour[ SW ].s1 = as_uint( heat_map[ neighbour[ SW ].s0 ] );		/* Temperature at SW cell. */
	neighbour[ S  ].s1 = as_uint( heat_map[ neighbour[ S  ].s0 ] );		/* Temperature at S  cell. */
	neighbour[ SE ].s1 = as_uint( heat_map[ neighbour[ SE ].s0 ] );		/* Temperature at SE cell. */
	neighbour[ W  ].s1 = as_uint( heat_map[ neighbour[ W  ].s0 ] );		/* Temperature at W  cell. */
	neighbour[ E  ].s1 = as_uint( heat_map[ neighbour[ E  ].s0 ] );		/* Temperature at E  cell. */
	neighbour[ NW ].s1 = as_uint( heat_map[ neighbour[ NW ].s0 ] );		/* Temperature at NW cell. */
	neighbour[ N  ].s1 = as_uint( heat_map[ neighbour[ N  ].s0 ] );		/* Temperature at N  cell. */
	neighbour[ NE ].s1 = as_uint( heat_map[ neighbour[ NE ].s0 ] );		/* Temperature at NE cell. */


	/** Find any / random free position and return that position along with his temperature. **/

	/* Right now, rand_bug_locus variable has the current bug's location. */

	/* Loop unroll. */
	if (HAS_NO_BUG( swarm_map[ neighbour[ NEIGHBOUR_IDX[ 0 ] ].s0 ] ))
		// rand_bug_locus = neighbour[ NEIGHBOUR_IDX[ 0 ] ];
		return neighbour[ NEIGHBOUR_IDX[ 0 ] ];

	if (HAS_NO_BUG( swarm_map[ neighbour[ NEIGHBOUR_IDX[ 1 ] ].s0 ] ))
		// rand_bug_locus = neighbour[ NEIGHBOUR_IDX[ 1 ] ];
		return neighbour[ NEIGHBOUR_IDX[ 1 ] ];

	if (HAS_NO_BUG( swarm_map[ neighbour[ NEIGHBOUR_IDX[ 2 ] ].s0 ] ))
		// rand_bug_locus = neighbour[ NEIGHBOUR_IDX[ 2 ] ];
		return neighbour[ NEIGHBOUR_IDX[ 2 ] ];

	if (HAS_NO_BUG( swarm_map[ neighbour[ NEIGHBOUR_IDX[ 3 ] ].s0 ] ))
		// rand_bug_locus = neighbour[ NEIGHBOUR_IDX[ 3 ] ];
		return neighbour[ NEIGHBOUR_IDX[ 3 ] ];

	if (HAS_NO_BUG( swarm_map[ neighbour[ NEIGHBOUR_IDX[ 4 ] ].s0 ] ))
		// rand_bug_locus = neighbour[ NEIGHBOUR_IDX[ 4 ] ];
		return neighbour[ NEIGHBOUR_IDX[ 4 ] ];

	if (HAS_NO_BUG( swarm_map[ neighbour[ NEIGHBOUR_IDX[ 5 ] ].s0 ] ))
		// rand_bug_locus = neighbour[ NEIGHBOUR_IDX[ 5 ] ];
		return neighbour[ NEIGHBOUR_IDX[ 5 ] ];

	if (HAS_NO_BUG( swarm_map[ neighbour[ NEIGHBOUR_IDX[ 6 ] ].s0 ] ))
		// rand_bug_locus = neighbour[ NEIGHBOUR_IDX[ 6 ] ];
		return neighbour[ NEIGHBOUR_IDX[ 6 ] ];

	if (HAS_NO_BUG( swarm_map[ neighbour[ NEIGHBOUR_IDX[ 7 ] ].s0 ] ))
		// rand_bug_locus = neighbour[ NEIGHBOUR_IDX[ 7 ] ];
		return neighbour[ NEIGHBOUR_IDX[ 7 ] ];


	return bug_locus;
}




/**
 * ************* KERNELS ******************
 * */


/**
 * Initiates the random generator by fill a vector of seeds / initial rng state.
 * Wang Hash, hash one integer into another.
 * Obliterate any correlation in the seeds when going wide (parallel), prior
 * any use of a PRNG (going deep).
 * Correlation comes from sequential global ID's of each workitem.
 * */
__kernel void init_random( __global uint *rng_state )
{
	__private uint seed;

	const uint gid = get_global_id( 0 );

	if (gid >= BUGS_NUMBER) return;

	seed = gid ^ INIT_SEED;      /* Add diversity at each run. */

	/* Use Wang Hash to decorrelate numbers. */
	seed = (seed ^ 61) ^ (seed >> 16);
	seed += (seed << 3);
	seed = seed ^ (seed >> 4);
	seed *= 0x27d4eb2d;
	seed = seed ^ (seed >> 15);

	/* Store result in global space. */
	rng_state[ gid ] = seed;

	return;
}



__kernel void init_maps( __global uint *swarm_map, __global float *heat_map, __global float *heat_buffer )
{
	const uint gid = get_global_id( 0 );

	if (gid >= WORLD_SIZE) return;

	swarm_map[ gid ] = EMPTY_CELL;	/* Clean all bugs from the map. */
	heat_map[ gid ] = 0.0;	     	/* Reset all temperatures from map. */
	heat_buffer[ gid ] = 0.0;    	/* Reset all temperatures from buffer. */

	return;
}



/**
 * Fill the world (swarm_map) with bugs, store their position (swarm_bugPosition).
 * Initiate the unhappiness for each bug.
 * */
__kernel void init_swarm( __global uint *swarm_bugPosition, __global uint *swarm_map, __global float *unhappiness,
				__global uint *rng_state )
{
	__private uint bug_locus;
	__private uint bug_ideal_temperature;	/* [0..200] */
	__private uint bug_output_heat;		/* [0..100] */
	__private uint bug_new;
	__private uint on_locus;


	const uint bug_id = get_global_id( 0 );

	if (bug_id >= BUGS_NUMBER) return;


	bug_ideal_temperature = (ushort) randomInt( BUGS_TEMPERATURE_MIN_IDEAL,
							BUGS_TEMPERATURE_MAX_IDEAL, &rng_state[ bug_id ] );

	bug_output_heat = (ushort) randomInt( BUGS_HEAT_MIN_OUTPUT, BUGS_HEAT_MAX_OUTPUT, &rng_state[ bug_id ] );

	/* Create a bug. */
	BUG_NEW( bug_new );
	SET_BUG_IDEAL_TEMPERATURE( bug_new, bug_ideal_temperature );
	SET_BUG_OUTPUT_HEAT( bug_new, bug_output_heat );

	/* Try, until succeed, to leave a bug in a empty space. */
	do {
		bug_locus = randomInt( 0, WORLD_SIZE, &rng_state[ bug_id ] );

		on_locus = atomic_cmpxchg( &swarm_map[ bug_locus ], EMPTY_CELL, bug_new );

	} while ( HAS_BUG( on_locus ) );

	barrier( CLK_GLOBAL_MEM_FENCE );	/* All still active workitems must arrive here to sync before go on. */

	/* Store bug position in the swarm. */
	swarm_bugPosition[ bug_id ] = bug_locus;

	/*
	   Compute initial unhappiness as abs(ideal_temperature - temperature).
	   Since world temperature is initially zero, it turns out that initial
	   unhappiness = ideal_temperature.
	 * */

	/* REPLACED: unhappiness[ bug_id ] = (float) bug_ideal_temperature; */
	/* With OpenCL type convertion.                                     */
	unhappiness[ bug_id ] = convert_float( bug_ideal_temperature );

	return;
}



/**
 * Change all bugs to 'want to move' state.
 *
 * This kernel changes the 8 LSB bits of all bug's representation, (as defined by the macro 'BUG'),
 * to the hexadecimal value 'aa', meaning the bug wants to move.
 * */
__kernel void prepare_bug_step( __global uint *swarm_bugPosition, __global uint *swarm_map )
{
	__private uint bug;
	__private uint bug_locus;


	const uint bug_id = get_global_id( 0 );

	if (bug_id >= BUGS_NUMBER) return;


	/* Get the bug location. */
	bug_locus = swarm_bugPosition[ bug_id ];

	/* Set the bug in the swarm to the 'want to move' state, preparing next iteration. */
	SET_BUG_TO_MOVE( swarm_map[ bug_locus ] );

	return;
}



/** Reset the 'bug_step_retry' flag to prepare agents to signal host to repeat bug_step kernel. */
__kernel void prepare_step_report( __global uint *bug_step_retry )
{
	const uint id = get_global_id( 0 );

	if (id > 0) return;

	/* Reset bug_step retry flag. */
	*bug_step_retry = RST_BUG_STEP_RETRY_FLAG;

	return;
}



/**
 * Perform a bug movement in the world.
 *
 * Netlogo completely separates the report of the 'best' / 'random' location from the actual bug movement. Only when
 * perform 'bug movevent', the availability (status) of the reported 'best' / 'random' location is checked, and only then
 * a new alternate free location, if exists, is computed if necessary.
 * */
__kernel void bug_step_best( __global uint *swarm_bugPosition, __global uint *swarm_map, __global float *heat_map,
				__global float *unhappiness, __global uint *bug_step_retry, __global uint *rng_state )
{
	__private uint2 bug_locus;		/* bug_locus.s0 is bug position, bug_locus.s1 is local temperature. */
	__private uint2 bug_new_locus;		/* bug_new_locus.s0 is bug position, bug_new_locus.s1 is local temperature. */
	__private uint bug_ideal_temperature;	/* 0,1,2,...,200 */
	__private uint bug_output_heat;		/* 0,1,2,...,100 */
	__private float bug_unhappiness;

	__private uint bug;
	__private uint on_locus;

	__private int todo;


	const uint bug_id = get_global_id( 0 );

	if (bug_id >= BUGS_NUMBER) return;


	/* Get the bug location. */
	bug_locus.s0 = swarm_bugPosition[ bug_id ];

	/*
	   Get a private copy of the bug while keeping the original in the swarm.
	   Until this bug resolves his movement, the original one with his original status and his position is kept in
	   swarm_map and swarm_bugPosition.

	   The 'resting' status of bug's private copy is setted. So, if bug stays or move to another position, this private
	   copy of the bug is then stored in that position and that means the bug will be at 'resting' state.
	   But if the bug does not move, the original in the swarm is kept to be used in a second bug step process, and
	   that means the bug still be in a 'moving' state.
	 * */
	bug = swarm_map[ bug_locus.s0 ];

	/* Get local temperature. Store float as uint using OpenCL type reinterpretation. */
	bug_locus.s1 = as_uint( heat_map[ bug_locus.s0 ] );

	bug_ideal_temperature = GET_BUG_IDEAL_TEMPERATURE( bug );
	bug_output_heat = GET_BUG_OUTPUT_HEAT( bug );

	SET_BUG_TO_REST( bug );						/* Set bug's private copy into resting state. */

	/* IDEA: use CL typecast ? */
	// bug_unhappiness = fabs( (float) bug_ideal_temperature - as_float( bug_locus.s1 ) );
	bug_unhappiness = fabs( convert_float( bug_ideal_temperature ) - as_float( bug_locus.s1 ) );

	/* Update bug's unhappiness vector in global memory. */
	unhappiness[ bug_id ] = bug_unhappiness;

	/*
	   Usually compare equality of floats is absurd. Netlogo wrapps the code with: if (unhappiness > 0) { do stuff... }
	   I decided to unwrap, terminating as soon as possible using if (bug_unhappiness == 0.0f) {...} and continue the
	   remaining code as a fall out case for (bug_unhappiness > 0.0f). After all, this is semantically equivalent to
	   netlogo version.
 	 * */
	if (bug_unhappiness == 0.0f)
	{
		/* Bug hasn't move, so we don't need to update swarm_bugPosition. */

		/* Put the resting bug in his old position, (that is 'resting' status override). */
		atomic_xchg( &swarm_map[ bug_locus.s0 ], bug );

		/* Leave heat in the old bug position. Remember bug_locus.s1 is temperature at bug position. */
		heat_map[ bug_locus.s0 ] = as_float( bug_locus.s1 ) + convert_float( bug_output_heat );

		return;
	}

	/** Arriving here, means (bug_unhappiness > 0.0f) **/

	/*
	   Find the best place for the bug to go. Tricky stuff.

	   In order to implement netlogo approach, WITHOUT branching in OpenCL kernel, that is (in netlogo order), to compute:
		1) random-move-chance,
		2) best-patch for (temp <  ideal_temp) (when bug is COLD),
		3) best-patch for (temp >= ideal_temp) (when bug is HOT),
	   the OpenCl kernel select(...) function must be used twice.

	   A variable is used to hold what to do, (1), (2) or (3). However the order must be reversed since (1), when happen,
	   takes precedence over (2) or (3), whatever (2) XOR (3) is true or not.

	   REMEMBER OpenCL specs:	select(a, b, c), implements:	(c) ? b : a
	 * */

	todo = select( GET_MIN_TEMP_NEIGHBOUR, GET_MAX_TEMP_NEIGHBOUR, as_float( bug_locus.s1 ) < (float) bug_ideal_temperature );
	todo = select( todo, GET_ANY_NEIGHBOUR, randomFloat( 0, 100, &rng_state[ bug_id ] ) < BUGS_RANDOM_MOVE_CHANCE );

	bug_new_locus = best_neighbour( todo, heat_map, bug_locus, &rng_state[ bug_id ] );


	/* If bug's current location is already the best one... */
	if (bug_new_locus.s0 == bug_locus.s0)
	{
		/* Bug hasn't move, we don't need to update swarm_bugPosition. */

		/* Put the resting bug in his old position, (that is 'resting' status override). */
		atomic_xchg( &swarm_map[ bug_locus.s0 ], bug );

		/* Leave heat in the old bug position. Remember bug_locus.s1 is temperature at bug position. */
		heat_map[ bug_locus.s0 ] = as_float( bug_locus.s1 ) + convert_float( bug_output_heat );

		return;
	}


	/* Otherwise, try to store the resting bug in his new 'best' location and return.
	   REMEMBER OpenCL specs:	atomic_cmpxchg( *p, cmp, val )   perform the operations:
	   	old = *p;
	   	*p = (old == cmp) ? val : old;
	   	return old;
	 * */
	on_locus = atomic_cmpxchg( &swarm_map[ bug_new_locus.s0 ], EMPTY_CELL, bug );

	if (HAS_NO_BUG( on_locus ))
	{
		/* SUCCESS! Reset old bug location. Should be atomic in case another work-item try to read. */
		atomic_xchg( &swarm_map[ bug_locus.s0 ], EMPTY_CELL );

		/* Update bug position in the swarm. */
		swarm_bugPosition[ bug_id ] = bug_new_locus.s0;

		/* Leave heat in the new bug position. */
		heat_map[ bug_new_locus.s0 ] = as_float( bug_new_locus.s1 ) + convert_float( bug_output_heat );

		return;
	}


	/**
	   Here, the best place become or was unavailable.
	   The bug did'n move, and the bug 'want to move' status remain the same.
	 * */

	/* Signal the host to call bug_step_any_free(...) kernel. */
	REPORT_REPEAT_STEP( *bug_step_retry );

	return;
}



__kernel void bug_step_any_free( __global uint *swarm_bugPosition, __global uint *swarm_map, __global float *heat_map,
					 __global uint *bug_step_retry, __global uint *rng_state )
{
	__private uint2 bug_locus;		/* bug_locus.s0 is bug position, bug_locus.s1 is local temperature. */
	__private uint2 bug_new_locus;		/* bug_new_locus.s0 is bug position, bug_new_locus.s1 is local temperature. */
	__private uint bug_output_heat;		/* 0,1,2,...,100 */
	__private uint bug;
	__private uint on_locus;


	const uint bug_id = get_global_id( 0 );

	if (bug_id >= BUGS_NUMBER) return;


	/* Get the bug location. */
	bug_locus.s0 = swarm_bugPosition[ bug_id ];

	/*  Get a private copy of the bug while keeping the original in the swarm. */
	bug = swarm_map[ bug_locus.s0 ];

	/* Check if bug has already moved in this iteration to a best position. If it has moved, then exit. */
	if (BUG_HAS_MOVED( bug )) return;


	/** Arriving here means, the bug has not yet moved in this iteration. */

	/* Get local temperature. Store float as uint using OpenCL type reinterpretation. */
	bug_locus.s1 = as_uint( heat_map[ bug_locus.s0 ] );

	bug_output_heat = GET_BUG_OUTPUT_HEAT( bug );

	SET_BUG_TO_REST( bug );						/* Set bug's private copy into resting state. */


	/* Find random available position or return the same position. */
	bug_new_locus = any_free_neighbour( heat_map, swarm_map, bug_locus, &rng_state[ bug_id ] );


	/* If bug's new location is the current one, there are no available neighbours. */
	if (bug_new_locus.s0 == bug_locus.s0)
	{
		/* Bug hasn't move, we don't need to update swarm_bugPosition. */

		/* Put the resting bug in his old position, (that is 'resting' status override). */
		atomic_xchg( &swarm_map[ bug_locus.s0 ], bug );

		/* Leave heat in the old bug position. Remember bug_locus.s1 is temperature at bug position. */
		heat_map[ bug_locus.s0 ] = as_float( bug_locus.s1 ) + convert_float( bug_output_heat );

		return;
	}


	/* Otherwise, try to store the bug in his new 'free' location and return.
	   REMEMBER OpenCL specs:	atomic_cmpxchg( *p, cmp, val )   perform the operations:
	   	old = *p;
	   	*p = (old == cmp) ? val : old;
	   	return old;
	 * */
	on_locus = atomic_cmpxchg( &swarm_map[ bug_new_locus.s0 ], EMPTY_CELL, bug );

	if (HAS_NO_BUG( on_locus ))
	{
		/* SUCCESS! Reset old bug location. Should be atomic in case another work-item try to read. */
		atomic_xchg( &swarm_map[ bug_locus.s0 ], EMPTY_CELL );

		/* Update bug position in the swarm. */
		swarm_bugPosition[ bug_id ] = bug_new_locus.s0;

		/* Leave heat in the new bug position. */
		heat_map[ bug_new_locus.s0 ] = as_float( bug_new_locus.s1 ) + convert_float( bug_output_heat );

		return;
	}


	/* Signal the host to call bug_step_any_free(...) kernel. */
	REPORT_REPEAT_STEP( *bug_step_retry );


	return;
}



__kernel void comp_world_heat( __global float *heat_map, __global float *heat_buffer )
{
	__private const uint cc = get_global_id( 0 );	/* Column at Center. */
	__private const uint rc = get_global_id( 1 );	/* Row at Center.   */

	if (rc >= WORLD_HEIGHT || cc >= WORLD_WIDTH) return;


	/* Compute required neighbouring coodinates. */
	__private const uint rn = (rc + 1) % WORLD_HEIGHT;			/* Row at North.    */
	__private const uint rs = (rc + WORLD_HEIGHT - 1) % WORLD_HEIGHT;	/* Row at South.    */
	__private const uint ce = (cc + 1) % WORLD_WIDTH;			/* Column at East.  */
	__private const uint cw = (cc + WORLD_WIDTH - 1) % WORLD_WIDTH;		/* Columns at West. */

	__private uint pos;

	__private float heat = 0.0f;


	/** Compute Diffusion */

	/* Store heat from neighbouring cells. */

	/* SW */
	pos = rs * WORLD_WIDTH + cw;
	heat = heat + heat_map[ pos ];

	/* S  */
	pos = rs * WORLD_WIDTH + cc;
	heat = heat + heat_map[ pos ];

	/* SE */
	pos = rs * WORLD_WIDTH + ce;
	heat = heat + heat_map[ pos ];

	/* W  */
	pos = rc * WORLD_WIDTH + cw;
	heat = heat + heat_map[ pos ];

	/* E  */
	pos = rc * WORLD_WIDTH + ce;
	heat = heat + heat_map[ pos ];

	/* NW */
	pos = rn * WORLD_WIDTH + cw;
	heat = heat + heat_map[ pos ];

	/* N  */
	pos = rn * WORLD_WIDTH + cc;
	heat = heat + heat_map[ pos ];

	/* NE */
	pos = rn * WORLD_WIDTH + ce;
	heat = heat + heat_map[ pos ];


	/* Get the 8th part of diffusion percentage from all neighbour cells. */
	heat = heat * WORLD_DIFFUSION_RATE / 8;

	/* Add cell's remaining heat. */
	pos = rc * WORLD_WIDTH + cc;
	heat = heat + heat_map[ pos ] * (1 - WORLD_DIFFUSION_RATE);

	/* Compute Evaporation */
	heat = heat * (1 - WORLD_EVAPORATION_RATE);


	/* Double buffer it. */
	heat_buffer[pos] = heat;


	return;
}




/*
 *
 */
__kernel void unhappiness_step1_reduce( __global float *unhappiness, __local float *partial_sums,
						__global float *unhapp_reduced )
{
	const uint gid = get_global_id( 0 );
	const uint lid = get_local_id( 0 );
	const uint global_size = get_global_size( 0 );
	const uint group_size = get_local_size( 0 );

	__private uint serialCount, index, iter;

	__private float sum = 0.0f;

	/*
	   The size of vector unhappiness is the number of bugs (BUGS_NUMBER), so we must take care of both cases, when
	   BUGS_NUMBER > global_size, and when global_size > BUGS_NUMBER.
	   This is what happen in the next loop ('sum' is private to each workitem)...

	   When: (BUGS_NUMBER < global_size) -> serialCount = 1
	   	sum[0 .. BUGS_NUMBER - 1] = unhappiness[0 .. BUGS_NUMBER - 1]
	   	sum[BUGS_NUMBER .. global_size - 1] = 0

	   When: (BUGS_NUMBER == global_size) -> serialCount = 1
	   	sum[0 .. global_size - 1] = unhappiness[0 .. global_size - 1]

	   When: (global_size < BUGS_NUMBER < 2*global_size) -> serialCount = 2
	   	sum[0 .. BUGS_NUMBER - global_size - 1] = unhappiness[0 .. BUGS_NUMBER - global_size - 1]
	   							+ unhappiness[global_size .. BUGS_NUMBER - 1]
	   	sum[BUGS_NUMBER - global_size .. global_size - 1] = unhappiness[BUGS_NUMBER - global_size .. global_size - 1]

	   Where each indexed sum, is a work-item | lid, private variable 'sum'.
	*/

	/* Serial sum. */
	serialCount = DIV_CEIL( BUGS_NUMBER, global_size );

	for (iter = 0; iter < serialCount; iter++)
	{
		index = iter * global_size + gid;

		/* If workitem is out of range. */
		if (index < BUGS_NUMBER)
			sum += unhappiness[ index ];
	}

	/* All workitems (including out of range (sum = 0.0), store 'sum' in local memory. */
	partial_sums[ lid ] = sum;

	/* Wait for all workitems to perform previous operations before proceed with local group sums.   */
	barrier( CLK_LOCAL_MEM_FENCE );


	/* Reduce. */
	for (iter = group_size / 2; iter > 0; iter >>= 1)
	{
		if (lid < iter)
			partial_sums[ lid ] += partial_sums[ lid + iter ];

		barrier( CLK_LOCAL_MEM_FENCE );
	}

	/* Store in global memory. */
	if (lid == 0) {
		unhapp_reduced[ get_group_id( 0 ) ] = partial_sums[ 0 ];
	}

	return;
}



__kernel void unhappiness_step2_average( __global float *unhapp_reduced, __local float *partial_sums,
						__global float *unhapp_average )
{
	__private uint iter;


	const uint lid = get_local_id( 0 );
	const uint group_size = get_local_size( 0 );


	partial_sums[ lid ] = select( 0.0f, unhapp_reduced[ lid ], lid < REDUCE_NUM_WORKGROUPS );

	barrier( CLK_LOCAL_MEM_FENCE );

	/* Further reduce. */
	for (iter = group_size / 2; iter > 0; iter >>= 1)
	{
		if (lid < iter)
			partial_sums[ lid ] += partial_sums[ lid + iter ];

		barrier( CLK_LOCAL_MEM_FENCE );
	}

	/* Compute average and store final result in global memory. */
	if (lid == 0) {
		*unhapp_average = partial_sums[ 0 ] / BUGS_NUMBER;
	}

	return;
}
