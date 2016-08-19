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
 * http://www.reedbeta.com/blog/2013/01/12/quick-and-easy-gpu-random-numbers-in-
 *		d3d11/
 * http://http.developer.nvidia.com/GPUGems3/gpugems3_ch37.html
 * https://math.stackexchange.com/questions/337782/pseudo-random-number-
 *		generation-on-the-gpu
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


/*!	Kernel version: 8	!*/



/* Enable doubles datatype in OpenCL kernel. */
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

/* Enable extended atomic operation on both, global and local memory. */
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable



/* Used to drive what shall happen to the agent at each step. */
#define FIND_ANY_FREE		0x00ffffff
#define FIND_MAX_TEMPERATURE	0x00ffff00
#define FIND_MIN_TEMPERATURE	0x00ff00ff

/* Number of neighboring cell that surround the agent's position. */
#define NUM_NEIGHBOURS	8


/*
 * An uint data type carries 8 bits for bug ideal temperature; 8 bits filled
 * with 1's, for bug presence; 8 bits for bug output heat; 8 bits with hex 'dd'
 * for bugs presence.
 * If there is no bug, the uint variable for that bug should be zero.
 * */

 /* Don't change this values. */
#define BUG		0x00ff00dd
#define EMPTY_CELL	0x00000000


/* Macros for 32 bits / unsigned int. */

#define BUG_NEW( uint_reg ) uint_reg = BUG

#define SET_BUG( uint_reg ) uint_reg = (uint_reg & 0xff00ff00) | BUG
#define SET_BUG_IDEAL_TEMPERATURE( uint_reg, value ) uint_reg = (uint_reg & 0x00ffffff) | ((value) << 24)
#define SET_BUG_OUTPUT_HEAT( uint_reg, value ) uint_reg = (uint_reg & 0xffff00ff) | ((value) << 8)

#define HAS_BUG( uint_reg ) ((uint_reg) != EMPTY_CELL)
#define HAS_NO_BUG( uint_reg ) ((uint_reg) == EMPTY_CELL)


#define GET_BUG_IDEAL_TEMPERATURE( uint_reg ) ( ((uint_reg) & 0xff000000) >> 24 )
#define GET_BUG_OUTPUT_HEAT( uint_reg ) ( ((uint_reg) & 0x0000ff00) >> 8 )



/**
 * @brief Performs integer division return the ceiling instead of the floor if
 * it is not an exact division.
 *
 * @param a Integer numerator.
 * @param b Integer denominator.
 * */
#define DIV_CEIL( a, b ) ( ((a) + (b) - 1) / (b) )

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
	return (max - min) * (double) rand_xorShift32( rng_state )
						* ( 1.0 / 4294967296.0 ) + min;
}



/*
 * Return random integer in range [min .. max[.
 * The rng_state is updated.
 * */
inline uint randomInt( uint min, uint max, __global uint *rng_state )
{
	return (max - min) * (double) rand_xorShift32( rng_state )
						* ( 1.0 / 4294967296.0 ) + min;
}



/*
 * It returns either:
 *		1- A free location with best (MAX Temperature | MIN Temperature).
 *		2- Any random free location.
 *		3- Self location.
 * for any agent.
 * World is a vector mapped to a matrix growing from (0,0) toward NorthEast,
 * that is, first cartesian quadrant.
 * */
inline uint best_Free_Neighbour( int todo, __global float *heat_map,
			__global uint *swarm_map, __private uint bug_locus,
			__global uint *rng_state )
{
	/* Agent vector position into the world / 2D position. */
	__private const uint rc = bug_locus / WORLD_WIDTH;    /* Central row. */
	__private const uint cc = bug_locus % WORLD_WIDTH;    /* Central col. */

	/* Neighbouring rows and columns. */

	/* Row at North.   */
	__private const uint rn = (rc + 1) % WORLD_HEIGHT;
	/* Row at South.   */
	__private const uint rs = (rc + WORLD_HEIGHT - 1) % WORLD_HEIGHT;
	/* Column at East. */
	__private const uint ce = (cc + 1) % WORLD_WIDTH;
	/* Column at West. */
	__private const uint cw = (cc + WORLD_WIDTH - 1) % WORLD_WIDTH;

	/* IMPORTANT best.s0 is position, best.s1 is temperature. */
	__private uint2 best;
	/* neighbour[..].s0 is position, neighbour[..].s1 is temperature. */
	__private uint2 neighbour[ NUM_NEIGHBOURS ];
	/* To index neighbour's. Randomly picks the first free neighbouring cell. */
	__private uint NEIGHBOUR_IDX[ NUM_NEIGHBOURS ] =
						{SW, S, SE, W, E, NW, N, NE};


	/* Compute back the vector positions. Used on both, best */
	/* location and random location.                         */
	neighbour[ SW ].s0 = rs * WORLD_WIDTH + cw;  /* SW neighbour position. */
	neighbour[ S  ].s0 = rs * WORLD_WIDTH + cc;  /* S  neighbour position. */
	neighbour[ SE ].s0 = rs * WORLD_WIDTH + ce;  /* SE neighbour position. */
	neighbour[ W  ].s0 = rc * WORLD_WIDTH + cw;  /* W  neighbour position. */
	neighbour[ E  ].s0 = rc * WORLD_WIDTH + ce;  /* E  neighbour position. */
	neighbour[ NW ].s0 = rn * WORLD_WIDTH + cw;  /* NW neighbour position. */
	neighbour[ N  ].s0 = rn * WORLD_WIDTH + cc;  /* N  neighbour position. */
	neighbour[ NE ].s0 = rn * WORLD_WIDTH + ce;  /* NE neighbour position. */

	if (todo != FIND_ANY_FREE)
	{
		/* Fetch temperature of all neighbours.                       */
		/* Store float in uint using OpenCL type reinterpretation.    */
		neighbour[ SW ].s1 = as_uint( heat_map[ neighbour[ SW ].s0 ] );
 		neighbour[ S  ].s1 = as_uint( heat_map[ neighbour[ S  ].s0 ] );
 		neighbour[ SE ].s1 = as_uint( heat_map[ neighbour[ SE ].s0 ] );
 		neighbour[ W  ].s1 = as_uint( heat_map[ neighbour[ W  ].s0 ] );
 		neighbour[ E  ].s1 = as_uint( heat_map[ neighbour[ E  ].s0 ] );
 		neighbour[ NW ].s1 = as_uint( heat_map[ neighbour[ NW ].s0 ] );
 		neighbour[ N  ].s1 = as_uint( heat_map[ neighbour[ N  ].s0 ] );
 		neighbour[ NE ].s1 = as_uint( heat_map[ neighbour[ NE ].s0 ] );

 		/* Actual bug location is the best location, until otherwise. */
		best.s0 = bug_locus;			   /* Bug position.   */
		best.s1 = as_uint( heat_map[ best.s0 ] );  /* Temperature at  */
							   /* bug position. */

		/* Loop unroll.  */
		if (todo == FIND_MAX_TEMPERATURE)
		{
			if (as_float( neighbour[ SW ].s1 )
				> as_float( best.s1 ))  best = neighbour[ SW ];

			if (as_float( neighbour[ S  ].s1 )
				> as_float( best.s1 ))  best = neighbour[ S  ];

			if (as_float( neighbour[ SE ].s1 )
				> as_float( best.s1 ))  best = neighbour[ SE ];

			if (as_float( neighbour[ W  ].s1 )
				> as_float( best.s1 ))  best = neighbour[ W  ];

			if (as_float( neighbour[ E  ].s1 )
				> as_float( best.s1 ))  best = neighbour[ E  ];

			if (as_float( neighbour[ NW ].s1 )
				> as_float( best.s1 ))  best = neighbour[ NW ];

			if (as_float( neighbour[ N  ].s1 )
				> as_float( best.s1 ))  best = neighbour[ N  ];

			if (as_float( neighbour[ NE ].s1 )
				> as_float( best.s1 ))  best = neighbour[ NE ];
		}
		else	/* todo == FIND_MIN_TEMPERATURE */
		{
			if (as_float( neighbour[ SW ].s1 )
				< as_float( best.s1 ))  best = neighbour[ SW ];

			if (as_float( neighbour[ S  ].s1 )
				< as_float( best.s1 ))  best = neighbour[ S  ];

			if (as_float( neighbour[ SE ].s1 )
				< as_float( best.s1 ))  best = neighbour[ SE ];

			if (as_float( neighbour[ W  ].s1 )
				< as_float( best.s1 ))  best = neighbour[ W  ];

			if (as_float( neighbour[ E  ].s1 )
				< as_float( best.s1 ))  best = neighbour[ E  ];

			if (as_float( neighbour[ NW ].s1 )
				< as_float( best.s1 ))  best = neighbour[ NW ];

			if (as_float( neighbour[ N  ].s1 )
				< as_float( best.s1 ))  best = neighbour[ N  ];

			if (as_float( neighbour[ NE ].s1 )
				< as_float( best.s1 ))  best = neighbour[ NE ];
		}

		/* Return, if bug is already in the best local or if new best local is bug free, */
		if ((best.s0 == bug_locus)
		    || HAS_NO_BUG( swarm_map[ best.s0 ] )) return best.s0;

	}	/* end_if (todo != GOTO_ANY_FREE) */


	/*
	 * Here: there is a random moving chance, or best neighbour is not free.
	 * Try to find any available free place.
	 * */

	/*
	 * Fisher-Yates shuffle algorithm to shuffle an index vector so we can
	 * pick a random free neighbour by checking each by index until find a
	 * first free.
	 * */
	for (uint i = 0; i < NUM_NEIGHBOURS; i++)
	{
		uint rnd = randomInt( i, NUM_NEIGHBOURS, rng_state );

		if (rnd == i) continue;

		char tmp = NEIGHBOUR_IDX[ i ];
		NEIGHBOUR_IDX[ i ] = NEIGHBOUR_IDX[ rnd ];
		NEIGHBOUR_IDX[ rnd ] = tmp;
	}

	/* Loop unroll. */

	/* Find a first free neighbour. Index over the 8 neighbours. */
	best.s0 = neighbour[ NEIGHBOUR_IDX[ 0 ] ].s0;
	if (HAS_NO_BUG( swarm_map[ best.s0 ] )) return best.s0;

	best.s0 = neighbour[ NEIGHBOUR_IDX[ 1 ] ].s0;
	if (HAS_NO_BUG( swarm_map[ best.s0 ] )) return best.s0;

	best.s0 = neighbour[ NEIGHBOUR_IDX[ 2 ] ].s0;
	if (HAS_NO_BUG( swarm_map[ best.s0 ] )) return best.s0;

	best.s0 = neighbour[ NEIGHBOUR_IDX[ 3 ] ].s0;
	if (HAS_NO_BUG( swarm_map[ best.s0 ] )) return best.s0;

	best.s0 = neighbour[ NEIGHBOUR_IDX[ 4 ] ].s0;
	if (HAS_NO_BUG( swarm_map[ best.s0 ] )) return best.s0;

	best.s0 = neighbour[ NEIGHBOUR_IDX[ 5 ] ].s0;
	if (HAS_NO_BUG( swarm_map[ best.s0 ] )) return best.s0;

	best.s0 = neighbour[ NEIGHBOUR_IDX[ 6 ] ].s0;
	if (HAS_NO_BUG( swarm_map[ best.s0 ] )) return best.s0;

	best.s0 = neighbour[ NEIGHBOUR_IDX[ 7 ] ].s0;
	if (HAS_NO_BUG( swarm_map[ best.s0 ] )) return best.s0;

	return bug_locus;	/* There is no free neighbour. */
}




/*
 * ************* KERNELS ******************
 * */


/*
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



__kernel void init_maps( __global uint *swarm_map, __global float *heat_map,
				__global float *heat_buffer )
{
	const uint gid = get_global_id( 0 );

	if (gid >= WORLD_SIZE) return;

	swarm_map[ gid ] = EMPTY_CELL;	/* Clean all bugs from the map. */
	heat_map[ gid ] = 0.0;	     	/* Reset all temperatures from map. */
	heat_buffer[ gid ] = 0.0;    	/* Reset all temperatures from buffer. */

	return;
}



/*
 * Fill the world (swarm_map) with bugs, store their position (swarm).
 * Reset the swarm intention vector (stores the intended bug movement.
 * Initiate the unhappiness for each bug.
 * */
__kernel void init_swarm( __global uint *swarm_bugPosition,
				__global uint *swarm_map,
				__global float *unhappiness,
				__global uint *rng_state )
{
	__private uint bug_locus;
	__private uint bug_ideal_temperature;	/* [0..200] */
	__private uint bug_output_heat;		/* [0..100] */
	__private uint bug_new;
	__private uint bug_old;


	const uint bug_id = get_global_id( 0 );

	if (bug_id >= BUGS_NUMBER) return;


	bug_ideal_temperature = (ushort) randomInt( BUGS_TEMPERATURE_MIN_IDEAL,
			BUGS_TEMPERATURE_MAX_IDEAL, &rng_state[ bug_id ] );

	bug_output_heat = (ushort) randomInt( BUGS_HEAT_MIN_OUTPUT,
			BUGS_HEAT_MAX_OUTPUT, &rng_state[ bug_id ] );

	/* Create a bug. */
	BUG_NEW( bug_new );
	SET_BUG_IDEAL_TEMPERATURE( bug_new, bug_ideal_temperature );
	SET_BUG_OUTPUT_HEAT( bug_new, bug_output_heat );

	/* Try, until succeed, to leave a bug in a empty space. */
	do {
		bug_locus = randomInt( 0, WORLD_SIZE, &rng_state[ bug_id ] );

		bug_old = atomic_cmpxchg( &swarm_map[ bug_locus ],
							EMPTY_CELL, bug_new );

	} while ( HAS_BUG( bug_old ) );

	barrier( CLK_GLOBAL_MEM_FENCE );    /* All still active workitems     */
					    /* must arrive here before go on. */

	/* Store bug position in the swarm. */
	swarm_bugPosition[ bug_id ] = bug_locus;

	/*
	 * Compute initial unhappiness as abs(ideal_temperature - temperature).
	 * Since world temperature is initially zero, it turns out that initial
	 * unhappiness = ideal_temperature.
	 * */

	/* REPLACED: unhappiness[ bug_id ] = (float) bug_ideal_temperature; */
	/* With OpenCL type convertion.                                     */
	unhappiness[ bug_id ] = convert_float( bug_ideal_temperature );

	return;
}



/*
 * Perform a bug movement in the world.
 *
 * Netlogo completely separates the report of the "best" / "random" location,
 * from the actual bug movement. Only when perform "bug-move", the availability
 * (status) of the reported location is checked, so an alternate free location,
 * if exists, is computed if necessary.
 *
 * This GPU version of the "bug-move" a.k.a "bug_step", uses the auxiliary
 * function find_Best_Neighbour(..), that does both at the same time, it checks
 * for the "winner", a location with max/min temperature that is also free of
 * bugs, or "any random" free location.
 *
 * The function best_Free_Neighbour(..) does its best to mimic netlogo behavior,
 * by packing the location report and the availability check into the same
 * function. When executed in serie, the returned result is guaranted since
 * only one bug is processed at a time.
 *
 * However the paralell execution in GPU means that a previously free location
 * may become unavailable, so it is required a second call to the same function,
 * best_Free_Neighbour(..), this time looking just for any free neighbour (as
 * if a random moving chance is taking place).
 * A second call to the the function may be necessary to exactly mimic Netlogo's
 * behavior, that is, search for a new free location, after the previous one
 * become occupied by a bug, setted by a concurrent thread.
 * So, in parallel execution, a 2-step operation may be required.
 * */
__kernel void bug_step(	__global uint *swarm_bugPosition,
			__global uint *swarm_map,
			__global float *heat_map,
			__global float *unhappiness,
			__global uint *rng_state )
{
	__private uint bug_locus;
	__private uint bug_new_locus;
	__private uint bug_ideal_temperature;	/* [0..200] */
	__private uint bug_output_heat;		/* [0..100] */
	__private float bug_unhappiness;
	__private float locus_temperature;
	__private uint bug;
	__private uint new_locus;

	__private int todo;


	const uint bug_id = get_global_id( 0 );

	if (bug_id >= BUGS_NUMBER) return;


	bug_locus = swarm_bugPosition[ bug_id ];

	bug = swarm_map[ bug_locus ];
	locus_temperature = heat_map[ bug_locus ];

	bug_ideal_temperature = GET_BUG_IDEAL_TEMPERATURE( bug );
	bug_output_heat = GET_BUG_OUTPUT_HEAT( bug );

	/* IDEA: use CL typecast ? */
	bug_unhappiness =
		fabs( (float) bug_ideal_temperature - locus_temperature );

	/* Update bug's unhappiness vector in global memory. */
	unhappiness[ bug_id ] = bug_unhappiness;

	/*
	 * Usually compare equality of floats is absurd. Netlogo wrapps the
	 * code with: if (unhappiness > 0) { do stuff... }
	 * I decided to unwrap, terminating as soon as possible using
	 * if (bug_unhappiness == 0.0f) {...}  and continue the remaining code
	 * as a fall out case for (bug_unhappiness > 0.0f). After all, this
	 * is semantically equivalent to netlogo version.
 	 * */
	if (bug_unhappiness == 0.0f)
	{
		/* Bug hasn't move, we don't need to update swarm_bugPosition. */
		heat_map[ bug_locus ] += bug_output_heat;
		return;
	}

	/* Arriving here, means (bug_unhappiness > 0.0f) */


	/* INFO: Mem fence here? */

	/*
		Find the best place for the bug to go. Tricky stuff.

		In order to implement netlogo approach, WITHOUT branching in
		OpenCL kernel, that is (in netlogo order), to compute:
		1) random-move-chance,
		2) best-patch for (temp <  ideal_temp) (when bug is in COLD),
		3) best-patch for (temp >= ideal_temp) (when bug is in HOT),
		the OpenCl kernel select(...) function must be used twice.

		A variable is used to hold what to do, (1), (2) or (3). However
		the order must be reversed since (1), when happen, takes
		precedence over	(2) or (3), whatever (2) XOR (3) is true or not.

		select(a, b, c) implements: 	(c) ? b : a
	*/

	todo = select( FIND_MIN_TEMPERATURE, FIND_MAX_TEMPERATURE,
				locus_temperature < bug_ideal_temperature );

	todo = select( todo, FIND_ANY_FREE,
		randomFloat( 0, 100, &rng_state[ bug_id ] ) < BUGS_RANDOM_MOVE_CHANCE );


	bug_new_locus = best_Free_Neighbour( todo, heat_map, swarm_map,
					bug_locus, &rng_state[ bug_id ] );


	/* If bug's current location is already the best one... */
	if (bug_new_locus == bug_locus)
	{
		/* Bug hasn't move, we don't need to update swarm_bugPosition. */
		heat_map[ bug_locus ] += bug_output_heat;
		return;
	}


	/* Otherwise, try to store the bug in his new 'best' location and return. */
	new_locus = atomic_cmpxchg( &swarm_map[ bug_new_locus ], EMPTY_CELL, bug );

	if (HAS_NO_BUG( new_locus ))
	{
		/* SUCCESS! Reset old bug location. */
		/* Should be atomic in case another work-item try to read. */
		atomic_xchg( &swarm_map[ bug_locus ], EMPTY_CELL );
		heat_map[ bug_new_locus ] += bug_output_heat;

		/* Update bug location in the swarm. */
		swarm_bugPosition[ bug_id ] = bug_new_locus;

		return;
	}


	/* Here, the best place become unavailable! Repeat the last actions. */

	/*
	   Call best_Free_Neighbour(..) function looking just for ANY FREE
	   neighbour or drop bug_step if it is impossible to find any.
	 * */

	todo = FIND_ANY_FREE;
	bug_new_locus = best_Free_Neighbour( todo, heat_map, swarm_map,
					bug_locus, &rng_state[ bug_id ] );

	/* Here, there is NO free neighbour. Bug must stay in same location */
	if (bug_new_locus == bug_locus)
	{
		/* Bug hasn't move we don't need to update swarm_bugPosition. */
		heat_map[ bug_locus ] += bug_output_heat;
		return;
	}


	/* Otherwise, try to store the bug in his new 'random' */
	/* location and return.                                */
	new_locus = atomic_cmpxchg( &swarm_map[ bug_new_locus ], EMPTY_CELL, bug );

	if (HAS_NO_BUG( new_locus ))
	{
		/* SUCCESS! Reset old bug location. */
		/* Must be atomic in case another workitem is trying to read. */
		atomic_xchg( &swarm_map[ bug_locus ], EMPTY_CELL );
		heat_map[ bug_new_locus ] += bug_output_heat;

		/* Update bug location in the swarm. */
		swarm_bugPosition[ bug_id ] = bug_new_locus;

		return;
	}

	/* Bug failled to move and stay at current location. */
	heat_map[ bug_locus ] += bug_output_heat;

	/* Bug hasn't move we don't need to update swarm_bugPosition. */

	return;
}



__kernel void comp_world_heat( __global float *heat_map,
					__global float *heat_buffer )
{
	float heat = 0;
	uint pos;
	uint ln, ls, ce, cw;	/* Line at North/South, Column at East/West. */

	const uint cc = get_global_id( 0 );	/* Column at Center. */
	const uint lc = get_global_id( 1 );	/* Line at Center.   */

	if (lc >= WORLD_HEIGHT || cc >= WORLD_WIDTH) return;


	/* Compute required coodinates. */
	ln = (lc + 1) % WORLD_HEIGHT;			/* Line at North.   */
	ls = (lc + WORLD_HEIGHT - 1) % WORLD_HEIGHT;	/* Line at South.   */
	ce = (cc + 1) % WORLD_WIDTH;			/* Column at East.  */
	cw = (cc + WORLD_WIDTH - 1) % WORLD_WIDTH;	/* Columns at West. */


	/* Compute Diffusion */

	/* NW */
	pos = ln * WORLD_WIDTH + cw;
	heat += heat_map[pos];

	/* N  */
	pos = ln * WORLD_WIDTH + cc;
	heat += heat_map[pos];

	/* NE */
	pos = ln * WORLD_WIDTH + ce;
	heat += heat_map[pos];

	/* W  */
	pos = lc * WORLD_WIDTH + cw;
	heat += heat_map[pos];

	/* E  */
	pos = lc * WORLD_WIDTH + ce;
	heat += heat_map[pos];

	/* SW */
	pos = ls * WORLD_WIDTH + cw;
	heat += heat_map[pos];

	/* S  */
	pos = ls * WORLD_WIDTH + cc;
	heat += heat_map[pos];

	/* SE */
	pos = ls * WORLD_WIDTH + ce;
	heat += heat_map[pos];

	/* Get the 8th part of diffusion percentage of all neighbour cells. */
	heat = heat * WORLD_DIFFUSION_RATE / 8;

	/* Add cell's remaining heat. */
	pos = lc * WORLD_WIDTH + cc;
	heat += heat_map[pos] * (1 - WORLD_DIFFUSION_RATE);


	barrier( CLK_GLOBAL_MEM_FENCE );	/* TODO: Is it needed? */


	/* Compute Evaporation */

	heat = heat * (1 - WORLD_EVAPORATION_RATE);


	/* Double buffer it. */
	heat_buffer[pos] = heat;

	barrier( CLK_GLOBAL_MEM_FENCE );	/* TODO: Is it needed? */

	return;
}




/*
 *
 */
__kernel void unhappiness_step1_reduce( __global float *unhappiness,
					__local float *partial_sums,
					__global float *unhapp_reduced )
{
	const uint gid = get_global_id( 0 );
	const uint lid = get_local_id( 0 );
	const uint global_size = get_global_size( 0 );
	const uint group_size = get_local_size( 0 );

	__private uint serialCount, index, iter;
	__private float sum = 0.0f;

	/*
	   The size of vector unhappiness is the number of bugs (BUGS_NUMBER).
	   So we must take care of both cases, when BUGS_NUMBER > global_size,
	   and when global_size > BUGS_NUMBER.
	   This is what happen in the next loop ('sum' is private to each
	   workitem)...

	   When: (BUGS_NUMBER < global_size) -> serialCount = 1
	   	sum[0 .. BUGS_NUMBER - 1] = unhappiness[0 .. BUGS_NUMBER - 1]
	   	sum[BUGS_NUMBER .. global_size - 1] = 0

	   When: (BUGS_NUMBER == global_size) -> serialCount = 1
	   	sum[0 .. global_size - 1] = unhappiness[0 .. global_size - 1]

	   When: (global_size < BUGS_NUMBER < 2*global_size) -> serialCount = 2
	   	sum[0 .. BUGS_NUMBER - global_size - 1] =
	   		unhappiness[0 .. BUGS_NUMBER - global_size - 1]
	   		+ unhappiness[global_size .. BUGS_NUMBER - 1]
	   	sum[BUGS_NUMBER - global_size .. global_size - 1] =
	   		unhappiness[BUGS_NUMBER - global_size .. global_size - 1]

	   Where each indexed sum, is a work-item | lid, private variable 'sum'.
	*/

	/* Serial sum. */
	serialCount = DIV_CEIL( BUGS_NUMBER, global_size );

	for (iter = 0; iter < serialCount; iter++)
	{
		index = iter * global_size + gid;

		/* For workitems out of range, sum = 0.0 */
		if (index < BUGS_NUMBER)
			sum += unhappiness[ index ];
	}

	/* All workitems (including out of range (sum = 0.0), store */
	/* 'sum' in local memory.                                   */
	partial_sums[ lid ] = sum;

	/* Wait for all workitems to perform previous operations.   */
	barrier( CLK_LOCAL_MEM_FENCE );


	/* Reduce. */
	for (iter = group_size / 2; iter > 0; iter >>= 1)
	{
		if (lid < iter)	{
			partial_sums[ lid ] += partial_sums[ lid + iter ];
		}
		barrier( CLK_LOCAL_MEM_FENCE );
	}

	/* Store in global memory. */
	if (lid == 0) {
		unhapp_reduced[ get_group_id( 0 ) ] = partial_sums[ 0 ];
	}

	return;
}



__kernel void unhappiness_step2_average( __global float *unhapp_reduced,
						__local float *partial_sums,
						__global float *unhapp_average )
{
	__private uint iter;


	const uint lid = get_local_id( 0 );
	const uint group_size = get_local_size( 0 );


	/* Load partial sum in local memory */
//	if (lid < REDOX_NUM_WORKGROUPS)
//		partial_sums[ lid ] = unhappiness[ lid ];
//	else
//		partial_sums[ lid ] = 0;

	partial_sums[ lid ] =
		select( 0.0f, unhapp_reduced[ lid ], lid < REDUCE_NUM_WORKGROUPS );

	barrier( CLK_LOCAL_MEM_FENCE );

	/* Further reduce. */
	for (iter = group_size / 2; iter > 0; iter >>= 1)
	{
		if (lid < iter)	{
			partial_sums[ lid ] += partial_sums[ lid + iter ];
		}
		barrier( CLK_LOCAL_MEM_FENCE );
	}

	/* Compute average and put final result in global memory. */
	if (lid == 0) {
		*unhapp_average = partial_sums[ 0 ] / BUGS_NUMBER;
	}

	return;
}
