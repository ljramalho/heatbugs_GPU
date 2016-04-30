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
 *	from:
 *		http://www.reedbeta.com/blog/2013/01/12/quick-and-easy-gpu-random-numbers-in-d3d11/
 *		http://http.developer.nvidia.com/GPUGems3/gpugems3_ch37.html
 *		https://math.stackexchange.com/questions/337782/pseudo-random-number-generation-on-the-gpu
 * */

/**
 * The following constants are passed to kernel at compile time.
 *		INIT_SEED
 *		BUGS_TEMPERATURE_MIN_IDEAL
 *		BUGS_TEMPERATURE_MAX_IDEAL
 *		BUGS_HEAT_MIN_OUTPUT
 *		BUGS_HEAT_MAX_OUTPUT
 *		BUGS_RANDOM_MOVE_CHANCE
 *		WORLD_DIFFUSION_RATE
 *		WORLD_EVAPORATION_RATE
 *		BUGS_NUMBER
 *		WORLD_WIDTH
 *		WORLD_HEIGHT
 *		WORLD_SIZE
 * */





/* Enable doubles datatype in OpenCL kernel. */
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

/* Enable extended atomic operation on both, global and local memory. */
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable




/* Used to drive what shall happen to the agent at each step. */
#define GOTO_ANY_FREE			0x00ffffff
#define GOTO_MAX_TEMPERATURE	0x00ffff00
#define GOTO_MIN_TEMPERATURE	0x00ff00ff

/* Number of neighboring cell that surround the agent's position. */
#define NUM_NEIGHBOURS	8


/*
 * An uint data type carries 8 bits for bug ideal temperature; 8 bits filled
 * with 1's, for bug presence; 8 bits for bug output heat; 8 bits with hex 'dd'
 * for bugs presence.
 * If there is no bug, the uint variable for that bug should be zero.
 * */

 /* Don't change this values. */
#define BUG			0x00ff00dd
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
 * It returns either:
 *		1- A free location with best (MAX Temperature | MIN Temperature).
 *		2- Any random free location.
 *		3- Self location.
 * for any agent.
 * World is a vector mapped to a matrix growing from (0,0) toward NorthEast,
 * that is, first cartesian quadrant.
 * */
inline uint best_Free_Neighbour( int todo, __global float *heat_map, __global uint *swarm_map, __private uint bug_locus, __global uint *rng_state )
{
	/* Agent vector position into the world / 2D position. */
	__private const uint rc = bug_locus / WORLD_WIDTH;					/* Row at Center.   */
	__private const uint cc = bug_locus % WORLD_WIDTH;					/*Column at center. */

	/* Neighbouring rows and columns. */
	__private const uint rn = (rc + 1) % WORLD_HEIGHT;					/* Row at North.    */
	__private const uint rs = (rc + WORLD_HEIGHT - 1) % WORLD_HEIGHT;	/* Row at South.    */
	__private const uint ce = (cc + 1) % WORLD_WIDTH;					/* Column at East.  */
	__private const uint cw = (cc + WORLD_WIDTH - 1) % WORLD_WIDTH;		/* Column at West.  */

	/* best.s0 is position, best.s1 is temperature. */
	__private uint2 best;
	/* neighbour[..].s0 is position, neighbour[..].s1 is temperature. */
	__private uint2 neighbour[ NUM_NEIGHBOURS ];
	/* To index neighbour's. Randomly picks the first free neighbouring cell. */
	__private uint NEIGHBOUR_IDX[ NUM_NEIGHBOURS ] = {SW, S, SE, W, E, NW, N, NE};



	/* Back to vector positions. Used on both, best location and random location. */
	neighbour[ SW ].s0 = rs * WORLD_WIDTH + cw;		/* SW neighbour position. */
	neighbour[ S  ].s0 = rs * WORLD_WIDTH + cc;		/* S  neighbour Position. */
	neighbour[ SE ].s0 = rs * WORLD_WIDTH + ce;		/* SE neighbour position. */
	neighbour[ W  ].s0 = rc * WORLD_WIDTH + cw;		/* W  neighbour position. */
	neighbour[ E  ].s0 = rc * WORLD_WIDTH + ce;		/* E  neighbour position. */
	neighbour[ NW ].s0 = rn * WORLD_WIDTH + cw;		/* NW neighbour position. */
	neighbour[ N  ].s0 = rn * WORLD_WIDTH + cc;		/* N  neighbour position. */
	neighbour[ NE ].s0 = rn * WORLD_WIDTH + ce;		/* NE neighbour position. */


	if (todo != GOTO_ANY_FREE)
	{
		/* Fetch temperature of all neighbours. */
		neighbour[ SW ].s1 = as_uint( heat_map[ neighbour[ SW ].s0 ] );
 		neighbour[ S  ].s1 = as_uint( heat_map[ neighbour[ S  ].s0 ] );
 		neighbour[ SE ].s1 = as_uint( heat_map[ neighbour[ SE ].s0 ] );
 		neighbour[ W  ].s1 = as_uint( heat_map[ neighbour[ W  ].s0 ] );
 		neighbour[ E  ].s1 = as_uint( heat_map[ neighbour[ E  ].s0 ] );
 		neighbour[ NW ].s1 = as_uint( heat_map[ neighbour[ NW ].s0 ] );
 		neighbour[ N  ].s1 = as_uint( heat_map[ neighbour[ N  ].s0 ] );
 		neighbour[ NE ].s1 = as_uint( heat_map[ neighbour[ NE ].s0 ] );

 		/* Actual bug location is also the best location until otherwise. */
		best.s0 = bug_locus;							/* Bug position.    */
		best.s1 = as_uint( heat_map[ best.s0 ] );		/* Temperature at bug position. */

		/* Loop unroll.  */
		if (todo == GOTO_MAX_TEMPERATURE)
		{
			if (as_float( neighbour[ SW ].s1 ) > as_float( best.s1 ))	best = neighbour[ SW ];
			if (as_float( neighbour[ S  ].s1 ) > as_float( best.s1 ))	best = neighbour[ S  ];
			if (as_float( neighbour[ SE ].s1 ) > as_float( best.s1 ))	best = neighbour[ SE ];
			if (as_float( neighbour[ W  ].s1 ) > as_float( best.s1 ))	best = neighbour[ W ];
			if (as_float( neighbour[ E  ].s1 ) > as_float( best.s1 ))	best = neighbour[ E ];
			if (as_float( neighbour[ NW ].s1 ) > as_float( best.s1 ))	best = neighbour[ NW ];
			if (as_float( neighbour[ N  ].s1 ) > as_float( best.s1 ))	best = neighbour[ N  ];
			if (as_float( neighbour[ NE ].s1 ) > as_float( best.s1 ))	best = neighbour[ NE ];
		}
		else	/* todo == GOTO_MIN_TEMPERATURE */
		{
			if (as_float( neighbour[ SW ].s1 ) < as_float( best.s1 ))	best = neighbour[ SW ];
			if (as_float( neighbour[ S  ].s1 ) < as_float( best.s1 ))	best = neighbour[ S  ];
			if (as_float( neighbour[ SE ].s1 ) < as_float( best.s1 ))	best = neighbour[ SE ];
			if (as_float( neighbour[ W  ].s1 ) < as_float( best.s1 ))	best = neighbour[ W  ];
			if (as_float( neighbour[ E  ].s1 ) < as_float( best.s1 ))	best = neighbour[ E  ];
			if (as_float( neighbour[ NW ].s1 ) < as_float( best.s1 ))	best = neighbour[ NW ];
			if (as_float( neighbour[ N  ].s1 ) < as_float( best.s1 ))	best = neighbour[ N  ];
			if (as_float( neighbour[ NE ].s1 ) < as_float( best.s1 ))	best = neighbour[ NE ];
		}

		/* Return, if bug is already in the best local or if new best local is bug free, */
		if ((best.s0 == bug_locus) || HAS_NO_BUG( swarm_map[ best.s0 ] )) return best.s0;
	}	/* end_if (todo != GOTO_ANY_FREE) */


	/*
	 * Here: there is a random moving chance, or the best neighbour is not free.
	 * Try to find any avaiable free place.
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

		char tmp = N_IDX[ i ];
		NEIGHBOUR_IDX[ i ] = NEIGHBOUR_IDX[ rnd ];
		NEIGHBOUR_IDX[ rnd ] = tmp;
	}

	/* Loop unroll. */

	/* Find a first free neighbour. */
	best.s0 = neighbour[ NEIGHBOUR_IDX[0] ].s0;
	if (HAS_NO_BUG( swarm_map[ best.s0 ] )) return best.s0;

	best.s0 = neighbour[ NEIGHBOUR_IDX[1] ].s0;
	if (HAS_NO_BUG( swarm_map[ best.s0 ] )) return best.s0;

	best.s0 = neighbour[ NEIGHBOUR_IDX[2] ].s0;
	if (HAS_NO_BUG( swarm_map[ best.s0 ] )) return best.s0;

	best.s0 = neighbour[ NEIGHBOUR_IDX[3] ].s0;
	if (HAS_NO_BUG( swarm_map[ best.s0 ] )) return best.s0;

	best.s0 = neighbour[ NEIGHBOUR_IDX[4] ].s0;
	if (HAS_NO_BUG( swarm_map[ best.s0 ] )) return best.s0;

	best.s0 = neighbour[ NEIGHBOUR_IDX[5] ].s0;
	if (HAS_NO_BUG( swarm_map[ best.s0 ] )) return best.s0;

	best.s0 = neighbour[ NEIGHBOUR_IDX[6] ].s0;
	if (HAS_NO_BUG( swarm_map[ best.s0 ] )) return best.s0;

	best.s0 = neighbour[ NEIGHBOUR_IDX[7] ].s0;
	if (HAS_NO_BUG( swarm_map[ best.s0 ] )) return best.s0;

	return bug_locus;
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

	seed = gid ^ INIT_SEED;					/* Add diversity at each run. */

	/* Use Wang Hash to decorrelate numbers. */
	seed = (seed ^ 61) ^ (seed >> 16);
	seed += (seed << 3);
	seed = seed ^ (seed >> 4);
	seed *= 0x27d4eb2d;
	seed = seed ^ (seed >> 15);

	/* Store result in global space. */
	rng_state[gid] = seed;

	return;
}



__kernel void init_maps( __global uint *swarm_map, __global float *heat_map, __global float *heat_buffer )
{
	const uint gid = get_global_id( 0 );

	if (gid >= WORLD_SIZE) return;

	swarm_map[ gid ] = 0;
	heat_map[ gid ] = 0.0;
	heat_buffer[ gid ] = 0.0;

	return;
}



/*
 * Fill the world (swarm_map) with bugs, store their position (swarm).
 * Reset the swarm intention vector (stores the intended bug movement.
 * Initiate the unhappiness for each bug.
 * */
__kernel void init_swarm( __global uint *swarm_bugPosition, __global uint *swarm_bugGoto, __global uint *swarm_map, __global float *unhappiness, __global uint *rng_state )
{
	__private uint bug_locus;
	__private uint bug_ideal_temperature;	/* [0..200] */
	__private uint bug_output_heat;			/* [0..100] */
	__private uint bug_new;
	__private uint bug_old;


	const uint bug_idx = get_global_id( 0 );

	if (bug_idx >= BUGS_NUMBER) return;


	bug_ideal_temperature = (ushort) randomInt( BUGS_TEMPERATURE_MIN_IDEAL, BUGS_TEMPERATURE_MAX_IDEAL, &rng_state[ bug_idx ] );
	bug_output_heat = (ushort) randomInt( BUGS_HEAT_MIN_OUTPUT, BUGS_HEAT_MAX_OUTPUT, &rng_state[ bug_idx ] );

	/* Create a bug. */
	BUG_NEW( bug_new );
	SET_BUG_IDEAL_TEMPERATURE( bug_new, bug_ideal_temperature );
	SET_BUG_OUTPUT_HEAT( bug_new, bug_output_heat );

	/* Try, until succeed, to leave a bug in a empty space. */
	do {
		bug_locus = randomInt( 0, WORLD_SIZE, &rng_state[ bug_idx ] );
		bug_old = atomic_cmpxchg( &swarm_map[ bug_locus ], EMPTY_CELL, bug_new );
	} while ( HAS_BUG( bug_old ) );

	barrier( CLK_GLOBAL_MEM_FENCE );	/* All workitems should arrive here before go on. */

	/* Store bug position in the swarm. */
	swarm_bugPosition[ bug_idx ] = bug_locus;
	/* Reset... */
	swarm_bugGoto[ bug_idx ] = 0;

	/*
	 * Compute initial unhappiness as abs(ideal_temperature - temperature).
	 * Since world temperature is initially zero, it turns out that initial
	 * unhappiness = ideal_temperature.
	 * */
	unhappiness[ bug_idx ] = (float) bug_ideal_temperature;

	return;
}



/*
 * Perform a bug movement in the world.
 * */
__kernel void bug_step( __global uint *swarm_bugPosition, __global uint *swarm_bugGoto, __global uint *swarm_map,
						__global float *heat_map, /*__global float *heat_buffer, */
						__global float *unhappiness, __global uint *rng_state )
{
	__private uint bug_locus;
	__private uint bug_new_locus;
	__private uint bug_ideal_temperature;	/* [0..200] */
	__private uint bug_output_heat;			/* [0..100] */
	__private float bug_unhappiness;
	__private float locus_temperature;
	__private uint bug;
	__private uint bug_old;

	__private int todo;


	const uint bug_idx = get_global_id( 0 );

	if (bug_idx >= BUGS_NUMBER) return;


	bug_locus = swarm_bugPosition[ bug_idx ];

	bug = swarm_map[ bug_locus ];
	locus_temperature = heat_map[ bug_locus ];

	bug_ideal_temperature = GET_BUG_IDEAL_TEMPERATURE( bug );
	bug_output_heat = GET_BUG_OUTPUT_HEAT( bug );

	bug_unhappiness = fabs( (float) bug_ideal_temperature - locus_temperature );
	unhappiness[ bug_idx ] = bug_unhappiness;

	/* INFO: Mem fence here? */

	/*
	 * Find the best place for the bug to go. Tricky stuff!
	 *
	 * In order to implement netlogo approach, WITHOUT branching in
	 * OpenCL kernel, that is (in netlogo order) to compute:
	 * 1) random-move-chance,
	 * 2) best-patch for (temp < ideal-temp) (when bug is in COLD),
	 * 3) best patch for (temp >= ideal-temp) (when bug is in HOT),
	 * the OpenCl kernel select(...) function must be used twice.
	 *
	 * A variable is used to hold what to do, (1), (2) or (3). However the
	 * order must be reversed since (1), when happen, takes precedence over
	 * (2) or (3), whatever (2) XOR (3) is true or not.
	 * */

	if (bug_unhappiness > 0.0)
	{
		todo = select( GOTO_MIN_TEMPERATURE, GOTO_MAX_TEMPERATURE, locus_temperature < bug_ideal_temperature );
		todo = select( todo, GOTO_ANY_FREE, randomFloat( 0, 100, &rng_state[ bug_idx ] ) < BUGS_RANDOM_MOVE_CHANCE );

		bug_new_locus = best_Free_Neighbour( todo, heat_map, swarm_map, bug_locus );

		/* Store bug the bug to the new location. */
		if (bug_new_locus != bug_locus)
		{
			bug_old = atomic_cmpxchg( &swarm_map[ bug_locus ], EMPTY_CELL, bug );

			swarm_map[ bug_locus ] = 0;
		}
	}

	heat_map[ bug_locus ] += bug_output_heat;

	return;
}



__kernel void comp_world_heat( __global float *heat_map, __global float *heat_buffer )
{
	float heat = 0;
	uint pos;
	uint ln, ls, ce, cw;	/* Line at North/South, Column at East/West. */

	const uint cc = get_global_id( 0 );				/* Column at Center. */
	const uint lc = get_global_id( 1 );				/* Line at Center.   */

	if (lc >= WORLD_HEIGHT || cc >= WORLD_WIDTH) return;


	/* Compute required coodinates. */
	ln = (lc + 1) % WORLD_HEIGHT;					/* Line at North.   */
	ls = (lc + WORLD_HEIGHT - 1) % WORLD_HEIGHT;	/* Line at South.   */
	ce = (cc + 1) % WORLD_WIDTH;					/* Column at East.  */
	cw = (cc + WORLD_WIDTH - 1) % WORLD_WIDTH;		/* Columns at West. */


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
__kernel void mean_unhappiness()
{
}


__kernel void test( __global int *vSeeds )
{
	const int gid = get_global_id( 0 );

	if (gid < BUGS_NUM) {
		vSeeds[gid] += 2;
	}
}
*/
