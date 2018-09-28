// call as:
//
// ./GFreduction.out G.bin J.bin M N
//
// (G.bin = filename where to store G in binary, same as J.bin)

// exits:
//   -1 ....... Cannot open file.
//   -2 ....... Cannot allocate memory.
//   -3 ....... Zero column in G
//   -4 ....... Algorithm broke down.
//----------------------------------------------------------------------------------------

// TO DO: find a way to compute g*Jg more efficiently

#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <float.h>
#include "omp.h"

#define MKL_Complex16 complex

#include <mkl.h>
#include <mkl_types.h>

#define EPSILON DBL_EPSILON
#define DIGITS DBL_DIG
#define eps 0.2
#define D 64
#define refresh 30
#define COND 2.0


void printMatrix(double complex *G, int M, int N){

	int i, j;
	for( i = 0; i < M; ++i ){
		for( j = 0; j < N; ++j ){
			printf("%8.2g + i%8.2g ", creal(G[i+M*j]), cimag(G[i+M*j]));
		}
		printf("\n");
	}
	printf("\n");
}

void printJ(double *J, int M){

	int i;
	for( i = 0; i < M; ++i ) printf("%3d  ", (int)J[i]);
	printf("\n");
}



//----------------------------------------------------------------------------------------

int main(int argc, char* argv[]){

	double ALPHA = (1.0 + csqrt(17.0))/8.0; //Bunch-Parlett alpha
	omp_set_nested(1);
	omp_set_dynamic(0);
	mkl_set_dynamic(0);
	omp_set_max_active_levels(2);

	// read variables from command line
	int M = atoi(argv[3]);
	int N = atoi(argv[4]);

	FILE *readG = fopen(argv[1], "rb");
	FILE *readJ = fopen(argv[2], "rb");


	printf("\n\n--------------------------------- ALGORITHM ------------------------------------\n");
	//printf("\nReading data...\n");

	// allocate memory
	double complex *G = (double complex*) mkl_malloc(M*N*sizeof(double complex), 64);
	double *J = (double*) mkl_malloc(M*sizeof(double), 64);
	long int *Prow = (long int*) mkl_malloc(M*sizeof(long int), 64);	// for row permutation
	long int *Pcol = (long int*) mkl_malloc(N*sizeof(long int), 64);	// for column permutation
	double complex *f = (double complex*) mkl_malloc(M*sizeof(double complex), 64);	// vector f
	double complex *T = (double complex*) mkl_malloc(2*M*sizeof(double complex), 64);	// temporary matrix
	double complex *norm = (double complex*) mkl_malloc(N*sizeof(double complex), 64);	// for quadrates of J-norms of columns
	double complex *K = (double complex*) mkl_malloc(2*M*sizeof(double complex), 64);	// temporary matrix
	double complex *C = (double complex*) mkl_malloc(4*sizeof(double complex), 64);	// temporary matrix
	double complex *E = (double complex*) mkl_malloc(2*M*sizeof(double complex), 64);	// temporary matrix
	int *ipiv = (int*) mkl_malloc(16*sizeof(int), 64);
	double complex *work = (double complex*) mkl_malloc(6*sizeof(double complex), 64);	// temporary matrix
	double *rwork = (double*) mkl_malloc(10*sizeof(double), 64);
	double *s = (double*) mkl_malloc(2*sizeof(double), 64);
	double complex *U = (double complex*) mkl_malloc(16*sizeof(double complex), 64);	// matrix of rotatoins
	int *p = (int*) mkl_malloc(M*sizeof(int), 64);	// for location of +1 in J for givens reduction
	int *n = (int*) mkl_malloc(M*sizeof(int), 64);  // for location of -1 in J for givens reduction

	// check if files are opened

	if(readG == NULL || readJ == NULL){
		printf("Cannot open file.\n");
		exit(-1);
	}

	// check if memory is allocated

	if(G == NULL || J == NULL || Pcol == NULL || Prow == NULL || f == NULL ){
		printf("Cannot allocate memory.\n");
		exit(-2);
	}

	// --------------------------------------- file reading ----------------------------------------

	double start = omp_get_wtime();

	#pragma omp parallel num_threads(3)
	{
		if(omp_get_thread_num() == 0){
			// prepare Pcol
			int j;
			for(j = 0; j < N; ++j ){
				Pcol[j] = j;
			}
		}

		if(omp_get_thread_num() == 1){
			// read matrix G 
			int i, j;
			for(j = 0; j < N; ++j ){
				for(i = 0; i < M; ++i ){
					double x, y;
					fscanf(readG, "%lg %lg ", &x, &y);
					G[i+M*j] = x + I*y;
				}
			}
		}

		if(omp_get_thread_num() == 2){
		 	// read vector J and prepare permutation vectors
			int i;
			for(i = 0; i < M; ++i){
				long int itemp;
				fscanf(readJ, "%ld ", &itemp);
				J[i] = 1.0 * itemp;
				Prow[i] = i;
			}
		}
	}

	double seconds = (double)(omp_get_wtime() - start);
	printf("reading time = %lg s\n", seconds);


	// ---------------------------------------------------------- ALGORITHM ----------------------------------------------------------

	//printf("Pivoting QR...\n\n");

	double pivot2time = 0;
	double pivot1time = 0;
	double pivotiranje = 0;
	double redukcijatime = 0;
	double H2 = 0;
	int pivot_1_count = 0;
	int pivot_2_count = 0;
	int last_pivot = -1;
	double err1 = 0, err2 = 0, err0 = 0, errk = 0;
	double max1 = 0, max2 = 0, max0 = 0, maxk = 0;
	start = omp_get_wtime();

	int i, j, k, nthreads, mkl_nthreads;

	// first compute J-norms of matrix G

	nthreads = N/D > omp_get_max_threads() ? N/D : omp_get_max_threads();
	if (N/D == 0) nthreads = 1;
	double norm_time = omp_get_wtime();

	#pragma omp parallel for num_threads( nthreads )
	for(j = 0; j < N; ++j){
		norm[j] = 0;
		for(i = 0; i < M; ++i) norm[j] += conj(G[i+M*j]) * J[i] * G[i+M*j];
	}
	printf("Racunanje normi = %lg s\n", omp_get_wtime() - norm_time);


	for(k = 0; k < N; ++k){

		// ------------------------ choosing a pivoting strategy (partial pivoting) -------------------------------

		// ------------------------ update J-norms of columns ------------------------

		nthreads =(N-k)/D > omp_get_max_threads() ? (N-k)/D : omp_get_max_threads();
		if ((N-k)/D == 0) nthreads = 1;

		if( k && !( k % refresh == 0 || (k % refresh == 1 && last_pivot == 2) ) ){	// if we have something to update

			#pragma omp parallel num_threads( nthreads )
			{
				#pragma omp for nowait
				for( j = k; j < N; ++j){

					// pivot 1 was last
					if( last_pivot == 1 ){

						double denomi = conj(G[k-1+M*j]) * J[k-1] * G[k-1+M*j];
						double frac = cabs(norm[j]) / cabs(denomi);

						// stable to update the norm
						if( creal(norm[j]) * denomi < 0 || cabs(frac - 1) > eps )
							norm[j] -= denomi;

						// else compute the norm again 
						else{
							norm[j] = 0;
							for(i = k; i < M; ++i) norm[j] += conj(G[i+M*j]) * J[i] * G[i+M*j];
						}
					}

					// pivot 2 was last
					else if( last_pivot == 2 ){

						double denomi = conj(G[k-1+M*j]) * J[k-1] * G[k-1+M*j] + conj(G[k-2+M*j]) * J[k-2] * G[k-2+M*j];
						double frac = cabs(norm[j]) / cabs(denomi);
						
						// stable to update the norm
						if( creal(norm[j]) * denomi < 0 || cabs(frac - 1) > eps )
							norm[j] -= denomi;

						// else compute the norm again 
						else{
							norm[j] = 0;
							for(i = k; i < M; ++i) norm[j] += conj(G[i+M*j]) * J[i] * G[i+M*j];
						}
					}
				}
			}
		}

		// refresh norms
		else{

			#pragma omp parallel for num_threads( nthreads )
			for(j = k; j < N; ++j){
				norm[j] = 0;
				for(i = k; i < M; ++i) norm[j] += conj(G[i+M*j]) * J[i] * G[i+M*j];
			}
		}



		// ------------------------ start the pivoting strategy ------------------------
		
		double pp = omp_get_wtime();
		double pivot_lambda = 0;
		double pivot_sigma = 0;
		int pivot_r = -1;	// 2nd column for partial pivoting
							// will be used for column swap k+1 <-> pivot_r when PIVOT_2 begins

		double Akk = (double) norm[k];

		if(k == N-1) goto PIVOT_1;

		// ------------------------ find pivot_lambda ------------------------

		nthreads = (M-k)/D > omp_get_max_threads() ? (M-k)/D : omp_get_max_threads();
		if ((M-k)/D == 0) nthreads = 1;

		#pragma omp parallel for num_threads( nthreads )
		for(i = k; i < M; ++i)	f[i] = J[i] * G[i+M*k];

		nthreads = (N-k)/D > omp_get_max_threads() ? (N-k)/D : omp_get_max_threads();
		if ((N-k)/D == 0) nthreads = 1;

		mkl_nthreads = (M-k)/D > mkl_get_max_threads()/nthreads ? (M-k)/D : mkl_get_max_threads()/nthreads;
		if ((M-k)/D == 0 || mkl_get_max_threads()/nthreads == 0) mkl_nthreads = 1;

		#pragma omp parallel num_threads( nthreads )
		{
			mkl_set_num_threads_local( mkl_nthreads );
			int Mk = M-k;
			int inc = 1;

			#pragma omp for nowait
			for(i = k+1; i < N; ++i){

				double complex Aik = 0;
				zdotc(&Aik, &Mk, &G[k+M*i], &inc, &f[k], &inc); //Aik = gi* J gk, but on a submatrix G[k:M, k:N]
				
				#pragma omp critical
				if(pivot_lambda < cabs(Aik)){
					pivot_lambda = cabs(Aik);
					pivot_r = i;
				}
			}
		}
		mkl_set_num_threads_local(0);	//return global value
		
		if(cabs(Akk) >= ALPHA * pivot_lambda) goto PIVOT_1;


		// ------------------------ find pivot_sigma ------------------------

		nthreads = (M-k)/D > omp_get_max_threads() ? (M-k)/D : omp_get_max_threads();
		if ((M-k)/D == 0) nthreads = 1;

		#pragma omp parallel for num_threads( nthreads )
		for(i = k; i < M; ++i)	f[i] = J[i] * G[i+M*pivot_r];

		nthreads = (N-k)/D > omp_get_max_threads() ? (N-k)/D : omp_get_max_threads();
		if ((N-k)/D == 0) nthreads = 1;

		mkl_nthreads = (M-k)/D > mkl_get_max_threads()/nthreads ? (M-k)/D : mkl_get_max_threads()/nthreads;
		if ((M-k)/D == 0 || mkl_get_max_threads()/nthreads == 0) mkl_nthreads = 1;

		#pragma omp parallel num_threads( nthreads ) 
		{
			mkl_set_num_threads_local( mkl_nthreads );

			#pragma omp for reduction(max:pivot_sigma)
			for(i = k; i < N; ++i){

				if(i == pivot_r) continue;

				double complex Air = 0;
				int Mk = M-k;
				int inc = 1;
				zdotc(&Air, &Mk, &G[k+M*i], &inc, &f[k], &inc);

				if(pivot_sigma < cabs(Air)) pivot_sigma = cabs(Air);
			}
		}
		mkl_set_num_threads_local(0);	//return global value

		if(cabs(Akk) * pivot_sigma >= ALPHA * pivot_lambda * pivot_lambda) goto PIVOT_1;

		double Arr = (double) norm[pivot_r];

		if(cabs(Arr) >= ALPHA * pivot_sigma){
			// gr is the pivot column 
			// swap columns k <-> r
			// then do PIVOT_1 with Householder

			long int itemp = Pcol[pivot_r];
			Pcol[pivot_r] = Pcol[k];
			Pcol[k] = itemp;

			double complex ctemp = norm[pivot_r];
			norm[pivot_r] = norm[k];
			norm[k] = ctemp;


			int inc = 1;
			int mkl_nthreads = M/D > mkl_get_max_threads() ? M/D : mkl_get_max_threads(); 
			if(M/D == 0) mkl_nthreads = 1;
			mkl_set_num_threads(mkl_nthreads);
			zswap(&M, &G[M*pivot_r], &inc, &G[M*k], &inc);
			Akk = Arr;

			goto PIVOT_1;
		}
		
		// ----------------------------------------------PIVOT_2-----------------------------------------------------

		pivotiranje = pivotiranje + omp_get_wtime() - pp;
		pivot_2_count += 1;
		double start2 = omp_get_wtime();
		last_pivot = 2;

		// do a column swap pivot_r <-> k+1 if needed

		if(pivot_r != k+1){

			long int itemp = Pcol[pivot_r];
			Pcol[pivot_r] = Pcol[k+1];
			Pcol[k+1] = itemp;

			double complex ctemp = norm[pivot_r];
			norm[pivot_r] = norm[k+1];
			norm[k+1] = ctemp;

			int inc = 1;
			int mkl_nthreads = M/D > mkl_get_max_threads() ? M/D : mkl_get_max_threads();
			if(M/D == 0) mkl_nthreads = 1;
			mkl_set_num_threads(mkl_nthreads);
			zswap(&M, &G[M*pivot_r], &inc, &G[M*(k+1)], &inc);
		}


		// copy columns of G into K
		int Mk = M-k;
		int inc = 1;
		mkl_nthreads = Mk/D > mkl_get_max_threads() ? Mk/D : mkl_get_max_threads();
		if(Mk/D == 0) mkl_nthreads = 1;

		mkl_set_num_threads(mkl_nthreads);
		zcopy(&Mk, &G[k+M*k], &inc, &K[k], &inc);
		zcopy(&Mk, &G[k+M*(k+1)], &inc, &K[k+M], &inc);
		

		int first_non_zero_idx = -1;	// index of the first non zero element in column k

		// [SEQUENTIAL] find the first non zero element in the kth column
		for(i = k; i < M; ++i){

			if(cabs(G[i+M*k]) < EPSILON) continue;
			first_non_zero_idx = i;
			break;
		}

		if(first_non_zero_idx == -1){
			printf("Zero column\n");
			exit(-3);
		}


		// do row swap if needed, so thath Gkk != 0

		if(first_non_zero_idx != k){ 

			long int itemp = Prow[first_non_zero_idx];
			Prow[first_non_zero_idx] = Prow[k];
			Prow[k] = itemp;

			double dtemp = J[first_non_zero_idx];
			J[first_non_zero_idx] = J[k];
			J[k] = dtemp;

			int Nk = N - k;
			int mkl_nthreads = Nk/D > mkl_get_max_threads() ? Nk/D : mkl_get_max_threads();
			if(Nk/D == 0) mkl_nthreads = 1;
			mkl_set_num_threads(mkl_nthreads);
			zswap(&Nk, &G[k + M*k], &M, &G[first_non_zero_idx + M*k], &M);

			mkl_set_num_threads(1);
			int n_ = 2;
			//zswap(&n_, &K[k], &M, &K[first_non_zero_idx], &M);
		}


		int np = 0;	// number of 1 in J[k:M, k:M] 
		int nn = 0;	// number of -1 in J[k:M, k:M]

		// update the signum arrays
		if( J[k] < 0 ) n[nn++] = k;
		else p[np++] = k;


		// find the first i so that Ji = -Jk and G(i, k) != 0
		// then swap rows k+1 <-> i

		first_non_zero_idx = -1;	// first non zero element in the -Jk class

		// [SEQUENTIAL]
		for(i = k+1; i < M; ++i){

			if(J[k] == J[i] || cabs(G[i+M*k]) < EPSILON ) continue;

			first_non_zero_idx = i;
			if(i == k+1) break; 	// no swapping needed, everythinig already in the right position


			// else, swap rows i <-> k+1

			long int itemp = Prow[i];
			Prow[i] = Prow[k+1];
			Prow[k+1] = itemp;

			double dtemp = J[i];
			J[i] = J[k+1];
			J[k+1] = dtemp;

			int Nk = N-k;
			int mkl_nthreads = Nk/D > mkl_get_max_threads() ? Nk/D : mkl_get_max_threads();
			if(Nk/D == 0) mkl_nthreads = 1;
			mkl_set_num_threads(mkl_nthreads);
			zswap(&Nk, &G[k+1 + M*k], &M, &G[i + M*k], &M);

			mkl_set_num_threads(1);
			int n_ = 2;
			//zswap(&n_, &K[k+1], &M, &K[i], &M);

			break;
		}

		// update the signum arrays
		if( J[k+1] < 0) n[nn++] = k+1;
		else p[np++] = k+1;

		// fill and count the signums (not necessary to be ordered)
	
		#pragma omp parallel for num_threads(nthreads)
		for(i = k+2; i < M; ++i){

			if(cabs(G[i+M*k]) < EPSILON) continue;

			else if(J[i] < 0){
				#pragma omp critical
				n[nn++] = i;
			}

			else{
				#pragma omp critical
				p[np++] = i;
			}
		}


		// [REDUCTION] do plane rotations with Gkk on all elements with signum Jk with reduction with the p array
		// do the sam thing with n array (at the same time)

		double rr = omp_get_wtime();
		#pragma omp parallel num_threads(2)
		{

			// first thread kills positives
			if(omp_get_thread_num() == 0){
				int offset;
				for(offset = 1; offset < np; offset *= 2){

					int nthreads_loc = np/(2*offset);
					if(nthreads_loc == 0) nthreads_loc = 1;
					else if ( nthreads_loc > (omp_get_max_threads()-2)/2 ) nthreads_loc = (omp_get_max_threads()-2)/2;

					#pragma omp parallel for num_threads( nthreads_loc )
					for(i = 0; i < np - offset; i += 2*offset){

						mkl_set_num_threads_local(1);

						// G[p[i], k] destroys G[p[i+offset], k]

						double c;
						double complex s;
						double complex eliminator = G[p[i] + M*k];
						double complex eliminated = G[p[i + offset] + M*k];
						zrotg(&eliminator, &eliminated, &c, &s);

						// apply the rotation
						int Nk = 2;
						zrot(&Nk, &G[p[i] + M*k], &M, &G[p[i + offset] + M*k], &M, &c, &s);
						G[p[i + offset] + M*k] = 0;
					}
				}
			}

			// second thread kills negatives
			else{
				int offset;
				for(offset = 1; offset < nn; offset *= 2){

					int nthreads_loc = nn/(2*offset);
					if(nthreads_loc == 0) nthreads_loc = 1;
					else if ( nthreads_loc > (omp_get_max_threads()-2)/2 ) nthreads_loc = (omp_get_max_threads()-2)/2;

					#pragma omp parallel for num_threads( nthreads_loc )
					for(i = 0; i < nn - offset; i += 2*offset){

						mkl_set_num_threads_local(1);

						// G[n[i], k] destroys G[n[i+offset], k]

						double c;
						double complex s;
						double complex eliminator = G[n[i] + M*k];
						double complex eliminated = G[n[i + offset] + M*k];
						zrotg(&eliminator, &eliminated, &c, &s);

						// apply the rotation
						int Nk = 2;
						zrot(&Nk, &G[n[i] + M*k], &M, &G[n[i + offset] + M*k], &M, &c, &s);
						G[n[i + offset] + M*k] = 0;
					}
				}
			}
		}

		mkl_set_num_threads_local(0);	//return global value

		rr = omp_get_wtime() - rr;


		int kth_nonzeros = 2;
		if(np == 0 || nn == 0) kth_nonzeros = 1;	// just one of them is 0. at this point one od them is nonzero
													// if not, the program would exit with -4 (before this point)
													// then A is maybe singular?

		// do the same thing on a SECOND COLUMN
		// wee need to know if the kth column has 0 or 1 or 2 nonzeros elements
		// that determines in which case we are... (A1), (A2), (A3), (B1) or (B2)


		first_non_zero_idx = -1;	// index of the first non zero element in column k+1
									// will be filled with the first nonzero index, or stay -1

		//[SEQUENTIAL]
		for(i = k + kth_nonzeros; i < M; ++i){
            if(cabs(G[i+M*(k+1)]) < EPSILON) continue;
			first_non_zero_idx = i;
			break;
		}


		if(first_non_zero_idx == -1){	// we have the needed form alredy, continue to other columns
			goto HOUSEHOLDER;
		}

		int kkth_nonzeros = 1;	// number of nonzero elements in the (k+1)st column, but those below kth_nonzeros
								// this value is at least one, because we didnt exit in the previous if
 

		// do row swap if needed, so thath G(k+kth_nonzeros, k+1) != 0
		// if first_non_zero_idx == k + kth_nonzeros, then that's fine, nothing to swap

		if(first_non_zero_idx != k + kth_nonzeros){ 

			long int itemp = Prow[first_non_zero_idx];
			Prow[first_non_zero_idx] = Prow[k + kth_nonzeros];
			Prow[k + kth_nonzeros] = itemp;

			double dtemp = J[first_non_zero_idx];
			J[first_non_zero_idx] = J[k + kth_nonzeros];
			J[k + kth_nonzeros] = dtemp;

			int Nk = N - k - 1;
			int mkl_nthreads = Nk/D > mkl_get_max_threads() ? Nk/D : mkl_get_max_threads();
			if(Nk/D == 0) mkl_nthreads = 1;
			mkl_set_num_threads(mkl_nthreads);
			zswap(&Nk, &G[k + kth_nonzeros + M*(k+1)], &M, &G[first_non_zero_idx + M*(k+1)], &M);

			mkl_set_num_threads(1);
			int n_ = 1;
			//zswap(&n_, &K[k + kth_nonzeros], &M, &K[first_non_zero_idx], &M);
		}


		// update the signum arrays needed for reduction

		nn = 0;
		np = 0;

		if(J[k + kth_nonzeros] < 0) n[nn++] = k + kth_nonzeros;
		else p[np++] = k + kth_nonzeros;


		// find the first non zero element in the -J(k + kth_nonzeros) class

		first_non_zero_idx = -1;

		// [SEQUENTIAL]
		for(i = k + kth_nonzeros + 1; i < M; ++i){

			if(J[k + kth_nonzeros] == J[i] || cabs(G[i+M*(k+1)]) < EPSILON ) continue;

			first_non_zero_idx = i;
			if(i == k + kth_nonzeros + 1) break; 	// no swapping needed, everythinig already in the right position


			// else, swap rows i <-> k + kth_nonzeros + 1

			long int itemp = Prow[i];
			Prow[i] = Prow[k + kth_nonzeros + 1];
			Prow[k + kth_nonzeros + 1] = itemp;

			double dtemp = J[i];
			J[i] = J[k + kth_nonzeros + 1];
			J[k + kth_nonzeros + 1] = dtemp;

			int Nk = N-k-1;
			int mkl_nthreads = Nk/D > mkl_get_max_threads() ? Nk/D : mkl_get_max_threads();
			if(Nk/D == 0) mkl_nthreads = 1;
			mkl_set_num_threads(mkl_nthreads);
			zswap(&Nk, &G[k + kth_nonzeros + 1 + M*(k+1)], &M, &G[i + M*(k+1)], &M);

			mkl_set_num_threads(1);
			int n_ = 1;
			//zswap(&n_, &K[k + kth_nonzeros + 1], &M, &K[i], &M);
			break;
		}


		// update the kkth_nonzeros
		if(first_non_zero_idx != -1) kkth_nonzeros = 2;


		// update the signum arrays needed for reduction
		if(J[k + kth_nonzeros + 1] < 0) n[nn++] = k + kth_nonzeros + 1;
		else p[np++] = k + kth_nonzeros + 1;


		// fill and count the signums (not necessary to be ordered)
		#pragma omp parallel for num_threads(nthreads)
		for(i = k + kth_nonzeros + 2; i < M; ++i){

			if(cabs(G[i+M*(k+1)]) < EPSILON) continue;

			else if(J[i] < 0){
				#pragma omp critical
				n[nn++] = i;
			}

			else{
				#pragma omp critical
				p[np++] = i;
			}
		}


		// [REDUCTION] do plane rotations with Gkk on all elements with signum Jk with reduction with the p array
		// do the sam thing with n array (at the same time)

		double rrr = omp_get_wtime();
		#pragma omp parallel num_threads(2)
		{

			// first thread kills positives
			if(omp_get_thread_num() == 0){
				int offset;
				for(offset = 1; offset < np; offset *= 2){

					int nthreads_loc = np/(2*offset);
					if(nthreads_loc == 0) nthreads_loc = 1;
					else if ( nthreads_loc > (omp_get_max_threads()-2)/2 ) nthreads_loc = (omp_get_max_threads()-2)/2;

					#pragma omp parallel for num_threads( nthreads_loc )
					for(i = 0; i < np - offset; i += 2*offset){

						// G[p[i], k+1] destroys G[p[i+offset], k+1]

						mkl_set_num_threads_local(1);
							
						double c;
						double complex s;
						double complex eliminator = G[p[i] + M*(k+1)];
						double complex eliminated = G[p[i + offset] + M*(k+1)];
						zrotg(&eliminator, &eliminated, &c, &s);

						// apply the rotation
						int Nk = 1;
						zrot(&Nk, &G[p[i] + M*(k+1)], &M, &G[p[i + offset] + M*(k+1)], &M, &c, &s);
						G[p[i + offset] + M*(k+1)] = 0;
					}
				}
			}

			// second thread kills negatives
			else{
				int offset;
				for(offset = 1; offset < nn; offset *= 2){

					int nthreads_loc = nn/(2*offset);
					if(nthreads_loc == 0) nthreads_loc = 1;
					else if ( nthreads_loc > (omp_get_max_threads()-2)/2) nthreads_loc = (omp_get_max_threads()-2)/2;

					#pragma omp parallel for num_threads( nthreads_loc )
					for(i = 0; i < nn - offset; i += 2*offset){

						// G[n[i], k+1] destroys G[n[i+offset], k+1]

						mkl_set_num_threads_local(1);

						double c;
						double complex s;
						double complex eliminator = G[n[i] + M*(k+1)];
						double complex eliminated = G[n[i + offset] + M*(k+1)];
						zrotg(&eliminator, &eliminated, &c, &s);

						// apply the rotation
						int Nk = 1;
						zrot(&Nk, &G[n[i] + M*(k+1)], &M, &G[n[i + offset] + M*(k+1)], &M, &c, &s);
						G[n[i + offset] + M*(k+1)] = 0;
					}
				}
			}
		}
		mkl_set_num_threads_local(0);

		rrr = omp_get_wtime() - rrr;
		redukcijatime = redukcijatime + rrr + rr;


		// -------- check forms od 2x2 pivot ---------


		// if we are in (A3) or (B2) forms, then the 2x2 reduction is finished
		// in that case continue the main loop

		// condition (A3)
		if(kth_nonzeros == 2 && kkth_nonzeros == 0) goto HOUSEHOLDER;

		// condition (B2)
		if(kth_nonzeros == 1 && kkth_nonzeros == 1) goto HOUSEHOLDER;

		//make rows real which need to be
		
		
		for(i = k; i < k + kth_nonzeros; ++i){

			if( cabs(cimag(G[i + M*k])) < EPSILON ) continue;

			double complex scal = conj(G[i + M*k]) / cabs(G[i + M*k]);
			G[i + M*k] = cabs(G[i + M*k]);
			G[i + M*(k+1)] *= scal;
		}
	
		for(i = k + kth_nonzeros; i < k + kth_nonzeros + kkth_nonzeros; ++i){

			if( cabs(cimag(G[i + M*(k+1)])) < EPSILON) continue;

			double complex scal = conj(G[i + M*(k+1)]) / cabs(G[i + M*(k+1)]);
			G[i + M*(k+1)] = cabs(G[i + M*(k+1)]);
		}
		

		// handle the (A1) form

		if(kth_nonzeros == 2 && kkth_nonzeros == 2){

			//printf("\tPIVOT (A1)\n");

			// check if its a proper form
			// if not, fix it

			if( cabs(G[k+M*k] * G[k+1+M*(k+1)] - G[k+1+M*k] * G[k+M*(k+1)]) < EPSILON){

				// swap columns k <-> k+1

				long int itemp = Pcol[k];
				Pcol[k] = Pcol[k+1];
				Pcol[k+1] = itemp;

				double complex ctemp = norm[k];
				norm[k] = norm[k+1];
				norm[k+1] = ctemp;

				int n_ = k + 4;
				int inc = 1;
				int mkl_nthreads = n_/D > mkl_get_max_threads() ? n_/D : mkl_get_max_threads();
				if(n_/D == 0) mkl_nthreads = 1;
				mkl_set_num_threads(mkl_nthreads);
				zswap(&n_, &G[M*k], &inc, &G[M*(k+1)], &inc);


				int Mk = M - k;
				mkl_nthreads = Mk/D > mkl_get_max_threads() ? Mk/D : mkl_get_max_threads();
				if(Mk/D == 0) mkl_nthreads = 1;
				mkl_set_num_threads(mkl_nthreads);
				//zswap(&Mk, &K[k], &inc, &K[k+M], &inc);

				// make the kth rows k, k+1 real (k+3 and k+2 are already real)
				for(i = k; i < k+2; ++i){

					if( cabs(cimag(G[i+M*k])) < EPSILON ) continue; //the element is already real

					double complex scal = conj(G[i+M*k]) / cabs(G[i+M*k]);
					G[i+M*k] = cabs(G[i+M*k]);	// to be exact, so that the Img part is really = 0
					G[i+M*(k+1)] *= scal;
				}

				// do plane rotations with kth row

				int idx = k+2; 	// idx is the row that will be eliminated with the kth row
				if(J[k] == J[k+3]) idx = k+3;

				// generate plane rotation
				double c;
				double complex s;
				double complex eliminator = G[k+M*k];
				double complex eliminated = G[idx+M*k];
				zrotg(&eliminator, &eliminated, &c, &s);

				// apply the rotation
				n_ = 2;
				mkl_set_num_threads(1);
				zrot(&n_, &G[k+M*k], &M, &G[idx+M*k], &M, &c, &s);
				G[idx+M*k] = 0;

				// do plane rotations with (k+1)st row

				idx = k+2; 	// idx it the row that will be eliminated with the kth row
				if(J[k+1] == J[k+3]) idx = k+3;

				// generate plane rotation
				eliminator = G[k+1+M*k];
				eliminated = G[idx+M*k];
				zrotg(&eliminator, &eliminated, &c, &s);

				// apply the rotation
				zrot(&n_, &G[k+1+M*k], &M, &G[idx+M*k], &M, &c, &s);
				G[idx+M*k] = 0;	
			}


			// now do the final reduction

			double complex g11 = G[k+M*k];
			double complex g21 = G[k+1+M*k];
			double complex g12 = G[k+M*(k+1)];
			double complex g22 = G[k+1+M*(k+1)];
			double complex g32 = G[k+2+M*(k+1)];
			double complex g42 = G[k+3+M*(k+1)];

			double complex z = 1.0 / (g11 * g22 - g21 * g12);	// z = 1/detG1
			double a = J[k] * J[k+2] * cabs(z*z) * (g21*g21 - g11*g11) * (g32*g32 - g42*g42);
			double complex q = - J[k] * J[k+2] * conj(z) * (g32*g32 - g42*g42) / (1 + csqrt(a+1));

			// copmute F1
			G[k+M*(k+1)] = g12 + g21*q;
			G[k+1+M*(k+1)] = g22 + g11*q;

			// put zeros explicitly in the right places
			G[k+2+M*k] = 0;
			G[k+3+M*k] = 0;
			G[k+2+M*(k+1)] = 0;
			G[k+3+M*(k+1)] = 0;

			double end2 = omp_get_wtime();
			pivot2time += (double) (end2 - start2);
			goto HOUSEHOLDER;
		}


		// handle forms (A2) and (B1)

		else if(kth_nonzeros == 2 && kkth_nonzeros == 1 || kth_nonzeros == 1 && kkth_nonzeros == 2){

			// check if its a proper form
			// if not, fix it

			//printf("\tPIVOT (A2) or (B1)\n");

			// (B1) is already in proper form, so fix just form (A2) if needed

			if( kth_nonzeros == 2 && cabs(G[k+M*k] * G[k+1+M*(k+1)] - G[k+1+M*k] * G[k+M*(k+1)]) < EPSILON ){

				printf("(not optimal) A2 proper form, k = %d\n", k);

				// swap columns k <-> k+1

				long int itemp = Pcol[k];
				Pcol[k] = Pcol[k+1];
				Pcol[k+1] = itemp;

				double complex ctemp = norm[k];
				norm[k] = norm[k+1];
				norm[k+1] = ctemp;

				int n_ = k + 3;
				int inc = 1;
				int mkl_nthreads = n_/D > mkl_get_max_threads() ? n_/D : mkl_get_max_threads();
				if(n_/D == 0) mkl_nthreads = 1;
				mkl_set_num_threads(mkl_nthreads);
				zswap(&n_, &G[M*k], &inc, &G[M*(k+1)], &inc);

				int Mk = M - k;
				mkl_nthreads = Mk/D > mkl_get_max_threads() ? Mk/D : mkl_get_max_threads();
				if(Mk/D == 0) mkl_nthreads = 1;
				mkl_set_num_threads(mkl_nthreads);
				//zswap(&Mk, &K[k], &inc, &K[k+M], &inc);

				// make the kth rows k, k+1 real (k+2 is already real)
				for(i = k; i < k+2; ++i){

					if( cabs(cimag(G[i+M*k])) < EPSILON ) continue; //the element is already real

					double complex scal = conj(G[i+M*k]) / cabs(G[i+M*k]);
					G[i+M*k] = cabs(G[i+M*k]);	// to be exact, so that the Img part is really = 0
					G[i+M*(k+1)] *= scal;	
				}

				// do a plane rotation, eliminate row k+2 with row idx

				int idx = k; 	// idx is the row that will eliminate row k+2
				if(J[idx] != J[k+2]) idx = k+1;

				// generate plane rotation		
				double c;
				double complex s;
				double complex eliminator = G[idx+M*k];
				double complex eliminated = G[k+2+M*k];
				zrotg(&eliminator, &eliminated, &c, &s);

				// apply the rotation
				mkl_set_num_threads(1);	
				n_ = 2;
				zrot(&n_, &G[idx+M*k], &M, &G[k+2+M*k], &M, &c, &s);
				G[k+2+M*k] = 0;
			}


			// now do the final reduction

			double complex g11 = G[k+M*k];
			double complex g21 = G[k+1+M*k];
			double complex g12 = G[k+M*(k+1)];
			double complex g22 = G[k+1+M*(k+1)];
			double complex g32 = G[k+2+M*(k+1)];

			double complex z = 1 / (g11 * g22 - g21 * g12);	// z = 1/detG1
			double a = J[k] * J[k+2] * cabs(z*z) * (g21*g21 - g11*g11) * g32*g32;
			double complex q = - J[k] * J[k+2] * conj(z) * g32*g32 / (1 + csqrt(a+1));

			// copmute F1
			G[k+M*(k+1)] = g12 + g21*q;
			G[k+1+M*(k+1)] = g22 + g11*q;


			// put zeros explicitly in the right places
			G[k+2+M*k] = 0;
			G[k+2+M*(k+1)] = 0;

			double end2 = omp_get_wtime();
			pivot2time += (double) (end2 - start2);
			goto HOUSEHOLDER;
		}

		// if here, something broke down
		printf("\n\n\nUPS, SOMETHING WENT WRONG... STOPPING...Is A maybe singular?\n\n\n");
		exit(-4);


		HOUSEHOLDER:






		// E = F1
		E[0] = G[k+M*k];
		E[1] = G[k+1+M*k];
		E[2] = G[k+M*(k+1)];
		E[3] = G[k+1+(k+1)*M];

		//printf("k = %d\n", k);
		double complex Akr = 0;
		for(i = k; i < M; ++i) Akr += conj(G[i+M*k]) * J[i] * G[i+M*(k+1)];

		T[0] = conj(E[0])*J[k]*E[0] + conj(E[1])*J[k+1]*E[1];
		T[1] = conj(E[2])*J[k]*E[0] + conj(E[3])*J[k+1]*E[1];
		T[2] = conj(E[0])*J[k]*E[2] + conj(E[1])*J[k+1]*E[3];
		T[3] = conj(E[2])*J[k]*E[2] + conj(E[3])*J[k+1]*E[3];
		double d1 = cabs(Akk - T[0]);
		double d2 = cabs(conj(Akr)-T[1]);
		double d3 = cabs(Akr - T[2]);
		double d4 = cabs(Arr - T[3]);
		//printf("|A2 - F*JF| = %lg\n", csqrt(d1*d1+d2*d2+d3*d3+d4*d4));
		err2 += csqrt(d1*d1+d2*d2+d3*d3+d4*d4);
		if(max2 < err2) max2 = err2;

		// C = old G1
		C[0] = K[k];
		C[1] = K[k+1];
		C[2] = K[k+M];
		C[3] = K[k+1+M];

		// T = G1*JF
		T[0] = conj(C[0])*J[k]*E[0] + conj(C[1])*J[k+1]*E[1];
		T[1] = conj(C[2])*J[k]*E[0] + conj(C[3])*J[k+1]*E[1];
		T[2] = conj(C[0])*J[k]*E[2] + conj(C[1])*J[k+1]*E[3];
		T[3] = conj(C[2])*J[k]*E[2] + conj(C[3])*J[k+1]*E[3];
		//printMatrix(T, 2, 2);

		// f = F*JG1
		f[0] = conj(E[0])*J[k]*C[0] + conj(E[1])*J[k+1]*C[1];
		f[1] = conj(E[2])*J[k]*C[0] + conj(E[3])*J[k+1]*C[1];
		f[2] = conj(E[0])*J[k]*C[2] + conj(E[1])*J[k+1]*C[3];
		f[3] = conj(E[2])*J[k]*C[2] + conj(E[3])*J[k+1]*C[3];
		d1 = cabs(f[0] - T[0]);
		d2 = cabs(f[1] - T[1]);
		d3 = cabs(f[2] - T[2]);
		d4 = cabs(f[3] - T[3]);
		//printMatrix(f, 2, 2);
		//printf("|F*JG - G*JF| = %lg\n", csqrt(d1*d1+d2*d2+d3*d3+d4*d4));
		err1 += csqrt(d1*d1+d2*d2+d3*d3+d4*d4);
		if(max1 < err1) max1 = err1;










		// K = the difference operator for the J Householder
		K[k] -= G[k+M*k];
		K[k+1] -= G[k+1+M*k];
		K[k+M] -= G[k+M*(k+1)];
		K[k+1+M] -= G[k+1+M*(k+1)];

		// compute K*JK, first we need T = JK
		// fill the rest of the G with zeros
		Mk = M - k;
		nthreads = Mk/D > omp_get_max_threads() ? Mk/D : omp_get_max_threads();
		if(Mk/D == 0) nthreads = 1;
		#pragma omp parallel for num_threads( nthreads )
		for(i = k; i < M; ++i){
			T[i] = J[i] * K[i];
			T[i+M] = J[i] * K[i+M];
		}

		// compute K*T, where T = JK
		// C = K*JK
		Mk = M - k;
		mkl_nthreads = Mk/D > mkl_get_max_threads() ? Mk/D : mkl_get_max_threads();
		if(Mk/D == 0) mkl_nthreads = 1;
		mkl_set_num_threads( mkl_nthreads );

		double complex alpha = 1;
		double complex beta = 0;
		int n_ = 2;
		char nontrans = 'N';
		char trans = 'C';
		zgemm(&trans, &nontrans, &n_, &n_, &Mk, &alpha, &K[k], &M, &T[k], &M, &beta, C, &n_);
		C[0] = creal(C[0]);
		C[3] = creal(C[3]);

		// C = C^(-1) = (K*JK)^+ 
		mkl_set_num_threads(1);
		int lwork = 4;
		int info;
		zgetrf(&n_, &n_, C, &n_, ipiv, &info);
		if( info ) printf("LU of K*JK unstable. Proceeding.\n");
		zgetri(&n_, C, &n_, ipiv, work, &lwork, &info);
		if( info ) printf("(K*JK)^+ unstable. Proceeding.\n");
		C[0] = creal(C[0]);
		C[3] = creal(C[3]);


		// apply the reflector
		int Nk = (N - k - 2)/2;
		nthreads = Nk/D > omp_get_max_threads() ? Nk/D : omp_get_max_threads();
		if(Nk/D == 0) nthreads = 1;

		Mk = M - k;
		mkl_nthreads = Mk/D > mkl_get_max_threads()/nthreads ? Mk/D : mkl_get_max_threads()/nthreads;
		if(Mk/D == 0 || mkl_get_max_threads()/nthreads == 0) mkl_nthreads = 1;

		// compute E = KC
		alpha = 1;
		beta = 0;
		mkl_set_num_threads( mkl_nthreads );
		zgemm(&nontrans, &nontrans, &Mk, &n_, &n_, &alpha, &K[k], &M, C, &n_, &beta, &E[k], &M);

		for(i = 0; i < 2*M; ++i) K[i] = 0;

		// E = K(K*JK)^+
		// T = JK
		double ss = omp_get_wtime();
		#pragma omp parallel num_threads( nthreads )
		{
			mkl_set_num_threads_local( mkl_nthreads );

			#pragma omp for nowait
			for(j = k+2; j < N; ++j){

				// c = T*g
				alpha = 1;
				beta = 0;
				zgemv(&trans, &Mk, &n_, &alpha, &T[k], &M, &G[k+M*j], &inc, &beta, &K[2*j], &inc);
			}
		}

		#pragma omp parallel num_threads( nthreads )
		{
			mkl_set_num_threads_local( mkl_nthreads );

			#pragma omp for nowait
			for(j = k+2; j < N; ++j){

				// g = g - 2E c
				alpha = -2;
				beta = 1;
				zgemv(&nontrans, &Mk, &n_, &alpha, &E[k], &M, &K[2*j], &inc, &beta, &G[k+M*j], &inc);
			}
		}

		mkl_set_num_threads_local(0);
		H2 += omp_get_wtime() - ss;

		k = k+1;
		double end2 = omp_get_wtime();
		pivot2time += (double) (end2 - start2);
		goto LOOP_END;
	
		// ----------------------------------------------PIVOT_1----------------------------------------------------

		PIVOT_1:

		last_pivot = 1;
		pivotiranje = pivotiranje + omp_get_wtime() - pp;
		pivot_1_count += 1;
		double start1 = omp_get_wtime();


		// check the condition sign(Akk) = Jk
		// if not, do row swap and diagonal swap in J

		if( Akk > 0 && J[k] < 0){

			int i;
			for(i = k+1; i < M; ++i) if(J[i] > 0) break;

			J[k] = 1.0;
			J[i] = -1.0;

			// update Prow
			long int itemp = Prow[i];
			Prow[i] = Prow[k];
			Prow[k] = itemp;

			// swap rows in G 
			int Nk = N - k;
			mkl_nthreads = Nk/D > mkl_get_max_threads() ? Nk/D : mkl_get_max_threads();
			if(Nk/D == 0) mkl_nthreads = 1;
			mkl_set_num_threads(mkl_nthreads);
			zswap(&Nk, &G[k + M*k], &M, &G[i + M*k], &M);
		}

		else if( Akk < 0 && J[k] > 0){

			for(i = k+1; i < M; ++i) if(J[i] < 0) break;

			J[k] = -1.0;
			J[i] = 1.0;


			// update Prow
			long int itemp = Prow[i];
			Prow[i] = Prow[k];
			Prow[k] = itemp;

			// swap rows in G 
			int Nk = N - k;
			mkl_nthreads = Nk/D > mkl_get_max_threads() ? Nk/D : mkl_get_max_threads();
			if(Nk/D == 0) mkl_nthreads = 1;
			mkl_set_num_threads(mkl_nthreads);
			zswap(&Nk, &G[k + M*k], &M, &G[i + M*k], &M);
		}
		

		// compute reflector constant H_sigma
		// compute vector f, where g*Jg = f*Jf must hold
		// that's why we need fk that looks like Hg = H_sigma*f = H_sigma*(sqrt(|sumk|), 0, ..., 0)

		double complex gkk;
		if(cabs(G[k+M*k]) >= EPSILON) gkk = -G[k+M*k] * csqrt(cabs(Akk)) / cabs(G[k+M*k]);
		else gkk = csqrt(cabs(Akk));

		// save the J norm of the vector
		double fJf = Akk + J[k] * (cabs(Akk) + 2 * csqrt(cabs(Akk)) * cabs(G[k+M*k]));

		// make the reflector vector and save it
		alpha = -1;
		inc = 1;
		Mk = M - k;
		mkl_nthreads = Mk/D > mkl_get_max_threads() ? Mk/D : mkl_get_max_threads();
		if(Mk/D == 0) mkl_nthreads = 1;
		mkl_set_num_threads(mkl_nthreads);

		zcopy(&Mk, &G[k+M*k], &inc, &f[k], &inc);
		f[k] -= gkk;

		// update G
		G[k + M*k] = gkk;
		//printf("k = %d\n|Akk - gkk| = %lg\n", k, cabs(Akk - conj(gkk)*J[k]*gkk) );
		err0 += cabs(Akk - conj(gkk)*J[k]*gkk);
		if(max0 < err0) max0 = err0;

		nthreads = (Mk-1)/D > omp_get_max_threads() ? (Mk-1)/D : omp_get_max_threads();
		if ( (Mk-1)/D == 0) nthreads = 1;

		T[k] = J[k] * f[k];
		#pragma omp parallel for num_threads(nthreads)
		for(i = k+1; i < M; ++i){
			G[i + M*k] = 0;
			T[i] = J[i] * f[i];
		}

		// apply the rotation on the rest of the matrix
		nthreads = (N-k-1)/D > omp_get_max_threads() ? (N-k-1)/D : omp_get_max_threads();
		if ((N-k-1)/D == 0) nthreads = 1;

		mkl_nthreads = Mk/D > mkl_get_max_threads()/nthreads ? Mk/D : mkl_get_max_threads()/nthreads;
		if (Mk/D == 0 || mkl_get_max_threads()/nthreads == 0) mkl_nthreads = 1;

		#pragma omp parallel num_threads( nthreads )
		{
			mkl_set_num_threads_local(mkl_nthreads);

			#pragma omp for nowait
			for(j = k+1; j < N; ++j){

				// T = Jf
				// alpha = f*Jg
				int Mk = M - k;
				int inc = 1;
				double complex alpha;
				zdotc(&alpha, &Mk, &T[k], &inc, &G[k+M*j], &inc);
				alpha = - 2 * alpha / fJf;
				// G[k + M*j] = alpha * f[k] + G[k + M*k]
				zaxpy(&Mk, &alpha, &f[k], &inc, &G[k + M*j], &inc);
			}
		}
		mkl_set_num_threads_local(0);

		pivot1time += (double)(omp_get_wtime() - start1);
		LOOP_END: continue;

	}	// END OF MAIN LOOP

	// ----------------------------------------- PRINT TIMEs -----------------------------------------

	seconds = (double)(omp_get_wtime() - start);
	printf("algorithm time = %lg s\n", seconds);
	printf("PIVOT_1 (%d)	time = %lg s (%lg %%)\n", pivot_1_count, pivot1time, pivot1time / seconds * 100);
	printf("PIVOT_2 (%d)	time = %lg s (%lg %%)\n", pivot_2_count, pivot2time, pivot2time / seconds * 100);
	printf("PIVOT_2 reflektor time = %lg s (udio relativnog = %lg %%, udio apsolutnog = %lg %%)\n", H2, H2/pivot2time * 100, H2/seconds * 100);
	printf("pivotiranje time = %lg s (%lg %%)\n", pivotiranje, pivotiranje/seconds * 100);
	printf("prosjecna greska |A2-F*JF| = %lg, max = %lg\n", err2/pivot_2_count, max2);
	printf("prosjecna greska |F*JG-G*JF| = %lg, max = %lg\n", err1/pivot_2_count, max1);
	printf("prosjecna greska |Akk - gkk| = %lg, max = %lg\n", err0/pivot_1_count, max0);


	// ----------------------------------------- writing -----------------------------------------

	start = omp_get_wtime();

	FILE *writeG = fopen("data/reducedG.bin", "wb");
	FILE *writeJ = fopen("data/reducedJ.bin", "wb");
	FILE *writeCol = fopen("data/Pcol.bin", "wb");
	FILE *writeRow = fopen("data/Prow.bin", "wb");


	#pragma omp parallel num_threads(4)
	{

		if(omp_get_thread_num() == 0){
			// write J
			int i;
			for(i = 0; i < M; ++i){
				fprintf(writeJ, "%ld ", (long int)J[i]);
			}
		}

		if(omp_get_thread_num() == 1){
			// write J and Prow
			int i;
			for(i = 0; i < M; ++i){
				fprintf(writeRow, "%ld ", Prow[i]);
			}
		}

		if(omp_get_thread_num() == 2){
			// write G
			int i, j;
			for(j = 0; j < N; ++j){
				for(i = 0; i < M; ++i){
					fprintf(writeG, "%.*g %.*g ", DIGITS, DIGITS, creal(G[i+M*j]), cimag(G[i+M*j]));
				}
			}
		}

		if(omp_get_thread_num() == 3){
			// write pcol
			int j;
			for(j = 0; j < N; ++j){
				fprintf(writeCol, "%ld ", Pcol[j]);
			}
		}
	}

	seconds = (double)(omp_get_wtime() - start);
	printf("writing time = %lg s\n", seconds);


	// ----------------------------------------- cleaning -----------------------------------------

	fclose(readG);
	fclose(readJ);
	fclose(writeG);
	fclose(writeJ);
	fclose(writeCol);
	fclose(writeRow);
	
	mkl_free(Prow);
	mkl_free(Pcol);
	mkl_free(G);
	mkl_free(J);
	mkl_free(f);

	printf("\n-------------------------------------------------------------------------------\n\n");
	return(0);
}
