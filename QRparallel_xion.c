// call as:
//
// ./GFreduction.out G.bin J.bin M N
//
// (G.bin = filename where to store G in binary, same as J.bin)

// exits:
//   -1 ....... Cannot open file.
//   -2 ....... Cannot allocate memory.
//   -3 ....... Zero column in G
//	 -4 ....... Algorithm broke down.
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


#define EPSILON DBL_EPSILON
#define DIGITS DBL_DIG
#define NTHREADS_FOR_COL_COPY 8
#define SEQ_TRESHOLD_FOR_COL_COPY 25	// if M < NTHREADS_FOR_COL_COPY * SEQ_TRESHOLD_FOR_COL_COPY then col swap/copy is sequential


double ALPHA = (1.0 + csqrt(17.0))/8.0; //Bunch-Parlett alpha

void printMatrix(double complex *G, int M, int N){

	int i, j;
	for( i = 0; i < M; ++i ){
		for( j = 0; j < N; ++j ){
			printf("%10.5g + i%10.5g  ", creal(G[i+M*j]), cimag(G[i+M*j]));
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

	// read variables from command line
	int M = atoi(argv[3]);
	int N = atoi(argv[4]);
	FILE *readG = fopen(argv[1], "rb");
	FILE *readJ = fopen(argv[2], "rb");

	//printf("\n\n--------------------------------- ALGORITHM ------------------------------------\n");
	//printf("\nReading data...\n");

	// allocate memory
	double complex *G = (double complex*) malloc(M*N*sizeof(double complex));
	double complex *H = (double complex*) malloc(M*M*sizeof(double complex));	// reflector
	double complex *T = (double complex*) malloc(M*N*sizeof(double complex));	// temporary matrix
	double complex *U = (double complex*) malloc(16*sizeof(double complex));	// matrix of rotatoins
	int *p = (int*) malloc(M*sizeof(int));	// for location of +1 in J for givens reduction
	int *n = (int*) malloc(M*sizeof(int));  // for location of -1 in J for givens reduction
	double *J = (double*) malloc(M*sizeof(double));
	long int *Prow = (long int*) malloc(M*sizeof(long int));	// for row permutation
	long int *Pcol = (long int*) malloc(N*sizeof(long int));	// for column permutation
	double complex *f = (double complex*) malloc(M*sizeof(double complex));	// vector f
	double complex *tempf = (double complex*) malloc(M*sizeof(double complex));	// vector tempf, fisrt column after Householder transform


	// check if files are opened

	if(readG == NULL || readJ == NULL){
		printf("Cannot open file.\n");
		exit(-1);
	}

	// check if memory is allocated

	if(G == NULL || J == NULL || Pcol == NULL || Prow == NULL || T == NULL || H == NULL || f == NULL || p == NULL || n == NULL){
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
				for(int i = 0; i < M; ++i ){
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

	double end = omp_get_wtime();
	double seconds = (double)(end - start);
	printf("reading time = %lg s\n", seconds);


	// ---------------------------------------------------------- ALGORITHM ----------------------------------------------------------

	//printf("Pivoting QR...\n\n");

	double pivot2time = 0;
	double pivot1time = 0;
	int pivot_1_count = 0;
	int pivot_2_count = 0;
	start = omp_get_wtime();

	// first count +-1 and store their locations

	int i, j, k;
	for(k = 0; k < N; ++k){

		// ------------------------ choosing a pivoting strategy (partial pivoting) -------------------------------
		// we need to know the signum of the J-norm of the first column
		// because the pivoting element, Akk, will have to satisfy
		// sign(Jk) = sign(gk*Jgk)

		double Akk = 0;	// Akk = gk* J gk, on a working submatrix G[k:M, k:N]
		double pivot_lambda = 0;
		double pivot_sigma = 0;
		int pivot_r = -1;	// 2nd column for partial pivoting
							// will be used for column swap k+1 <-> pivot_r when PIVOT_2 begins
		
		// compute Akk for the working submatrix G[k:M, k:N]
		#pragma omp parallel for reduction(+:Akk)
		for(i = k; i < M; ++i) Akk += conj(G[i+M*k]) * J[i] * G[i+M*k];		

		if(k == N-1) goto PIVOT_1;


		// find pivot_lambda

		#pragma omp parallel for
		for(i = k+1; i < N; ++i){
			double complex Aik = 0;		//Aik = gi* J gk, but on a submatrix G[k:M, k:N]

			#pragma omp parallel for reduction(+:Aik)
			for(j = k; j < M; ++j) Aik += conj(G[j+M*i]) * J[j] * G[j+M*k];

			#pragma omp critical
			if(pivot_lambda < cabs(Aik)){
				pivot_lambda = cabs(Aik);
				pivot_r = i;
			}
		}

		if(cabs(Akk) >= ALPHA * pivot_lambda) goto PIVOT_1;


		// find pivot_sigma

	
		#pragma omp parallel for
		for(i = k; i < N; ++i){

			if(i == pivot_r) continue;
			double complex Air = 0;  //Air = gi* J gr, but on a submatrix G[k:M, k:N]

			#pragma omp parallel for reduction(+:Air)
			for(int j = k; j < M; ++j)	Air += conj(G[j+M*i]) * J[j] * G[j+M*pivot_r];

			#pragma omp critical
			if(pivot_sigma < cabs(Air)) pivot_sigma = cabs(Air);
		}

		if(cabs(Akk) * pivot_sigma >= ALPHA * pivot_lambda * pivot_lambda) goto PIVOT_1;


		double Arr = 0; // on a working submatrix G[k:M, k:N]

		#pragma omp parallel for reduction(+:Arr)
		for(i = k; i < M; ++i) Arr += conj(G[i+M*pivot_r]) * J[i] * G[i+M*pivot_r];

		if(cabs(Arr) >= ALPHA * pivot_sigma){
			// gr is the pivot column 
			// swap columns k <-> r
			// then do PIVOT_1 with Householder

			long int itemp = Pcol[pivot_r];
			Pcol[pivot_r] = Pcol[k];
			Pcol[k] = itemp;

			int inc = 1;
			int offset = M / NTHREADS_FOR_COL_COPY;
			int leftovers;

			// this means that if a single thread needs to copy less than 25 elements, 
			// we shift it to a sequental way (this is the case when M < 25*8 = 200)
			// M < 200 -> sequential
			// M >= 200 -> parallel
			if(offset < SEQ_TRESHOLD_FOR_COL_COPY) leftovers = M;
			else leftovers = M % offset;

			if(offset >= SEQ_TRESHOLD_FOR_COL_COPY){
				#pragma omp parallel for num_threads(NTHREADS_FOR_COL_COPY)
				for(i = 0; i <= M - offset; i += offset){
					zswap_(&offset, &G[i + M*pivot_r], &inc, &G[i + M*k], &inc);
				}
			}

			zswap_(&leftovers, &G[M - leftovers + M*pivot_r], &inc, &G[M - leftovers + M*k], &inc);

			Akk = Arr;

			goto PIVOT_1;
		}

		
		// ----------------------------------------------PIVOT_2-----------------------------------------------------

		pivot_2_count += 1;
		double start2 = omp_get_wtime();

		// do a column swap pivot_r <-> k+1 if needed

		if(pivot_r != k+1){

			long int itemp = Pcol[pivot_r];
			Pcol[pivot_r] = Pcol[k+1];
			Pcol[k+1] = itemp;

			int inc = 1;
			int offset = M / NTHREADS_FOR_COL_COPY;
			int leftovers;

			// this means that if a single thread needs to copy less than 25 elements, 
			// we shift it to a sequental way (this is the case when M < 25*8 = 200)
			// M < 200 -> sequential
			// M >= 200 -> parallel
			if(offset < SEQ_TRESHOLD_FOR_COL_COPY) leftovers = M;
			else leftovers = M % offset;

			if(offset >= SEQ_TRESHOLD_FOR_COL_COPY){
				#pragma omp parallel for num_threads(NTHREADS_FOR_COL_COPY)
				for(i = 0; i <= M - offset; i += offset){
					zswap_(&offset, &G[i + M*pivot_r], &inc, &G[i + M*(k+1)], &inc);
				}
			}

			zswap_(&leftovers, &G[M - leftovers + M*pivot_r], &inc, &G[M - leftovers + M*(k+1)], &inc);
		}


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


		// making the kth column real

		#pragma omp parallel for
		for(i = first_non_zero_idx; i < M; ++i){

			if(cabs(G[i+M*k]) < EPSILON) continue;

			double complex scal = conj(G[i+M*k]) / cabs(G[i+M*k]);
			G[i+M*k] = cabs(G[i+M*k]);	// to be exact, so that the Img part is really = 0
			int Nk = N-k-1;
			zscal_(&Nk, &scal, &G[i+M*(k+1)], &M);
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
			int offset = Nk / NTHREADS_FOR_COL_COPY;
			int leftovers;

			// this means that if a single thread needs to copy less than 25 elements, 
			// we shift it to a sequental way (this is the case when Nk < 25*8 = 200)
			// Nk < 200 -> sequential
			// Nk >= 200 -> parallel
			if(offset < SEQ_TRESHOLD_FOR_COL_COPY) leftovers = Nk;
			else leftovers = Nk % offset;

			if(offset >= SEQ_TRESHOLD_FOR_COL_COPY){
				#pragma omp parallel for num_threads(NTHREADS_FOR_COL_COPY)
				for(i = 0; i <= Nk - offset; i += offset){
					zswap_(&offset, &G[k + M*(k + i)], &M, &G[first_non_zero_idx + M*(k + i)], &M);
				}
			}

			zswap_(&leftovers, &G[k + M*(N - leftovers)], &M, &G[first_non_zero_idx + M*(N - leftovers)], &M);
		}


		int np = 0;	// number of 1 in J[k:M, k:M] 
		int nn = 0;	// number of -1 in J[k:M, k:M]

		// update the signum arrays
		if( J[k] < 0) n[nn++] = k;
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
			int offset = Nk / NTHREADS_FOR_COL_COPY;
			int leftovers;

			// this means that if a single thread needs to copy less than 25 elements, 
			// we shift it to a sequental way (this is the case when Nk < 25*8 = 200)
			// Nk < 200 -> sequential
			// Nk >= 200 -> parallel
			if(offset < SEQ_TRESHOLD_FOR_COL_COPY) leftovers = Nk;
			else leftovers = Nk % offset;

			if(offset >= SEQ_TRESHOLD_FOR_COL_COPY){
				#pragma omp parallel for num_threads(NTHREADS_FOR_COL_COPY)
				for(j = 0; j <= Nk - offset; j += offset){
					zswap_(&offset, &G[k + 1 + M*(k + j)], &M, &G[i + M*(k + j)], &M);
				}
			}

			zswap_(&leftovers, &G[k + 1 + M*(N - leftovers)], &M, &G[i + M*(N - leftovers)], &M);
			break;
		}

		// update the signum arrays
		if( J[k+1] < 0) n[nn++] = k+1;
		else p[np++] = k+1;

		// fill and count the signums (not necessary to be ordered)
		#pragma omp parallel for
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

		#pragma omp parallel num_threads(2)
		{

			// first thread kills positives
			if(omp_get_thread_num() == 0){
				int offset;
				for(offset = 1; offset < np; offset *= 2){

					int nthreads = np/(2*offset);
					if ( nthreads > omp_get_max_threads()) nthreads = omp_get_max_threads();

					#pragma omp parallel for num_threads( nthreads )
					for(i = 0; i < np - offset; i += 2*offset){

						// G[p[i], k] destroys G[p[i+offset], k]

						double c;
						double complex s;
						double complex eliminator = G[p[i] + M*k];
						double complex eliminated = G[p[i + offset] + M*k];
						zrotg_(&eliminator, &eliminated, &c, &s);

						// apply the rotation
						int Nk = N-k;
						zrot_(&Nk, &G[p[i] + M*k], &M, &G[p[i + offset] + M*k], &M, &c, &s);
						G[p[i + offset] + M*k] = 0;
					}
				}
			}

			// second thread kills negatives
			else{
				int offset;
				for(offset = 1; offset < nn; offset *= 2){

					int nthreads = nn/(2*offset);
					if ( nthreads > omp_get_max_threads()) nthreads = omp_get_max_threads();

					#pragma omp parallel for num_threads( nthreads )
					for(i = 0; i < nn - offset; i += 2*offset){

						// G[p[i], k] destroys G[p[i+offset], k]

						double c;
						double complex s;
						double complex eliminator = G[n[i] + M*k];
						double complex eliminated = G[n[i + offset] + M*k];
						zrotg_(&eliminator, &eliminated, &c, &s);

						// apply the rotation
						int Nk = N-k;
						zrot_(&Nk, &G[n[i] + M*k], &M, &G[n[i + offset] + M*k], &M, &c, &s);
						G[n[i + offset] + M*k] = 0;
					}
				}
			}
		}


		int kth_nonzeros = 2;
		if(np == 0 || nn == 0) kth_nonzeros = 1;	// just one of them is 0. at this point one od them is nonzero
													// if not, the program would exit with -4 (before this point)
													// then A is maybe singular?



		// do the same thing on a SECOND COLUMN
		// wee need to know if the kth column has 0 or 1 or 2 nonzeros elements
		// that determines in which case we are... (A1), (A2), (A3), (B1) or (B2)


		// making the (k+1)th column real

		first_non_zero_idx = -1;	// index of the first non zero element in column k+1
									// will be filled with the first nonzero index, or stay -1

		#pragma omp parallel for
		for(i = k + kth_nonzeros; i < M; ++i){

			if(cabs(G[i+M*(k+1)]) < EPSILON) continue;
			if(first_non_zero_idx == -1) first_non_zero_idx = i;

			double complex scal = conj(G[i+M*(k+1)]) / cabs(G[i+M*(k+1)]);
			G[i+M*(k+1)] = cabs(G[i+M*(k+1)]);	// to be exact, so that the Img part is really = 0
			int Nk = N - k - 2;
			zscal_(&Nk, &scal, &G[i+M*(k+2)], &M);
		}

	
		if(first_non_zero_idx == -1){	// we have the needed form alredy, continue to other columns
			k = k + 1;
			goto LOOP_END;
		}

		int kkth_nonzeros = 1;	// number of nonzero elements in the (k+1)st column, but those below kth_nonzeros
								// at least one, because we didnt exit in the previous if
 

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
			int offset = Nk / NTHREADS_FOR_COL_COPY;
			int leftovers;

			// this means that if a single thread needs to copy less than 25 elements, 
			// we shift it to a sequental way (this is the case when Nk < 25*8 = 200)
			// Nk < 200 -> sequential
			// Nk >= 200 -> parallel
			if(offset < SEQ_TRESHOLD_FOR_COL_COPY) leftovers = Nk;
			else leftovers = Nk % offset;

			if(offset >= SEQ_TRESHOLD_FOR_COL_COPY){
				#pragma omp parallel for num_threads(NTHREADS_FOR_COL_COPY)
				for(i = 0; i <= Nk - offset; i += offset){
					zswap_(&offset, &G[k + kth_nonzeros + M*(k + 1 + i)], &M, &G[first_non_zero_idx + M*(k + 1 + i)], &M);
				}
			}

			zswap_(&leftovers, &G[k + kth_nonzeros + M*(N - leftovers)], &M, &G[first_non_zero_idx + M*(N - leftovers)], &M);
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
			int offset = Nk / NTHREADS_FOR_COL_COPY;
			int leftovers;

			// this means that if a single thread needs to copy less than 25 elements, 
			// we shift it to a sequental way (this is the case when Nk < 25*8 = 200)
			// Nk < 200 -> sequential
			// Nk >= 200 -> parallel
			if(offset < SEQ_TRESHOLD_FOR_COL_COPY) leftovers = Nk;
			else leftovers = Nk % offset;

			if(offset >= SEQ_TRESHOLD_FOR_COL_COPY){
				#pragma omp parallel for num_threads(NTHREADS_FOR_COL_COPY)
				for(j = 0; j <= Nk - offset; j += offset){
					zswap_(&offset, &G[k + kth_nonzeros + 1 + M*(k + 1 + j)], &M, &G[i + M*(k + 1 + j)], &M);
				}
			}

			zswap_(&leftovers, &G[k + kth_nonzeros + 1 + M*(N - leftovers)], &M, &G[i + M*(N - leftovers)], &M);
			break;
		}


		// update the kkth_nonzeros
		if(first_non_zero_idx != -1) kkth_nonzeros = 2;


		// update the signum arrays needed for reduction
		if(J[k + kth_nonzeros + 1] < 0) n[nn++] = k + kth_nonzeros + 1;
		else p[np++] = k + kth_nonzeros + 1;


		// fill and count the signums (not necessary to be ordered)
		#pragma omp parallel for
		for(int i = k + kth_nonzeros + 2; i < M; ++i){

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

		#pragma omp parallel num_threads(2)
		{

			// first thread kills positives
			if(omp_get_thread_num() == 0){
				int offset;
				for(offset = 1; offset < np; offset *= 2){

					int nthreads = np/(2*offset);
					if ( nthreads > omp_get_max_threads()) nthreads = omp_get_max_threads();

					#pragma omp parallel for num_threads( nthreads )
					for(i = 0; i < np - offset; i += 2*offset){

						// G[p[i], k+1] destroys G[p[i+offset], k+1]

						double c;
						double complex s;
						double complex eliminator = G[p[i] + M*(k+1)];
						double complex eliminated = G[p[i + offset] + M*(k+1)];
						zrotg_(&eliminator, &eliminated, &c, &s);

						// apply the rotation
						int Nk = N - k - 1;
						zrot_(&Nk, &G[p[i] + M*(k+1)], &M, &G[p[i + offset] + M*(k+1)], &M, &c, &s);
						G[p[i + offset] + M*(k+1)] = 0;
					}
				}
			}

			// second thread kills negatives
			else{
				int offset;
				for(offset = 1; offset < nn; offset *= 2){

					int nthreads = nn/(2*offset);
					if ( nthreads > omp_get_max_threads()) nthreads = omp_get_max_threads();

					#pragma omp parallel for num_threads( nthreads )
					for(i = 0; i < nn - offset; i += 2*offset){

						// G[n[i], k+1] destroys G[n[i+offset], k+1]

						double c;
						double complex s;
						double complex eliminator = G[n[i] + M*(k+1)];
						double complex eliminated = G[n[i + offset] + M*(k+1)];
						zrotg_(&eliminator, &eliminated, &c, &s);

						// apply the rotation
						int Nk = N - k - 1;
						zrot_(&Nk, &G[n[i] + M*(k+1)], &M, &G[n[i + offset] + M*(k+1)], &M, &c, &s);
						G[n[i + offset] + M*(k+1)] = 0;
					}
				}
			}
		}


		// -------- check forms od 2x2 pivot ---------


		// if we are in (A3) or (B2) forms, then the 2x2 reduction is finished
		// in that case continue the main loop

		// condition (A3)
		if(kth_nonzeros == 2 && kkth_nonzeros == 0) goto LOOP_END;

		// condition (B2)
		if(kth_nonzeros == 1 && kkth_nonzeros == 1) goto LOOP_END;


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

				int n = k + 4;
				int inc = 1;
				zswap_(&n, &G[k+M*k], &inc, &G[k+M*(k+1)], &inc);

				// make the kth rows k, k+1 real (k+3 and k+2 are already real)

				for(int i = k; i < k+2; ++i){

					if( cabs(cimag(G[i+M*k])) < EPSILON ) continue; //the element is already real

					double complex scal = conj(G[i+M*k]) / cabs(G[i+M*k]);
					G[i+M*k] = cabs(G[i+M*k]);	// to be exact, so that the Img part is really = 0
					int Nk = N - k - 1;
					zscal_(&Nk, &scal, &G[i+M*(k+1)], &M);	
				}

				// do plane rotations with kth row

				int idx = k+2; 	// idx is the row that will be eliminated with the kth row
				if(J[k] == J[k+3]) idx = k+3;

				// generate plane rotation
				double c;
				double complex s;
				double complex eliminator = G[k+M*k];
				double complex eliminated = G[idx+M*k];
				zrotg_(&eliminator, &eliminated, &c, &s);

				// apply the rotation
				int Nk = N-k;
				zrot_(&Nk, &G[k+M*k], &M, &G[idx+M*k], &M, &c, &s);
				G[idx+M*k] = 0;

				// do plane rotations with (k+1)st row

				idx = k+2; 	// idx it the row that will be eliminated with the kth row
				if(J[k+1] == J[k+3]) idx = k+3;

				// generate plane rotation
				eliminator = G[k+1+M*k];
				eliminated = G[idx+M*k];
				zrotg_(&eliminator, &eliminated, &c, &s);

				// apply the rotation
				Nk = N-k;
				zrot_(&Nk, &G[k+1+M*k], &M, &G[idx+M*k], &M, &c, &s);
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
			double complex a = J[k] * J[k+2] * cabs(z*z) * (g21*g21 - g11*g11) * (g32*g32 - g42*g42);
			double complex r = 1.0 / csqrt(1+a);
			double complex y = J[k] * J[k+2] * r * conj(z);

			double complex i_xy = (J[k] * J[k+2] * r * cabs(z*z) * (g21*g21 - g11*g11)) / (1 + csqrt(1+a));
			double complex i_yx = (J[k] * J[k+2] * r * cabs(z*z) * (g32*g32 - g42*g42)) / (1 + csqrt(1+a));

			// making the matrix U dimensions 4x4
			// first column
			U[0] = 1 - i_yx * g21 * g21;
			U[1] = -i_yx * g11 * g21;
			U[2] = r * z * g21 * g32;
			U[3] = r * z * g21 * g42;
			// second column
			U[4] = i_yx * g11 * g21;
			U[5] = 1 + i_yx * g11 * g11;
			U[6] = -r * z * g11 * g32;
			U[7] = -r * z * g11 * g42;
			// third column
			U[8] = -y * g21 * g32;
			U[9] = -y * g11 * g32;
			U[10] = 1 - i_xy * g32 * g32;
			U[11] = -i_xy * g32 * g42;
			// fourth column
			U[12] = y * g21 * g42;
			U[13] = y * g11 * g42;
			U[14] = i_xy * g32 * g42;
			U[15] = 1 + i_xy * g42 * g42;

			// apply the reduction

			int n = 4;
			int Nk = N - k;
			char non_trans = 'N';
			double complex alpha = 1;
			double complex beta = 0;
			zgemm_(&non_trans, &non_trans, &n, &Nk, &n, &alpha, U, &n, &G[k+M*k], &M, &beta, T, &n);


			// copy rows of T back into G
			for(int i = 0; i < 4; ++i) zcopy_(&Nk, &T[i], &n, &G[k + i + M*k], &M);

			// put zeros explicitly in the right places
			G[k+2+M*k] = 0;
			G[k+3+M*k] = 0;
			G[k+2+M*(k+1)] = 0;
			G[k+3+M*(k+1)] = 0;

			k = k+1;

			double end2 = omp_get_wtime();
			pivot2time += (double) (end2 - start2);

			goto LOOP_END;
		}


		// handle forms (A2) and (B1)

		else if(kth_nonzeros == 2 && kkth_nonzeros == 1 || kth_nonzeros == 1 && kkth_nonzeros == 2){

			// check if its a proper form
			// if not, fix it
			// treba li na drugi nacin izracunat determinanticu??

			//printf("\tPIVOT (A2) or (B1)\n");

			// (B1) is already in proper form, so fix just form (A2) if needed

			if( kth_nonzeros == 2 && cabs(G[k+M*k] * G[k+1+M*(k+1)] - G[k+1+M*k] * G[k+M*(k+1)]) < EPSILON ){

				// swap columns k <-> k+1

				long int itemp = Pcol[k];
				Pcol[k] = Pcol[k+1];
				Pcol[k+1] = itemp;

				int n = k + 3;
				int inc = 1;
				zswap_(&n, &G[k+M*k], &inc, &G[k+M*(k+1)], &inc);

				// make the kth rows k, k+1 real (k+2 is already real)

				for(i = k; i < k+2; ++i){

					if( cabs(cimag(G[i+M*k])) < EPSILON ) continue; //the element is already real

					double complex scal = conj(G[i+M*k]) / cabs(G[i+M*k]);
					G[i+M*k] = cabs(G[i+M*k]);	// to be exact, so that the Img part is really = 0
					int Nk = N - k - 1;
					zscal_(&Nk, &scal, &G[i+M*(k+1)], &M);	
				}

				// do a plane rotation, eliminate row k+2 with row idx

				int idx = k; 	// idx is the row that will eliminate row k+2
				if(J[idx] != J[k+2]) idx = k+1;

				// generate plane rotation
				double c;
				double complex s;
				double complex eliminator = G[idx+M*k];
				double complex eliminated = G[k+2+M*k];
				zrotg_(&eliminator, &eliminated, &c, &s);

				// apply the rotation
				int Nk = N-k;
				zrot_(&Nk, &G[idx+M*k], &M, &G[k+2+M*k], &M, &c, &s);
				G[k+2+M*k] = 0;
			}


			// now do the final reduction

			double complex g11 = G[k+M*k];
			double complex g21 = G[k+1+M*k];
			double complex g12 = G[k+M*(k+1)];
			double complex g22 = G[k+1+M*(k+1)];
			double complex g32 = G[k+2+M*(k+1)];

			double complex z = 1 / (g11 * g22 - g21 * g12);	// z = 1/detG1
			double complex a = J[k] * J[k+2] * cabs(z*z) * (g21*g21 - g11*g11) * g32*g32;
			double complex r = 1/csqrt(1+a);
			double complex y = J[k] * J[k+2] * r * conj(z);

			double complex i_yx = (J[k] * J[k+2] * r * cabs(z*z) * g32*g32) / (1 + csqrt(1+a));

			// making the matrix U dimensions 3x3
			// first column
			U[0] = 1 - i_yx * g21 * g21;
			U[1] = -i_yx * g11 * g21;
			U[2] = r * z * g21 * g32;
			// second column
			U[3] = i_yx * g11 * g21;
			U[4] = 1 + i_yx * g11 * g11;
			U[5] = -r * z * g11 * g32;
			// third column
			U[6] = -y * g21 * g32;
			U[7] = -y * g11 * g32;
			U[8] = r;

			// apply the reduction
			// g11 and g21 remain unchanged

			int n = 3;
			int Nk = N - k - 1;
			char non_trans = 'N';
			double complex alpha = 1;
			double complex beta = 0;
			zgemm_(&non_trans, &non_trans, &n, &Nk, &n, &alpha, U, &n, &G[k+M*(k+1)], &M, &beta, T, &n);


			// copy rows of T back into G

			for(int i = 0; i < 3; ++i) zcopy_(&Nk, &T[i], &n, &G[k + i + M*(k+1)], &M);

			// put zeros explicitly in the right places
			G[k+2+M*k] = 0;
			G[k+2+M*(k+1)] = 0;

			k = k+1;

			double end2 = omp_get_wtime();
			pivot2time += (double) (end2 - start2);

			goto LOOP_END;
		}

		// if here, something broke down
		printf("\n\n\nUPS, SOMETHING WENT WRONG... STOPPING...Is A maybe singular?\n\n\n");
		exit(-4);
	
		// ----------------------------------------------PIVOT_1-----------------------------------------------------
		PIVOT_1: 

		pivot_1_count += 1;
		double start1 = omp_get_wtime();

		// check the condition sign(Akk) = Jk
		// if not, do row swap and diagonal swap in J

		//printf("PIVOT_1, k = %d\n", k);

		if( Akk > 0 && J[k] < 0){

			int i;
			for(i = k+1; i < M; ++i)
				if(J[i] > 0) break;

			J[k] = 1.0;
			J[i] = -1.0;

			// update Prow
			long int itemp = Prow[i];
			Prow[i] = Prow[k];
			Prow[k] = itemp;

			// swap rows in G 
			int Nk = N - k;
			int offset = Nk / NTHREADS_FOR_COL_COPY;
			int leftovers;

			// this means that if a single thread needs to copy less than 25 elements, 
			// we shift it to a sequental way (this is the case when Nk < 25*8 = 200)
			// Nk < 200 -> sequential
			// Nk >= 200 -> parallel
			if(offset < SEQ_TRESHOLD_FOR_COL_COPY) leftovers = Nk;
			else leftovers = Nk % offset;

			if(offset >= SEQ_TRESHOLD_FOR_COL_COPY){
				#pragma omp parallel for num_threads(NTHREADS_FOR_COL_COPY)
				for(j = 0; j <= Nk - offset; j += offset){
					zswap_(&offset, &G[k + M*(k + j)], &M, &G[i + M*(k + j)], &M);
				}
			}

			zswap_(&leftovers, &G[k + M*(N - leftovers)], &M, &G[i + M*(N - leftovers)], &M);
		}

		else if( Akk < 0 && J[k] > 0){

			for(i = k+1; i < M; ++i)
				if(J[i] < 0) break;

			J[k] = -1.0;
			J[i] = 1.0;


			// update Prow
			long int itemp = Prow[i];
			Prow[i] = Prow[k];
			Prow[k] = itemp;

			// swap rows in G 
			int Nk = N - k;
			int offset = Nk / NTHREADS_FOR_COL_COPY;
			int leftovers;

			// this means that if a single thread needs to copy less than 25 elements, 
			// we shift it to a sequental way (this is the case when Nk < 25*8 = 200)
			// Nk < 200 -> sequential
			// Nk >= 200 -> parallel
			if(offset < SEQ_TRESHOLD_FOR_COL_COPY) leftovers = Nk;
			else leftovers = Nk % offset;

			if(offset >= SEQ_TRESHOLD_FOR_COL_COPY){
				#pragma omp parallel for num_threads(NTHREADS_FOR_COL_COPY)
				for(j = 0; j <= Nk - offset; j += offset){
					zswap_(&offset, &G[k + M*(k + j)], &M, &G[i + M*(k + j)], &M);
				}
			}

			zswap_(&leftovers, &G[k + M*(N - leftovers)], &M, &G[i + M*(N - leftovers)], &M);
		}
		

		// compute reflector constant H_sigma
		// compute vector f, where g*Jg = f*Jf must hold
		// that's why we need fk that looks like Hg = H_sigma*f = H_sigma*(sqrt(|sumk|), 0, ..., 0)

		double complex H_sigma = 1;
		if(cabs(G[k+M*k]) >= EPSILON) H_sigma = -G[k+M*k] / cabs(G[k+M*k]);
		f[k] = csqrt(cabs(Akk)) * H_sigma;

		#pragma omp parallel for
		for(i = k+1; i < M; ++i) f[i] = 0;


		// make the vector f(k:M)
		// copy f into tempf, so we dont need tu multyply the first column of G with H
		// and do f(k:M) = f(k:M) - g(k:M)

		double complex alpha = -1;
		int inc = 1;
		int Mk = M - k;
		int offset = Mk / NTHREADS_FOR_COL_COPY;
		int leftovers;
		if(offset < SEQ_TRESHOLD_FOR_COL_COPY) leftovers = Mk;
		else leftovers = Mk % offset;

		if(offset >= SEQ_TRESHOLD_FOR_COL_COPY){
			int j = 0;
			#pragma omp parallel for num_threads(NTHREADS_FOR_COL_COPY)
			for(j = 0; j <= Mk - offset; j += offset){
				zcopy_(&offset, &f[k + j], &inc, &tempf[k + j], &inc);	// copy f into tempf
				zaxpy_(&offset, &alpha, &G[k + j + M*k], &inc, &f[k + j], &inc);	// f(k:M) = f(k:M) - g(k:M)
			}
		}
		zcopy_(&leftovers, &f[M - leftovers], &inc, &tempf[M - leftovers], &inc);
		zaxpy_(&leftovers, &alpha, &G[M - leftovers + M*k], &inc, &f[M - leftovers], &inc);


		// constant needed to make the reflector H
		double complex wJw = Akk + J[k] * (cabs(Akk) + 2 * csqrt(cabs(Akk)) * cabs(G[k+M*k]));


		// make the reflector

		#pragma omp parallel for collapse(2)
		for(int i = k; i < M; ++i)
			for(int j = k; j < M; ++j)
				H[i+M*j] = -2 * f[i] * conj(f[j]) * J[j] / wJw;
			
		#pragma omp parallel for
		for(int i = k; i < M; ++i) H[i+M*i] += 1;


		// apply the reflector on a submatrix

		char non_trans = 'N';
		alpha = 1.0;
		double complex beta = 0;
		int Nk = N - k - 1;

		offset = Mk / NTHREADS_FOR_COL_COPY;
		if(offset < SEQ_TRESHOLD_FOR_COL_COPY) leftovers = Mk;
		else leftovers = Mk % offset;

		if(offset >= SEQ_TRESHOLD_FOR_COL_COPY){

			#pragma omp parallel for num_threads(NTHREADS_FOR_COL_COPY)
			for(int i = 0; i <= Mk - offset; i += offset){

				#pragma omp parallel for
				for(int j = 0; j < Nk; ++j){
					zgemv_(&non_trans, &offset, &Mk, &alpha, &H[k + i + M*k], &M, &G[k + M*(k+j+1)], &inc, &beta, &T[i+j*Mk], &inc);
				}
			}
		}

	
		#pragma omp parallel for
		for(j = 0; j < Nk; ++j){
			zgemv_(&non_trans, &leftovers, &Mk, &alpha, &H[k + Mk - leftovers + M*k], &M, &G[k + M*(k+j+1)], &inc, &beta, &T[Mk - leftovers + j*Mk], &inc);
		}


		// copy back things from T to G

		inc = 1;
		Mk = M - k;
		offset = Mk / NTHREADS_FOR_COL_COPY;
		if(offset < SEQ_TRESHOLD_FOR_COL_COPY) leftovers = Mk;
		else leftovers = Mk % offset;

		if(offset >= SEQ_TRESHOLD_FOR_COL_COPY){
			#pragma omp parallel for num_threads(NTHREADS_FOR_COL_COPY)
			for(j = 0; j <= Mk - offset; j += offset){
				zcopy_(&offset, &tempf[k + j], &inc, &G[k + j + M*k], &inc);
			}
		}
		zcopy_(&leftovers, &tempf[M - leftovers], &inc, &G[M - leftovers + M*k], &inc);


		// G = T (copy columns)
		#pragma omp parallel for
		for(j = 0; j < Nk; ++j) zcopy_(&Mk, &T[j*Mk], &inc, &G[k + M*(j+k+1)], &inc);	

		double end1 = omp_get_wtime();
		pivot1time += (double)(end1 - start1);

		LOOP_END: continue;
	}

	end = omp_get_wtime();
	seconds = (double)(end - start);
	printf("algorithm time = %lg s\n", seconds);
	printf("PIVOT_1 (%d)	time = %lg s (%lg %%)\n", pivot_1_count, pivot1time, pivot1time / seconds * 100);
	printf("PIVOT_2 (%d)	time = %lg s (%lg %%)\n", pivot_2_count, pivot2time, pivot2time / seconds * 100);


	// -------------------------------- writing -------------------------------- 	

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

	end = omp_get_wtime();
	seconds = (double)(end - start);
	printf("writing time = %lg s\n", seconds);


	// ------------------------------- cleaning -------------------------------

	fclose(readG);
	fclose(readJ);
	fclose(writeG);
	fclose(writeJ);
	fclose(writeCol);
	fclose(writeRow);
	
	free(Prow);
	free(Pcol);
	free(G);
	free(J);
	free(T);
	free(H);
	free(f);
	free(tempf);
	free(U);
	free(p);
	free(n);


	//printf("\nFinished.\n");
	//printf("\n-------------------------------------------------------------------------------\n\n");
	return(0);
}