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
#define D 64


void printMatrix(double complex *G, int M, int N){

	int i, j;
	for( i = 0; i < M; ++i ){
		for( j = 0; j < N; ++j ){
			printf("%15.10g + i%15.10g  ", creal(G[i+M*j]), cimag(G[i+M*j]));
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
	double complex *v = (double complex*) mkl_malloc(M*N*sizeof(double complex), 64);	// reflector vectors
	int *t = (int*) mkl_malloc(N*sizeof(int), 64);		// which transformations on which column were applied
	double complex *vJv = (double complex*) mkl_malloc(M*sizeof(double complex), 64);	// reflector vector J norms
	double *J = (double*) mkl_malloc(M*sizeof(double), 64);
	long int *Prow = (long int*) mkl_malloc(M*sizeof(long int), 64);	// for row permutation
	long int *Pcol = (long int*) mkl_malloc(N*sizeof(long int), 64);	// for column permutation
	double complex *f = (double complex*) mkl_malloc(M*sizeof(double complex), 64);	// vector f


	// check if files are opened

	if(readG == NULL || readJ == NULL){
		printf("Cannot open file.\n");
		exit(-1);
	}

	// check if memory is allocated

	if(G == NULL || v == NULL || t == 0 || vJv == NULL || J == NULL || Pcol == NULL || Prow == NULL || f == NULL ){
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
				t[j] = 0;
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
	int pivot_1_count = 0;
	int pivot_2_count = 0;
	int last_pivot = -1;
	start = omp_get_wtime();

	int i, j, k, nthreads, mkl_nthreads;

	for(k = 0; k < N; ++k){

		// TODO: refreashaj normu ako jako padne ispod pocetne....

		// ------------------------ choosing a pivoting strategy (partial pivoting) -------------------------------
		
		double pp = omp_get_wtime();
		double pivot_lambda = 0;
		double pivot_sigma = 0;
		int pivot_r = -1;	// 2nd column for partial pivoting
							// will be used for column swap k+1 <-> pivot_r when PIVOT_2 begins


		// first apply the previous rotations
		// [SEQUENTIAL] outer loop
		for(i = t[k]; i < k; ++i){

			if(k == 2){
			printf("v%d = \n", i);
			printMatrix(&v[i+M*i], M-i, 1);
			printf("g = \n");
			printMatrix(&G[i+M*k], M-i, 1);}

			#pragma omp parallel for
			for(j = i; j < M; ++j) f[j] = J[j] * G[j+M*k];

			double complex alpha;
			int Mi = M - i;
			int inc = 1;

			mkl_nthreads = Mi/D > mkl_get_max_threads() ? Mi/D : mkl_get_max_threads();
			if(Mi/D == 0) mkl_nthreads = 1;
			mkl_set_num_threads(mkl_nthreads);

			zdotc(&alpha, &Mi, &v[i + M*i], &inc, &f[i], &inc);
			alpha = - 2 * alpha / vJv[i];

			zaxpy(&Mi, &alpha, &v[i + M*i], &inc, &G[i + M*k], &inc);	// G[i + M*k] = alpha * v[i + M*i] + G[i + M*k]
		}

		if( t[k] < k ) t[k] = k;

		#pragma omp parallel for
		for(i = k; i < M; ++i) f[i] = J[i] * G[i+M*k];

		double complex akk;
		int Mk = M - k;
		int inc = 1;
		mkl_nthreads = Mk/D > mkl_get_max_threads() ? Mk/D : mkl_get_max_threads();
		if(Mk/D == 0) mkl_nthreads = 1;
		mkl_set_num_threads(mkl_nthreads);
		zdotc(&akk, &Mk, &G[k+M*k], &inc, &f[k], &inc);

		double Akk = (double) akk;

		// DELETE
		printf("A%d = %lg\n", k, Akk);
		goto PIVOT_1;

		if(k == N-1) goto PIVOT_1;

		// ------------------------ find pivot_lambda ------------------------


		nthreads = (N-k)/D > omp_get_max_threads() ? (N-k)/D : omp_get_max_threads();
		if ((N-k)/D == 0) nthreads = 1;

		#pragma omp parallel num_threads( nthreads )
		{
			#pragma omp for nowait
			for(i = k+1; i < N; ++i){

				double complex Aik = 0;
				int Mk = M-k;
				int inc = 1;
				mkl_nthreads = mkl_get_max_threads() / nthreads;
				if(mkl_nthreads == 0) mkl_nthreads = 1;
				mkl_set_num_threads_local( mkl_nthreads );
				zdotc(&Aik, &Mk, &G[k+M*i], &inc, &f[k], &inc); //Aik = gi* J gk, but on a submatrix G[k:M, k:N]
				
				#pragma omp critical
				if(pivot_lambda < cabs(Aik)){
					pivot_lambda = cabs(Aik);
					pivot_r = i;
				}
			}
		}
		mkl_set_num_threads_local(0);

		if(cabs(Akk) >= ALPHA * pivot_lambda) goto PIVOT_1;


		// ------------------------ find pivot_sigma ------------------------

		// first apply previous rotation
		// [SEQUENTIAL] outer loop
		for(i = t[pivot_r]; i < k; ++i){

			#pragma omp parallel for
			for(j = i; j < M; ++j) f[j] = J[j] * G[j+M*pivot_r];

			double complex alpha;
			int Mi = M - i;
			int inc = 1;

			mkl_nthreads = Mi/D > mkl_get_max_threads() ? Mi/D : mkl_get_max_threads();
			if(Mi/D == 0) mkl_nthreads = 1;
			mkl_set_num_threads(mkl_nthreads);

			zdotc(&alpha, &Mi, &v[i + M*i], &inc, &f[i], &inc);
			alpha = - 2 * alpha / vJv[i];

			zaxpy(&Mi, &alpha, &v[i + M*i], &inc, &G[i + M*pivot_r], &inc);	// G[i + M*r] = alpha * v[i + M*i] + G[i + M*r]
		}

		if( t[pivot_r] < k ) t[pivot_r] = k;

		#pragma omp parallel for
		for(i = k; i < M; ++i) f[i] = J[i] * G[i+M*pivot_r];

		nthreads = (N-k)/D > omp_get_max_threads() ? (N-k)/D : omp_get_max_threads();
		if ((N-k)/D == 0) nthreads = 1;

		#pragma omp parallel for reduction(max:pivot_sigma) num_threads( nthreads ) 
		for(i = k; i < N; ++i){

			if(i == pivot_r) continue;

			double complex Air = 0;
			int Mk = M-k;
			int inc = 1;
			mkl_nthreads = mkl_get_max_threads() / nthreads;
			if(mkl_nthreads == 0) mkl_nthreads = 1;
			mkl_set_num_threads_local( mkl_nthreads );
			zdotc(&Air, &Mk, &G[k+M*i], &inc, &f[k], &inc);

			if(pivot_sigma < cabs(Air)) pivot_sigma = cabs(Air);
		}
		mkl_set_num_threads_local(0);	//return global value

		if(cabs(Akk) * pivot_sigma >= ALPHA * pivot_lambda * pivot_lambda) goto PIVOT_1;

		double complex arr;
		Mk = M - k;
		inc = 1;
		mkl_nthreads = Mk/D > mkl_get_max_threads() ? Mk/D : mkl_get_max_threads();
		if(Mk/D == 0) mkl_nthreads = 1;
		mkl_set_num_threads(mkl_nthreads);
		zdotc(&akk, &Mk, &G[k+M*pivot_r], &inc, &f[k], &inc);

		double Arr = (double) arr;

		if(cabs(Arr) >= ALPHA * pivot_sigma){

			long int itemp = Pcol[pivot_r];
			Pcol[pivot_r] = Pcol[k];
			Pcol[k] = itemp;

			itemp = t[pivot_r];
			t[pivot_r] = t[k];
			t[k] = itemp;

			printf("RADIM Pcol\n");

			int inc = 1;
			mkl_nthreads = M/D > mkl_get_max_threads() ? M/D : mkl_get_max_threads(); 
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

			itemp = t[pivot_r];
			t[pivot_r] = t[k];
			t[k] = itemp;

			int inc = 1;
			mkl_nthreads = M/D > mkl_get_max_threads() ? M/D : mkl_get_max_threads();
			if(M/D == 0) mkl_nthreads = 1;
			mkl_set_num_threads(mkl_nthreads);
			zswap(&M, &G[M*pivot_r], &inc, &G[M*(k+1)], &inc);
		}

		pivot2time += (double) (omp_get_wtime() - start2);
		goto LOOP_END;

	


		// ----------------------------------------------PIVOT_1----------------------------------------------------

		PIVOT_1: 

		last_pivot = 1;
		pivotiranje = pivotiranje + omp_get_wtime() - pp;
		pivot_1_count += 1;
		double start1 = omp_get_wtime();


		// first apply the previous rotations
		// [SEQUENTIAL] outer loop
		for(i = t[k]; i < k; ++i){

			#pragma omp parallel for
			for(j = i; j < M; ++j) f[j] = J[j] * G[j+M*k];

			double complex alpha;
			int Mi = M - i;
			int inc = 1;

			mkl_nthreads = Mi/D > mkl_get_max_threads() ? Mi/D : mkl_get_max_threads();
			if(Mi/D == 0) mkl_nthreads = 1;
			mkl_set_num_threads(mkl_nthreads);

			zdotc(&alpha, &Mi, &v[i + M*i], &inc, &f[i], &inc);
			alpha = - 2 * alpha / vJv[i];

			zaxpy(&Mi, &alpha, &v[i + M*i], &inc, &G[i + M*k], &inc);	// G[i + M*k] = alpha * v[i + M*i] + G[i + M*k]
		}

		if( t[k] < k) t[k] = k;


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

			printf("RADIM Prow\n");
			printMatrix(v, M, k-1);

			// swap rows in G 
			int Nk = N - k;
			mkl_nthreads = Nk/D > mkl_get_max_threads() ? Nk/D : mkl_get_max_threads();
			if(Nk/D == 0) mkl_nthreads = 1;
			mkl_set_num_threads(mkl_nthreads);
			zswap(&Nk, &G[k + M*k], &M, &G[i + M*k], &M);

			// swap rows in v 
			int kk = k-1;
			mkl_nthreads = kk/D > mkl_get_max_threads() ? kk/D : mkl_get_max_threads();
			if(kk/D == 0) mkl_nthreads = 1;
			mkl_set_num_threads(mkl_nthreads);
			zswap(&kk, &v[k], &M, &v[i], &M);

			printMatrix(v, M, k-1);
		}

		else if( Akk < 0 && J[k] > 0){

			for(i = k+1; i < M; ++i) if(J[i] < 0) break;

			J[k] = -1.0;
			J[i] = 1.0;


			// update Prow
			long int itemp = Prow[i];
			Prow[i] = Prow[k];
			Prow[k] = itemp;

			printf("RADIM Prow\n");

			// swap rows in G 
			int Nk = N - k;
			mkl_nthreads = Nk/D > mkl_get_max_threads() ? Nk/D : mkl_get_max_threads();
			if(Nk/D == 0) mkl_nthreads = 1;
			mkl_set_num_threads(mkl_nthreads);
			zswap(&Nk, &G[k + M*k], &M, &G[i + M*k], &M);

			printMatrix(v, M, k-1);

			// swap rows in v 
			int kk = k-1;
			mkl_nthreads = kk/D > mkl_get_max_threads() ? kk/D : mkl_get_max_threads();
			if(kk/D == 0) mkl_nthreads = 1;
			mkl_set_num_threads(mkl_nthreads);
			zswap(&kk, &v[k], &M, &v[i], &M);

			printMatrix(v, M, k-1);
		}
		

		// compute reflector constant H_sigma
		// compute vector f, where g*Jg = f*Jf must hold
		// that's why we need fk that looks like Hg = H_sigma*f = H_sigma*(sqrt(|sumk|), 0, ..., 0)

		double complex H_sigma = 1;
		if(cabs(G[k+M*k]) >= EPSILON) H_sigma = -G[k+M*k] / cabs(G[k+M*k]);
		double complex gkk = csqrt(cabs(Akk)) * H_sigma;

		// save the J norm of the vector
		vJv[k] = Akk + J[k] * (cabs(Akk) + 2 * csqrt(cabs(Akk)) * cabs(G[k+M*k]));


		// make the reflector vector and save it
		double complex alpha = -1;
		inc = 1;
		Mk = M - k;
		mkl_nthreads = Mk/D > mkl_get_max_threads() ? Mk/D : mkl_get_max_threads();
		if(Mk/D == 0) mkl_nthreads = 1;
		mkl_set_num_threads(mkl_nthreads);

		zcopy(&Mk, &G[k+M*k], &inc, &v[k+M*k], &inc);
		v[k + M*k] -= gkk;

		nthreads = Mk/D > omp_get_max_threads() ? Mk/D : omp_get_max_threads();
		if (Mk/D == 0) nthreads = 1;

		// update G
		G[k + M*k] = gkk;
		#pragma omp parallel for num_threads(nthreads)
		for(i = k+1; i < M; ++i) G[i + M*k] = 0;
	
		pivot1time += (double)(omp_get_wtime() - start1);
		LOOP_END: continue;

	}	// END OF MAIN LOOP


	// ----------------------------------------- PRINT TIMEs -----------------------------------------

	seconds = (double)(omp_get_wtime() - start);
	printf("algorithm time = %lg s\n", seconds);
	printf("PIVOT_1 (%d)	time = %lg s (%lg %%)\n", pivot_1_count, pivot1time, pivot1time / seconds * 100);
	printf("PIVOT_2 (%d)	time = %lg s (%lg %%)\n", pivot_2_count, pivot2time, pivot2time / seconds * 100);
	printf("pivotiranje time = %lg s (%lg %%)\n", pivotiranje, pivotiranje/seconds * 100);


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
	mkl_free(v);

	printf("\n-------------------------------------------------------------------------------\n\n");
	return(0);
}
