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
#define eps 1e-1
#define D 64
#define refresh 30


void printMatrix(double complex *G, int M, int N){

	int i, j;
	for( i = 0; i < M; ++i ){
		for( j = 0; j < N; ++j ){
			printf("%7.2g + i%7.2g  ", creal(G[i+M*j]), cimag(G[i+M*j]));
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
	int pivot_1_count = 0;
	int pivot_2_count = 0;
	int last_pivot = -1;
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

		//printf("k = %d, G = \n", k);
		//printMatrix(G, M, N);

		// ------------------------ choosing a pivoting strategy (partial pivoting) -------------------------------

		// ------------------------ update J-norms of columns ------------------------

		if( k && ( k % refresh == 0 || (k % refresh == 1 && last_pivot == 2) ) ){	// if we have something to update

			#pragma omp parallel num_threads( nthreads )
			{
				#pragma omp for nowait
				for( j = k; j < N; ++j){

					// pivot 1 was last
					if( last_pivot == 1 ){

						double denomi = conj(G[k-1+M*j]) * J[k-1] * G[k-1+M*j];
						double frac = cabs(norm[j]) / cabs(denomi);

						// not a case of catastrophic cancellation
						if( creal(norm[j]) * denomi < 0 || cabs(frac - 1) < eps )
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
						
						// not a case of catastrophic cancellation
						if( creal(norm[j]) * denomi < 0 || cabs(frac - 1) < eps)
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

			nthreads = (N-k)/D > omp_get_max_threads() ? (N-k)/D : omp_get_max_threads();
			if ((N-k)/D == 0) nthreads = 1;

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

		#pragma omp parallel num_threads( nthreads )
		{
			#pragma omp for nowait
			for(i = k+1; i < N; ++i){

				double complex Aik = 0;
				int Mk = M-k;
				int inc = 1;
				int mkl_nthreads = mkl_get_max_threads() / nthreads;
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
		mkl_set_num_threads_local(0);	//return global value
		
		if(cabs(Akk) >= ALPHA * pivot_lambda) goto PIVOT_1;


		// ------------------------ find pivot_sigma ------------------------

		nthreads = (M-k)/D > omp_get_max_threads() ? (M-k)/D : omp_get_max_threads();
		if ((M-k)/D == 0) nthreads = 1;

		#pragma omp parallel for num_threads( nthreads )
		for(i = k; i < M; ++i)	f[i] = J[i] * G[i+M*pivot_r];

		nthreads = (N-k)/D > omp_get_max_threads() ? (N-k)/D : omp_get_max_threads();
		if ((N-k)/D == 0) nthreads = 1;

		#pragma omp parallel for reduction(max:pivot_sigma) num_threads( nthreads ) 
		for(i = k; i < N; ++i){

			if(i == pivot_r) continue;

			double complex Air = 0;
			int Mk = M-k;
			int inc = 1;
			int mkl_nthreads = mkl_get_max_threads() / nthreads;
			if(mkl_nthreads == 0) mkl_nthreads = 1;
			mkl_set_num_threads_local( mkl_nthreads );
			zdotc(&Air, &Mk, &G[k+M*i], &inc, &f[k], &inc);

			if(pivot_sigma < cabs(Air)) pivot_sigma = cabs(Air);
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

			int inc = 1;
			int mkl_nthreads = M/D > mkl_get_max_threads() ? M/D : mkl_get_max_threads();
			if(M/D == 0) mkl_nthreads = 1;
			mkl_set_num_threads(mkl_nthreads);
			zswap(&M, &G[M*pivot_r], &inc, &G[M*(k+1)], &inc);
		}

		// compute A2
		int Mk = M - k;
		int inc = 1;
		mkl_nthreads = Mk/D > mkl_get_max_threads() ? Mk/D : mkl_get_max_threads();
		if(Mk/D == 0) mkl_nthreads = 1;
		mkl_set_num_threads(mkl_nthreads);
		double complex Akr = 0;
		for(i = k; i < M; ++i) Akr += conj(G[i+M*k]) * J[i] * G[i+M*(k+1)];

		// K = inverse of A2
		double detA = Akk * Arr - cabs(Akr) * cabs(Akr); 

		K[0] = Arr / detA;
		K[1] = -conj(Akr) / detA;
		K[2] = -Akr / detA;
		K[3] = Akk / detA;

		// find pivot G1
		int idx = -1;
		for(i = k+1; i < M; ++i){

			double complex detG1 = G[k+M*k]*G[i+M*(k+1)] - G[k+M*(k+1)]*G[i+M*k];
			if( J[k] == J[i] || cabs(detG1) < EPSILON ) continue;

			// try finding maximal trace for K^2
			// trK^2 = K0 x*J1x + K3 y*J1y + 2*Re( conj(K1) x*J1y )
			// detK^2 = |detG1|^2 / detA
			// x is the first column of G1, y is the second

			double xJx = conj(G[k+M*k]) * J[k] * G[k+M*k] + conj(G[i+M*k]) * J[i] * G[i+M*k];
			double yJy = conj(G[k+M*(k+1)]) * J[k] * G[k+M*(k+1)] + conj(G[i+M*(k+1)]) * J[i] * G[i+M*(k+1)];
			double complex xJy = conj(G[k+M*k]) * J[k] * G[k+M*(k+1)] + conj(G[i+M*k]) * J[i] * G[i+M*(k+1)];
			double trace = K[0] * xJx + K[3] * yJy + 2 * creal( K[1] * xJy );
			double det = -cabs(detG1) * cabs(detG1) / detA;

			//printf("k = %d, trace + 2 * creal(csqrt(det)) = %lg , trace = %lg, det = %lg\n", k, trace + 2 * creal(csqrt(det)), trace, det);
			//printMatrix(K, 2, 2);
			//printf("xJx = %lg, yJy = %lg, xJy = %lg + i %lg\n-----------------\n", xJx, yJy, creal(xJy), cimag(xJy));
		
			// condition that a sqrt exists
			// see: https://www.maa.org/sites/default/files/pdf/cms_upload/Square_Roots-Sullivan13884.pdf
			if(trace + 2 * creal(csqrt(det)) >= 0 ){
				idx = i;
				break;
			}
		}

		if(idx == -1){
			printf("No more altering signs in J or no such G1 for finding a sqrt(K^2) ... (in pivot 2) ... Exiting\n");
			printJ(&J[k], M-k);
			exit(-5);
		}

		// swap rows
		if( idx != k+1 ){

			double dtemp = J[idx];
			J[idx] = J[k+1];
			J[k+1] = dtemp;

			// update Prow
			long int itemp = Prow[idx];
			Prow[idx] = Prow[k+1];
			Prow[k+1] = itemp;

			// swap rows in G 
			int Nk = N - k;
			mkl_nthreads = Nk/D > mkl_get_max_threads() ? Nk/D : mkl_get_max_threads();
			if(Nk/D == 0) mkl_nthreads = 1;
			mkl_set_num_threads(mkl_nthreads);
			zswap(&Nk, &G[k+1 + M*k], &M, &G[idx + M*k], &M);
		}

		int n = 2;
		double complex alpha = 1, beta = 0;
		char nontrans = 'N';
		char trans = 'C';

		mkl_set_num_threads(1);
		zgemm(&nontrans, &nontrans, &n, &n, &n, &alpha, &G[k+M*k], &M, K, &n, &beta, T, &n);	// T = G1 K
		zgemm(&nontrans, &trans, &n, &n, &n, &alpha, T, &n, &G[k+M*k], &M, &beta, K, &n);	// K = T G1^H

		K[0] *= J[k];
		K[1] *= J[k];
		K[2] *= J[k+1];
		K[3] *= J[k+1];

		K[0] = creal(K[0]);
		K[3] = creal(K[3]);

		int kontrola = 0;
		if(kontrola){
			printf("K (kvadrat) = \n");
			printMatrix(K, 2, 2);
		}

		// sqrt(K) = T
		// first solve K^2 = G1 M^(-1) G1* J1
		double detK = (double)(K[0]*K[3] - K[1]*K[2]);
		double trK = (double) (K[0] + K[3]);

		if( cabs(trK * trK - 4 * detK) > EPSILON ){

			double a = creal(csqrt(trK + 2 * csqrt(detK)));

			T[0] = (K[0] + csqrt(detK)) / a;
			T[1] = K[1] / a;
			T[2] = K[2] / a;
			T[3] = (K[3] + csqrt(detK)) / a;
		}
		else{

			double a;

			if(trK < 0) a = creal(csqrt(- 2 * trK));
			else a = creal(csqrt( 2 * trK ));
			
			T[0] = (K[0] + 0.5 * trK) / a;
			T[1] = K[1] / a;
			T[2] = K[2] / a;
			T[3] = (K[3] + 0.5 * trK) / a;
		}

		if(kontrola){
			printf("T = \n");
			printMatrix(T, 2, 2);
		}

		T[0] = creal(T[0]);
		T[3] = creal(T[3]);
		if( creal(T[0] + T[3]) > 0){

			T[0] *= -1.0;
			T[1] *= -1.0;
			T[2] *= -1.0;
			T[3] *= -1.0;
		}

		double detT = creal(T[0]*T[3] - T[1]*T[2]);
		K[0] = T[3] / detT;
		K[1] = -T[1] / detT;
		K[2] = -T[2] / detT;
		K[3] = T[0] / detT;

		if(kontrola){
			printf("T^2 = \n");
			C[0] = T[0]*T[0] + T[2]*T[1];
			C[1] = T[0]*T[1] + T[3]*T[1];
			C[2] = T[0]*T[2] + T[2]*T[3];
			C[3] = T[1]*T[2] + T[3]*T[3];
			printMatrix(C, 2, 2);
		}

		n = 2;
		alpha = 1, beta = 0;
		nontrans = 'N';
		mkl_set_num_threads(1);
		zgemm(&nontrans, &nontrans, &n, &n, &n, &alpha, K, &n, &G[k+M*k], &M, &beta, T, &n);	// T = K G1


		if(kontrola){

			printf("trK = %lg, detK = %lg\n", trK, detK);
			printf("detA = %lg\n", detA);
			double complex detG1 = G[k+M*k]*G[k+1+M*(k+1)] - G[k+1+M*k]*G[k+M*(k+1)];
			printf("detG1 = %lg + i%lg\n", creal(G[k+M*k]*G[k+1+M*(k+1)] - G[k+1+M*k]*G[k+M*(k+1)]), cimag(G[k+M*k]*G[k+1+M*(k+1)] - G[k+1+M*k]*G[k+M*(k+1)]));
			printf("Jk = %lg, jk+1 = %lg\n", J[k], J[k+1]);

			printf("F1 = \n");
			printMatrix(T, 2, 2);

			printf("F1* J F1 = \n");
			C[0] = J[k]*T[0]*conj(T[0]) + J[k+1]*T[1]*conj(T[1]);
			C[1] = J[k]*T[0]*conj(T[2]) + J[k+1]*T[1]*conj(T[3]);
			C[2] = J[k]*T[2]*conj(T[0]) + J[k+1]*T[3]*conj(T[1]);
			C[3] = J[k]*T[2]*conj(T[2]) + J[k+1]*T[3]*conj(T[3]);
			printMatrix(C, 2, 2);
			
			printf("A2 = \n");
			C[0] = Akk;
			C[1] = conj(Akr);
			C[2] = Akr;
			C[3] = Arr;
			printMatrix(C, 2, 2);
			double complex G0 = G[k+M*k];
			double complex G1 = G[k+1+M*k];
			double complex G2 = G[k+M*(k+1)];
			double complex G3 = G[k+1+M*(k+1)];
			printf("F1* J G1 = \n");
			C[0] = J[k]*G0*conj(T[0]) + J[k+1]*G1*conj(T[1]);
			C[1] = J[k]*G0*conj(T[2]) + J[k+1]*G1*conj(T[3]);
			C[2] = J[k]*G2*conj(T[0]) + J[k+1]*G3*conj(T[1]);
			C[3] = J[k]*G2*conj(T[2]) + J[k+1]*G3*conj(T[3]);
			printMatrix(C, 2, 2);
			printf("G1* J F1 = \n");
			C[0] = J[k]*T[0]*conj(G0) + J[k+1]*T[1]*conj(G1);
			C[1] = J[k]*T[0]*conj(G2) + J[k+1]*T[1]*conj(G3);
			C[2] = J[k]*T[2]*conj(G0) + J[k+1]*T[3]*conj(G1);
			C[3] = J[k]*T[2]*conj(G2) + J[k+1]*T[3]*conj(G3);
			printMatrix(C, 2, 2);
		}

		// copy columns of G into K
		Mk = M-k;
		inc = 1;
		mkl_nthreads = Mk/D > mkl_get_max_threads()/2 ? Mk/D : mkl_get_max_threads()/2;
		if(Mk/D == 0) mkl_nthreads = 1;

		#pragma omp parallel num_threads(2)
		{
			mkl_set_num_threads_local(mkl_nthreads);
			if(omp_get_thread_num() == 0) zcopy(&Mk, &G[k+M*k], &inc, &K[k], &inc);
			else zcopy(&Mk, &G[k+M*(k+1)], &inc, &K[k+M], &inc);
		}
		mkl_set_num_threads_local(0);

		// K = the difference operator for the J Householder
		K[k] -= T[0];
		K[k+1] -= T[1];
		K[k + M] -= T[2];
		K[k+1 + M] -= T[3];


		double complex T0, T1, T2, T3;
		int provjera = 1;
		if(provjera){
			T0 = T[0];
			T1 = T[1];
			T2 = T[2];
			T3 = T[3];
		}


		// fill first two columns of G
		//G[k+M*k] = T[0];
		//G[k+1+M*k] = T[1];
		//G[k+M*(k+1)] = T[2];
		//G[k+1+M*(k+1)] = T[3];


		// compute K*JK, first we need T = JK
		// fill the rest of the G with zeros
		nthreads = Mk/D > omp_get_max_threads() ? Mk/D : omp_get_max_threads();
		if(Mk/D == 0) nthreads = 1;
		#pragma omp parallel for num_threads( nthreads )
		for(i = k; i < M; ++i){
			T[i] = J[i] * K[i];
			T[i+M] = J[i] * K[i+M];
			//if( i >= k+2 ){
			//	G[i+M*k] = 0;
			//	G[i+M*(k+1)] = 0;
			//}
		}

		// compute K*T, where T = JK
		// C = K*JK
		Mk = M - k;
		mkl_nthreads = Mk/D > mkl_get_max_threads() ? Mk/D : mkl_get_max_threads();
		if(Mk/D == 0) mkl_nthreads = 1;
		mkl_set_num_threads( mkl_nthreads );
		alpha = 1;
		beta = 0;
		zgemm(&trans, &nontrans, &n, &n, &Mk, &alpha, &K[k], &M, &T[k], &M, &beta, C, &n);	// C = K*T (T = JK)

		C[0] = creal(C[0]);
		C[3] = creal(C[3]);


		if(provjera){
			printf("PIVOT_2, k = %d\n", k);
			printf("detA = %lg\n", detA);
			printf("det(D*JD) = %lg\n", C[0]*C[3] - cabs(C[1])*cabs(C[1]));

			double complex G0 = G[k+M*k];
			double complex G1 = G[k+1+M*k];
			double complex G2 = G[k+M*(k+1)];
			double complex G3 = G[k+1+M*(k+1)];

			// F*JG
			double complex a1 = J[k]*G0*conj(T0) + J[k+1]*G1*conj(T1);
			double complex a2 = J[k]*G0*conj(T2) + J[k+1]*G1*conj(T3);
			double complex a3 = J[k]*G2*conj(T0) + J[k+1]*G3*conj(T1);
			double complex a4 = J[k]*G2*conj(T2) + J[k+1]*G3*conj(T3);
		
			// G*JF
			double complex b1 = J[k]*T0*conj(G0) + J[k+1]*T1*conj(G1);
			double complex b2 = J[k]*T0*conj(G2) + J[k+1]*T1*conj(G3);
			double complex b3 = J[k]*T2*conj(G0) + J[k+1]*T3*conj(G1);
			double complex b4 = J[k]*T2*conj(G2) + J[k+1]*T3*conj(G3);

			double d1 = cabs(a1-b1);
			double d2 = cabs(a2-b2);
			double d3 = cabs(a3-b3);
			double d4 = cabs(a4-b4);

			double err = csqrt(d1*d1 + d2*d2 + d3*d3 + d4*d4);
			printf("|F*JG-G*JF| = %lg\n", err);

			// F*JF
			a1 = J[k]*T0*conj(T0) + J[k+1]*T1]*conj(T1);
			a2 = J[k]*T0*conj(T2) + J[k+1]*T1*conj(T3);
			a3 = J[k]*T2*conj(T0) + J[k+1]*T3*conj(T1);
			a4 = J[k]*T2*conj(T2) + J[k+1]*T3*conj(T3);

			d1 = cabs(a1-Akk);
			d2 = cabs(a2-conj(Akr));
			d3 = cabs(a3-Akr);
			d4 = cabs(a4-Arr);

			err = csqrt(d1*d1 + d2*d2 + d3*d3 + d4*d4);
			printf("|A2-F*JF| = %lg\n--------------------------\n", err);
		}

		// C = C^(-1) = (K*JK)^+
		double detC = C[0]*C[3] - cabs(C[1])*cabs(C[1]);

		if(kontrola) printf("detC = %lg\n", detC);

		double complex C0 = C[3] / detC;
		double complex C1 = -C[1] / detC;
		double complex C2 = -C[2] / detC;
		double complex C3 = C[0] / detC;

		C[0] = C0;
		C[1] = C1;
		C[2] = C2;
		C[3] = C3;

		// apply the reflector
		int Nk = (N - k - 2)/2;
		nthreads = Nk/D > omp_get_max_threads() ? Nk/D : omp_get_max_threads();
		if(Nk/D == 0) nthreads = 1;

		Mk = M - k;
		mkl_nthreads = Mk/D > mkl_get_max_threads()/nthreads ? Mk/D : mkl_get_max_threads()/nthreads;
		if(Mk/D == 0) mkl_nthreads = 1;

		// compute E = KC
		alpha = 1;
		beta = 0;
		zgemm(&nontrans, &nontrans, &Mk, &n, &n, &alpha, &K[k], &M, C, &n, &beta, &E[k], &M);

		if(kontrola){
			printf("E = \n");
			printMatrix(E, M, 2);
			printf("JK = \n");
			printMatrix(T, M, 2);
		}

		// K = W (Mk x 2 matrix)
		// C = (W*JW)^+ (2x2 matrix)
		// T = JK
		//#pragma omp parallel num_threads( nthreads )
		//{
			//#pragma omp for nowait
			for(j = k; j < N; j += 1){

				//mkl_set_num_threads_local(mkl_nthreads);
				//double complex *CC = (double complex*) mkl_malloc(4*sizeof(double complex), 64);

				// case when we have 2 columns of G to work with
				if(0){//j != N-1

					// CC  = T*G
					alpha = 1;
					beta = 0;
					zgemm(&trans, &nontrans, &n, &n, &Mk, &alpha, &T[k], &M, &G[k+M*j], &M, &beta, C, &n);

					// G = G - 2E CC
					alpha = -2;
					beta = 1;
					zgemm(&nontrans, &nontrans, &Mk, &n, &n, &alpha, &E[k], &M, C, &n, &beta, &G[k+M*j], &M);
				}

				// case when we are in the last column
				else{
					
					// CC = T*g
					alpha = 1;
					beta = 0;
					inc = 1;
					zgemv(&trans, &Mk, &n, &alpha, &T[k], &M, &G[k+M*j], &inc, &beta, C, &inc);

					// g = g - 2E CC
					alpha = -2;
					beta = 1;
					zgemv(&nontrans, &Mk, &n, &alpha, &E[k], &M, C, &inc, &beta, &G[k+M*j], &inc);
				}

				//mkl_free(CC);
			}
		//}
		mkl_set_num_threads_local(0);

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

		double complex H_sigma = 1;
		if(cabs(G[k+M*k]) >= EPSILON) H_sigma = -G[k+M*k] / cabs(G[k+M*k]);
		double complex gkk = csqrt(cabs(Akk)) * H_sigma;

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

		#pragma omp parallel num_threads( nthreads )
		{
			#pragma omp for nowait
			for(j = k+1; j < N; ++j){

				mkl_nthreads = Mk/D > mkl_get_max_threads()/nthreads ? Mk/D : mkl_get_max_threads()/nthreads;
				if (Mk/D == 0) mkl_nthreads = 1;
				mkl_set_num_threads_local(mkl_nthreads);

				// T = Jf
				// alpha = f*Jg
				int Mk = M - k;
				int inc = 1;
				double complex alpha;
				zdotc(&alpha, &Mk, &T[k], &inc, &G[k+M*j], &inc);
				alpha = - 2 * alpha / fJf;
				zaxpy(&Mk, &alpha, &f[k], &inc, &G[k + M*j], &inc);	// G[k + M*j] = alpha * f[k] + G[k + M*k]
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
	printf("redukcija u PIVOT_2 time = %lg s (udio relativnog = %lg %%, udio apsolutnog = %lg %%)\n", redukcijatime, redukcijatime/pivot2time * 100, redukcijatime/seconds * 100);
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

	printf("\n-------------------------------------------------------------------------------\n\n");
	return(0);
}
