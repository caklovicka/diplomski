// call as:
//
// ./generateG.out G.bin J.bin M N
//
// (G.bin = filename where to store G in binary, same as J.bin)

// exits:
//   -1 ....... Cannot open file.
//   -2 ....... Cannot allocate memory.
//   -3 ....... A = G*JG is singular.
//----------------------------------------------------------------------------------------

#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <time.h>
#include <float.h>
#include "omp.h"
#include <unistd.h>


#define EPSILON DBL_EPSILON
#define DIGITS DBL_DIG


void printMatrix(double complex *G, int M, int N){

	int i, j;
	for( i = 0; i < M; ++i ){
		for( j = 0; j < N; ++j ){
			printf("%6.2f + i%6.2f    ", creal(G[i+M*j]), cimag(G[i+M*j]));
		}
		printf("\n");
	}
	printf("\n");
}

void printSingular(double *s, int N){

	int i;
	for(i = 0; i < N; ++i) printf("%.*g\n", DIGITS, s[i+N*i]);
	printf("\n");
}
void printVector(double* J, int M){
	int i;
	for(i = 0; i < M; ++i){
		printf("%d ", (int)J[i]);
	}
	printf("\n");
}

//----------------------------------------------------------------------------------------

int main(int argc, char* argv[]){


	// read variables from command line
	int M = atoi(argv[3]);
	int N = atoi(argv[4]);
	FILE *writeG = fopen(argv[1], "wb");
	FILE *writeJ = fopen(argv[2], "wb");

	//printf("\n\n-------------------------------- GENERATING -----------------------------------\n\n");
	//printf("Generating...\n");

	double start = omp_get_wtime();

	// allocate memory
	double complex *G = (double complex*) malloc(M*N*sizeof(double complex));
	double *J = (double*) malloc(M*sizeof(double));


	// check if files are opened
	if(writeG == NULL || writeJ == NULL){
		printf("Cannot open file.\n");
		exit(-1);
	}

	// check if memory is allocated
	if(G == NULL || J == NULL){
		printf("Cannot allocate memory.\n");
		exit(-2);
	}

	// fill matrix G randomly
	srand((unsigned int)time(NULL));
	double x, y;
	int i, j;

	#pragma omp parallel for shared(G)
	for( i = 0; i < N*M; ++i ){
		x = 50.0*rand()/(RAND_MAX) - 5;
		y = 50.0*rand()/(RAND_MAX) - 5;
		G[i] = x + I*y;
	}
 
 	// fill vector J randomly
 	
 	#pragma omp parallel for private(i) shared(J)
	for(i = 0; i < M; ++i){
		if(1.0*rand()/(RAND_MAX) < 0.5)	J[i] = -1.0;
		else J[i] = 1.0;
	}

	
	// ----------------------------------- computing A -----------------------------------

	//printf("Checking that A is nonslingular...\n");

	
	// make T
	double complex *T = (double complex*)malloc(M*N*sizeof(double complex));
	if(T == NULL){
		printf("Cannot allocate memory.\n");
		exit(-2);
	}

	// T = JG

	#pragma omp parallel for private(i) shared(T)
	for(i = 0; i < M*N; ++i){
		T[i] = J[i%M] * G[i];
	}

	// make A

	double complex *A = (double complex*) malloc(N*N*sizeof(double complex));
	if(A == NULL){
		printf("Cannot allocate memory.\n");
		exit(-2);
	}


	// do A = G*T

	char trans = 'C';
	char non_trans = 'N';
	double complex alpha = 1;
	double complex beta = 0;

	int p = 50;
	int n = 1;
	int m = N / p;
	int m_last = N % p;
	

	int pp = N;
	if(pp > 500) pp = 500;

	#pragma omp parallel for private(i) num_threads(pp)
	for(i = 0; i < N; ++i){
		
		if(m != 0){
			#pragma omp parallel for private(j) num_threads(p)
			for(j = 0; j <= N-m; j += m){
				zgemm_(&trans, &non_trans, &n, &m, &M, &alpha, &G[i*M], &M, &T[j*M], &M, &beta, &A[i+N*j], &N);	//A = G*T = G*JG (NxN)
			}
		}

		zgemm_(&trans, &non_trans, &n, &m_last, &M, &alpha, &G[i*M], &M, &T[ (p*m)*M ], &M, &beta, &A[i+N*(p*m)], &N);

	}

	double end = omp_get_wtime();
	double seconds = (double)(end - start);
	printf("assembly time = %lg s\n", seconds);


	// ---------------------------------------------------- SVD ----------------------------------------------------

	/*

	char jobz = 'N';
	double *s = (double*) malloc(N*sizeof(double));	// for singular values, they will be sorted in s as s(i) >= s(i+1)
	int lwork = 6*N;
	double complex *work = (double complex*) malloc(lwork*sizeof(double complex));
	double *rwork = (double*) malloc(lwork*sizeof(double));
	int *iwork = (int*)malloc(8*N*sizeof(int));

	if(work == NULL || rwork == NULL || iwork == NULL || s == NULL){
		printf("Cannot allocate memory.\n");
		exit(-2);
	}

	int info;
	zgesdd_(&jobz, &N, &N, A, &N, s, NULL, &N, NULL, &N, work, &lwork, rwork, iwork, &info);


	if(s[N-1] < EPSILON){
		printf("A = G*JG is singular.\n");
		exit(-3);
	}
	else printf("Smallest singular value of A: %.*g\n", DIGITS, s[N-1]);

	free(s);
	free(work);
	free(rwork);
	free(iwork);
	
	*/

	// ------------------------------------- write A G J in files ---------------------------------------------- 

	start = omp_get_wtime();

	FILE *writeA = fopen("data/A.bin", "wb");

	//#pragma omp parallel num_threads(3)
	{
		if(omp_get_thread_num() == 0){
			int i, j;
			for(j = 0; j < N; ++j){
				for(i = 0; i < N; ++i){
					fprintf(writeA, "%.*g %.*g ", DIGITS, creal(A[i+N*j]), DIGITS, cimag(A[i+N*j]));
				}
			}
		}

		if(omp_get_thread_num() == 1){
			int i, j;
			for(j = 0; j < N; ++j){
				for(i = 0; i < M; ++i){
					fprintf(writeG, "%.*g %.*g ", DIGITS, creal(G[i+M*j]), DIGITS, cimag(G[i+M*j]));
				}
			}
		}

		if(omp_get_thread_num() == 2){
			int i;
			for(i = 0; i < M; ++i) fprintf(writeJ, "%ld ", (long int)J[i]);
		}
	}

	end = omp_get_wtime();
	seconds = (float)(end - start);
	printf("writing time = %lg s\n", seconds);

	// -------------------------------------------- cleaning --------------------------------------------

	fclose(writeJ);
	fclose(writeG);
	fclose(writeA);

	free(A);
	//free(T);
	free(J);
	free(G);

	//printf("Finished.\n");
	//printf("\n-------------------------------------------------------------------------------\n\n");

return(0);
}
