#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>
#include <float.h>
#include "omp.h"


#define EPSILON DBL_EPSILON
#define DIGITS DBL_DIG

double dznrm2_(int* N, double complex* X, int* inc);

void printMatrix(double complex *G, int M, int N){

	int i, j;
	for( i = 0; i < M; ++i ){
		for( j = 0; j < N; ++j ){
			printf("%23.*g + i%23.*g    ", DIGITS, DIGITS, creal(G[i+M*j]), cimag(G[i+M*j]));
		}
		printf("\n");
	}
	printf("\n");
}

void printVector(long int *J, int M){
	int i;
	for(i = 0; i < M; ++i){
		printf("%ld ", J[i]);
	}
	printf("\n");
}

//----------------------------------------------------------------------------------------

int main(int argc, char* argv[]){

	// read variables from command line
	FILE *readG = fopen(argv[1], "rb");
	FILE *readJ = fopen(argv[2], "rb");
	FILE *readA = fopen(argv[3], "rb");
	FILE *readPcol = fopen(argv[4], "rb");
	int M = atoi(argv[5]);
	int N = atoi(argv[6]);


	//printf("\n\n------------------------------------ CHECK ------------------------------------\n\n");
	//printf("Reading data...\n");

	// allocate memory
	double complex *AA = (double complex*) malloc(N*N*sizeof(double complex));
	double complex *A = (double complex*) malloc(N*N*sizeof(double complex));
	double complex *PA = (double complex*) malloc(N*N*sizeof(double complex));
	double complex *G = (double complex*) malloc(M*N*sizeof(double complex));
	double complex *T = (double complex*) malloc(M*N*sizeof(double complex));	// temporary matrix
	long int *J = (long int*) malloc(M*sizeof(long int));
	long int *Pcol = (long int*) malloc(N*sizeof(long int));	// for column permutation


	// check if files are opened
	if(readG == NULL || readJ == NULL || readA == NULL || readPcol == NULL){
		printf("Cannot open file.\n");
		exit(-1);
	}

	// check if memory is allocated
	if(G == NULL || J == NULL || T == NULL || A == NULL || AA == NULL || Pcol == NULL){
		printf("Cannot allocate memory.\n");
		exit(-2);
	}

	int i, j;
	// read vector J and Prow
	for(i = 0; i < M; ++i){
		fscanf(readJ, "%ld ", &J[i]);
	}

	// read Pcol
	for(i = 0; i < N; ++i){
		fscanf(readPcol, "%ld ", &Pcol[i]);
	}

	// read matrix G
	double x, y;
	for(j = 0; j < N; ++j){
		for( i = 0; i < M; ++i ){
			fscanf(readG, "%lg %lg ", &x, &y);
			G[i+M*j] = x + I*y;
		}
	}
	
	// read AA
	for(j = 0; j < N; ++j){
		for( i = 0; i < N; ++i){
			fscanf(readA, "%lg %lg ", &x, &y);
			AA[i+N*j] = x + I*y;
		}
	}

	/*printf("Pcol = \n");
	printVector(Pcol, N);
	printf("J' = \n");
	printVector(J, M);
	printf("\nG = \n");
	printMatrix(G, M, N);
	*/

	// ------------------------------------ compute G*JG ------------------------------------

	//printf("computing G*JG...\n");

	char trans = 'C';
	char non_trans = 'N';
	double complex alpha = 1;
	double complex beta = 0;

	
	double complex *JJ = (double complex*)malloc(M*M*sizeof(double complex));

	if(JJ == NULL){
		printf("Cannot allocate memory.\n");
		exit(-2);
	}

	#pragma omp parallel for collapse(2)
	for(i = 0; i < M; ++i){
		for(j = 0; j < M; ++j){
			if(i == j) JJ[i+M*j] = (double complex)J[i];
			else JJ[i+M*j] = 0;
		}
	}

	for(i = N; i < M; ++i){

		if( cabs(G[i+M*(N-1)]) >= EPSILON ){
			printf("\nERR: G not triangular!!! (checked last column)\n\n");
			break;
		}
	}

	zgemm_(&trans, &non_trans, &N, &M, &M, &alpha, G, &M, JJ, &M, &beta, T, &N);	//T = G*J (NxM)
	zgemm_(&non_trans, &non_trans, &N, &N, &M, &alpha, T, &N, G, &M, &beta, A, &N);	//A = TG = G*JG (NxN)

	// ------------------------------------------ apply permutation on A ----------------------------

	// compute perm on the upper triangle, and use A*=A on the lower

	#pragma omp parallel for
	for( i = 0; i < N; ++i ){
		for( j = i; j < N; ++j ){
			PA[Pcol[i]+N*Pcol[j]] = A[i+N*j];
			PA[Pcol[j]+N*Pcol[i]] = conj(A[i+N*j]);
		}
	}
	
	// ------------------------------------------ residual ------------------------------------------

	/*printf("\nA (izracunata) = \n");
	printMatrix(A, N, N);
	
	printf("\nPA (permutirana, izracunata) = \n");
	printMatrix(PA, N, N);

	printf("AA (prava matrica) = \n");
	printMatrix(AA, N, N);
	*/

	double norm = 0; 
	double max = 0;
	int ii = -1, jj = -1;

	#pragma omp parallel for collapse(2)
	for(i = 0; i < N; ++i){
		for(j = 0; j < N; ++j){

		#pragma omp critical
		{
			if(cabs(PA[i+N*j] - AA[i+N*j]) > max){
				max = cabs(PA[i+N*j] - AA[i+N*j]);
				ii = i;
				jj = j;
			} 
			norm += cabs(PA[i+N*j] - AA[i+N*j]) * cabs(PA[i+N*j] - AA[i+N*j]);
		}
		}
	}

	int NN = N*N;
	int inc = 1;
	double norm2 = dznrm2_(&NN, AA, &inc);

	printf("maximum coordinate difference in (%d, %d): %.5e\n", ii, jj, max);
	printf("norm(PA-AA): %.5e\n", csqrt(norm));
	printf("norm(PA-AA)/norm(AA) = %.5e\n", csqrt(norm)/norm2);
	//printf("dnrm2_(AA) = %.5e\n", norm2);

	// ------------------------------- cleaning -------------------------------

	fclose(readG);
	fclose(readJ);
	fclose(readA);
	fclose(readPcol);

	free(Pcol);
	free(A);	
	free(G);
	free(J);
	free(T);


	//printf("Finished.\n");
	printf("-------------------------------------------------------------------------------\n\n");
	return(0);
}
