#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>
#include <float.h>

#define MKL_Complex16 complex

#include <mkl.h>
#include "omp.h"


#define EPSILON DBL_EPSILON
#define DIGITS 3 //DBL_DIG

double dznrm2(int* N, double complex* X, int* inc);

void printMatrix(double complex *G, int M, int N){

	int i, j;
	for( i = 0; i < M; ++i ){
		for( j = 0; j < N; ++j ){
			printf("%11.*g + i%11.*g    ", DIGITS, DIGITS, creal(G[i+M*j]), cimag(G[i+M*j]));
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

	mkl_set_dynamic(1);
	omp_set_dynamic(1);

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
	double complex *AA = (double complex*) mkl_malloc(N*N*sizeof(double complex), 64);
	double complex *A = (double complex*) mkl_malloc(N*N*sizeof(double complex), 64);
	double complex *PA = (double complex*) mkl_malloc(N*N*sizeof(double complex), 64);
	double complex *G = (double complex*) mkl_malloc(M*N*sizeof(double complex), 64);
	double complex *T = (double complex*) mkl_malloc(M*N*sizeof(double complex), 64);	// temporary matrix
	long int *J = (long int*)mkl_malloc(M*sizeof(long int), 64);
	long int *Pcol = (long int*) mkl_malloc(N*sizeof(long int), 64);	// for column permutation


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

	zgemm(&trans, &non_trans, &N, &M, &M, &alpha, G, &M, JJ, &M, &beta, T, &N);	//T = G*J (NxM)
	zgemm(&non_trans, &non_trans, &N, &N, &M, &alpha, T, &N, G, &M, &beta, A, &N);	//A = TG = G*JG (NxN)

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
	printMatrix(AA, N, N);*/	

	double norm = 0; 
	double max = 0;
	double max_rel = 0;

	int ii = -1, jj = -1, ir = -1, jr = -1;

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
			if(cabs(PA[i+N*j] - AA[i+N*j])/cabs(AA[i+N*j]) > max_rel){
				max_rel = cabs(PA[i+N*j] - AA[i+N*j])/cabs(AA[i+N*j]);
				ir = i;
				jr = j;
			} 
			norm += cabs(PA[i+N*j] - AA[i+N*j]) * cabs(PA[i+N*j] - AA[i+N*j]);
		}
		}
	}

	int NN = N*N;
	int inc = 1;
	double norm2 = dznrm2(&NN, AA, &inc);

	printf("\n\nmaximum coordinate difference in (%d, %d): %.5e\n", ii, jj, max);
	printf("maximum relative coordinate difference in (%d, %d): %.5e\n", ir, jr, max_rel);
	printf("norm(PA-AA): %.5e\n", csqrt(norm));
	printf("norm(PA-AA)/norm(AA) = %.5e\n", csqrt(norm)/norm2);
	//printf("dnrm2_(AA) = %.5e\n", norm2);


	// ---------------------------------------------------- SVD ----------------------------------------------------

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
	zgesdd_(&jobz, &N, &N, PA, &N, s, NULL, &N, NULL, &N, work, &lwork, rwork, iwork, &info);

	if(s[N-1] < EPSILON){
		printf("\n\n\nA = G*JG is singular.\n\n\n");
		exit(-3);
	}
	else printf("\nSmallest singular value of A: %.*g, cond = %lg\n", DIGITS, s[N-1], csqrt(s[0]/s[N-1]));

	double ss;
	FILE *svd = fopen("data/svd.bin", "rb");
	for(i = 0; i < N; ++i){
		fscanf(svd, "%lg ", &ss);
		s[i] -= ss;
		printf("s[i] = %lg\n", s[i]);
	}
	inc = 1;
	double norm_svd = dznrm2(&N, s, &inc);
	printf("norm(singulars - s) = %lg\n", norm_svd);

	free(s);
	free(work);
	free(rwork);
	free(iwork);

	// ---------------------------------------------------- END SVD ----------------------------------------------------

	// ------------------------------- cleaning -------------------------------

	fclose(readG);
	fclose(readJ);
	fclose(readA);
	fclose(readPcol);

	mkl_free(Pcol);
	mkl_free(A);	
	mkl_free(G);
	mkl_free(J);
	mkl_free(T);


	//printf("Finished.\n");
	printf("-------------------------------------------------------------------------------\n\n");
	return(0);
}
