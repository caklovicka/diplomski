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


void printMatrix(double complex *G, int M, int N){

	int i, j;
	for( i = 0; i < M; ++i ){
		for( j = 0; j < N; ++j ){
			printf("%.*g + i%.*g    ", DBL_DIG, DBL_DIG, creal(G[i+M*j]), cimag(G[i+M*j]));
		}
		printf("\n");
	}
	printf("\n");
}

void printSingular(double *s, int N){

	int i;
	for(i = 0; i < N; ++i) printf("%.*g\n", DBL_DIG, s[i+N*i]);
	printf("\n");
}
void printVector(double complex *J, int M){
	int i;
	for(i = 0; i < M; ++i){
		printf("%ld ", (long int)J[i+M*i]);
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

	printf("Generating...\n");

	// allocate memory
	double complex *G = (double complex*) malloc(M*N*sizeof(double complex));
	double complex *J = (double complex*) malloc(M*M*sizeof(double complex));


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
	for( i = 0; i < M; ++i ){
		for( j = 0; j < N; ++j ){
			x = 50.0*rand()/(RAND_MAX) - 5;
			y = 50.0*rand()/(RAND_MAX) - 5;
			G[i+M*j] = x + I*y;
		}
	}
 
 	// fill vector J randomly
 	for( i = 0; i < M; ++i ){
		for( j = 0; j < M; ++j ){
			J[i+M*j] = 0;
		}
	}
	for(i = 0; i < M; ++i){
		if(1.0*rand()/(RAND_MAX) < 0.5)	J[i+M*i] = -1;
		else J[i+M*i] = 1;
		
		//scale blas koristi, a ne sa matricom J mnoziti
	}

	
	// ----------------------------------- computing A -----------------------------------

	printf("Checking that A is nonslingular...\n");

	char trans = 'C';
	char non_trans = 'N';
	double complex alpha = 1;
	double complex beta = 0;
	double complex *T = (double complex*)malloc(M*N*sizeof(double complex));

	if(T == NULL){
		printf("Cannot allocate memory.\n");
		exit(-2);
	}

	zgemm_(&trans, &non_trans, &N, &M, &M, &alpha, G, &M, J, &M, &beta, T, &N);	//T = G*J (NxM)
	double complex *A = (double complex*) malloc(N*N*sizeof(double complex));

	if(A == NULL){
		printf("Cannot allocate memory.\n");
		exit(-2);
	}

	zgemm_(&non_trans, &non_trans, &N, &N, &M, &alpha, T, &N, G, &M, &beta, A, &N);	//A = TG = G*JG (NxN)


	// ------------------------------------ write A into file ------------------------------------

	FILE *writeA = fopen("data/A.bin", "wb");

	printf("G(generirana, friško) = \n");
	printMatrix(G, M, N);

	for(j = 0; j < N; ++j)
		for(i = 0; i < N; ++i)	
			fprintf(writeA, "%.*g %.*g ", DBL_DIG, DBL_DIG, creal(A[i+N*j]), cimag(A[i+N*j]));


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
	zgesdd_(&jobz, &N, &N, A, &N, s, NULL, &N, NULL, &N, work, &lwork, rwork, iwork, &info);


	if(s[N-1] < DBL_EPSILON){
		printf("A = G*JG is singular.\n");
		exit(-3);
	}
	else printf("Smallest singular value: %.*g\n", DBL_DIG, s[N-1]);


	// -------------------------- write G and J in files -------------------------- 

	for(j = 0; j < N; ++j){
		for(i = 0; i < M; ++i){
			fprintf(writeG, "%.*g %.*g ", DBL_DIG, DBL_DIG, creal(G[i+M*j]), cimag(G[i+M*j]));
		}
	}

	for(i = 0; i < M; ++i) fprintf(writeJ, "%ld ", (long int)J[i+M*i]);

	// -------------------------------------------- cleaning --------------------------------------------

	printVector(J, M);

	fclose(writeJ);
	fclose(writeG);
	fclose(writeA);

	free(A);
	free(T);
	free(J);
	free(G);
	free(s);
	free(work);
	free(rwork);
	free(iwork);
	printf("Finished.\n");

return(0);
}