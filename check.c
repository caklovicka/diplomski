#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>


void printMatrix(double complex *G, int M, int N){

	int i, j;
	for( i = 0; i < M; ++i ){
		for( j = 0; j < N; ++j ){
			printf("%7.2f + i%7.2f    ", creal(G[i+M*j]), cimag(G[i+M*j]));
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
	FILE *readP = fopen(argv[4], "rb");
	int M = atoi(argv[5]);
	int N = atoi(argv[6]);

	printf("Reading data...\n");

	// allocate memory
	double complex *AA = (double complex*) malloc(N*N*sizeof(double complex));
	double complex *A = (double complex*) malloc(N*N*sizeof(double complex));
	double complex *PA = (double complex*) malloc(N*N*sizeof(double complex));
	double complex *G = (double complex*) malloc(M*N*sizeof(double complex));
	double complex *T = (double complex*) malloc(M*N*sizeof(double complex));	// temporary matrix
	long int *J = (long int*) malloc(M*sizeof(long int));
	long int *P = (long int*) malloc(N*sizeof(long int));	// for column permutation


	// check if files are opened
	if(readG == NULL || readJ == NULL || readA == NULL || readP == NULL || AA == NULL){
		printf("Cannot open file.\n");
		exit(-1);
	}

	// check if memory is allocated
	if(G == NULL || J == NULL || P == NULL || T == NULL || A == NULL){
		printf("Cannot allocate memory.\n");
		exit(-2);
	}

	int i, j;
	// read vector J and P
	for(i = 0; i < M; ++i){
		fscanf(readJ, "%ld ", &J[i]);
		fscanf(readP, "%ld ", &P[i]);
	}

	// read matrix G
	double x, y;
	for(j = 0; j < N; ++j){
		for( i = 0; i < M; ++i ){
			fscanf(readG, "%lf %lf ", &x, &y);
			G[i+M*P[j]] = x + I*y;
		}
	}

	printVector(P, N);
	
	// read AA
	for(j = 0; j < N; ++j){
		for( i = 0; i < N; ++i){
			fscanf(readA, "%lf %lf ", &x, &y);
			AA[i+N*j] = x + I*y;
		}
	}

	// ------------------------------------ compute G*JG ------------------------------------

	printf("computing G*JG...\n");

	char trans = 'C';
	char non_trans = 'N';
	double complex alpha = 1;
	double complex beta = 0;

	
	double complex *JJ = (double complex*)malloc(M*M*sizeof(double complex));

	if(JJ == NULL){
		printf("Cannot allocate memory.\n");
		exit(-2);
	}

	for(i = 0; i < M; ++i){
		for(j = 0; j < M; ++j){
			if(i == j) JJ[i+M*j] = (double complex)J[i];
			else JJ[i+M*j] = 0;
		}
	}

	zgemm_(&trans, &non_trans, &N, &M, &M, &alpha, G, &M, JJ, &M, &beta, T, &N);	//T = G*J (NxM)
	zgemm_(&non_trans, &non_trans, &N, &N, &M, &alpha, T, &N, G, &M, &beta, A, &N);	//A = TG = G*JG (NxN)


	// ------------------------------------------ residual ------------------------------------------

	printf("A = \n");
	printMatrix(A, N, N);

	printf("AA = \n");
	printMatrix(AA, N, N);

	double norm = 0; 
	double max = 0;
	for(i = 0; i < N; ++i){
		for(j = 0; j < N; ++j){
			printf("%.2lf + i%.2lf   -   %.2lf + i%.2lf   =   %.2lf + i%.2lf\n", creal(A[P[i]+N*P[j]]), cimag(A[P[i]+N*P[j]]), creal(AA[i+N*j]), cimag(AA[i+N*j]), creal(A[P[i]+N*P[j]] - AA[i+N*j]), cimag(A[P[i]+N*P[j]] - AA[i+N*j]));
			if(cabs(A[P[i]+N*P[j]] - AA[i+N*j]) > max) max = cabs(A[P[i]+N*P[j]] - AA[i+N*j]);
			norm += cabs(A[P[i]+N*P[j]] - AA[i+N*j]) * cabs(A[P[i]+N*P[j]] - AA[i+N*j]);
		}
	}

	printf("P = \n");
	printVector(P, N);

	printf("maximum coordinate difference: %lf\n", max);
	printf("norm(A-AA): %lf\n", sqrt(norm));

	// ------------------------------- cleaning -------------------------------

	fclose(readG);
	fclose(readJ);
	fclose(readA);
	fclose(readP);

	free(P);
	free(A);	
	free(G);
	free(J);
	free(T);


	printf("Finished.\n");
	return(0);
}