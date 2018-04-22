// call as:
//
// ./GFreduction.out G.bin J.bin M N
//
// (G.bin = filename where to store G in binary, same as J.bin)

// exits:
//   -1 ....... Cannot open file.
//   -2 ....... Cannot allocate memory.
//----------------------------------------------------------------------------------------



#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>
#include <time.h>

double eps = 1.0;	// ovaj je za g*Jg >= eps
double eps0 = 1.0e-6;	// ovo je nula

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
	int M = atoi(argv[3]);
	int N = atoi(argv[4]);
	FILE *readG = fopen(argv[1], "rb");
	FILE *readJ = fopen(argv[2], "rb");

	printf("Reading data...\n");

	// allocate memory
	double complex *G = (double complex*) malloc(M*N*sizeof(double complex));
	double complex *H = (double complex*) malloc(M*M*sizeof(double complex));	// reflector
	double complex *T = (double complex*) malloc(M*N*sizeof(double complex));	// temporary matrix
	long int *J = (long int*) malloc(M*sizeof(long int));
	long int *Prow = (long int*) malloc(M*sizeof(long int));	// for row permutation
	long int *Pcol = (long int*) malloc(N*sizeof(long int));	// for column permutation
	double complex *f = (double complex*) malloc(M*sizeof(double complex));	// vector f


	// check if files are opened
	if(readG == NULL || readJ == NULL){
		printf("Cannot open file.\n");
		exit(-1);
	}

	// check if memory is allocated
	if(G == NULL || J == NULL || Pcol == NULL || Prow == NULL || T == NULL || H == NULL || f == NULL){
		printf("Cannot allocate memory.\n");
		exit(-2);
	}

	// read matrix G and prepare Pcol
	double x, y;
	int i, j;
	for( j = 0; j < N; ++j ){
		Pcol[j] = j;
		for( i = 0; i < M; ++i ){
			fscanf(readG, "%lf %lf ", &x, &y);
			G[i+M*j] = x + I*y;
		}
	}

 	// read vector J and prepare permutation vectors
	for(i = 0; i < M; ++i){
		fscanf(readJ, "%ld ", &J[i]);
		Prow[i] = i;
	}

	// ---------------------------------------------------------- ALGORITHM ----------------------------------------------------------

	printf("Pivoting QR...\n");

	for(int k = 0; k < N; ++k){

		// choosing a pivoting strategy (partial pivoting)

		
	}


	// ------------------------------- cleaning -------------------------------

	fclose(readG);
	fclose(readJ);
	/*fclose(writeG);
	fclose(writeJ);
	fclose(writeCol);
	fclose(writeRow);
	*/

	free(Prow);
	free(Pcol);
	free(G);
	free(J);
	free(T);
	free(H);
	free(f);


	printf("Finished.\n");
	return(0);
}