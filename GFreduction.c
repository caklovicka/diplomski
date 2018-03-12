// call as:
//
// ./GFreduction.out G.bin J.bin M N
//
// (G.bin = filename where to store G in binary, same as J.bin)

// exits:
//   -1 ....... Cannot open file.
//   -2 ....... Cannot allocate memory.
//----------------------------------------------------------------------------------------

// swapaj
// permutiraj retke (za -1)
// onda ubaci partial


#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>
#include <time.h>

double eps = 1.0e-2;

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

	// read matrix G
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

	printf("G = \n");
	printMatrix(G, M, N);
	printf("J = \n");
	printVector(J, M);

	// ---------------------------------------------------------- ALGORITHM ----------------------------------------------------------

	printf("Pivoting QR...\n");

	int k;
	for(k = 0; k < N; ++k){

		// choosing one column as pivot. the first one that is gk* J gk != 0
		// if pivot_col = -1 after thene first for loop, then we need 2x2 pivot
		printf("----------------------------- %d. iteration ----------------------\n", k);

		int pivot_col = -1;	
		double sumk;

		for(j = k; j < N; ++j){

			for(i = k; i < M; ++i) f[i] = conj(G[i+M*j]) * G[i+M*j] * J[i];

			sumk = 0;
			for(i = k; i < M; ++i) sumk += creal(f[i]);		// sumk = g*Jg for that block

			if(k == N-1){
				pivot_col = N-1;
				break;
			}

			if(abs(sumk) >= eps){
				pivot_col = j;
				break;
			}
		}

		printf("sumk = %lf\n", sumk);
		printf("pivot_col = %d\n", pivot_col);

		// ---------------------------------------------------------- 1x1 pivot ----------------------------------------------------------
		if(pivot_col != -1){

			int len_of_f = M - k;

			// swap
			if(pivot_col != k){

				int inc = 1;
				zswap_(&len_of_f, &G[k+M*pivot_col], &inc, &G[k+M*k], &inc);	// G(k:M, k) <-> G(k:M, pivot_col)

				long int temp = Pcol[pivot_col];
				Pcol[pivot_col] = Pcol[k];
				Pcol[k] = temp;

				printf("after colum pivot...\n");
				printMatrix(G, M, N);
			}

			f[k] = csqrt(cabs(sumk));  // g*Jg = f*Jf must hold, that's why we need fk that looks like Hg = sigma*f = sigma*(sqrt(|sumk|), 0, ..., 0)
			for(i = k+1; i < M; ++i) f[i] = 0;	


			// ----------------------- row pivot -----------------------

			if(k < M-1 && sumk < 0 && J[k] == 1){

				for(i = k+1; i < M; ++i)
					if(J[i] == -1) break;

				// swap
				int inc = M;
				int Nk = N - k;
				zswap_(&Nk, &G[i+M*k], &inc, &G[k+M*k], &inc);

				J[k] = -1;
				J[i] = 1;

				long int temp = Prow[i];
				Prow[i] = Prow[k];
				Prow[k] = temp;

				printf("pivoting rows...\n");
				printMatrix(G, M, N);
			}
			else if(k < M-1 && sumk > 0 && J[k] == -1){

				for(i = k+1; i < M; ++i)
					if(J[i] == 1) break;

				// swap
				int inc = M;
				int Nk = N - k;
				zswap_(&Nk, &G[i+M*k], &inc, &G[k+M*k], &inc);

				J[k] = 1;
				J[i] = -1;

				long int temp = Prow[i];
				Prow[i] = Prow[k];
				Prow[k] = temp;

				printf("pivoting rows...\n");
				printMatrix(G, M, N);
			}

			// ----------------------- compute reflector constant sigma -----------------------

			double complex sigma = f[k] * J[k] * G[k+M*k];	// sigma = f*Jg
			if(sumk > 0) sigma = -sigma;
			if(cabs(sigma) >= eps)	sigma = sigma/cabs(sigma);
			else{
				sigma = 1;
				printf("sigma = 0 --> sigma = 1!\n");
			}

			int inc = 1;
			zscal_(&len_of_f, &sigma, &f[k], &inc); // f(k:M) = sigma*f(k:M)

			printf("f = \n");
			printMatrix(&f[k], len_of_f, 1);


			// ----------------------- make the reflector -----------------------

			double complex alpha = -1;
			zaxpy_(&len_of_f, &alpha, &G[k+M*k], &inc, &f[k], &inc);	// f(k:M) = f(k:M) - g(k:M)

			alpha = 0;
			for(i = k; i < M; ++i) alpha += conj(f[i]) * J[i] * f[i];

			for(i = k; i < M; ++i){
				for(j = k; j < M; ++j){

					if(i != j) H[i+M*j] = -2 * f[i] * conj(f[j]) * J[j] / alpha;
					else H[i+M*j] = 1 - 2 * f[i] * conj(f[j]) * J[j] / alpha;
				}
			}

			// ------------------------------- apply the reflector -------------------------------

			char non_trans = 'N';
			alpha = 1;
			double complex beta = 0;
			int Nk = N - k;
			zgemm_(&non_trans, &non_trans, &len_of_f, &Nk, &len_of_f, &alpha, &H[k+M*k], &M, &G[k+M*k], &M, &beta, T, &len_of_f);	// T = HG

			inc = 1;
			for(j = 0; j < Nk; ++j) zcopy_(&len_of_f, &T[j*len_of_f], &inc, &G[k+M*(j+k)], &inc);	// G = T (copy blocks)

			printf("HG = \n");
			printMatrix(G, M, N);
			printf("Pcol = \n");
			printVector(Pcol, N);
			printf("Prow = \n");
			printVector(Prow, M);
			printf("J = \n");
			printVector(J, M);
			printf("\n");
		}
	}

	// -------------------------------- writing -------------------------------- 	

	FILE *writeG = fopen("data/reducedG.bin", "wb");
	FILE *writeJ = fopen("data/reducedJ.bin", "wb");
	FILE *writeCol = fopen("data/Pcol.bin", "wb");
	FILE *writeRow = fopen("data/Prow.bin", "wb");

	// write J and Prow
	for(i = 0; i < M; ++i){
		fprintf(writeJ, "%ld ", J[i]);
		fprintf(writeRow, "%ld ", Prow[i]);
	}

	// write G and pcol
	for(j = 0; j < N; ++j){
		fprintf(writeCol, "%ld ", Pcol[j]);
		for(i = 0; i < M; ++i){
			fprintf(writeG, "%lf %lf ", creal(G[i+M*j]), cimag(G[i+M*j]));
		}
	}

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


	printf("Finished.\n");
	return(0);
}