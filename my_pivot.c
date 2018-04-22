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
double eps0 = 1.0e-12;	// ovo je nula
double ALPHA = (1 + csqrt(17))/8; //Bunch-Parlett alpha

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

	printf("\nReading data...\n");

	// allocate memory
	double complex *G = (double complex*) malloc(M*N*sizeof(double complex));
	double complex *H = (double complex*) malloc(M*M*sizeof(double complex));	// reflector
	double complex *T = (double complex*) malloc(M*N*sizeof(double complex));	// temporary matrix
	double *J = (double*) malloc(M*sizeof(double));
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
	for(int j = 0; j < N; ++j ){
		Pcol[j] = j;

		for(int i = 0; i < M; ++i ){
			double x, y;
			fscanf(readG, "%lf %lf ", &x, &y);
			G[i+M*j] = x + I*y;
		}
	}

 	// read vector J and prepare permutation vectors
	for(int i = 0; i < M; ++i){
		long int itemp;
		fscanf(readJ, "%ld ", &itemp);
		J[i] = 1.0 * itemp;
		Prow[i] = i;
	}


	// ---------------------------------------------------------- ALGORITHM ----------------------------------------------------------

	printf("Pivoting QR...\n");

	for(int k = 0; k < N; ++k){

		// ------------------------ choosing a pivoting strategy (partial pivoting) -------------------------------
		// we need to know the signum of the J-norm of the first column
		// because the pivoting element, Akk, will have to satisfy
		// sign(Jk) = sign(gk*Jgk)

		double Akk = 0;	// Akk = gk* J gk, on a working submatrix G[k:M, k:N]
		double pivot_lambda = 0;
		double pivot_sigma = 0;
		int pivot_r = k + 1; // 2nd column for partial pivoting
		
		// compute Akk for the working submatrix G[k:M, k:N]
		for(int i = k; i < M; ++i) Akk += conj(G[i+M*k]) * J[i] * G[i+M*k];		

		if(k == N-1) goto PIVOT_1;

		// find pivot_lambda
		for(int i = k+1; i < M; ++i){
			double complex Aik = 0;	//Aik = gi* J gk, but on a submatrix G[k:M, k:N]
			for(int j = k; j < M; ++j)	Aik += conj(G[j+M*i]) * J[j] * G[j+M*k];
			if(pivot_lambda < cabs(Aik)) pivot_lambda = cabs(Aik);
		}

		if(cabs(Akk) >= ALPHA * pivot_lambda)	goto PIVOT_1;

		// find pivot_sigma
		for(int i = k; i < M; ++i){
			if(i == pivot_r) continue;
			double complex Air = 0;  //Air = gi* J gr, but on a submatrix G[k:M, k:N]
			for(int j = k; j < M; ++j)	Air += conj(G[j+M*i]) * J[j] * G[j+M*pivot_r];
			if(pivot_sigma < cabs(Air)) pivot_sigma = cabs(Air);
		}

		if(cabs(Akk) * pivot_sigma >= ALPHA * pivot_lambda * pivot_lambda)	goto PIVOT_1;

		double Arr = 0; // on a working submatrix G[k:M, k:N]
		for(int i = k; i < M; ++i) Arr += conj(G[i+M*pivot_r]) * J[i] * G[i+M*pivot_r];

		if(cabs(Arr) >= ALPHA * pivot_sigma){
			// gr is the pivot column
			// swap columns k <-> r
			// then do PIVOT_1 with Householder

			long int itemp = Pcol[pivot_r];
			Pcol[pivot_r] = Pcol[k];
			Pcol[k] = itemp;

			int inc = 1;
			zswap_(&M, &G[M*pivot_r], &inc, &G[M*k], &inc);
			Akk = Arr;

			goto PIVOT_1;
		}


		// ----------------------------------------------PIVOT_2-----------------------------------------------------
		// column swapping k+1 <-> pivot_r
		// then do givens

		printf("\nNEED PIVOT 2, but doing pivot 1 for now... k = %d\n\n", k);

		// k = k+1; (skip one)
		//goto LOOP_END;	// end of PIVOT_2, skipping PIVOT_1

		// ----------------------------------------------PIVOT_1-----------------------------------------------------
		PIVOT_1:

		// find gik so that |gik|/|Akk| closest to 1
		// swap rows i <-> k
		// but Ji should be sign(Akk)

		if( Akk > 0 ){

			int idx = k;
			double min = -1;

			for(int i = k; i < M; ++i){
				double temp = cabs(cabs(G[i+M*k])/cabs(Akk) - 1);
				if(J[i] > 0 && (min > temp || min < 0)){
					idx = i;
					min = temp;
				}
			}

			if(idx != k){

				double temp = J[k];
				J[k] = J[idx];
				J[idx] = temp;

				// swap rows in G 
				int inc = M;
				int Nk = N - k;
				zswap_(&Nk, &G[idx+M*k], &inc, &G[k+M*k], &inc);

				// update Prow
				long int itemp = Prow[idx];
				Prow[idx] = Prow[k];
				Prow[k] = itemp;
			}
		}

		else if( Akk < 0 ){

			int idx = k;
			double min = -1;

			for(int i = k; i < M; ++i){
				double temp = cabs(cabs(G[i+M*k])/cabs(Akk) - 1);
				if(J[i] < 0 && (min > temp || min < 0)){
					idx = i;
					min = temp;
				}
			}

			if(idx != k){

				double temp = J[k];
				J[k] = J[idx];
				J[idx] = temp;

				// swap rows in G 
				int inc = M;
				int Nk = N - k;
				zswap_(&Nk, &G[idx+M*k], &inc, &G[k+M*k], &inc);

				// update Prow
				long int itemp = Prow[idx];
				Prow[idx] = Prow[k];
				Prow[k] = itemp;
			}
		}


		// compute reflector constant H_sigma
		// compute vector f, where g*Jg = f*Jf must hold
		// that's why we need fk that looks like Hg = H_sigma*f = H_sigma*(sqrt(|sumk|), 0, ..., 0)

		double complex H_sigma = 1;
		if(cabs(G[k+M*k]) > eps0) H_sigma = -G[k+M*k] / cabs(G[k+M*k]);
		f[k] = csqrt(cabs(Akk)) * H_sigma;
		for(int i = k+1; i < M; ++i) f[i] = 0;


		// make the reflector
		// make the vector f(k:M)

		double complex alpha = -1;
		int inc = 1;
		int Mk = M - k;
		zaxpy_(&Mk, &alpha, &G[k+M*k], &inc, &f[k], &inc);	// f(k:M) = f(k:M) - g(k:M)


		double complex wJw = Akk + J[k] * (cabs(Akk) + 2 * csqrt(cabs(Akk)) * cabs(G[k+M*k]));
	
		for(int i = k; i < M; ++i)
			for(int j = k; j < M; ++j)
				H[i+M*j] = -2 * f[i] * conj(f[j]) * J[j] / wJw;
			
		for(int i = k; i < M; ++i) H[i+M*i] += 1;


		// apply the reflector on a submatrix

		char non_trans = 'N';
		alpha = 1;
		double complex beta = 0;
		int Nk = N - k;
		zgemm_(&non_trans, &non_trans, &Mk, &Nk, &Mk, &alpha, &H[k+M*k], &M, &G[k+M*k], &M, &beta, T, &Mk);	// T = HG

		inc = 1;
		for(int j = 0; j < Nk; ++j) zcopy_(&Mk, &T[j*Mk], &inc, &G[k+M*(j+k)], &inc);	// G = T (copy blocks)

		LOOP_END: continue;
	}


	// -------------------------------- writing -------------------------------- 	

	FILE *writeG = fopen("data/reducedG.bin", "wb");
	FILE *writeJ = fopen("data/reducedJ.bin", "wb");
	FILE *writeCol = fopen("data/Pcol.bin", "wb");
	FILE *writeRow = fopen("data/Prow.bin", "wb");

	// write J and Prow
	for(int i = 0; i < M; ++i){
		fprintf(writeJ, "%ld ", (long int)J[i]);
		fprintf(writeRow, "%ld ", Prow[i]);
	}

	// write G and pcol
	for(int j = 0; j < N; ++j){
		fprintf(writeCol, "%ld ", Pcol[j]);
		for(int i = 0; i < M; ++i){
			fprintf(writeG, "%lf %lf ", creal(G[i+M*j]), cimag(G[i+M*j]));
		}
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