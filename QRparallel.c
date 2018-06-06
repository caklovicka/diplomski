// call as:
//
// ./GFreduction.out G.bin J.bin M N
//
// (G.bin = filename where to store G in binary, same as J.bin)

// exits:
//   -1 ....... Cannot open file.
//   -2 ....... Cannot allocate memory.
//	 -4 ....... Algorithm broke down.
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


#define EPSILON DBL_EPSILON
#define DIGITS DBL_DIG
double ALPHA = (1.0 + csqrt(17.0))/8.0; //Bunch-Parlett alpha

void printMatrix(double complex *G, int M, int N){

	int i, j;
	for( i = 0; i < M; ++i ){
		for( j = 0; j < N; ++j ){
			printf("%10.5g + i%10.5g  ", creal(G[i+M*j]), cimag(G[i+M*j]));
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

	// read variables from command line
	int M = atoi(argv[3]);
	int N = atoi(argv[4]);
	FILE *readG = fopen(argv[1], "rb");
	FILE *readJ = fopen(argv[2], "rb");

	//printf("\n\n--------------------------------- ALGORITHM ------------------------------------\n");
	//printf("\nReading data...\n");

	// allocate memory
	double complex *G = (double complex*) malloc(M*N*sizeof(double complex));
	double complex *H = (double complex*) malloc(M*M*sizeof(double complex));	// reflector
	double complex *T = (double complex*) malloc(M*N*sizeof(double complex));	// temporary matrix
	double complex *U = (double complex*) malloc(16*sizeof(double complex));	// matrix of rotatoins
	double *J = (double*) malloc(M*sizeof(double));
	long int *Prow = (long int*) malloc(M*sizeof(long int));	// for row permutation
	long int *Pcol = (long int*) malloc(N*sizeof(long int));	// for column permutation
	double complex *f = (double complex*) malloc(M*sizeof(double complex));	// vector f
	double complex *tempf = (double complex*) malloc(M*sizeof(double complex));	// vector tempf, fisrt column after Householder transform


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

	// --------------------------------------- file reading ----------------------------------------

	double start = omp_get_wtime();

	// read matrix G and prepare Pcol

	for(int j = 0; j < N; ++j ){
		Pcol[j] = j;

		for(int i = 0; i < M; ++i ){
			double x, y;
			fscanf(readG, "%lg %lg ", &x, &y);
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

	double end = omp_get_wtime();
	double seconds = (double)(end - start);
	printf("reading time = %lg s\n", seconds);


	// ---------------------------------------------------------- ALGORITHM ----------------------------------------------------------

	//printf("Pivoting QR...\n\n");

	start = omp_get_wtime();

	for(int k = 0; k < N; ++k){

		// ------------------------ choosing a pivoting strategy (partial pivoting) -------------------------------
		// we need to know the signum of the J-norm of the first column
		// because the pivoting element, Akk, will have to satisfy
		// sign(Jk) = sign(gk*Jgk)

		double Akk = 0;	// Akk = gk* J gk, on a working submatrix G[k:M, k:N]
		double pivot_lambda = 0;
		double pivot_sigma = 0;
		int pivot_r = -1;	// 2nd column for partial pivoting
							// will be used for column swap k+1 <-> pivot_r when PIVOT_2 begins
		
		// compute Akk for the working submatrix G[k:M, k:N]
		#pragma omp parallel for reduction(+:Akk)
		for(int i = k; i < M; ++i) Akk += conj(G[i+M*k]) * J[i] * G[i+M*k];		

		if(k == N-1) goto PIVOT_1;


		// find pivot_lambda

		#pragma omp parallel for
		for(int i = k+1; i < N; ++i){
			double complex Aik = 0;		//Aik = gi* J gk, but on a submatrix G[k:M, k:N]

			#pragma omp parallel for reduction(+:Aik)
			for(int j = k; j < M; ++j) Aik += conj(G[j+M*i]) * J[j] * G[j+M*k];

			#pragma omp critical
			if(pivot_lambda < cabs(Aik)){
				pivot_lambda = cabs(Aik);
				pivot_r = i;
			}
		}

		if(cabs(Akk) >= ALPHA * pivot_lambda) goto PIVOT_1;


		// find pivot_sigma

		#pragma omp parallel for
		for(int i = k; i < N; ++i){

			if(i == pivot_r) continue;
			double complex Air = 0;  //Air = gi* J gr, but on a submatrix G[k:M, k:N]

			#pragma omp parallel for reduction(+:Air)
			for(int j = k; j < M; ++j)	Air += conj(G[j+M*i]) * J[j] * G[j+M*pivot_r];

			#pragma omp critical
			if(pivot_sigma < cabs(Air)) pivot_sigma = cabs(Air);
		}

		if(cabs(Akk) * pivot_sigma >= ALPHA * pivot_lambda * pivot_lambda) goto PIVOT_1;


		double Arr = 0; // on a working submatrix G[k:M, k:N]

		#pragma omp parallel for reduction(+:Arr)
		for(int i = k; i < M; ++i) Arr += conj(G[i+M*pivot_r]) * J[i] * G[i+M*pivot_r];

		if(cabs(Arr) >= ALPHA * pivot_sigma){
			// gr is the pivot column 
			// swap columns k <-> r
			// then do PIVOT_1 with Householder

			long int itemp = Pcol[pivot_r];
			Pcol[pivot_r] = Pcol[k];
			Pcol[k] = itemp;

			int inc = 1;
			int nthreads = 8;
			int offset = M / nthreads;
			int leftovers = M % offset;

			// this means that if a single thread needs to copy less than 25 elements, 
			// we shift it to a sequental way (this is the case when M < 25*8 = 200)
			// M < 200 -> sequential
			// M >= 200 -> parallel
			if(offset < 25){	
				offset = 0;
				leftovers = M;
			}

			printf("offset = %d, leftovers = %d\n", offset, leftovers);

			if(offset > 0){
				int i = 0;
				#pragma omp parallel for num_threads(nthreads)
				for(i = 0; i <= M - offset; i += offset){
					zswap_(&offset, &G[i + M*pivot_r], &inc, &G[i + M*k], &inc);
				}
			}

			zswap_(&leftovers, &G[M - leftovers + M*pivot_r], &inc, &G[M - leftovers + M*k], &inc);

			Akk = Arr;

			goto PIVOT_1;
		}


		// ----------------------------------------------PIVOT_2-----------------------------------------------------

		// do a column swap pivot_r <-> k+1 if needed

		if(pivot_r != k+1){

			long int itemp = Pcol[pivot_r];
			Pcol[pivot_r] = Pcol[k+1];
			Pcol[k+1] = itemp;

			int inc = 1;
			zswap_(&M, &G[M*pivot_r], &inc, &G[M*(k+1)], &inc);
		}


		// making the kth column real...

		int first_non_zero_idx = -1;	// index of the first non zero element in column k

		for(int i = k; i < M; ++i){

			if(cabs(G[i+M*k]) < EPSILON) continue;
			if(first_non_zero_idx == -1) first_non_zero_idx = i;

			double complex scal = conj(G[i+M*k]) / cabs(G[i+M*k]);
			G[i+M*k] = cabs(G[i+M*k]);	// to be exact, so that the Img part is really = 0
			int Nk = N-k-1;
			zscal_(&Nk, &scal, &G[i+M*(k+1)], &M);
		}


		// do row swap if needed, so thath Gkk != 0

		if(first_non_zero_idx != k && first_non_zero_idx != -1){ 

			long int itemp = Prow[first_non_zero_idx];
			Prow[first_non_zero_idx] = Prow[k];
			Prow[k] = itemp;

			double dtemp = J[first_non_zero_idx];
			J[first_non_zero_idx] = J[k];
			J[k] = dtemp;

			int Nk = N - k;
			zswap_(&Nk, &G[k+M*k], &M, &G[first_non_zero_idx + M*k], &M);
		}


		// do plane rotations with Gkk on all elements with signum Jk 
		// (if they are not 0 already), if they are, do nothing

		if(first_non_zero_idx != -1){

			for(int i = k+1; i < M; ++i){
			
				if(J[k] != J[i]) continue;

				// generate plane rotation
				double c;
				double complex s;
				double complex eliminator = G[k+M*k];
				double complex eliminated = G[i+M*k];
				zrotg_(&eliminator, &eliminated, &c, &s);

				// apply the rotation
				int Nk = N-k;
				zrot_(&Nk, &G[k+M*k], &M, &G[i+M*k], &M, &c, &s);
				G[i+M*k] = 0;
			}
		}


		// find the first i so that Ji = -Jk and G(i, k) != 0
		// then swap rows k+1 <-> i

		first_non_zero_idx = -1;	// first non zero element in the -Jk class
		for(int i = k+1; i < M; ++i){

			if(J[k] == J[i] || cabs(G[i+M*k]) < EPSILON ) continue;

			first_non_zero_idx = i;
			if(i == k+1) break; 	// no swapping needed, everythinig already in the right position


			// else, swap rows i <-> k+1

			long int itemp = Prow[i];
			Prow[i] = Prow[k+1];
			Prow[k+1] = itemp;

			double dtemp = J[i];
			J[i] = J[k+1];
			J[k+1] = dtemp;

			int Nk = N-k;
			zswap_(&Nk, &G[k+1+M*k], &M, &G[i+M*k], &M);
			break;
		}


		// do plane rotations on G(k+1, k+1), if they are not 0 already
		// if they are all 0 in the -Jk class, then first_non_zero_idx = -1;

		if(first_non_zero_idx != -1){

			for(int i = k+2; i < M; ++i){
			
				if(J[k+1] != J[i]) continue;

				// generate plane rotation
				double c;
				double complex s;
				double complex eliminator = G[k+1+M*k];
				double complex eliminated = G[i+M*k];
				zrotg_(&eliminator, &eliminated, &c, &s);

				// apply the rotation
				int Nk = N-k;
				zrot_(&Nk, &G[k+1+M*k], &M, &G[i+M*k], &M, &c, &s);
				G[i+M*k] = 0;
			}
		}


		// do the same thing on a second column
		// wee need to know if the kth column has 1 or 2 nonzeros elements
		// that determines in which case we are... (A1), (A2), (A3), (B1) or (B2)

		int kth_nonzeros = 2;
		if(first_non_zero_idx == -1) kth_nonzeros = 1;


		// making the (k+1)th column real

		first_non_zero_idx = -1;	// index of the first non zero element in column k+1
									// will be filled with the first nonzero index, or stay -1

		for(int i = k + kth_nonzeros; i < M; ++i){

			if(cabs(G[i+M*(k+1)]) < EPSILON) continue;
			if(first_non_zero_idx == -1) first_non_zero_idx = i;

			double complex scal = conj(G[i+M*(k+1)]) / cabs(G[i+M*(k+1)]);
			G[i+M*(k+1)] = cabs(G[i+M*(k+1)]);	// to be exact, so that the Img part is really = 0
			int Nk = N - k - 2;
			zscal_(&Nk, &scal, &G[i+M*(k+2)], &M);
		}

	
		int kkth_nonzeros = 0;	// number of nonzero elements in the (k+1)st column, but those below kth_nonzeros
		
		// if first_non_zero_idx != -1 at this point, then we >=1 elements != 0 in column (k+1) below (k+kth_nonzeros) 
		if(first_non_zero_idx != -1) kkth_nonzeros += 1; 
 

		// do row swap if needed, so thath G(k+kth_nonzeros, k+1) != 0
		// if first_non_zero_idx == k + kth_nonzeros, then thats fine, nothing to swap

		if(first_non_zero_idx != k + kth_nonzeros && first_non_zero_idx != -1){ 

			long int itemp = Prow[first_non_zero_idx];
			Prow[first_non_zero_idx] = Prow[k + kth_nonzeros];
			Prow[k + kth_nonzeros] = itemp;

			double dtemp = J[first_non_zero_idx];
			J[first_non_zero_idx] = J[k + kth_nonzeros];
			J[k + kth_nonzeros] = dtemp;

			int Nk = N - k - 1;
			zswap_(&Nk, &G[k + kth_nonzeros + M*(k+1)], &M, &G[first_non_zero_idx + M*(k+1)], &M);
		}


		// do plane rotations with G(k + kth_nonzeros, k+1) on all elements with sign J(k+kth_nonzeros) 
		// (if they are not 0 already), if they are, do nothing

		if(first_non_zero_idx != -1){

			for(int i = k + kth_nonzeros + 1; i < M; ++i){
			
				if(J[k + kth_nonzeros] != J[i]) continue;

				// generate plane rotation
				double c;
				double complex s;
				double complex eliminator = G[k + kth_nonzeros + M*(k+1)];
				double complex eliminated = G[i+M*(k+1)];
				zrotg_(&eliminator, &eliminated, &c, &s);

				// apply the rotation
				int Nk = N-k-1;
				zrot_(&Nk, &G[k + kth_nonzeros + M*(k+1)], &M, &G[i+M*(k+1)], &M, &c, &s);
				G[i+M*(k+1)] = 0;
			}
		}

		// find the first i so that Ji = -J(k + kth_nonzeros) and G(i, k + kth_nonzeros + 1) != 0
		// then swap rows k + kth_nonzeros + 1 <-> i

		first_non_zero_idx = -1;	// first non zero element in the -J(k + kth_nonzeros) class
		for(int i = k + kth_nonzeros + 1; i < M; ++i){

			if(J[k + kth_nonzeros] == J[i] || cabs(G[i+M*(k+1)]) < EPSILON ) continue;

			first_non_zero_idx = i;
			if(i == k + kth_nonzeros + 1) break; 	// no swapping needed, everythinig already in the right position


			// else, swap rows i <-> k + kth_nonzeros + 1

			long int itemp = Prow[i];
			Prow[i] = Prow[k + kth_nonzeros + 1];
			Prow[k + kth_nonzeros + 1] = itemp;

			double dtemp = J[i];
			J[i] = J[k + kth_nonzeros + 1];
			J[k + kth_nonzeros + 1] = dtemp;

			int Nk = N-k-1;
			zswap_(&Nk, &G[k + kth_nonzeros + 1 + M*(k+1)], &M, &G[i+M*(k+1)], &M);
			break;
		}


		// if first_non_zero_idx != -1 at this point, then we 2 elements != 0 in column (k+1) below (k+kth_nonzeros) 
		if(first_non_zero_idx != -1) kkth_nonzeros += 1; 

		// do plane rotations with G(k + kth_nonzeros + 1, k+1), if they are not 0 already
		// if they are all 0 in the -J(k + kth_nonzeros) class, then first_non_zero_idx = -1;

		if(first_non_zero_idx != -1){

			for(int i = k + kth_nonzeros + 2; i < M; ++i){
			
				if(J[k + kth_nonzeros + 1] != J[i]) continue;

				// generate plane rotation
				double c;
				double complex s;
				double complex eliminator = G[k + kth_nonzeros + 1 + M*(k+1)];
				double complex eliminated = G[i + M*(k+1)];
				zrotg_(&eliminator, &eliminated, &c, &s);

				// apply the rotation
				int Nk = N-k-1;
				zrot_(&Nk, &G[k + kth_nonzeros + 1 + M*(k+1)], &M, &G[i+M*(k+1)], &M, &c, &s);
				G[i+M*(k+1)] = 0;
			}
		}

		// if we are in (A3) or (B2) forms, then the 2x2 reduction is finished
		// in that case continue the main loop

		// condition (A3)
		if(kth_nonzeros == 2 && kkth_nonzeros == 0) goto LOOP_END;

		// condition (B2)
		if(kth_nonzeros == 1 && kkth_nonzeros == 1) goto LOOP_END;


		// handle the (A1) form

		if(kth_nonzeros == 2 && kkth_nonzeros == 2){

			//printf("\tPIVOT (A1)\n");

			// check if its a proper form
			// if not, fix it

			if( cabs(G[k+M*k] * G[k+1+M*(k+1)] - G[k+1+M*k] * G[k+M*(k+1)]) < EPSILON){

				// swap columns k <-> k+1

				long int itemp = Pcol[k];
				Pcol[k] = Pcol[k+1];
				Pcol[k+1] = itemp;

				int n = k + 4;
				int inc = 1;
				zswap_(&n, &G[k+M*k], &inc, &G[k+M*(k+1)], &inc);

				// make the kth rows k, k+1 real (k+3 and k+2 are already real)

				for(int i = k; i < k+2; ++i){

					if( cabs(cimag(G[i+M*k])) < EPSILON ) continue; //the element is already real

					double complex scal = conj(G[i+M*k]) / cabs(G[i+M*k]);
					G[i+M*k] = cabs(G[i+M*k]);	// to be exact, so that the Img part is really = 0
					int Nk = N - k - 1;
					zscal_(&Nk, &scal, &G[i+M*(k+1)], &M);	
				}

				// do plane rotations with kth row

				int idx = k+2; 	// idx is the row that will be eliminated with the kth row
				if(J[k] == J[k+3]) idx = k+3;

				// generate plane rotation
				double c;
				double complex s;
				double complex eliminator = G[k+M*k];
				double complex eliminated = G[idx+M*k];
				zrotg_(&eliminator, &eliminated, &c, &s);

				// apply the rotation
				int Nk = N-k;
				zrot_(&Nk, &G[k+M*k], &M, &G[idx+M*k], &M, &c, &s);
				G[idx+M*k] = 0;

				// do plane rotations with (k+1)st row

				idx = k+2; 	// idx it the row that will be eliminated with the kth row
				if(J[k+1] == J[k+3]) idx = k+3;

				// generate plane rotation
				eliminator = G[k+1+M*k];
				eliminated = G[idx+M*k];
				zrotg_(&eliminator, &eliminated, &c, &s);

				// apply the rotation
				Nk = N-k;
				zrot_(&Nk, &G[k+1+M*k], &M, &G[idx+M*k], &M, &c, &s);
				G[idx+M*k] = 0;	
			}


			// now do the final reduction

			double complex g11 = G[k+M*k];
			double complex g21 = G[k+1+M*k];
			double complex g12 = G[k+M*(k+1)];
			double complex g22 = G[k+1+M*(k+1)];
			double complex g32 = G[k+2+M*(k+1)];
			double complex g42 = G[k+3+M*(k+1)];

			double complex z = 1.0 / (g11 * g22 - g21 * g12);	// z = 1/detG1
			double complex a = J[k] * J[k+2] * cabs(z*z) * (g21*g21 - g11*g11) * (g32*g32 - g42*g42);
			double complex r = 1.0 / csqrt(1+a);
			double complex y = J[k] * J[k+2] * r * conj(z);

			double complex i_xy = (J[k] * J[k+2] * r * cabs(z*z) * (g21*g21 - g11*g11)) / (1 + csqrt(1+a));
			double complex i_yx = (J[k] * J[k+2] * r * cabs(z*z) * (g32*g32 - g42*g42)) / (1 + csqrt(1+a));

			// making the matrix U dimensions 4x4
			// first column
			U[0] = 1 - i_yx * g21 * g21;
			U[1] = -i_yx * g11 * g21;
			U[2] = r * z * g21 * g32;
			U[3] = r * z * g21 * g42;
			// second column
			U[4] = i_yx * g11 * g21;
			U[5] = 1 + i_yx * g11 * g11;
			U[6] = -r * z * g11 * g32;
			U[7] = -r * z * g11 * g42;
			// third column
			U[8] = -y * g21 * g32;
			U[9] = -y * g11 * g32;
			U[10] = 1 - i_xy * g32 * g32;
			U[11] = -i_xy * g32 * g42;
			// fourth column
			U[12] = y * g21 * g42;
			U[13] = y * g11 * g42;
			U[14] = i_xy * g32 * g42;
			U[15] = 1 + i_xy * g42 * g42;

			// apply the reduction

			int n = 4;
			int Nk = N - k;
			char non_trans = 'N';
			double complex alpha = 1;
			double complex beta = 0;
			zgemm_(&non_trans, &non_trans, &n, &Nk, &n, &alpha, U, &n, &G[k+M*k], &M, &beta, T, &n);


			// copy rows of T back into G
			for(int i = 0; i < 4; ++i) zcopy_(&Nk, &T[i], &n, &G[k + i + M*k], &M);

			// put zeros explicitly in the right places
			G[k+2+M*k] = 0;
			G[k+3+M*k] = 0;
			G[k+2+M*(k+1)] = 0;
			G[k+3+M*(k+1)] = 0;

			k = k+1;
			goto LOOP_END;
		}


		// handle forms (A2) and (B1)

		else if(kth_nonzeros == 2 && kkth_nonzeros == 1 || kth_nonzeros == 1 && kkth_nonzeros == 2){

			// check if its a proper form
			// if not, fix it
			// treba li na drugi nacin izracunat determinanticu??

			//printf("\tPIVOT (A2) or (B1)\n");

			// (B1) is already in proper form, so fix just form (A2) if needed

			if( kth_nonzeros == 2 && cabs(G[k+M*k] * G[k+1+M*(k+1)] - G[k+1+M*k] * G[k+M*(k+1)]) < EPSILON ){

				// swap columns k <-> k+1

				long int itemp = Pcol[k];
				Pcol[k] = Pcol[k+1];
				Pcol[k+1] = itemp;

				int n = k + 3;
				int inc = 1;
				zswap_(&n, &G[k+M*k], &inc, &G[k+M*(k+1)], &inc);

				// make the kth rows k, k+1 real (k+2 is already real)

				for(int i = k; i < k+2; ++i){

					if( cabs(cimag(G[i+M*k])) < EPSILON ) continue; //the element is already real

					double complex scal = conj(G[i+M*k]) / cabs(G[i+M*k]);
					G[i+M*k] = cabs(G[i+M*k]);	// to be exact, so that the Img part is really = 0
					int Nk = N - k - 1;
					zscal_(&Nk, &scal, &G[i+M*(k+1)], &M);	
				}

				// do a plane rotation, eliminate row k+2 with row idx

				int idx = k; 	// idx is the row that will eliminate row k+2
				if(J[idx] != J[k+2]) idx = k+1;

				// generate plane rotation
				double c;
				double complex s;
				double complex eliminator = G[idx+M*k];
				double complex eliminated = G[k+2+M*k];
				zrotg_(&eliminator, &eliminated, &c, &s);

				// apply the rotation
				int Nk = N-k;
				zrot_(&Nk, &G[idx+M*k], &M, &G[k+2+M*k], &M, &c, &s);
				G[k+2+M*k] = 0;
			}


			// now do the final reduction

			double complex g11 = G[k+M*k];
			double complex g21 = G[k+1+M*k];
			double complex g12 = G[k+M*(k+1)];
			double complex g22 = G[k+1+M*(k+1)];
			double complex g32 = G[k+2+M*(k+1)];

			double complex z = 1 / (g11 * g22 - g21 * g12);	// z = 1/detG1
			double complex a = J[k] * J[k+2] * cabs(z*z) * (g21*g21 - g11*g11) * g32*g32;
			double complex r = 1/csqrt(1+a);
			double complex y = J[k] * J[k+2] * r * conj(z);

			double complex i_yx = (J[k] * J[k+2] * r * cabs(z*z) * g32*g32) / (1 + csqrt(1+a));

			// making the matrix U dimensions 3x3
			// first column
			U[0] = 1 - i_yx * g21 * g21;
			U[1] = -i_yx * g11 * g21;
			U[2] = r * z * g21 * g32;
			// second column
			U[3] = i_yx * g11 * g21;
			U[4] = 1 + i_yx * g11 * g11;
			U[5] = -r * z * g11 * g32;
			// third column
			U[6] = -y * g21 * g32;
			U[7] = -y * g11 * g32;
			U[8] = r;

			// apply the reduction
			// g11 and g21 remain unchanged

			int n = 3;
			int Nk = N - k - 1;
			char non_trans = 'N';
			double complex alpha = 1;
			double complex beta = 0;
			zgemm_(&non_trans, &non_trans, &n, &Nk, &n, &alpha, U, &n, &G[k+M*(k+1)], &M, &beta, T, &n);


			// copy rows of T back into G

			for(int i = 0; i < 3; ++i) zcopy_(&Nk, &T[i], &n, &G[k + i + M*(k+1)], &M);

			// put zeros explicitly in the right places
			G[k+2+M*k] = 0;
			G[k+2+M*(k+1)] = 0;

			k = k+1;
			goto LOOP_END;
		}

		// if here, something broke down
		printf("\n\n\nUPS, SOMETHING WENT WRONG... STOPPING...Is A maybe singular?\n\n\n");
		exit(-4);

		// ----------------------------------------------PIVOT_1-----------------------------------------------------
		PIVOT_1:
		// check the condition sign(Akk) = Jk
		// if not, do row swap and diagonal swap in J

		//printf("PIVOT_1, k = %d\n", k);

		if( Akk > 0 && J[k] < 0){

			int i;
			for(i = k+1; i < M; ++i)
				if(J[i] > 0) break;

			J[k] = 1.0;
			J[i] = -1.0;

			// swap rows in G 
			int Nk = N - k;
			zswap_(&Nk, &G[i+M*k], &M, &G[k+M*k], &M);

			// update Prow
			long int itemp = Prow[i];
			Prow[i] = Prow[k];
			Prow[k] = itemp;
		}

		else if( Akk < 0 && J[k] > 0){

			int i;
			for(i = k+1; i < M; ++i)
				if(J[i] < 0) break;

			J[k] = -1.0;
			J[i] = 1.0;

			// swap rows in G 
			int Nk = N - k;
			zswap_(&Nk, &G[i+M*k], &M, &G[k+M*k], &M);

			// update Prow
			long int itemp = Prow[i];
			Prow[i] = Prow[k];
			Prow[k] = itemp;
		}
		

		// compute reflector constant H_sigma
		// compute vector f, where g*Jg = f*Jf must hold
		// that's why we need fk that looks like Hg = H_sigma*f = H_sigma*(sqrt(|sumk|), 0, ..., 0)

		double complex H_sigma = 1;
		if(cabs(G[k+M*k]) >= EPSILON) H_sigma = -G[k+M*k] / cabs(G[k+M*k]);
		f[k] = csqrt(cabs(Akk)) * H_sigma;
		for(int i = k+1; i < M; ++i) f[i] = 0;


		// make the reflector
		// make the vector f(k:M)

		double complex alpha = -1;
		int inc = 1;
		int Mk = M - k;

		zcopy_(&Mk, &f[k], &inc, &tempf[k], &inc); // copy f into tempf, so we dont need tu multyply the first column of G with H
		zaxpy_(&Mk, &alpha, &G[k+M*k], &inc, &f[k], &inc);	// f(k:M) = f(k:M) - g(k:M)


		double complex wJw = Akk + J[k] * (cabs(Akk) + 2 * csqrt(cabs(Akk)) * cabs(G[k+M*k]));

		for(int i = k; i < M; ++i)
			for(int j = k; j < M; ++j)
				H[i+M*j] = -2 * f[i] * conj(f[j]) * J[j] / wJw;
			
		for(int i = k; i < M; ++i) H[i+M*i] += 1;


		// apply the reflector on a submatrix

		char non_trans = 'N';
		alpha = 1.0;
		double complex beta = 0;

		int Nk = N - k - 1;
		zgemm_(&non_trans, &non_trans, &Mk, &Nk, &Mk, &alpha, &H[k+M*k], &M, &G[k+M*(k+1)], &M, &beta, T, &Mk);	// T = HG(k:M, k+1:N)

		inc = 1;
		zcopy_(&Mk, &tempf[k], &inc, &G[k+M*k], &inc);
		for(int j = 0; j < Nk; ++j) zcopy_(&Mk, &T[j*Mk], &inc, &G[k + M*(j+k+1)], &inc);	// G = T (copy blocks)

		LOOP_END: continue;
	}

	end = omp_get_wtime();
	seconds = (double)(end - start);
	printf("algortihm time = %lg s\n", seconds);


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
			fprintf(writeG, "%.*g %.*g ", DIGITS, DIGITS, creal(G[i+M*j]), cimag(G[i+M*j]));
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
	free(tempf);
	free(U);


	//printf("\nFinished.\n");
	//printf("\n-------------------------------------------------------------------------------\n\n");
	return(0);
}