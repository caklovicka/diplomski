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
#define eps 0.2
#define refresh 30

#define D 64


void printMatrix(double complex *G, int M, int N){

	int i, j;
	for( i = 0; i < M; ++i ){
		for( j = 0; j < N; ++j ){
			printf("%15.10g + i%15.10g  ", creal(G[i+M*j]), cimag(G[i+M*j]));
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
	double complex *norm = (double complex*) mkl_malloc(N*sizeof(double complex), 64);	// for quadrates of J-norms of columns
	//double complex *H = (double complex*) mkl_malloc(M*M*sizeof(double complex), 64);	// reflector
	double complex *T = (double complex*) mkl_malloc(4*M*sizeof(double complex), 64);	// temporary matrix
	double complex *U = (double complex*) mkl_malloc(16*sizeof(double complex), 64);	// matrix of rotatoins
	int *p = (int*) mkl_malloc(M*sizeof(int), 64);	// for location of +1 in J for givens reduction
	int *n = (int*) mkl_malloc(M*sizeof(int), 64);  // for location of -1 in J for givens reduction
	double *J = (double*) mkl_malloc(M*sizeof(double), 64);
	long int *Prow = (long int*) mkl_malloc(M*sizeof(long int), 64);	// for row permutation
	long int *Pcol = (long int*) mkl_malloc(N*sizeof(long int), 64);	// for column permutation
	double complex *f = (double complex*) mkl_malloc(M*sizeof(double complex), 64);	// vector f
	//double complex *tempf = (double complex*) mkl_malloc(M*sizeof(double complex), 64);	// vector tempf, fisrt column after Householder transform


	// check if files are opened

	if(readG == NULL || readJ == NULL){
		printf("Cannot open file.\n");
		exit(-1);
	}

	// check if memory is allocated

	if(G == NULL || J == NULL || Pcol == NULL || Prow == NULL || T == NULL || f == NULL || p == NULL || n == NULL || norm == NULL){
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

	double end = omp_get_wtime();
	double seconds = (double)(end - start);
	printf("reading time = %lg s\n", seconds);


	// ---------------------------------------------------------- ALGORITHM ----------------------------------------------------------

	//printf("Pivoting QR...\n\n");

	double pivot2time = 0;
	double pivot1time = 0;
	double mnozenjetime = 0;
	double redukcijatime = 0;
	double pivotiranje = 0;
	int pivot_1_count = 0;
	int pivot_2_count = 0;
	int last_pivot = -1;
	start = omp_get_wtime();

	int i, j, k, nthreads;


	// first compute J-norms of matrix G

	printf("poceo racunati norme...\n");

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

		//printf("k = %d\n", k);

		// ------------------------ choosing a pivoting strategy (partial pivoting) -------------------------------

		// ------------------------ update J-norms of columns ------------------------

		if( k && !( k % refresh == 0 || (k % refresh == 1 && last_pivot == 2) ) ){	// if we have something to update

			#pragma omp parallel num_threads( nthreads )
			{
				#pragma omp for nowait
				for( j = k; j < N; ++j){

					// pivot 1 was last
					if( last_pivot == 1 ){

						double denomi = conj(G[k-1+M*j]) * J[k-1] * G[k-1+M*j];
						double frac = cabs(norm[j]) / cabs(denomi);

						// not a case of catastrophic cancellation
						if( creal(norm[j]) * denomi < 0 || cabs(frac - 1) > eps )
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
						if( creal(norm[j]) * denomi < 0 || cabs(frac - 1) > eps)
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

			double complex ctemp = norm[pivot_r];
			norm[pivot_r] = norm[k+1];
			norm[k+1] = ctemp;

			int inc = 1;
			int mkl_nthreads = M/D > mkl_get_max_threads() ? M/D : mkl_get_max_threads();
			if(M/D == 0) mkl_nthreads = 1;
			mkl_set_num_threads(mkl_nthreads);
			zswap(&M, &G[M*pivot_r], &inc, &G[M*(k+1)], &inc);
		}

		int first_non_zero_idx = -1;	// index of the first non zero element in column k

		// [SEQUENTIAL] find the first non zero element in the kth column
		for(i = k; i < M; ++i){

			if(cabs(G[i+M*k]) < EPSILON) continue;
			first_non_zero_idx = i;
			break;
		}

		if(first_non_zero_idx == -1){
			printf("Zero column\n");
			exit(-3);
		}


		// do row swap if needed, so that Gkk != 0

		if(first_non_zero_idx != k){

			long int itemp = Prow[first_non_zero_idx];
			Prow[first_non_zero_idx] = Prow[k];
			Prow[k] = itemp;

			double dtemp = J[first_non_zero_idx];
			J[first_non_zero_idx] = J[k];
			J[k] = dtemp;

			int Nk = N - k;
			int mkl_nthreads = Nk/D > mkl_get_max_threads() ? Nk/D : mkl_get_max_threads();
			if(Nk/D == 0) mkl_nthreads = 1;
			mkl_set_num_threads(mkl_nthreads);
			zswap(&Nk, &G[k + M*k], &M, &G[first_non_zero_idx + M*k], &M);
		}

		first_non_zero_idx = -1;

		// [SEQUENTIAL] find the first non zero element in the kth column of class -J[k]
		for(i = k+1; i < M; ++i){

			if(cabs(G[i+M*k]) < EPSILON && J[k] == J[k+1]) continue;
			first_non_zero_idx = i;
			break;
		}

		// do row swap if needed, so that Gk,k+1 != 0

		if(first_non_zero_idx != k+1){

			long int itemp = Prow[first_non_zero_idx];
			Prow[first_non_zero_idx] = Prow[k];
			Prow[k] = itemp;

			double dtemp = J[first_non_zero_idx];
			J[first_non_zero_idx] = J[k];
			J[k] = dtemp;

			int Nk = N - k;
			int mkl_nthreads = Nk/D > mkl_get_max_threads() ? Nk/D : mkl_get_max_threads();
			if(Nk/D == 0) mkl_nthreads = 1;
			mkl_set_num_threads(mkl_nthreads);
			zswap(&Nk, &G[k + M*k], &M, &G[first_non_zero_idx + M*k], &M);
		}

		if (first_non_zero_idx != -1) kth_nonzeros = 2;
		else kth_nonzeros = 1;


		// make those rows real

		int mkl_nthreads = (N-k)/D > (mkl_get_max_threads()/kth_nonzeros) ? (N-k)/D : (mkl_get_max_threads()/kth_nonzeros);
		if((N-k)/D == 0 ) mkl_nthreads = 1;

		#pragma omp parallel for num_threads( kth_nonzeros )
		for(i = k; i < k + kth_nonzeros; ++i){

		    mkl_set_num_threads_local(mkl_nthreads);

		    double complex scal = conj(G[i + M*k]) / cabs(G[i + M*k]);
            G[i + M*k] = cabs(G[i + M*k]);
            int Nk = N - k - 1;
            zscal(&Nk, &scal, &G[i + M*(k+1)], &M);
		}


		// ------------ do Householder with kth_nonzeros elements in f -----------------

		if(kth_nonzeros == 1){

            // compute reflector constant H_sigma
            // compute vector f, where g*Jg = f*Jf must hold
            // that's why we need fk that looks like Hg = H_sigma*f = H_sigma*(sqrt(|sumk|), 0, ..., 0)

            double complex gkk;
            if(cabs(G[k+M*k]) >= EPSILON) gkk = -G[k+M*k] * csqrt(cabs(Akk)) / cabs(G[k+M*k]);
            else gkk = csqrt(cabs(Akk));

            // save the J norm of the vector
            double fJf = Akk + J[k] * (cabs(Akk) + 2 * csqrt(cabs(Akk)) * cabs(G[k+M*k]));

            // make the reflector vector and save it
            double complex alpha = -1;
            int inc = 1;
            int Mk = M - k;
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

            mkl_nthreads = Mk/D > mkl_get_max_threads()/nthreads ? Mk/D : mkl_get_max_threads()/nthreads;
            if (Mk/D == 0 || mkl_get_max_threads()/nthreads == 0) mkl_nthreads = 1;

            #pragma omp parallel num_threads( nthreads )
            {
                mkl_set_num_threads_local(1);

                #pragma omp for nowait
                for(j = k+1; j < N; ++j){

                    // T = Jf
                    // alpha = f*Jg
                    int Mk = M - k;
                    int inc = 1;
                    double complex alpha;
                    zdotc(&alpha, &Mk, &T[k], &inc, &G[k+M*j], &inc);
                    alpha = - 2 * alpha / fJf;
                    // G[k + M*j] = alpha * f[k] + G[k + M*k]
                    zaxpy(&Mk, &alpha, &f[k], &inc, &G[k + M*j], &inc);
                }
            }

		}

		else{

		    // compute reflector constant H_sigma
            // compute vector f, where g*Jg = f*Jf must hold
            // that's why we need fk that looks like Hg = H_sigma*f = H_sigma*(sqrt(|sum1|), sqrt(|sum2|), 0, ..., 0)
            // sum1 - sum2 = g*Jg, sum1 is of positives, sum2 of negatives

            double sum1 = 0;

            for(i = k; i < M; ++i){
                if(J[i] > 0) sum1 += conj(G[i+M*k]) * G[i+M*k];
            }

            double sum2 = cabs(norm[k] - sum1);

            if( J[k] < 0 ){

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

            if( J[k+1] > 0 ){

                for(i = k+1; i < M; ++i) if(J[i] < 0) break;

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

            double complex g1, g2;
            if(cabs(G[k+M*k]) >= EPSILON) g1 = -G[k+M*k] * csqrt(sum1) / cabs(G[k+M*k]);
            else g1 = csqrt(sum1);

            if(cabs(G[k+M*k]) >= EPSILON) g2 = -G[k+M*k] * csqrt(sum2) / cabs(G[k+M*k]);
            else g2 = csqrt(sum2);

            // save the J norm of the vector
            double fJf = Akk + J[k] * (cabs(Akk) + 2 * csqrt(cabs(Akk)) * cabs(G[k+M*k]));

            // make the reflector vector and save it
            double complex alpha = -1;
            int inc = 1;
            int Mk = M - k;
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

            mkl_nthreads = Mk/D > mkl_get_max_threads()/nthreads ? Mk/D : mkl_get_max_threads()/nthreads;
            if (Mk/D == 0 || mkl_get_max_threads()/nthreads == 0) mkl_nthreads = 1;

            #pragma omp parallel num_threads( nthreads )
            {
                mkl_set_num_threads_local(1);

                #pragma omp for nowait
                for(j = k+1; j < N; ++j){

                    // T = Jf
                    // alpha = f*Jg
                    int Mk = M - k;
                    int inc = 1;
                    double complex alpha;
                    zdotc(&alpha, &Mk, &T[k], &inc, &G[k+M*j], &inc);
                    alpha = - 2 * alpha / fJf;
                    // G[k + M*j] = alpha * f[k] + G[k + M*k]
                    zaxpy(&Mk, &alpha, &f[k], &inc, &G[k + M*j], &inc);
                }
            }

		}

		}


















		// -------- check forms od 2x2 pivot ---------


		// if we are in (A3) or (B2) forms, then the 2x2 reduction is finished
		// in that case continue the main loop

		// condition (A3)
		if(kth_nonzeros == 2 && kkth_nonzeros == 0) goto LOOP_END;

		// condition (B2)
		if(kth_nonzeros == 1 && kkth_nonzeros == 1) goto LOOP_END;


		//make rows real which need to be

		int mkl_nthreads = (N-k)/N > (mkl_get_max_threads()/2-2-kth_nonzeros-kkth_nonzeros) ? (N-k)/N : (mkl_get_max_threads()/2-2-kth_nonzeros-kkth_nonzeros);
		if((N-k)/N == 0 || mkl_nthreads <= 0) mkl_nthreads = 1;

		#pragma omp parallel num_threads(2)
		{
			if(omp_get_thread_num() == 0){

				#pragma omp parallel for num_threads( kth_nonzeros ) private(i)
				for(i = k; i < k + kth_nonzeros; ++i){

					if( cabs(cimag(G[i + M*k])) < EPSILON ) continue;

					mkl_set_num_threads_local(mkl_nthreads);

					double complex scal = conj(G[i + M*k]) / cabs(G[i + M*k]);
					G[i + M*k] = cabs(G[i + M*k]);
					int Nk = N - k - 1;
					zscal(&Nk, &scal, &G[i + M*(k+1)], &M);
				}
			}
			if(omp_get_thread_num() == 1){

				#pragma omp parallel for num_threads( kkth_nonzeros ) private(i)
				for(i = k + kth_nonzeros; i < k + kth_nonzeros + kkth_nonzeros; ++i){

					if( cabs(cimag(G[i + M*(k+1)])) < EPSILON) continue;

					mkl_set_num_threads_local(mkl_nthreads);

					double complex scal = conj(G[i + M*(k+1)]) / cabs(G[i + M*(k+1)]);
					G[i + M*(k+1)] = cabs(G[i + M*(k+1)]);
					int Nk = N - k - 2;
					zscal(&Nk, &scal, &G[i + M*(k+2)], &M);
				}
			}
		}

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

				double complex ctemp = norm[k];
				norm[k] = norm[k+1];
				norm[k+1] = ctemp;

				int n_ = k + 4;
				int inc = 1;
				int mkl_nthreads = n_/D > mkl_get_max_threads() ? n_/D : mkl_get_max_threads();
				if(n_/D == 0) mkl_nthreads = 1;

				mkl_set_num_threads(mkl_nthreads);
				zswap(&n_, &G[M*k], &inc, &G[M*(k+1)], &inc);

				mkl_nthreads = (N-k)/N > mkl_get_max_threads()-2 ? (N-k)/N : mkl_get_max_threads()-2;
				if((N-k)/D == 0 || mkl_nthreads <= 0) mkl_nthreads = 1;

				// make the kth rows k, k+1 real (k+3 and k+2 are already real)
				#pragma omp parallel for num_threads(2)
				for(i = k; i < k+2; ++i){

					if( cabs(cimag(G[i+M*k])) < EPSILON ) continue; //the element is already real

					double complex scal = conj(G[i+M*k]) / cabs(G[i+M*k]);
					G[i+M*k] = cabs(G[i+M*k]);	// to be exact, so that the Img part is really = 0
					int Nk = N - k - 1;

					mkl_set_num_threads_local(mkl_nthreads);
					zscal(&Nk, &scal, &G[i+M*(k+1)], &M);
				}

				// do plane rotations with kth row

				int idx = k+2; 	// idx is the row that will be eliminated with the kth row
				if(J[k] == J[k+3]) idx = k+3;

				// generate plane rotation
				double c;
				double complex s;
				double complex eliminator = G[k+M*k];
				double complex eliminated = G[idx+M*k];
				zrotg(&eliminator, &eliminated, &c, &s);

				// apply the rotation
				int Nk = N-k;
				mkl_nthreads = (N-k)/N > mkl_get_max_threads() ? (N-k)/N : mkl_get_max_threads();
				if((N-k)/D == 0) mkl_nthreads = 1;
				mkl_set_num_threads(mkl_nthreads);
				zrot(&Nk, &G[k+M*k], &M, &G[idx+M*k], &M, &c, &s);
				G[idx+M*k] = 0;

				// do plane rotations with (k+1)st row

				idx = k+2; 	// idx it the row that will be eliminated with the kth row
				if(J[k+1] == J[k+3]) idx = k+3;

				// generate plane rotation
				eliminator = G[k+1+M*k];
				eliminated = G[idx+M*k];
				zrotg(&eliminator, &eliminated, &c, &s);

				// apply the rotation
				Nk = N-k;
				zrot(&Nk, &G[k+1+M*k], &M, &G[idx+M*k], &M, &c, &s);
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

			int n_ = 4;
			int Nk = N - k;
			char non_trans = 'N';
			double complex alpha = 1;
			double complex beta = 0;

			int mkl_nthreads = Nk/D > mkl_get_max_threads() ? Nk/D : mkl_get_max_threads();
			if(Nk/D == 0) mkl_nthreads = 1;
			mkl_set_num_threads(mkl_nthreads);
			zgemm(&non_trans, &non_trans, &n_, &Nk, &n_, &alpha, U, &n_, &G[k+M*k], &M, &beta, T, &n_);

			mkl_nthreads = Nk/D > mkl_get_max_threads()-4 ? Nk/D : mkl_get_max_threads()-4;
			if(Nk/D == 0 || mkl_nthreads <= 0) mkl_nthreads = 1;

			// copy rows of T back into G
			#pragma omp parallel for num_threads(4)
			for(i = 0; i < 4; ++i){
				mkl_set_num_threads_local(mkl_nthreads);
				zcopy(&Nk, &T[i], &n_, &G[k + i + M*k], &M);
			}

			// put zeros explicitly in the right places
			G[k+2+M*k] = 0;
			G[k+3+M*k] = 0;
			G[k+2+M*(k+1)] = 0;
			G[k+3+M*(k+1)] = 0;

			k = k+1;

			double end2 = omp_get_wtime();
			pivot2time += (double) (end2 - start2);
			//printf("PIVOT_2 time = %lf\n", (double)(end2 - start2));
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

				printf("(not optimal) A2/B1 proper form, k = %d\n", k);

				// swap columns k <-> k+1

				long int itemp = Pcol[k];
				Pcol[k] = Pcol[k+1];
				Pcol[k+1] = itemp;

				double complex ctemp = norm[k];
				norm[k] = norm[k+1];
				norm[k+1] = ctemp;

				int n_ = k + 3;
				int inc = 1;
				int mkl_nthreads = n_/D > mkl_get_max_threads() ? n_/D : mkl_get_max_threads();
				if(n_/D == 0) mkl_nthreads = 1;
				mkl_set_num_threads(mkl_nthreads);
				zswap(&n_, &G[M*k], &inc, &G[M*(k+1)], &inc);

				mkl_nthreads = (N-k)/D > mkl_get_max_threads()-2 ? (N-k)/D : mkl_get_max_threads()-2;
				if((N-k)/D == 0 || mkl_nthreads <= 0) mkl_nthreads = 1;

				// make the kth rows k, k+1 real (k+2 is already real)
				#pragma omp parallel for num_threads(2)
				for(i = k; i < k+2; ++i){

					if( cabs(cimag(G[i+M*k])) < EPSILON ) continue; //the element is already real

					double complex scal = conj(G[i+M*k]) / cabs(G[i+M*k]);
					G[i+M*k] = cabs(G[i+M*k]);	// to be exact, so that the Img part is really = 0
					int Nk = N - k - 1;

					mkl_set_num_threads_local(mkl_nthreads);
					zscal(&Nk, &scal, &G[i+M*(k+1)], &M);
				}

				// do a plane rotation, eliminate row k+2 with row idx

				int idx = k; 	// idx is the row that will eliminate row k+2
				if(J[idx] != J[k+2]) idx = k+1;

				// generate plane rotation
				double c;
				double complex s;
				double complex eliminator = G[idx+M*k];
				double complex eliminated = G[k+2+M*k];
				zrotg(&eliminator, &eliminated, &c, &s);

				// apply the rotation
				mkl_nthreads = (N-k)/D > mkl_get_max_threads() ? (N-k)/D : mkl_get_max_threads();
				if((N-k)/D == 0) mkl_nthreads = 1;
				mkl_set_num_threads(mkl_nthreads);
				int Nk = N-k;
				zrot(&Nk, &G[idx+M*k], &M, &G[k+2+M*k], &M, &c, &s);
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

			int n_ = 3;
			int Nk = N - k - 1;
			char non_trans = 'N';
			double complex alpha = 1;
			double complex beta = 0;

			int mkl_nthreads = Nk/D > mkl_get_max_threads() ? Nk/D : mkl_get_max_threads();
			if(Nk/D == 0) mkl_nthreads = 1;
			mkl_set_num_threads(mkl_nthreads);
			zgemm_(&non_trans, &non_trans, &n_, &Nk, &n_, &alpha, U, &n_, &G[k+M*(k+1)], &M, &beta, T, &n_);

			mkl_nthreads = Nk/D > mkl_get_max_threads()-3 ? Nk/D : mkl_get_max_threads()-3;
			if(Nk/D == 0 || mkl_nthreads <= 0) mkl_nthreads = 1;

			//copy rows of T back into G
			#pragma omp parallel for num_threads(3)
			for(i = 0; i < 3; ++i){
				mkl_set_num_threads_local(mkl_nthreads);
				zcopy(&Nk, &T[i], &n_, &G[k + i + M*(k+1)], &M);
			}


			// put zeros explicitly in the right places
			G[k+2+M*k] = 0;
			G[k+2+M*(k+1)] = 0;

			k = k+1;

			double end2 = omp_get_wtime();
			pivot2time += (double) (end2 - start2);
			//printf("PIVOT_2 time = %lf\n", (double)(end2 - start2));

			goto LOOP_END;
		}

		// if here, something broke down
		printf("\n\n\nUPS, SOMETHING WENT WRONG... STOPPING...Is A maybe singular?\n\n\n");
		exit(-4);


		// ----------------------------------------------PIVOT_1-----------------------------------------------------

		PIVOT_1:

		pivotiranje = pivotiranje + omp_get_wtime() - pp;
		last_pivot = 1;
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

		double complex gkk;
		if(cabs(G[k+M*k]) >= EPSILON) gkk = -G[k+M*k] * csqrt(cabs(Akk)) / cabs(G[k+M*k]);
		else gkk = csqrt(cabs(Akk));

		// save the J norm of the vector
		double fJf = Akk + J[k] * (cabs(Akk) + 2 * csqrt(cabs(Akk)) * cabs(G[k+M*k]));

		// make the reflector vector and save it
		double complex alpha = -1;
		int inc = 1;
		int Mk = M - k;
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

		mkl_nthreads = Mk/D > mkl_get_max_threads()/nthreads ? Mk/D : mkl_get_max_threads()/nthreads;
		if (Mk/D == 0 || mkl_get_max_threads()/nthreads == 0) mkl_nthreads = 1;

		#pragma omp parallel num_threads( nthreads )
		{
			mkl_set_num_threads_local(1);

			#pragma omp for nowait
			for(j = k+1; j < N; ++j){

				// T = Jf
				// alpha = f*Jg
				int Mk = M - k;
				int inc = 1;
				double complex alpha;
				zdotc(&alpha, &Mk, &T[k], &inc, &G[k+M*j], &inc);
				alpha = - 2 * alpha / fJf;
				// G[k + M*j] = alpha * f[k] + G[k + M*k]
				zaxpy(&Mk, &alpha, &f[k], &inc, &G[k + M*j], &inc);
			}
		}

		pivot1time += (double)(omp_get_wtime() - start1);
		LOOP_END: continue;

	}	// END OF MAIN LOOP

	end = omp_get_wtime();
	seconds = (double)(end - start);
	printf("algorithm time = %lg s\n", seconds);
	printf("PIVOT_1 (%d)	time = %lg s (%lg %%)\n", pivot_1_count, pivot1time, pivot1time / seconds * 100);
	printf("PIVOT_2 (%d)	time = %lg s (%lg %%)\n", pivot_2_count, pivot2time, pivot2time / seconds * 100);
	printf("mnozenje u PIVOT_1 time = %lg s (udio relativnog = %lg %%, udio apsolutnog = %lg %%)\n", mnozenjetime, mnozenjetime/pivot1time * 100, mnozenjetime/seconds * 100);
	printf("redukcija u PIVOT_2 time = %lg s (udio relativnog = %lg %%, udio apsolutnog = %lg %%)\n", redukcijatime, redukcijatime/pivot2time * 100, redukcijatime/seconds * 100);
	printf("pivotiranje time = %lg s (%lg %%)\n", pivotiranje, pivotiranje/seconds * 100);



	// -------------------------------- writing --------------------------------

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

	end = omp_get_wtime();
	seconds = (double)(end - start);
	printf("writing time = %lg s\n", seconds);


	// ------------------------------- cleaning -------------------------------

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
	mkl_free(T);
	mkl_free(f);
	mkl_free(U);
	mkl_free(p);
	mkl_free(n);
	mkl_free(norm);

	//printf("\nFinished.\n");
	printf("\n-------------------------------------------------------------------------------\n\n");
	return(0);
}
