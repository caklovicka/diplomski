generate:
	rm -rf data
	mkdir data
	@echo $(M) $(N)
	gcc generateG.c -o generateG.out -lblas -llapack -w
	./generateG.out data/G.bin data/J.bin $(M) $(N) -w

check:
	@echo $(M) $(N)
	gcc check.c -o check.out -lm -lblas -llapack -w
	./check.out data/reducedG.bin data/reducedJ.bin data/A.bin data/Pcol.bin $(M) $(N) -w

runQR:
	@echo $(M) $(N)
	icc -mkl QRreduction.c -o QRreduction.out -lm -w
	./QRreduction.out data/G.bin data/J.bin $(M) $(N) -w

all_seq: 
	rm -rf data
	mkdir data
	@echo $(M) $(N)
	gcc generateG.c -o generateG.out -lblas -llapack -w
	./generateG.out data/G.bin data/J.bin $(M) $(N) -w
	gcc QRreduction.c -o QRreduction.out -lblas -llapack -lm -std=gnu11 -w
	./QRreduction.out data/G.bin data/J.bin $(M) $(N) -w
	gcc check.c -o check.out -lm -lblas -llapack -w
	./check.out data/reducedG.bin data/reducedJ.bin data/A.bin data/Pcol.bin $(M) $(N) -w

generate_par:
	rm -rf data
	mkdir data
	@echo $(M) $(N)
	gcc generateGparallel.c -o generateGparallel.out -lblas -llapack -fopenmp -w
	./generateGparallel.out data/G.bin data/J.bin $(M) $(N) -w

runQR_par:
	@echo $(M) $(N)
	gcc QRparallel.c -o QRparallel.out -lblas -llapack -lm -fopenmp -w
	./QRparallel.out data/G.bin data/J.bin $(M) $(N) -w

runH:
	@echo $(M) $(N)
	icc JHouseholder.c -o JHouseholder.out -mkl -lm -fopenmp -w
	./JHouseholder.out data/G.bin data/J.bin $(M) $(N) -w


runGH:
	@echo $(M) $(N)
	icc Givens_Householder.c -o Givens_Householder.out -mkl -lm -fopenmp -w
	./Givens_Householder.out data/G.bin data/J.bin $(M) $(N) -w


generate_xeon:
	rm -rf data
	mkdir data
	@echo $(M) $(N)
	icc -mkl generateGparallel.c -o generateGparallel.out -fopenmp
	./generateGparallel.out data/G.bin data/J.bin $(M) $(N)

runQR_xeon:
	@echo $(M) $(N)
	icc -mkl QRparallel_xeon_mkl.c -o QRparallel_xeon_mkl.out -fopenmp
	./QRparallel_xeon_mkl.out data/G.bin data/J.bin $(M) $(N)

runQR_exp:
	@echo $(M) $(N)
	icc -mkl QRparallel_experimental.c -o QRparallel_experimental.out -fopenmp
	./QRparallel_experimental.out data/G.bin data/J.bin $(M) $(N)

compile_exp:
	@echo $(M) $(N)
	icc -mkl QRparallel_experimental.c -o QRparallel_experimental.out -fopenmp
	icc -mkl QRparallel_0.c -o QRparallel_0.out -fopenmp
	icc -mkl QRparallel_1.c -o QRparallel_1.out -fopenmp
	icc -mkl QRparallel_2.c -o QRparallel_2.out -fopenmp

run_exp:
	@echo $(M) $(N)
	numactl -m 1 ./QRparallel_experimental.out data/G.bin data/J.bin $(M) $(N)

run0:
	@echo $(M) $(N)
	numactl -m 1 ./QRparallel_0.out data/G.bin data/J.bin $(M) $(N)

run1:
	@echo $(M) $(N)
	numactl -m 1 ./QRparallel_1.out data/G.bin data/J.bin $(M) $(N)

run2:
	@echo $(M) $(N)
	numactl -m 1 ./QRparallel_2.out data/G.bin data/J.bin $(M) $(N)

check_xeon:
	@echo $(M) $(N)
	icc -mkl check_mkl.c -o check_mkl.out -fopenmp -w
	./check_mkl.out data/reducedG.bin data/reducedJ.bin data/A.bin data/Pcol.bin $(M) $(N)


clean:
	rm -rf data 
	rm -f generateG.out check.out QRreduction.out generateGparallel.out QRparallel.out QRparallel_xeon_mkl.out check_mkl.out QRparallel_experimental.out

