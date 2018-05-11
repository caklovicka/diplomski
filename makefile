$(VERBOSE).SILENT:

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
	gcc QRreduction.c -o QRreduction.out -lblas -llapack -lm -w
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

generateG_par:
	rm -rf data
	mkdir data
	@echo $(M) $(N)
	gcc generateGparallel.c -o generateGparallel.out -lblas -llapack -fopenmp -w
	./generateGparallel.out data/G.bin data/J.bin $(M) $(N) -w


clean:
	rm -rf data 
	rm -f generateG.out GFreduction.out check.out QRreduction.out

