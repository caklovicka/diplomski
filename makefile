generate:
	rm -rf data
	mkdir data
	@echo $(M)
	@echo $(N)
	gcc generateG.c -o generateG.out -lblas -llapack -w
	./generateG.out data/G.bin data/J.bin $(M) $(N) -w

runGF:
	@echo $(M)
	@echo $(N)
	gcc GFreduction.c -o GFreduction.out -lblas -llapack -lm -std=gnu11 -w
	./GFreduction.out data/G.bin data/J.bin $(M) $(N) -w

check:
	@echo $(M)
	@echo $(N)
	gcc check.c -o check.out -lm -lblas -llapack -w
	./check.out data/reducedG.bin data/reducedJ.bin data/A.bin data/Pcol.bin $(M) $(N) -w

runQR:
	@echo $(M)
	@echo $(N)
	gcc QRreduction.c -o QRreduction.out -lm -lblas -llapack -std=gnu11 -w
	./QRreduction.out data/G.bin data/J.bin $(M) $(N) -w


all: 
	rm -rf data
	mkdir data
	@echo $(M)
	@echo $(N)
	gcc generateG.c -o generateG.out -lblas -llapack -w
	./generateG.out data/G.bin data/J.bin $(M) $(N) -w
	gcc GFreduction.c -o GFreduction.out -lblas -llapack -lm -std=gnu11 -w
	./GFreduction.out data/G.bin data/J.bin $(M) $(N) -w
	gcc check.c -o check.out -lm -lblas -llapack -w
	./check.out data/reducedG.bin data/reducedJ.bin data/A.bin data/Pcol.bin $(M) $(N) -w

clean:
	rm -rf data 
	rm -f generateG.out GFreduction.out check.out

