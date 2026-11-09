.PHONY: all smoke clean
all:
	bash run_all.sh
smoke:
	python tools/make_fake_data.py && MEDISCAN_DATA=data_fake QUICK=1 RES=64 EPOCHS=2 bash run_all.sh
clean:
	rm -rf results figures models .cache
