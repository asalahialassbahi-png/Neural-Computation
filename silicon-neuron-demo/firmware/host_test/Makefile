CC ?= gcc
CFLAGS ?= -O2 -Wall -std=c11 -I.. -I../generated

SRC = host_main.c ../nn_core.c ../trace_io.c ../generated/model_data.c

host_test: $(SRC) ../nn_core.h ../trace_io.h ../generated/model_data.h
	$(CC) $(CFLAGS) -o $@ $(SRC)

clean:
	rm -f host_test
