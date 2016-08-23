# Define required directories.
#export OBJDIR := ${CURDIR}/obj
#export BUILDDIR := ${CURDIR}/build
#SRCDIRS = ${CURDIR}/src

# -lm : Link with math library-
# -g  : Default debug information


# Variable definitions.
CC = gcc
#CFLAGS = -Wall -std=c99 -g
CFLAGS = -Wall -std=c99 -O3
# CFLAGS = -Wall -std=c99 -pedantic -g
RESULTSDIR = ../results
BUILDDIR = ../bin


.PHONY: all
all: mkdirs clean compile copy
	@echo MAKE Complete...


.PHONY: debug
debug: mkdirs clean compile_debug copy
	@echo MAKE Complete...


.PHONY: compile
compile:  heatbugs.c heatbugs.h
	$(CC) heatbugs.c heatbugs.h $(CFLAGS) `pkg-config --cflags --libs cf4ocl2 glib-2.0` -lOpenCL -o $(BUILDDIR)/heatbugs


.PHONY: compile_debug
compile_debug: heatbugs.c heatbugs.h
#	$(CC) heatbugs.c heatbugs.h $(CFLAGS) `pkg-config --cflags --libs glib-2.0` -o heatbugs
	$(CC) heatbugs.c heatbugs.h $(CFLAGS) -D DEBUG `pkg-config --cflags --libs cf4ocl2 glib-2.0` -lOpenCL -o $(BUILDDIR)/heatbugs


.PHONY: mkdirs
mkdirs:
	@if [ ! -d $(BUILDDIR) ]; then mkdir -p $(BUILDDIR); fi
	@if [ ! -d $(RESULTSDIR) ]; then mkdir -p $(RESULTSDIR); fi
#	mkdir -p $(OBJDIR)
#	mkdir -p $(BUILDDIR)


.PHONY: copy
copy:
	cp heatbugs.cl $(BUILDDIR)


.PHONY: clean
clean:
	rm -rf $(BUILDDIR)/*
#	rm -d $(BUILDDIR)
#	rm -drf $(BUILDDIR)
