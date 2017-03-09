# Define required directories.
#export OBJDIR := ${CURDIR}/obj
#export BUILDDIR := ${CURDIR}/build
#SRCDIRS = ${CURDIR}/src

# http://www.rapidtables.com/code/linux/gcc/gcc-g.htm
# -g    : No debug info.
# -lm   : Link with math library.
# -ansi : the same as -std=c89

#https://www.linuxquestions.org/questions/programming-9/gcc-warn_unused_result-attribute-917158/
# -U_FORTIFY_SOURCE
# or
# -D_FORTIFY_SOURCE=0
# Disable warnings like: warning: ignoring return value of ‘fread’...


# Variable definitions.
CC = gcc
#CFLAGS = -Wall -std=c99 -g
CFLAGS = -Wall -std=c99 -O3
# CFLAGS = -Wall -std=c99 -pedantic -g
BUILDDIR = ../bin
RESULTSDIR = ../results


.PHONY: all
all: mkdirs clean compile copy
	@echo MAKE Complete...


.PHONY: debug
debug: mkdirs clean compile_debug copy
	@echo MAKE Complete...


.PHONY: compile
compile:  heatbugs.c heatbugs.h
	@if [ ! -d $(BUILDDIR) ]; then mkdir $(BUILDDIR); fi
	$(CC) heatbugs.c heatbugs.h $(CFLAGS) `pkg-config --cflags --libs cf4ocl2 glib-2.0` -lOpenCL -o $(BUILDDIR)/heatbugs


.PHONY: compile_debug
compile_debug: heatbugs.c heatbugs.h
#	$(CC) heatbugs.c heatbugs.h $(CFLAGS) `pkg-config --cflags --libs glib-2.0` -o heatbugs
	$(CC) heatbugs.c heatbugs.h $(CFLAGS) -D DEBUG `pkg-config --cflags --libs cf4ocl2 glib-2.0` -lOpenCL -o $(BUILDDIR)/heatbugs


.PHONY: mkdirs
mkdirs:
#	@if [ ! -d $(BUILDDIR) ]; then mkdir -p $(BUILDDIR); fi
#	@if [ ! -d $(RESULTSDIR) ]; then mkdir -p $(RESULTSDIR); fi
	mkdir -p $(BUILDDIR)
	mkdir -p $(RESULTSDIR)


.PHONY: copy
copy:
	cp heatbugs.cl $(BUILDDIR)


.PHONY: clean
clean:
	rm -rf $(BUILDDIR)/*
#	rm -d $(BUILDDIR)
#	rm -drf $(BUILDDIR)
