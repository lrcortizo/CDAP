#!/usr/bin/env python
from mpi4py import MPI
comm = MPI.COMM_WORLD
my_rank = comm.rank
num_processes = comm.size
if my_rank != 0:
	message = "Saludos del proceso %d" % my_rank
	dest_rank = 0
	comm.send(message,dest=dest_rank)
else:
	print "Hola, soy el proceso %d (hay %d procesos) y recibo:" % (my_rank,num_processes)
	for source_rank in range(1,num_processes):
		message = comm.recv(source=source_rank)
		print message
