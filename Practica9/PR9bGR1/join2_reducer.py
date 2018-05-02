#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import re

# --------------------------------------------------------------------------
# Acepta un fichero <clave, valor>
#
# Lleva la cuenta del libro actual y el anterior, y si el libro cambia hace el
# join mostrando el libro y numero total de ejemplares vendidos para Amazon y FNAC.
#
# No hay que hacer comprobación de las claves porque hadoop las pone ordenadas
#
# No se hace comprobación de errores sobre la entrada
# --------------------------------------------------------------------------

prev_key          = "  "
cont = 0
numAmazon = 0
numFNAC = 0
labelFNAC = False
labelAmazon = False
best = ""

for line in sys.stdin:
    line       = line.strip()
    key_value  = line.split('\t')
    curr_key  = key_value[0]
    value   = key_value[1]

    #Comprueba que el libro es el mismo
    if curr_key == prev_key:
        #Comprueba si se trata del numero de ejemplares o de la librería
        if (re.match(r'^[0-9]+$', value)): #Si es el numero de ejemplares aumenta el contador
            cont += int(value)
        elif value == 'FNAC': #Si se corresponde con las librerías deseadas levanta un label
            labelFNAC = True
        elif  value == 'Amazon':
            labelAmazon = True

    elif prev_key: #existe prev_key
        if labelFNAC and labelAmazon: #Comprueba a que librería hace referencia y lo indica en el print
            print( "{0}\t{1}\tSe vende en FNAC y Amazon".format(prev_key, cont) )
        elif labelFNAC:
            print( "{0}\t{1}\tSe vende en FNAC".format(prev_key, cont) )
            numFNAC += cont
        elif labelAmazon:
            print( "{0}\t{1}\tSe vende en Amazon".format(prev_key, cont) )
            numAmazon += cont

        #Resetear valores
        cont = 0
        labelFNAC = False
        labelAmazon = False
        prev_key = curr_key

    if numAmazon > numFNAC:
        best = 'Amazon'
    else:
        best = 'FNAC'

print ("\nEjemplares vendidos de los libros que se venden en FNAC: {0}".format(numFNAC))
print ("Ejemplares vendidos de los libros que se venden en Amazon: {0}".format(numAmazon))
print ("La mejor seleccion de libros la hace: {0}".format(best))
