#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys

# --------------------------------------------------------------------------
# Acepta un fichero <palabra, valor> y hace un join de palabras
#
# Lleva la cuenta de la palabra actual y la anterior, y si la palabra cambia
# hace el join mostrando la palabra y las claves. No hay que hacer comprobación
# de las claves porqeu hadoop las pone ordenadas
#
#  No se hace comprobación de errores sobre la entrada
# --------------------------------------------------------------------------

prev_word          = "  " 
months             = ['Jan','Feb','Mar','Apr','Jun','Jul','Aug','Sep','Nov','Dec']
dates_to_output    = [] 
day_cnts_to_output = [] 
line_cnt           = 0  # contador de líneas

for line in sys.stdin:
    line       = line.strip() 
    key_value  = line.split('\t') 
    line_cnt   = line_cnt+1     

    curr_word  = key_value[0]  
    value_in   = key_value[1] 

    #-----------------------------------------------------
    # Comprueba que la palabra es nueva
    #----------------------------------------------------
    if curr_word != prev_word:

        # -----------------------     
	    # Escribe el resultado de join para nº línea > 1
        # -----------------------
        if line_cnt>1:
    	    for i in range(len(dates_to_output)):  #loop thru dates, indexes start at 0
    	         print('{0} {1} {2} {3}'.format(dates_to_output[i],prev_word,day_cnts_to_output[i],curr_word_total_cnt))
            dates_to_output=[]
            day_cnts_to_output=[]
            prev_word=curr_word  
	
    # ---------------------------------------------------------------
    # Procesa la palabra actual
    # ---------------------------------------------------------------

    if (value_in[0:3] in months): 
        date_day =value_in.split() 
        dates_to_output.append(date_day[0])
        day_cnts_to_output.append(date_day[1])
    else:
        curr_word_total_cnt = value_in  

# ---------------------------------------------------------------
# Escribe el último resultado del join
# ---------------------------------------------------------------
for i in range(len(dates_to_output)): 
         print('{0} {1} {2} {3}'.format(dates_to_output[i],prev_word,day_cnts_to_output[i],curr_word_total_cnt))