#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys

# ---------------------------------------------------------------------------
# Este mapper acepta valores <date word, value> y mueve la fecha con el valor
#
# No hay ninguna comprobación de errores en la entrada
# ---------------------------------------------------------------------------

for line in sys.stdin:
    line       = line.strip()   
    key_value  = line.split(",")   
    key_in     = key_value[0].split(" ")  
    value_in   = key_value[1]   

    if len(key_in)>=2:   # si key_in tiene dos elementos       
        date = key_in[0]      
        word = key_in[1]
        value_out = date+" "+value_in    
        print( '%s\t%s' % (word, value_out) )  
    else:   
        print( '%s\t%s' % (key_in[0], value_in) )  
