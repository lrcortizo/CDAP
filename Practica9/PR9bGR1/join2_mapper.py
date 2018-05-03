#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import re

# ---------------------------------------------------------------------------
# Este mapper acepta valores <key, value> y devuelve aquellos que se corresponden con Amazon o FNAC
# key se corresponde con nombres de libros
# value puede ser el numero de ejemlares vendidos o la librería donde se vende
# No hay ninguna comprobación de errores en la entrada
# ---------------------------------------------------------------------------

for line in sys.stdin:
    line       = line.strip()
    key_value  = line.split(",")
    key     = key_value[0]
    value   = key_value[1]

    if (re.match(r'^[A-Za-z]+$', value)):
        if value=='Amazon' or value=='FNAC':
            print( '%s,%s' % (key, value) )
    else:
        print( '%s,%s' % (key, value) )
