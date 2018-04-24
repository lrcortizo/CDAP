#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys

# ---------------------------------------------------------------------------
# Este mapper acepta valores <key, value> y el value puede hacer referencia a una librería o al numero de ejemplares
#
# No hay ninguna comprobación de errores en la entrada
# ---------------------------------------------------------------------------

for line in sys.stdin:
    line       = line.strip()
    key_value  = line.split(",")
    key     = key_value[0]
    value   = key_value[1]

    print( '%s\t%s' % (key, value) )
