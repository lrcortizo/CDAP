#!/bin/bash

cat join2_*.txt | ./join2_mapper.py | sort
