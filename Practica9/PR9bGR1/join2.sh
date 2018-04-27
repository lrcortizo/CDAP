#!/bin/bash

cat join2_*.txt | ./join2_mapper.py | sort | ./join2_reducer.py
