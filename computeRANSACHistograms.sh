#!/bin/bash

prefix="Results_"
if [ $# -ge 1 ] ; then
   prefix=$1
   shift
fi

models=($(echo {0..4}))
splits=($(echo {0..4}))

for testn in {0..6} ; do
    python computeMetrics.py -T $testn -M ${models[@]} -S ${splits[@]} -p $prefix --ensemble 2>&1 | tee RANSACFrequencyTest${testn}.txt
done

