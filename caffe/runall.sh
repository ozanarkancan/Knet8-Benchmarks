#!/bin/bash

printf "Caffe examples bulk running...\n"
for example in $( ls *.py ); do
    echo "$example running..."
    for i in `seq 1 5`; do
        output=`python $example`
        printf " - $output\n"
    done
done
