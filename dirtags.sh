#!/bin/bash

a=$(find $1 -maxdepth 1 -type f -name "*.cc" -print -o -type f -name "*.c" -print -o -type f -name "*.h" -print)
cd $1

first=1
list=

for i in $a; do
    if (( first == 1 )); then
        first=0
    fi
    list="${list}${i##*/} "
done

echo $list

if [[ ! -z $list ]]; then
    ctags --extra=+q $list
fi

