#!/bin/bash

if [ -z "$1" ]; then
    echo "student number is required.
Usage: ./CollectSubmission 20??-?????"
    exit 0
fi

files="AS1-Logistic_Regression.ipynb"

for file in $files
do
    if [ ! -f $file ]; then
        echo "Required $file not found."
        exit 0
    fi

    if [ ${file: -6} == ".ipynb" ]

    then
        echo "Converting $file to python file"
        jupyter nbconvert --to python "$file"
    fi
done


rm -f $1.tar.gz
mkdir $1
cp -r  ./*.py $1/
tar cvzf $1.tar.gz $1
