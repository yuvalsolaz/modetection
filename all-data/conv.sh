#!/bin/bash

[ -d utf8 ] || mkdir utf8
for f in *.csv;do
    echo "converting : $f to ./utf8/$f"
    iconv -f UTF-8 -t ISO-8859-1 -c $f > ./utf8/$f	
done
 

