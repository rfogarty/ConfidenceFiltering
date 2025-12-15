#!/bin/bash

function getBestModel() {
   testn=$1
   splitn=$2
   echo "$(ls best-comboloss-test${testn}-split${splitn}-* 2>/dev/null | cut -d- -f5 | sort -n | tail -n 1 | sed -E "s/^/best-comboloss-test${testn}-split${splitn}-/")"
   #echo "$(ls best-comboloss-test${testn}-split${splitn}-* 2>/dev/null | cut -d- -f5 | sort -n | tail -n 1 | xargs -IRPLC echo best-comboloss-test${testn}-split${splitn}-RPLC)"
}

export -f getBestModel

if [ $# -eq 2 ] ; then
   getBestModel "$@"
else 
   echo "Run with: $0 <0|1|2|...>(Test Number) <0|1|2|...>(Split Number)"
fi

