#!/bin/bash
###############################################################################################
#
# Tests the last 5 best models on (presumably) holdout data.
#
# Most likely want to run in persistent shell session such as screen like this:
#   > screen -d -m -S <sessionName> spawnTesting.sh <GPU> <FirstTest> [<NextTest>...<LastTest>]
#
# Author: Ryan Botet Fogarty
# Last Edited: 2024.01.05
###############################################################################################
if [ $# -lt 2 ] ; then
    echo "Run with: $0 <GPU> <TestNum> [<NextTestNum> ... <LastTestNum>]"
fi

gpuidx=$1
shift

splits=($(echo {0..4}))

for testn in $@ ; do
    for modeln in {0..4} ; do 
        ./runSplitTestsNoRANSAC.sh $gpuidx $testn $modeln ${splits[@]} 2>&1 | tee runLogHoldoutTest${testn}_Model${modeln}.txt
    done
done

