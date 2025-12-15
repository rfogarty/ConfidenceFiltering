#!/bin/bash
# Most likely want to run in persistent shell session such as screen like this:
#   > screen -d -m -S <sessionName> spawnTraining.sh <GPU> <FirstTest> [<NextTest>...<LastTest>]

if [ $# -lt 2 ] ; then
    echo "Run with: $0 <GPU> <TestNum> [<NextTestNum> ... <LastTestNum>]"
fi

gpuidx=$1
shift

splits=($(echo {0..4}))

for testn in $@ ; do
    ./runDockerTrain.sh $gpuidx $testn ${splits[@]} 2>&1 | tee runLogTrainTest${testn}.txt
done

