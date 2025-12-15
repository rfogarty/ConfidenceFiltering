#!/bin/bash

function getBestModel() {
   testn=$1
   splitn=$2
   echo "$(ls best-comboloss-test${testn}-split${splitn}-* 2>/dev/null | cut -d- -f5 | sort -n | tail -n 1 | sed -E "s/^/best-comboloss-test${testn}-split${splitn}-/")"
}

function dockerPython() {
    dockerImage="$1"
    shift
    
    ##NV_GPU=0,1
    ## -i
    #echo "Launching Docker Session:"
    #echo "   $@"
    #echo
    #docker run --cpus="64" --rm --user $(id -u):$(id -g) \
    #   --runtime=nvidia \
    #   --workdir $PWD \
    #   --entrypoint python \
    #   --mount type=bind,source=/raid/rfogarty,target=/raid/rfogarty \
    #   -e DISPLAY -t ${dockerImage} \
    #   $@
    python $@
}

gpu=0

if [ $# -gt 0 ] ; then
   # First grab the gpu we are running on:
   gpu=$1
   shift
fi

testn=0
if [ $# -gt 0 ] ; then
   # First grab the gpu we are running on:
   testn=$1
   shift
fi
echo "Dispatching to GPU: $gpu"

if [ $# -gt 0 ] ; then
   echo -n "Processing splits: "
   for i in "$@" ; do
      echo -n "$i "
      splits+=("$i")
   done
   echo #newline
else
   echo -n "Processing splits: $(echo {0..4})"
   splits=($(echo {0..4}))
fi

gpus=($gpu)

dockerImage="moffittml_tensorflow_plus_keras_and_opencv:1.4"

extra_args=( \
            -g ${gpus[@]} \
            --mask \
            --trainFeatures \
            --augment \
            -e 647 \
            -p 130 \
            -b 90 \
            -P 3 \
            -c 49 \
            -l 0.03 \
            --lrRedRate 0.0135)
#            --lrRedRate 0.0035)
#            -l 0.005)
#            -e 1040 \

entrytarget="fine_tune_multi.py"
if [ -e "$entrytarget" ] ; then
    # Keep current path, i.e. do nothing
    echo -n ""
elif [ -e "../$entrytarget" ] ; then
    entrytarget="../$entrytarget"
elif [ -e "../../$entrytarget" ] ; then
    entrytarget="../../$entrytarget"
elif [ -e "../../../$entrytarget" ] ; then
    entrytarget="../../../$entrytarget"
elif [ -e "../../../../$entrytarget" ] ; then
    entrytarget="../../../../$entrytarget"
else
    echo "Entry target: $entrytarget not found" 1>&2
    exit 1
fi

for splitn in "${splits[@]}" ; do
   splitmodel="$(getBestModel ${testn} ${splitn})"
   mkdir -p backup
   cp $splitmodel backup/
   restartEpoch=$(echo "$splitmodel" | cut -d- -f5 | cut -d. -f1)
   #((restartEpoch+=1)) - this could treat a number with leading 0s as octal, instead do the following, which forces decimal
   restartEpoch=$((10#$restartEpoch+1))
   echo "Using restartEpoch: $restartEpoch"
   time dockerPython "$dockerImage" $entrytarget -T $testn -s $splitn -R $restartEpoch "${extra_args[@]}" $splitmodel 2>&1 | tee "runLogFineTuneTest${testn}Split${splitn}.txt"
done

