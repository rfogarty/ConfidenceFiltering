#!/bin/bash
# Run with ./runSplitTestsNoRANSAC.sh [gpu(s)] $(echo {0..num})

function getBestModel() {
   testn=$1
   splitn=$2
   modeln=$3
   #echo "$(ls best-comboloss-test${testn}-split${splitn}-* 2>/dev/null | tail -n $((modeln+1)) | head -n1)"
   echo "$(ls best-comboloss-test${testn}-split${splitn}-* 2>/dev/null | cut -d- -f5 | sort -n | tail -n $((modeln+1)) | head -n1 | sed -E "s/^/best-comboloss-test${testn}-split${splitn}-/")"
   #echo "$(ls best-accuracy-split${split}-* 2>/dev/null | tail -n 1)"
   #echo "$(ls snapshot-weights-test${testn}-split${splitn}-* 2>/dev/null | tail -n $((modeln+1)) | head -n1)"
}

function dockerPython() {
    dockerImage="$1"
    shift
    
    #NV_GPU=0,1
    # -i
    echo "Launching Docker Session:"
    echo "   $@"
    echo

#    podman run --rm \
#       --cpus="64"  --gpus all \
#       --workdir $PWD \
#       --entrypoint python \
#       --mount type=bind,source=/raid/rfogarty,target=/raid/rfogarty \
#       --mount type=bind,source=/home/80022521/sessions/shared,target=/shared \
#       -e DISPLAY -t ${dockerImage} \
#       $@

    #docker run $(echo $DOCKER_DNS) --rm --user $(id -u):$(id -g) \
    #   --runtime=nvidia \
    #   --workdir $PWD \
    #   --entrypoint python \
    #   --mount type=bind,source=/raid/rfogarty,target=/raid/rfogarty \
    #   --mount type=bind,source=/home/80022521/sessions/shared,target=/shared \
    #   -e DISPLAY -t ${dockerImage} \
    #   $@
    python $@
}


gpu=6

if [ $# -gt 0 ] ; then
   # First grab the gpu we are running on:
   gpu=$1
   shift
fi

testn=0
if [ $# -gt 0 ] ; then
   # Next grab Test number:
   testn=$1
   shift
fi

modeln=-1
if [ $# -gt 0 ] ; then
   # Next grab what Model number:
   modeln=$1
   shift
fi

# Finally grab any remaining arguments as Splits to process
if [ $# -gt 0 ] ; then
   echo -n "Processing splits: "
   for i in "$@" ; do
      echo -n "$i "
      splits+=("$i")
   done
   echo #newline
else
   echo -n "Processing splits: $(echo {0..19})"
   splits=($(echo {0..19}))
fi

gpus=($gpu)

extra_args=( \
            --mask \
            -g ${gpus[@]} )


dockerImage="moffittml_tensorflow_plus_keras_and_opencv:1.5"

entrytarget="testSplitNoRANSAC.py"
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

echo "Computing output for all splits"
for splitn in "${splits[@]}" ; do
   splitmodel="$(getBestModel ${testn} ${splitn} ${modeln})"
   if [ ! -z "$splitmodel" ] ; then
      echo "Testing Split=$splitn Model(${modeln})=$splitmodel"
      time dockerPython "$dockerImage" $entrytarget -T ${testn} -s "${splitn}" -n ${modeln} $splitmodel ${extra_args[@]} 2>&1 | tee "runLogHoldoutTest${testn}Split${splitn}Model${modeln}.txt"
   else
      echo "WARNING: No model for Split=$splitn"
   fi
done


