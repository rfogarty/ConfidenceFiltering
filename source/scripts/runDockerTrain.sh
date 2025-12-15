#!/bin/bash


function dockerPython() {
    dockerImage="$1"
    shift
    
    #NV_GPU=0,1
    # -i
    echo "Launching Docker Session:"
    echo "   $@"
    echo
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
            --augment \
            --trainFeatures \
            -e 593 \
            -p 130 \
            -b 90 \
            -P 3 \
            -c 49 \
            -l 0.1 \
            --lrRedRate 0.0035)
#            -l 0.05)
#            -e 936 \
#            --mask \ argument ignored in most cases

# Note about --lrRedRate (LR reduction rate), useful formulas:
#    "--lrRedRate"=>rate, "-e"=>n, "-l"=>begval
#    endval=? # final LR value appreciably lower than begval (e.g. 0.1*begval, or 0.01*begval)
#
#  Use this to find reasonable value for --lrRedRate:
#    rate = -1 * ( ((endval/begval)**(1/n)) - 1 ) # extra parantheses added for clarity
#
#  Use this to confirm the ending LR 
#    endval = m.exp(n*m.log(-rate+1)+m.log(begval))

entrytarget="train_multi.py"
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
   time dockerPython "$dockerImage" $entrytarget -T $testn -s "${splitn}" ${extra_args[@]} 2>&1 | tee runLogTest${testn}Split${splitn}.txt
done

