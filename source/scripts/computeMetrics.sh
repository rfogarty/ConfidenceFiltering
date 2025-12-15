#!/bin/bash


# A POSIX variable
OPTIND=1         # Reset in case getopts has been used previously in the shell.

# Initialize our own variables:
prefix=""
filterConfidence=""
sfractionProp=""
relabels=""
logPrefix="LogMetrics"
calPrefix=""
calEnsPrefix=""

SCRIPTNAME=$0
NUMARGS=$#

while getopts "h?tp:f:s:r:l:c:" opt; do
  case "$opt" in
    h|\?)
      echo "Run with: $SCRIPTNAME [-p prefix] [-f confidenceThresh]|[-s sfractionThreshold] [-r relabels] [-l logoutPrefix]"
      exit 0
      ;;
    p)  prefix="-p $OPTARG"
      ;;
    f)  filterConfidence="-f $OPTARG"
      ;;
    s)  sfractionProp="-s $OPTARG"
      ;;
    r)  relabels="-r $OPTARG"
      ;;
    l)  logPrefix="$OPTARG"
      ;;
    c)  calPrefix="--log_calibration $OPTARG"
        calEnsPrefix="--log_calibration ${OPTARG}Ens"
      ;;
  esac
done

shift $((OPTIND-1))

[ "${1:-}" = "--" ] && shift

entrytarget="computeMetrics.py"
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
elif [ -e "../../../../../$entrytarget" ] ; then
    entrytarget="../../../../../$entrytarget"
else
    echo "Entry target: $entrytarget not found" 1>&2
    exit 1
fi


# Compute and show test/split metrics
if [ ! -e ${logPrefix}Test0Split0Model0.txt ] ; then
   for testn in {0..6} ; do 
      modeln=0
      for splitn in {0..4} ; do
      for modeln in {0..4} ; do
         echo "Running test:${testn}, split:${splitn}"
         #python ${entrytarget} -T $testn -M $modeln -S $splitn $prefix $sfractionProp $filterConfidence $relabels $calPrefix --calHistBins 20 2>&1 | tee ${logPrefix}Test${testn}Split${splitn}.txt
         #python ${entrytarget} -T $testn -M $modeln -S $splitn $prefix $sfractionProp $filterConfidence $relabels $calPrefix --calFilterN 40 2>&1 | tee ${logPrefix}Test${testn}Split${splitn}.txt
         #python ${entrytarget} -T $testn -M $modeln -S $splitn $prefix $sfractionProp $filterConfidence $relabels $calPrefix --read_calibration PreConfCal 2>&1 | tee ${logPrefix}Test${testn}Split${splitn}.txt
         python ${entrytarget} -T $testn -M $modeln -S $splitn $prefix $sfractionProp $filterConfidence $relabels $calPrefix --read_calibration PreConfCal 2>&1 | tee ${logPrefix}Test${testn}Split${splitn}Model${modeln}.txt
         #python ${entrytarget} -T $testn -M $modeln -S $splitn $prefix $sfractionProp $filterConfidence $relabels $calPrefix 2>&1 | tee ${logPrefix}Test${testn}Split${splitn}Model${modeln}.txt
         #python ${entrytarget} -T $testn -M $modeln -S $splitn $prefix $sfractionProp $filterConfidence $relabels $calPrefix --read_calibration PreConfCal 2>&1 | tee ${logPrefix}Test${testn}Split${splitn}Model${modeln}.txt
      done
      done
   done
fi

echo "Accuracies" > SummaryNonEnsemble.txt
grep Accuracy ${logPrefix}Test* | cut -d" " -f2 >> SummaryNonEnsemble.txt
#read -n 1 -p "Press any key to continue "

echo "F1" >> SummaryNonEnsemble.txt
grep F1 ${logPrefix}Test* | cut -d" " -f2 >> SummaryNonEnsemble.txt
#read -n 1 -p "Press any key to continue "

echo "AUC" >> SummaryNonEnsemble.txt
grep AUC ${logPrefix}Test* | cut -d" " -f2 >> SummaryNonEnsemble.txt
#read -n 1 -p "Press any key to continue "

echo "Filtered,Total" > FilterSummaryNonEnsemble.txt
grep Filtered ${logPrefix}Test* | sed -E "s/.*Filtered //" | sed -E "s/ of total /,/" | sed -E "s/ from data//" >> FilterSummaryNonEnsemble.txt

# Now compute and show ensemble metrics
if [ ! -e ${logPrefix}EnsembleTest0.txt ] ; then
   #for testn in {0..6} ; do 
   #for testn in {0..6} ; do 
   #for testn in {0,3} ; do 
   for testn in {0..6} ; do 
      echo "Running test:${testn}"
      #                                                                                              Smart ensemble turns out to do a little worse than normal ensemble?
      #python ${entrytarget} -T $testn -M 0 -S {0..4} $prefix $sfractionProp $filterConfidence $relabels --ensemble --smartEnsemble 2>&1 | tee ${logPrefix}EnsembleTest${testn}.txt
      #python ${entrytarget} -T $testn -M 0 -S {0..4} $prefix $sfractionProp $filterConfidence $relabels --ensemble $calEnsPrefix --calHistBins 20 2>&1 | tee ${logPrefix}EnsembleTest${testn}.txt
      #python ${entrytarget} -T $testn -M 0 -S {0..4} $prefix $sfractionProp $filterConfidence $relabels --ensemble $calEnsPrefix --calFilterN 40 2>&1 | tee ${logPrefix}EnsembleTest${testn}.txt
      #python ${entrytarget} -T $testn -M 0 -S {0..4} $prefix $sfractionProp $filterConfidence $relabels --ensemble $calEnsPrefix --read_calibration PreConfCal 2>&1 | tee ${logPrefix}EnsembleTest${testn}.txt
      #python ${entrytarget} -T $testn -M 0 -S {0..4} $prefix $sfractionProp $filterConfidence $relabels --ensemble $calEnsPrefix 2>&1 | tee ${logPrefix}EnsembleTest${testn}.txt
      python ${entrytarget} -T $testn -M {0..0} -S {0..4} $prefix $sfractionProp $filterConfidence $relabels --ensemble $calEnsPrefix 2>&1 | tee ${logPrefix}EnsembleTest${testn}.txt
      #python ${entrytarget} -T $testn -M {0..4} -S {0..4} $prefix $sfractionProp $filterConfidence $relabels --ensemble $calEnsPrefix --read_calibration PreConfCal 2>&1 | tee ${logPrefix}EnsembleTest${testn}.txt
      #python ${entrytarget} -T $testn -M 0 -S {0..4} $prefix $sfractionProp $filterConfidence $relabels --ensemble $calEnsPrefix --read_calibration PreConfCal 2>&1 | tee ${logPrefix}EnsembleTest${testn}.txt
   done
fi

echo "Ensemble Accuracies" > SummaryEnsemble.txt
grep Accuracy ${logPrefix}EnsembleTest* | cut -d" " -f2 >> SummaryEnsemble.txt
#read -n 1 -p "Press any key to continue "

echo "Ensemble F1" >> SummaryEnsemble.txt
grep F1 ${logPrefix}EnsembleTest* | cut -d" " -f2 >> SummaryEnsemble.txt
#read -n 1 -p "Press any key to continue "

echo "Ensemble AUC" >> SummaryEnsemble.txt
grep AUC ${logPrefix}EnsembleTest* | cut -d" " -f2 >> SummaryEnsemble.txt
#read -n 1 -p "Press any key to continue "

echo "Filtered,Total" > FilterSummaryEnsemble.txt
grep Filtered ${logPrefix}EnsembleTest* | sed -E "s/.*Filtered //" | sed -E "s/ of total /,/" | sed -E "s/ from data//" >> FilterSummaryEnsemble.txt


