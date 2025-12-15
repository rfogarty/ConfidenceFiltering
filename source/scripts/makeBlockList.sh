#!/bin/bash

if [ $# -lt 3 ] ; then
   echo "Run with: $0 <NUM_FOLDS> <MIN_REJ> <MAX_REJ> [InferencePerFile.txt] [blocklist.txt]"
   exit 1
fi

numFolds=$1
minRej=$2
maxRej=$3
shift
shift
shift
#minToDrop=$((minAcc-1))
#minToDrop=$((maxRej-1))

inputfile="InferencePerFile.txt"
if [ $# -ge 1 ] ; then
    inputfile="$1"
    shift
fi

outputfile="blocklist.txt"
if [ $# -eq 1 ] ; then
    outputfile="$1"
fi

function addToBlocklist() {
   inf="$1"
   outf="$2"
   blocknum=$3
   
   numMatched=0
   # prune full path
   searchStringGS="/GS"
   searchStringBenign="/Benign"
   while read -r line ; do
      if [ ! -z "$line" ] ; then
         patch="$(echo "$line" | cut -d: -f1)"
         if [[ $patch =~ GS[0345] ]] ; then
             base="GS${patch#*$searchStringGS}"
         else
             base="Benign${patch#*$searchStringBenign}"
         fi
         echo "$base" >> "$outf"
         numMatched=$((numMatched+1))
      fi
   done < <(grep -E "${blocknum}[0-9]*\$" "${inf}" | sort)
   echo "$numMatched"
}

function countMatched() {
   inf="$1"
   blocknum=$2
   
   #numMatched=0
   #while read -r line ; do
   #   if [ ! -z "$line" ] ; then
   #      numMatched=$((numMatched+1))
   #   fi
   #done < <(grep -E "${blocknum}\$" "${inf}" | sort)
   #echo "$numMatched"
   echo "$(grep -cE "${blocknum}[0-9]*\$" "${inf}")"
}

function blockIt() {

   N=$1
   i=$2
   ofs=$3
   ifs=$4
   if [ $i -eq 0 ] ; then
      echo "# No matches in $N" >> "$ofs"
   else
      echo "# Only $i matches in $N" >> "$ofs"   
   fi
   #accuracy="$(echo "scale=10; $i/$N" | bc -l | awk '{printf "%f", $0}'| sed -E "s/0*$//g" | sed -E "s/\.$/.0/")"
   accuracy=$(echo "print(f'{$i/$N}')" | python - | cut -c1-10)
   echo "Checking accuracy: $accuracy" 1>&2
   num=$(addToBlocklist "$ifs" "$ofs" "$accuracy")
   echo "$num"
}

function computeHistogram() {
   N=$1
   i=$2
   #ofs=$3
   #ifs=$4
   ifs=$3
   #accuracy="$(echo "scale=10; $i/$N" | bc -l | awk '{printf "%f", $0}'| sed -E "s/0*$//g" | sed -E "s/\.$/.0/")"
   accuracy=$(echo "print(f'{$i/$N}')" | python - | cut -c1-10)
   #echo "Checking accuracy: $accuracy" 1>&2
   num=$(countMatched "$ifs" "$accuracy")
   echo "$i  $num" >> MatchedHistogram.txt
   echo "$num($accuracy)"
}

echo "########################## Adding to Blocklist ##########################"
#for idx in $(echo 0;seq $minToDrop) ; do
for idx in $(seq $minRej $maxRej) ; do
   echo "Adding to $outputfile: $idx/$numFolds"
   num=$(blockIt $numFolds $idx "$outputfile" "$inputfile")
done

echo "########################## Computing Histogram ##########################"
#echo "# Histogram Table:" > MatchedHistogram.txt
echo "Hits  Frequency" > MatchedHistogram.txt
echo "Histogram:"
for idx in $(echo 0;seq $numFolds) ; do
#for idx in $(seq $minRej $maxRej) ; do
   #num="$(computeHistogram $numFolds $idx "$outputfile" "$inputfile")"
   num="$(computeHistogram $numFolds $idx "$inputfile")"
   echo -n " $num"
done
echo


