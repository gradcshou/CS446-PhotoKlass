#!/bin/bash

# this shell script runs experiments and saves the evaluation results to corresponding directories
image_size=32

dir=experiments/image_size_$image_size
cp -p $dir/photo_klass_input.py .
# run different experiments
for i in {1..4}
do
   for j in 3 5
   do
      cp -p $dir/config${i}_field$j/photo_klass.py .
      echo Start training CNN for architecture $i with receptive field = $j ...
      echo
      python photo_klass_train.py
      echo Evaluting trained model for architecture $i with receptive field = $j ...
      python photo_klass_eval.py > $dir/config${i}_field$j/eval_result.txt
      echo Done. The performance of the model has been saved to $dir/config${i}_field$j/eval_result.txt 
      echo
   done
done
