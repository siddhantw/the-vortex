#!/bin/bash

echo "######### Started: $(date) #########"

export DISPLAY=:99.0

host_url="$2"
results_folder="$3"
results_folder_xml="$3/output.xml"
results_folder_rerun_xml="$3/rerun.xml"
results_rerun_to_merge_xml="$results_folder_xml"
tests_folder="$4"

environment="$1"

mkdir -p "$results_folder"

robot --name UltronProTests --variable ENV:$environment --variable HOST_URL:$host_url --include critical --exclude wpgintegration -d $results_folder -x junit_format.xml $tests_folder

sleep 5s #to generate report

number_failed=$(python ./result_robots.py $results_folder_xml)
echo "Failed Tests: $number_failed"

if [ "$number_failed" != 0 ]; then

  robot --variable ENV:$environment --name UltronProTests --rerunfailed $results_folder_xml --output $results_folder_rerun_xml $tests_folder
  results_rerun_to_merge_xml="$results_folder_rerun_xml"

  sleep 5s #to generate report

  rebot -x junit_format.xml -d $results_folder --merge $results_folder_xml $results_rerun_to_merge_xml

fi

echo "######### Completed: $(date) #########"
