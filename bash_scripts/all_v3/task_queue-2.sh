seed=${1:-0}
log_file="./task_queue-2.log"

time=$(date "+%Y-%m-%d %H:%M:%S")
echo "Task Begin at $time" > $log_file

bash bash_scripts/all_v3/input_search-3.sh $seed
time=$(date "+%Y-%m-%d %H:%M:%S")
echo "input_search-3 is finished at $time" >> $log_file

bash bash_scripts/all_v3/input_search-7.sh $seed
time=$(date "+%Y-%m-%d %H:%M:%S")
echo "input_search-7 is finished at $time" >> $log_file

bash bash_scripts/all_v3/input_search-11.sh $seed
time=$(date "+%Y-%m-%d %H:%M:%S")
echo "input_search-11 is finished at $time" >> $log_file