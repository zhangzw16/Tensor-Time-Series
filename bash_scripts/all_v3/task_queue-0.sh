seed=${1:-0}
log_file="./task_queue-0.log"

time=$(date "+%Y-%m-%d %H:%M:%S")
echo "Task Begin at $time" > $log_file

bash bash_scripts/all_v3/input_search-1.sh $seed
time=$(date "+%Y-%m-%d %H:%M:%S")
echo "input_search-1 is finished at $time" >> $log_file

bash bash_scripts/all_v3/input_search-5.sh $seed
time=$(date "+%Y-%m-%d %H:%M:%S")
echo "input_search-5 is finished at $time" >> $log_file

bash bash_scripts/all_v3/input_search-9.sh $seed
time=$(date "+%Y-%m-%d %H:%M:%S")
echo "input_search-9 is finished at $time" >> $log_file