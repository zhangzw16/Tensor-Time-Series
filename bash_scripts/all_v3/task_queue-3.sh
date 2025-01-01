seed=${1:-0}
log_file="./task_queue-3.log"

time=$(date "+%Y-%m-%d %H:%M:%S")
echo "Task Begin at $time" > $log_file

bash bash_scripts/all_v3/input_search-4.sh $seed
time=$(date "+%Y-%m-%d %H:%M:%S")
echo "input_search-4 is finished at $time" >> $log_file

bash bash_scripts/all_v3/input_search-8.sh $seed
time=$(date "+%Y-%m-%d %H:%M:%S")
echo "input_search-8 is finished at $time" >> $log_file

bash bash_scripts/all_v3/input_search-12.sh $seed
time=$(date "+%Y-%m-%d %H:%M:%S")
echo "input_search-12 is finished at $time" >> $log_file