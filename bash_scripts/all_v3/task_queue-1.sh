seed=${1:-0}
log_file="./task_queue-1.log"

time=$(date "+%Y-%m-%d %H:%M:%S")
echo "Task Begin at $time" > $log_file

bash bash_scripts/all_v3/input_search-2.sh $seed
time=$(date "+%Y-%m-%d %H:%M:%S")
echo "input_search-2 is finished at $time" >> $log_file

bash bash_scripts/all_v3/input_search-6.sh $seed
time=$(date "+%Y-%m-%d %H:%M:%S")
echo "input_search-6 is finished at $time" >> $log_file

bash bash_scripts/all_v3/input_search-10.sh $seed
time=$(date "+%Y-%m-%d %H:%M:%S")
echo "input_search-10 is finished at $time" >> $log_file