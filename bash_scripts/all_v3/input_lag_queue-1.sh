
log_file='input_lag_queue-1.log'
seed=67
echo "Starting input_lag_queue-1.sh" > $log_file
bash bash_scripts/all_v3/input_lag-1.sh $seed
echo "input_lag-1.sh $seed FINISHED!" >> $log_file

seed=109
echo "Starting input_lag_queue-1.sh" >> $log_file
bash bash_scripts/all_v3/input_lag-1.sh $seed
echo "input_lag-1.sh $seed FINISHED!" >> $log_file