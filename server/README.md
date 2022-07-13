# Running code on server (BC4)

calls `submit_job.sh` for a single gpu or `submit_job_2.sh` for dual gpu

# Train all gans/no gans on a single task
to submit all jobs:
```
$ ./server/run_single_task.sh
```

# Copying results to local machine
Run the script:
```
$ ./server/get_models_from_server.sh
```
to download to ``~/Downloads`
