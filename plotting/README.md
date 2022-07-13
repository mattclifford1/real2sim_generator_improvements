# Graph creation

## training graphs
to compare the training graphs first get the training csvs from BC4 using `server/get_models_from_server.sh` then add/remove graphs you want to show in the dict `curves_to_plot` then:
```
$ python plotting/training_graphs_from_list.py --dir ~/Downloads/matt/
```

## weights difference graphs
```
$python plotting/compare_weights_from_list.py --dir ~/Downloads/matt
```
and adjust dict `models` to change what models to compare
