# Downstream task
Evaluation of downstream task. Currently in a bit of a hacky way by using its own dataloader to get the eval data (since it relies on getting the labels and normalising them which the sim2real dataloader doesn't get).

## Usage
```
from downstream_task.evaller import evaller
# eval same domain space
e = evaller(ARGS.dir, data_task=('surface_3d','shear','sim'), model_task=('surface_3d','shear','sim'))
mae =e.get_MAE()  # pass sim data

# eval after transformed dataspace
e = evaller(ARGS.dir, data_task=('surface_3d','shear','real'), model_task=('surface_3d','shear','sim'))
mae =e.get_MAE(real2sim_model)  # pass real2sim model to eval the transform
```
