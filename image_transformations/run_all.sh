#!/bin/bash

# takes about 2mins per iteration step on desktop
STEPS=101     # ~3 hours
python image_transformations/eval_tranforms.py --trans rotation --min -50 --max 50 --steps $STEPS  --ram &
python image_transformations/eval_tranforms.py --trans brightness --min -1 --max 1 --steps $STEPS --ram &
python image_transformations/eval_tranforms.py --trans blur --min 0 --max 30 --steps 31 --ram &
python image_transformations/eval_tranforms.py --trans zoom --min 0.5 --max 3 --steps $STEPS --ram &
python image_transformations/eval_tranforms.py --trans x_shift --min -0.5 --max 0.5 --steps $STEPS --ram &
python image_transformations/eval_tranforms.py --trans y_shift --min -0.5 --max 0.5 --steps $STEPS --ram &
