#!/bin/bash

# ======= 1% setting ==========
# ./script/command.sh simmatch-1p   8 1 "python3 -u simmatch.py --nesterov --lambda_in 5 --lr 0.03  --epochs 400 --cos --warmup-epoch 5 --st 0.1  --tt 0.1 --anno-percent 0.01 --c_smooth 0.9  --DA --checkpoint simmatch-1p.pth "

# ======= 10% setting ==========
# ./script/command.sh simmatch-10p  8 1 "python3 -u simmatch.py --nesterov --lambda_in 5 --lr 0.03  --epochs 400 --cos --warmup-epoch 5 --st 0.1  --tt 0.1 --anno-percent 0.1  --c_smooth 0.9  --DA --checkpoint simmatch-10p.pth"


# ======= evaluate 1% ==========
# ./script/command.sh eval-simmatch-1p    8 1  "python3 -u simmatch.py --evaluate --anno-percent 0.01  --DA --checkpoint simmatch-1p.pth"
# * Acc@1 67.200 Acc@5 87.082

# ======= evaluate 10% ==========
# ./script/command.sh eval-simmatch-10p   8 1  "python3 -u simmatch.py --evaluate --anno-percent 0.1   --DA --checkpoint simmatch-10p.pth"
# * Acc@1 74.464 Acc@5 91.648
