#!/usr/bin/env bash
python my_try.py --horizon 6 --data data/electricity.txt --save save/elec.pt --output_fun Linear --model LSTNet --skip 0 --batch_size 1 --hidCNN 50 --hidRNN 50
