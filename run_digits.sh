nohup python exp_digits.py -m "DAN-A"  -e 150 > logs/dan_a_pyhidiff_digits.out  2>&1 & 
nohup python exp_digits.py -m "DANN-A" -e 150 > logs/dann_a_pyhidiff_digits.out  2>&1 & 
nohup python exp_digits.py -m "DSAN-A" -e 150 > logs/dsan_a_pyhidiff_digits.out  2>&1 & 
nohup python exp_digits.py -m "FT-A"   -e 150 > logs/ft_a_pyhidiff_digits.out  2>&1 & 
nohup python exp_digits.py -m "MCD-A"  -e 150 > logs/mcd_a_pyhidiff_digits.out  2>&1 & 

nohup python exp_digits.py -m "DAN-B"  -e 150 > logs/dan_b_pyhidiff_digits.out  2>&1 & 
nohup python exp_digits.py -m "DANN-B" -e 150 > logs/dann_b_pyhidiff_digits.out 2>&1 & 
nohup python exp_digits.py -m "DSAN-B" -e 150 > logs/dsan_b_pyhidiff_digits.out 2>&1 & 
nohup python exp_digits.py -m "FT-B"   -e 150 > logs/ft_b_pyhidiff_digits.out   2>&1 & 
nohup python exp_digits.py -m "MCD-B"  -e 150 > logs/mcd_b_pyhidiff_digits.out  2>&1 & 
nohup python exp_digits.py -m "NN-A"   -e 150 > logs/nn_a_pyhidiff_digits.out   2>&1 & 
nohup python exp_digits.py -m "NN-B"   -e 150 > logs/nn_b_pyhidiff_digits.out   2>&1 & 
nohup python exp_digits.py -m "ABMSDA" -e 150 > logs/abmsda_pyhidiff_digits.out 2>&1 & 
nohup python exp_digits.py -m "M3SDA"  -e 150 > logs/m3sda_pyhidiff_digits.out  2>&1 & 
nohup python exp_digits.py -m "MDAN"   -e 150 > logs/mdan_pyhidiff_digits.out   2>&1 & 
nohup python exp_digits.py -m "MFSAN"  -e 150 > logs/mfsan_pyhidiff_digits.out  2>&1 & 

nohup python exp_digits.py -m "OURS"      -e 150 > logs/ours_pyhidiff_digits.out 2>&1 &
nohup python exp_digits.py -m "OURS-beta" -e 150 > logs/ours_beta_pyhidiff_digits.out 2>&1 &