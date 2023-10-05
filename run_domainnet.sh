nohup python exp_domainnet.py -m "DAN-A"  > logs/dan_a_domainnet.out  2>&1 & 
nohup python exp_domainnet.py -m "DANN-A" > logs/dann_a_domainnet.out  2>&1 & 
nohup python exp_domainnet.py -m "DSAN-A" > logs/dsan_a_domainnet.out  2>&1 & 
nohup python exp_domainnet.py -m "FT-A"   > logs/ft_a_domainnet.out  2>&1 & 
nohup python exp_domainnet.py -m "MCD-A"  > logs/mcd_a_domainnet.out  2>&1 & 
nohup python exp_domainnet.py -m "NN-A"   > logs/nn_a_domainnet.out   2>&1 & 

nohup python exp_domainnet.py -m "NN-B"   > logs/nn_b_domainnet.out   2>&1 & 
nohup python exp_domainnet.py -m "DAN-B"  > logs/dan_b_domainnet.out  2>&1 & 
nohup python exp_domainnet.py -m "DANN-B" > logs/dann_b_domainnet.out 2>&1 & 
nohup python exp_domainnet.py -m "DSAN-B" > logs/dsan_b_domainnet.out 2>&1 & 
nohup python exp_domainnet.py -m "FT-B"   > logs/ft_b_domainnet.out   2>&1 & 
nohup python exp_domainnet.py -m "MCD-B"  > logs/mcd_b_domainnet.out  2>&1 & 
nohup python exp_domainnet.py -m "ABMSDA" > logs/abmsda_domainnet.out 2>&1 & 
nohup python exp_domainnet.py -m "M3SDA"  > logs/m3sda_domainnet.out  2>&1 & 
nohup python exp_domainnet.py -m "MDAN"   > logs/mdan_domainnet.out   2>&1 & 
nohup python exp_domainnet.py -m "MFSAN"  > logs/mfsan_domainnet.out  2>&1 & 

nohup python exp_domainnet.py -m "OURS"      > logs/ours_domainnet.out 2>&1 &
nohup python exp_domainnet.py -m "OURS-beta" > logs/ours_beta_domainnet.out 2>&1 &