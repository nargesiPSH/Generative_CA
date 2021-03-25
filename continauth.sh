#!/bin/sh
export LD_LIBRARY_PATH=/vol/vssp/signsrc/externalLibs/cudnn-8.0-linux-x64-v6.0/lib64:$LD_LIBRARY_PATH                                                                                                                                                                          
exec /vol/research/MCMC/anaconda3/envs/continauthH/bin/python /vol/research/MCMC/PyCharmCodes/Internship/ContinAuth-master/codes/CA-system.py
#make sure to update anaconda and python location to your local system file location

