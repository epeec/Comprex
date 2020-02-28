#shopt -s expand_aliases
#alias ssh="ssh -i ${HOME}/.ssh/id_rsa_${SLURM_JOB_ID}"
gaspi_run -m nodelist `pwd`/run_aux.sh
