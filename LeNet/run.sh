echo "#!/bin/bash -i" > run_aux.sh
echo "cd `pwd`
conda activate ${CONDA_DEFAULT_ENV}
python main.py" >> run_aux.sh
chmod +x run_aux.sh

echo "localhost
localhost
" > nodelist

gaspi_run -m nodelist `pwd`/run_aux.sh
