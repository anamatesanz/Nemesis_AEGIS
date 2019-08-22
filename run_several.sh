# Parallel thread database training running
# $bash run_several.sh

# This part is mandatory to every opened terminal
cd Desktop/IGEM/Rosetta/PyRosetta4.Debug.python36.mac.release-225/
export ROSETTA=/Users/anuska/Desktop/IGEM/Rosetta/rosetta_src_code/
export RNA_TOOLS=$ROSETTA/tools/rna_tools/
export PATH=$RNA_TOOLS/bin/:$PATH
export ROSETTA3=$ROSETTA/main/source/bin/
export PYTHONPATH=$PYTHONPATH:$RNA_TOOLS/bin/
source ~/.bashrc
python $RNA_TOOLS/sym_link.py

for i in $(seq 1 10)
do
    echo "Starting thread num " $i
    python3 prueba35_DB_Creation_Threads.py $i
done


