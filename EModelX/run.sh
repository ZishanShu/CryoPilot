mapfile='data/inputs/maps/emd_32336.map.gz'
fasta_file='data/inputs/fastas/7w72'
python run.py --protocol=temp_free --EM_map="${mapfile}" --fasta="${fasta_file}" --output_dir=data/outputs --run_pulchra --pulchra_path modules/pulchra304/src/pulchra