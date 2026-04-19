# EModelX
EModelX is a method for automatic cryo-EM protein complex structure modeling.
![EModelX](paper.png)
![EMDB-32336](case.jpg)
## Web Server
EModelX's web server is accessible at <https://bio-web1.nscc-gz.cn/app/EModelX>   
## Colab
EModelX can be run in Colab: [Minimal Example](https://colab.research.google.com/github/biomed-AI/EModelX/blob/main/minimal_example.ipynb)  
## Environment
`conda env create -f EModelX.yml`  
For EModelX(+AF), you may need to run AlphaFold following <https://github.com/deepmind/alphafold> or get AlphaFold-predicted single-chain structures from AlphaFoldDB (<https://alphafold.ebi.ac.uk/>).  

## Download NN model weights
Download NN model weights from <https://drive.google.com/file/d/13BKzEBfL0uubYgcJJTGSZ-PkO9hXAB9X/view?usp=drive_link> and place them in `./models`.  

## If you don't have standard fasta file  
Run:  
`python get_fasta.py  gen_fasta_path  'fasta_name_1;chain_num_1;sequence_1'  'fasta_name_2;chain_num_2;sequence_2'  ...;...;...  'fasta_name_n;chain_num_n;sequence_n'`  
, where `gen_fasta_path` is the path to save the generated fasta file (e.g. `data/inputs/fastas/7w72`)    
, `chain_num_n` should be an integer that indicates how many chains belong to this sequence    
, and `sequence_n` should be a protein sequence    
, and quotation marks `'` or `"` would be ignored    
, and make sure `;`s are only used as separtion marks.    


## Minimal Example: Protein complex structure modeling for a cryo-EM map
**For EModelX**:   
`python run.py --protocol=temp_free --EM_map=data/inputs/maps/emd_32336.map.gz --fasta=data/inputs/fastas/7w72 --output_dir=data/outputs --run_pulchra --pulchra_path modules/pulchra304/src/pulchra` 

**For EModelX(+AF)**, now we support automatically download the most similar alphafold structure from alphafoldDB within a certain similarity cutoff:   
`python run.py --protocol=temp_flex --EM_map=data/inputs/maps/emd_32336.map.gz --fasta=data/inputs/fastas/7w72 --template_dir=data/inputs/templates --download_afdb --afdb_allow_seq_id 0.6 --output_dir=data/outputs --run_pulchra --pulchra_path=modules/pulchra304/src/pulchra`   
, where you can replace `--EM_map` with your target EM map   
, and `--fasta` with your target fasta   
, and `--template_dir`: directory of the template folder, only needed when --protocol == temp_flex   
, and `--output_dir`: the output directory for modeling results   

**Notice:** If you want to run **EModelX(+AF)**, please either use `--download_afdb` or prepare the `--template_dir` by yourself. To prepare the `--template_dir` by yourself, please follow our example in `./inputs/templates`. More detailedly, you need to prepare a directory with a consistent name matching the header lines in the fasta file, and the PDB file needed to be renamed as ranked_0.pdb.   

## To enable postprocessing by phenix.real_space_refinement
### Environment
Install phenix following <https://phenix-online.org/> into a directory, e.g. `modules/phenix-1.20.1-4487`  

### Example
**For EModelX**:   
`python run.py --protocol=temp_free --EM_map=data/inputs/maps/emd_32336.map.gz --fasta=data/inputs/fastas/7w72 --output_dir=data/outputs --run_pulchra --pulchra_path=modules/pulchra304/src/pulchra --run_phenix --phenix_act=modules/phenix-1.20.1-4487/phenix_env.sh --resolution=3.1`  
**For EModelX(+AF)**:   
`python run.py --protocol=temp_flex --EM_map=data/inputs/maps/emd_32336.map.gz --fasta=data/inputs/fastas/7w72 --template_dir=data/inputs/templates --output_dir=data/outputs --download_afdb --afdb_allow_seq_id 0.6 --run_pulchra --pulchra_path=modules/pulchra304/src/pulchra --run_phenix --phenix_act=modules/phenix-1.20.1-4487/phenix_env.sh --resolution=3.1`


