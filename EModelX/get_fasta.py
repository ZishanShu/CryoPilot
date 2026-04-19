import sys
from modules.utils import chainID_list
# fasta_name||chain_num||sequence
chain_ind=0
gen_fasta_path=sys.argv[1]
with open(gen_fasta_path,'w') as w:
    for arg in sys.argv[2:]:
        try:
            arg.replace('\'','').replace('\"','')
            seq_s=arg.split(',')
            assert len(seq_s)==3
            fasta_name,chain_num,sequence=seq_s[0],seq_s[1],seq_s[2]
            fasta_name=fasta_name.strip().replace(' ','')
            chain_num=int(chain_num.strip().replace(' ',''))
            sequence=sequence.strip().replace(' ','')
            
            chain_list=[]
            for i in range(chain_num):
                chain_list.append(chainID_list[chain_ind])
                chain_ind+=1
            chain_str=', '.join(chain_list)
            w.write(f'>{fasta_name}|Chains {chain_str}|generated\n{sequence}\n')
        except Exception as ex:
            print(ex)

