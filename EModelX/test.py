from Bio.PDB.PDBParser import PDBParser
import argparse
from Bio.PDB import DSSP
import warnings
import numpy as np
from modules.utils import calc_dis,parseMMscore,parseChainCompscore
import pandas as pd
from copy import copy
import os
from multiprocessing import Pool
from subprocess import run, DEVNULL
import glob

np.set_printoptions(threshold=np.inf,suppress=True,precision=2)
warnings.filterwarnings('ignore')

def get_model(pdbid,emid):
    models={}
    models['EModelX']=f'data/built_models/EModelX/{pdbid}_EModelX_all_atom_model_real_space_refined_000.pdb'
    models['EModelX(init)']=f'data/built_models/EModelX(init)/{pdbid}_EModelX(init)_all_atom_model_real_space_refined_000.pdb'
    models['EModelX(+AF)']=f'data/built_models/EModelX(+AF)/{pdbid}_EModelX(+AF)_all_atom_model_real_space_refined_000.pdb'
    models['Phenix']=f'data/built_models/Phenix/ph_{pdbid}.pdb'
    models['MAINMAST']=f'data/built_models/MAINMAST/mm_{pdbid}_rebuild_allchain.pdb'
    models['DeepTracer']=f'data/built_models/DeepTracer/emd_{emid}.pdb'
    models['ModelAngelo']=f'data/built_models/ModelAngelo/{pdbid}.pdb'
    models['PDB']=f'data/built_models/PDB/prot_only_{pdbid}_only_atom.pdb'
    models['PDB_refined']=f'data/built_models/PDB/prot_only_{pdbid}_only_atom_real_space_refined_000.pdb'
    models['ModelAngelo_refined']=f'data/built_models/ModelAngelo/{pdbid}_real_space_refined_000.pdb'
    return models


def dssp(file_name):
    p = PDBParser()

    structure = p.get_structure('structure_name',file_name)
    # 解析需要的pdb文件
    model = structure[0]
    with open(file_name,'r') as r:
        lines=r.readlines()
    name=file_name.split('/')[-1]
    tmp_file=f'temp_files/tmp_{name}'
    with open(tmp_file,'w') as w:
        for line in lines:
            w.write(f'{line.strip()}  \n')
    
    dssp_res=DSSP(model,f'{tmp_file}', dssp='mkdssp')
    os.system(f'rm \'{tmp_file}\'')

    return dssp_res,model
    

def get_ss_recall(acc_dict,f1,f2):
    ss1,model1=dssp(f1)
    ss2,model2=dssp(f2)
    
    coords_list1=[]
    ss_list1=[]
    key_list1=list(ss1.keys())
    key_list1_new=[]
    for key in key_list1:
        try:
            ss_list1.append(ss1[key][2])
            cid=key[0]
            res_num=key[1]
            ca=model1[cid][res_num]['CA']
            coords_list1.append(ca.get_coord())
            key_list1_new.append(key)
        except:
            continue
    key_list1=key_list1_new

    key_list2_new=[]
    coords_list2=[]
    ss_list2=[]
    key_list2=list(ss2.keys())
    for key in key_list2:
        try:
            ss_list2.append(ss2[key][2])
            cid=key[0]
            res_num=key[1]
            ca=model2[cid][res_num]['CA']
            coords_list2.append(ca.get_coord())
            key_list2_new.append(key)
        except:
            continue
    key_list2=key_list2_new

    dis_mat=calc_dis(coords_list1,coords_list2)
    nearest=np.argmin(dis_mat,axis=1)
    min_dis=np.min(dis_mat,axis=1)

    for i,key in enumerate(key_list1):
        ss=ss_list1[i]
        if min_dis[i] < 3:
            j=nearest[i]
            score=1 if ss_list2[j]==ss else 0
        else:
            score=0

        if ss in acc_dict:
            acc_dict[ss].append(score)
        else:
            acc_dict['-'].append(score)
        acc_dict['all'].append(score)
    return acc_dict

def job3(z):
    acc_dict={}
    for key in ['G','H','I','T','E','B','S','-','all']:
        acc_dict[key]=[]
    try:
        acc_dict=get_ss_recall(acc_dict,z[0],z[1])
    except:
        for key in ['G','H','I','T','E','B','S','-','all']:
            acc_dict[key]=[0]
    return acc_dict 


def ss_recall(args,methods,df):
    for method in methods:
        data_list=[]
        for emid,pdbid in zip(df['emid'],df['pdbid']):
            models= get_model(pdbid,emid)
            f1=models['PDB']
            f2=models[method]
            data_list.append((f1,f2))

        pool=Pool(args.ncpu)
        results=pool.map(job3, data_list)

        acc_list=[]
        for acc_dict in results:
            for key in ['G','H','I','T','E','B','S','-']:
                print('{}:{:.2f}'.format(key,np.mean(acc_dict[key])),end='\t')
            print('all:{:.2f}'.format(np.mean(acc_dict['all'])))
            acc_list.append(np.mean(acc_dict['all']))
        df[f'SS recall by {method}']=acc_list
        print(method,np.mean(acc_list))

    return df

def get_ramalyze_rotalyze(cmd):
    try:
        out_rate=1000
        results=os.popen(cmd).readlines()
        for i,res in enumerate(results):
            if 'SUMMARY:' in res and 'outliers' in res:
                s=res.split()
                out_rate=float(s[1][:-1])
        return out_rate
    except:
        return 1000

def get_clash(cmd):
    try:
        clash_score=1000
        results=os.popen(cmd).readlines()
        for i,res in enumerate(results):
            if 'clashscore = ' in res:
                s=res.split()
                clash_score=float(s[2])
        return clash_score
    except:
        return 1000
    

def phenix_rama_rota_clash(args,methods,df,metric):
    phenix_dir=os.path.dirname(args.phenix_act)
    act_cmd=f'export PHENIX=\"{phenix_dir}\" && export PHENIX_VERSION=1.20.1-4487 && export DISPLAY=:0.0 && . $PHENIX/build/setpaths.sh'
    for method in methods:
        
        
        cmd_list=[]
        for emid,pdbid in zip(df['emid'],df['pdbid']):
            models= get_model(pdbid,emid)
            path=os.path.abspath(models[method])
            cmd=f'{act_cmd} && cd temp_files  && phenix.{metric} \'{path}\''
            cmd_list.append(cmd)
            
        rate_list=[]
        pool=Pool(args.ncpu)
        if metric=='ramalyze' or metric=='rotalyze':
            key=f'ramachandran outliers by {method} (%)' if metric=='ramalyze' else f'rotalyze outliers by {method} (%)'
            results=pool.map(get_ramalyze_rotalyze, cmd_list)
        elif metric=='clashscore':
            key=f'calshscore outliers by {method}'
            results=pool.map(get_clash, cmd_list)

        for rate in results:
            rate_list.append(rate)

        df[key]=rate_list
    return df

def job1(z):
    return parseMMscore(z[0],z[1])


def mmalign(args,methods,df):
    for method in methods:
        keys=[f'TM-score by {method}',f'RMSD by {method}',f'Coverage by {method}',f'SeqID by {method}']
        list_dict={}
        for key in keys:
            list_dict[key]=[]
        
        data_list=[]
        for emid,pdbid in zip(df['emid'],df['pdbid']):
            models= get_model(pdbid,emid)
            f1=models['PDB']
            f2=models[method]
            data_list.append((f1,f2))
        pool=Pool(args.ncpu)
        results=pool.map(job1, data_list)

        for res in results:
            ResNum_pdb,ResNum_pred,Align_len,MM1,MM2,RMSD,SeqID=res
            list_dict[f'TM-score by {method}'].append(MM1)
            list_dict[f'RMSD by {method}'].append(RMSD)
            list_dict[f'Coverage by {method}'].append(Align_len/ResNum_pdb)
            list_dict[f'SeqID by {method}'].append(SeqID)
        for key in keys:
            df[key]=list_dict[key]

    return df

def get_cc(cmd):
    try:
        cc_mask=0
        cc_box=0
        results=os.popen(cmd).readlines()
        for i,res in enumerate(results):
            if res.startswith('  CC_mask'):
                cc_mask=float(res.split(':')[1].strip())
            if res.startswith('  CC_box'):
                cc_box=float(res.split(':')[1].strip())
        return cc_mask,cc_box
    except:
        return 0,0
    

def phenix_cc(args,methods,df):
    phenix_dir=os.path.dirname(args.phenix_act)
    act_cmd=f'export PHENIX=\"{phenix_dir}\" && export PHENIX_VERSION=1.20.1-4487 && export DISPLAY=:0.0 && . $PHENIX/build/setpaths.sh'
    for method in methods:
        cmd_list=[]
        for emid,pdbid,resol in zip(df['emid'],df['pdbid'],df['resolution']):
            em_path=os.path.abspath(os.path.join(args.map_dir,f'emd_{emid}.map.gz'))
            models= get_model(pdbid,emid)
            path=os.path.abspath(models[method])
            cmd=f'{act_cmd} && cd temp_files  && phenix.map_model_cc \'{path}\' {em_path} resolution={resol}'
            cmd_list.append(cmd)
            
        cc_mask_list=[]
        cc_box_list=[]
        pool=Pool(args.ncpu)
        results=pool.map(get_cc, cmd_list)

        for res in results:
            cc_mask, cc_box =res
            cc_mask_list.append(cc_mask)
            cc_box_list.append(cc_box)

        df[f'CC_MASK by {method}']=cc_mask_list
        df[f'CC_BOX by {method}']=cc_box_list
    return df

def get_mt(cmd):
    try:
        fsc05=999
        results=os.popen(cmd).readlines()
        for i,res in enumerate(results):
            line=res.strip()
            if line.startswith('FSC(map,model map)=0.5'):
                fsc05=float(line.split()[3])
        return fsc05
    except:
        return 999

def phenix_mt(args,methods,df):
    phenix_dir=os.path.dirname(args.phenix_act)
    act_cmd=f'export PHENIX=\"{phenix_dir}\" && export PHENIX_VERSION=1.20.1-4487 && export DISPLAY=:0.0 && . $PHENIX/build/setpaths.sh'
    for method in methods:
        cmd_list=[]
        for emid,pdbid,resol in zip(df['emid'],df['pdbid'],df['resolution']):
            em_path=os.path.abspath(os.path.join(args.map_dir,f'emd_{emid}.map.gz'))
            models= get_model(pdbid,emid)
            path=os.path.abspath(models[method])
            cmd=f'{act_cmd} && cd temp_files  && phenix.mtriage \'{path}\' {em_path}'
            # print(get_mt(cmd))
            cmd_list.append(cmd)
            
        fsc05_list=[]
        pool=Pool(args.ncpu)
        results=pool.map(get_mt, cmd_list)

        for fsc05 in results:
            fsc05_list.append(fsc05)

        df[f'd_FSC05 MASKED by {method}']=fsc05_list
    return df


def phenix_chain_comp(args,methods,df):
    phenix_dir=os.path.dirname(args.phenix_act)
    act_cmd=f'export PHENIX=\"{phenix_dir}\" && export PHENIX_VERSION=1.20.1-4487 && export DISPLAY=:0.0 && . $PHENIX/build/setpaths.sh'
    for method in methods:
        keys=[f'Mean Length by {method}',f'Forward Rate by {method}',f'CA Score by {method}',f'Found by {method}',f'Sequence Matching by {method}',f'Close RMSD by {method}']
        list_dict={}
        for key in keys:
            list_dict[key]=[]
        
        data_list=[]
        for emid,pdbid in zip(df['emid'],df['pdbid']):
            models= get_model(pdbid,emid)
            f1=os.path.abspath(models['PDB'])
            f2=os.path.abspath(models[method])
            cmd=f'{act_cmd} && cd temp_files  && phenix.chain_comparison \'{f1}\' \'{f2}\''
            data_list.append(cmd)
        pool=Pool(args.ncpu)
        results=pool.map(parseChainCompscore, data_list)

        for res in results:
            Close_RMSD, Close_N, Far_N, Close_Forward_N, Close_Reverse_N, Close_Mixed_N, Found, CA_Score, Seq_Match, Seq_Score, Mean_length, Fragments, Bad_Connections= res
            list_dict[f'Found by {method}'].append(Found)
            list_dict[f'Close RMSD by {method}'].append(Close_RMSD)
            list_dict[f'Sequence Matching by {method}'].append(Seq_Match)
            list_dict[f'Mean Length by {method}'].append(Mean_length)
            if Close_N==0:
                list_dict[f'Forward Rate by {method}'].append(0)
            else:
                list_dict[f'Forward Rate by {method}'].append(Close_Forward_N/Close_N)
            list_dict[f'CA Score by {method}'].append(CA_Score)
        for key in keys:
            df[key]=list_dict[key]

    return df

def seq_recall(args,methods,df):
    ignore_pdbid=['7n06','7b03','7a4a','7nv0','7mxy','7n9z'] #ModelAngelo failed to conform the map

    results={}
    for method in methods:
        results[method]=[]
    
    bfactors=[]
    pdbParser=PDBParser(PERMISSIVE=1)
    for pdbid, emid in zip(df['pdbid'],df['emid']):
        if pdbid in ignore_pdbid:
            continue
        print(pdbid, emid)
        models=get_model(pdbid,emid)
        gt=pdbParser.get_structure(pdbid,models['PDB'])
        pred={}
        for method in methods:
            pred[method]=pdbParser.get_structure(pdbid,models[method])

        coords={}
        aas={}
        for method in methods:
            coords[method]=[]
            aas[method]=[]

        for method in methods:
            for chain in pred[method][0].get_list():
                for residue in chain.get_list():
                    coords[method].append(residue['CA'].get_coord())
                    aas[method].append(residue.get_resname())
            coords[method]=np.array(coords[method])
            aas[method]=np.array(aas[method])
        
        for chain in gt[0].get_list():
            for residue in chain.get_list():
                if 'CA' in residue:
                    if residue['CA'].get_bfactor()>=1:
                        bfactors.append(residue['CA'].get_bfactor())
                        for method in methods:
                            c=residue['CA'].get_coord()
                            delta_coords=coords[method]-np.array([c for _ in range(coords[method].shape[0])])
                            distance=np.sqrt(np.sum(delta_coords*delta_coords,axis=1))
                            recall=np.int(np.sum((aas[method]==residue.get_resname())*(distance<=3))>0)
                            results[method].append(recall)
        
    argind=np.argsort(bfactors)
    yy={}
    for method in methods:
        yy[method]=np.array(results[method])[argind]
    xx=np.sort(bfactors)
    x=[]
    y={}
    for method in methods:
        y[method]=[]
    for i in range(len(xx)//1000):
        x.append(np.mean(xx[i*1000:(i+1)*1000]))
        for method in methods:
            y[method].append(np.mean(yy[method][i*1000:(i+1)*1000]))
    
    df=pd.DataFrame()
    df['Bfactor']=x
    for method in methods:
        df[f'Sequence Recall by {method}']=y[method]
    
    return df

def run_refine(cmd):
    run([cmd], stdout=DEVNULL, stderr=DEVNULL, shell=True)


def phenix_real_space_refine(args,methods,df):
    phenix_dir=os.path.dirname(args.phenix_act)
    act_cmd=f'export PHENIX=\"{phenix_dir}\" && export PHENIX_VERSION=1.20.1-4487 && export DISPLAY=:0.0 && . $PHENIX/build/setpaths.sh'
    for method in methods:
        cmd_list=[]
        for emid,pdbid,resol in zip(df['emid'],df['pdbid'],df['resolution']):
            em_path=os.path.abspath(os.path.join(args.map_dir,f'emd_{emid}.map.gz'))
            phenix_param=os.path.abspath(args.phenix_param)
            models= get_model(pdbid,emid)
            path=os.path.abspath(models[method])
            path_dir=os.path.dirname(path)
            if len(glob.glob(os.path.join(path_dir,f'*{pdbid}*real_space_refined_000.pdb')))>0:
                continue
            print(pdbid)
            path_name=path.split('/')[-1]
            cmd=f'{act_cmd} && cd \'{path_dir}\'  && phenix.real_space_refine \'{path_name}\' {em_path} {phenix_param} resolution={resol}'
            cmd_list.append(cmd)
            
        pool=Pool(args.ncpu)
        results=pool.map(run_refine, cmd_list)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_csv', type=str, default='data/inputs/csv/test.csv', help='path of test set list')
    parser.add_argument('--results_dir', type=str, default='data/displays/tables/', help='path of test set list')
    parser.add_argument('--map_dir', type=str, default='data/inputs/maps', help='directory of maps')
    
    parser.add_argument('--MMalign', action='store_true', help='whether to test MMalign')
    parser.add_argument('--phenix_cc', action='store_true', help='whether to test phenix_cc')
    parser.add_argument('--phenix_mt', action='store_true', help='whether to test phenix_mt')
    parser.add_argument('--phenix_chain_comp', action='store_true', help='whether to test phenix_chain_comp')
    parser.add_argument('--phenix_ramalyze', action='store_true', help='whether to test phenix_ramalyze')
    parser.add_argument('--phenix_rotalyze', action='store_true', help='whether to test phenix_rotalyze')
    parser.add_argument('--phenix_clashscore', action='store_true', help='whether to test phenix_clashscore')
    parser.add_argument('--phenix_refine', action='store_true', help='whether to test phenix_real_space_refine')
    parser.add_argument('--phenix_act',type=str, default='modules/phenix-1.20.1-4487/phenix_env.sh', help='script to activate phenix environment, e.g.: modules/phenix-1.20.1-4487/phenix_env.sh')
    parser.add_argument('--phenix_param', default='data/inputs/phenix.eff',type=str, help='param for phenix.real_space_refine')


    parser.add_argument('--seq_recall', action='store_true', help='whether to test sequenceRecall')
    parser.add_argument('--secondaryStructure', action='store_true', help='whether to test secondaryStructure')
    parser.add_argument('--ss_recall', action='store_true', help='whether to test secondaryStructure')

    parser.add_argument('--ncpu', type=int, required=True, help='number of cpu to run')
    
    args = parser.parse_args()

    df=pd.read_csv(args.test_csv)

    if args.ss_recall:
        methods=['EModelX','EModelX(+AF)','Phenix','MAINMAST','DeepTracer','ModelAngelo']
        df=ss_recall(args,methods,df)
        df.to_csv(os.path.join(args.results_dir,'results_ss_recall.csv'),index=False)
    
    if args.phenix_ramalyze:
        methods=['EModelX','EModelX(+AF)','Phenix','MAINMAST','DeepTracer','ModelAngelo','PDB']
        df=phenix_rama_rota_clash(args,methods,df,'ramalyze')
        df.to_csv(os.path.join(args.results_dir,'results_ramalyze.csv'),index=False)
    
    if args.phenix_rotalyze:
        methods=['EModelX','EModelX(+AF)','Phenix','MAINMAST','DeepTracer','ModelAngelo','PDB']
        df=phenix_rama_rota_clash(args,methods,df,'rotalyze')
        df.to_csv(os.path.join(args.results_dir,'results_rotalyze.csv'),index=False)

    if args.phenix_clashscore:
        methods=['EModelX','EModelX(+AF)','Phenix','MAINMAST','DeepTracer','ModelAngelo','PDB']
        df=phenix_rama_rota_clash(args,methods,df,'clashscore')
        df.to_csv(os.path.join(args.results_dir,'results_clashscore.csv'),index=False)
    
    if args.MMalign:
        methods=['EModelX(init)','EModelX','EModelX(+AF)','Phenix','MAINMAST','DeepTracer','ModelAngelo']
        df=mmalign(args,methods,df)
        df.to_csv(os.path.join(args.results_dir,'results_mmalign.csv'),index=False)

    if args.phenix_cc:
        methods=['EModelX','EModelX(+AF)','Phenix','MAINMAST','DeepTracer','ModelAngelo','PDB']
        df=phenix_cc(args,methods,df)
        df.to_csv(os.path.join(args.results_dir,'results_cc.csv'),index=False)

    if args.phenix_mt:
        methods=['EModelX','EModelX(+AF)','Phenix','MAINMAST','DeepTracer','ModelAngelo','PDB']
        df=phenix_mt(args,methods,df)
        df.to_csv(os.path.join(args.results_dir,'results_mt.csv'),index=False)

    if args.phenix_chain_comp:
        methods=['EModelX(init)','EModelX','EModelX(+AF)','Phenix','MAINMAST','DeepTracer','ModelAngelo']
        df=phenix_chain_comp(args,methods,df)
        df.to_csv(os.path.join(args.results_dir,'results_chain_comp.csv'),index=False)
    
    if args.seq_recall:
        methods=['EModelX(init)','EModelX','EModelX(+AF)','Phenix','MAINMAST','DeepTracer','ModelAngelo']
        df=pd.read_csv(args.test_csv)
        df=seq_recall(args,methods,df)
        df.to_csv(os.path.join(args.results_dir,'results_seq_recall.csv'),index=False)
