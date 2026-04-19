import argparse
import warnings
import torch
import numpy as np
import os
from modules.AutoEM import Solver
from unet3d.emmodel import ResUNet3D4EM

warnings.filterwarnings('ignore')
np.set_printoptions(threshold=np.inf,suppress=True,precision=2)

def run_emodelx(dynamic_config,BB_model,CA_model,AA_model):
    if dynamic_config.protocol not in ['temp_free','temp_flex','seq_free']:
        print('Wrong protocol! protocol should be [temp_free,temp_flex,seq_free]')
        return 'Wrong protocol! protocol should be [temp_free,temp_flex,seq_free]'
    if dynamic_config.protocol in ['temp_free','temp_flex']:
        if not dynamic_config.fasta:
            print('--fasta is required when protocol is not seq_free')
            return '--fasta is required when protocol is not seq_free'
        elif not os.path.exists(dynamic_config.fasta):
            print('--fasta: path not exisit!')
            return '--fasta: path not exisit!'
        if dynamic_config.protocol =='temp_flex':
            if not dynamic_config.template_dir:
                print('--template_dir is required when protocol is not seq_free')
                return '--template_dir is required when protocol is not seq_free'
            elif not os.path.exists(dynamic_config.template_dir):
                print('--template_dir: path not exisit!')
                return '--template_dir: path not exisit!'
    
    if dynamic_config.run_phenix:
        dynamic_config.run_pulchra=True
        if not dynamic_config.resolution:
            print('--resolution is required for run.phenix_real_space_refine')
            return '--resolution is required for run.phenix_real_space_refine'
        if not dynamic_config.phenix_act:
            print('--phenix_act is required for run.phenix_real_space_refine')
            return '--phenix_act is required for run.phenix_real_space_refine'
        
    if dynamic_config.run_pulchra:
        if not dynamic_config.pulchra_path:
            print('--pulchra_path is required for run.phenix_real_space_refine')
            return '--pulchra_path is required for run.phenix_real_space_refine'


    AutoEM_solver = Solver(dynamic_config)
    return AutoEM_solver.run(BB_model,CA_model,AA_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=str, default='temp_free', help='choose among temp_free,temp_flex,seq_free')

    parser.add_argument('--EM_map', type=str, required=True, help='path of EM map')
    parser.add_argument('--fasta', type=str, default='', help='path of fasta file, required when protocol is temp_free or temp_flex')
    parser.add_argument('--template_dir', type=str, default='./inputs/templates', help='dir of template folder, required when --protocol is temp_flex, path format for different chain please reference to ./inputs/templates')
    
    parser.add_argument('--run_pulchra', action='store_true', help='whether to run pulchra for all_atom construction')
    parser.add_argument('--download_afdb', action='store_true', help='whether to download_afdb')
    parser.add_argument('--pulchra_path',type=str, help='directory of pulchra, e.g.: modules/pulchra304/src/pulchra')

    parser.add_argument('--run_phenix', action='store_true', help='whether to run phenix.real_space_refine')
    parser.add_argument('--resolution', type=float, help='resolution of EM map, required when run_phenix_real_space_refine is open')
    parser.add_argument('--phenix_act',type=str, help='script to activate phenix environment, e.g.: modules/phenix-1.20.1-4487/phenix_env.sh')
    parser.add_argument('--phenix_param', default='data/inputs/phenix.eff',type=str, help='param for phenix.real_space_refine')

    parser.add_argument('--output_dir', type=str, default='./data/outputs/example', help='dir of output pdbs')
    
    parser.add_argument('--best_CA_model', type=str, default='./models/best_CA_model.ckpt', help='set as default')
    parser.add_argument('--best_BB_model', type=str, default='./models/best_BB_model.ckpt', help='set as default')
    parser.add_argument('--best_AA_model', type=str, default='./models/best_AA_model.ckpt', help='set as default')
    parser.add_argument('--seed', type=int, default=2022, help='set as default')
    parser.add_argument('--cluster_eps', type=int, default=10, help='set as default')
    parser.add_argument('--cluster_min_points', type=int, default=10, help='set as default')
    parser.add_argument('--nms_radius', type=int, default=9, help='set as default')
    parser.add_argument('--CA_score_thrh', type=float, default=0.35, help='set as default')
    parser.add_argument('--frags_len', type=int, default=150, help='set as defaul')
    parser.add_argument('--n_hop', type=int, default=6, help='set as default')
    parser.add_argument('--neigh_mat_thrh', type=float, default=0.7, help='set as default')
    parser.add_argument('--mul_proc_num', type=int, default=30, help='set as default')
    parser.add_argument('--score_thrh', type=float, default=2, help='set as default')
    parser.add_argument('--gap_len', type=int, default=3, help='set as default')
    parser.add_argument('--struct_len', type=int, default=5, help='set as default')
    parser.add_argument('--afdb_allow_seq_id', type=float, default=0.6, help='set as default')
    
    dynamic_config = parser.parse_args()

    torch.manual_seed(dynamic_config.seed)
    BB_model = ResUNet3D4EM().to('cuda')
    CA_model = ResUNet3D4EM().to('cuda')
    AA_model = ResUNet3D4EM().to('cuda')
    BB_model.load_state_dict(torch.load(dynamic_config.best_BB_model))
    CA_model.load_state_dict(torch.load(dynamic_config.best_CA_model))
    AA_model.load_state_dict(torch.load(dynamic_config.best_AA_model))
    BB_model.eval()
    CA_model.eval()
    AA_model.eval()

    result = run_emodelx(dynamic_config,BB_model,CA_model,AA_model)
    if result!='success':
        raise Exception(result)