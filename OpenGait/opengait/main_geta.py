import os
import argparse
import torch
import torch.nn as nn
from modeling import models
from utils import config_loader, get_ddp_module, init_seeds, params_count, get_msg_mgr

parser = argparse.ArgumentParser(description='Main program for GaitGL with GETA compression.')
parser.add_argument('--local_rank', type=int, default=0,
                    help="passed by torch.distributed.launch module")
parser.add_argument('--local-rank', type=int, default=0,
                    help="passed by torch.distributed.launch module, for pytorch >=2.0")
parser.add_argument('--cfgs', type=str,
                    default='configs/gaitgl/gaitgl_geta.yaml', help="path of config file")
parser.add_argument('--phase', default='train',
                    choices=['train', 'test'], help="choose train or test phase")
parser.add_argument('--log_to_file', action='store_true',
                    help="log to file, default path is: output/<dataset>/<model>/<save_name>/<logs>/<Datetime>.txt")
parser.add_argument('--iter', default=0, help="iter to restore")
opt = parser.parse_args()


def initialization(cfgs, training):
    msg_mgr = get_msg_mgr()
    engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg']
    output_path = os.path.join('output/', cfgs['data_cfg']['dataset_name'],
                               cfgs['model_cfg']['model'], engine_cfg['save_name'])
    if training:
        msg_mgr.init_manager(output_path, opt.log_to_file, engine_cfg['log_iter'],
                             engine_cfg['restore_hint'] if isinstance(engine_cfg['restore_hint'], (int)) else 0)
    else:
        msg_mgr.init_logger(output_path, opt.log_to_file)

    msg_mgr.log_info(engine_cfg)

    seed = torch.distributed.get_rank()
    init_seeds(seed)


def run_model(cfgs, training):
    msg_mgr = get_msg_mgr()
    model_cfg = cfgs['model_cfg']
    msg_mgr.log_info(model_cfg)
    Model = getattr(models, model_cfg['model'])
    model = Model(cfgs, training)
    
    # Create a dummy input for the model - necessary for OTO
    batch_size = 4
    seq_len = 30  # Frames per sequence
    height = 64   # Height of silhouette
    width = 44    # Width of silhouette
    
    # Silhouette images in batch form
    sils = torch.rand(batch_size, seq_len, 1, height, width).cuda()
    # Labels, types, views, seqL
    labs = torch.zeros(batch_size).long().cuda()
    typs = torch.zeros(batch_size).long().cuda()
    vies = torch.zeros(batch_size).long().cuda()
    seqL = torch.full((batch_size,), seq_len).long().cuda()
    
    dummy_input = [sils, labs, typs, vies, seqL]
    
    if training and cfgs['trainer_cfg']['sync_BN']:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    if 'compression_optimizer' in cfgs and training:
        # Initialize the compression optimizer (GETA or HESSO)
        if cfgs['compression_optimizer'] == 'geta':
            optimizer = model.init_geta(dummy_input, cfgs['geta_optimizer_cfg'])
            msg_mgr.log_info("Using GETA optimizer for compression")
        elif cfgs['compression_optimizer'] == 'hesso':
            optimizer = model.init_hesso(dummy_input, cfgs['hesso_optimizer_cfg'])
            msg_mgr.log_info("Using HESSO optimizer for compression")
        else:
            msg_mgr.log_info("No compression optimizer specified, using standard optimizer")
            optimizer = None
    else:
        optimizer = None
    
    if cfgs['trainer_cfg']['fix_BN']:
        model.fix_BN()
    
    model = get_ddp_module(model, cfgs['trainer_cfg']['find_unused_parameters'])
    msg_mgr.log_info(params_count(model))
    msg_mgr.log_info("Model Initialization Finished!")

    if training:
        # Run training with the appropriate optimizer
        Model.run_train(model, optimizer=optimizer)
    else:
        # After training is complete, construct the compressed model if testing with GETA
        if hasattr(model.module, 'construct_compressed_model') and not training:
            compression_dir = cfgs['trainer_cfg'].get('compression_dir', './output/compressed_models')
            os.makedirs(compression_dir, exist_ok=True)
            compressed_model_path = model.module.construct_compressed_model(compression_dir)
            if compressed_model_path:
                msg_mgr.log_info(f"Compressed model saved to {compressed_model_path}")
        
        # Run standard testing
        Model.run_test(model)


if __name__ == '__main__':
    torch.distributed.init_process_group('nccl', init_method='env://')
    if torch.distributed.get_world_size() != torch.cuda.device_count():
        raise ValueError("Expect number of available GPUs({}) equals to the world size({}).".format(
            torch.cuda.device_count(), torch.distributed.get_world_size()))
    cfgs = config_loader(opt.cfgs)
    if opt.iter != 0:
        cfgs['evaluator_cfg']['restore_hint'] = int(opt.iter)
        cfgs['trainer_cfg']['restore_hint'] = int(opt.iter)

    training = (opt.phase == 'train')
    initialization(cfgs, training)
    run_model(cfgs, training)