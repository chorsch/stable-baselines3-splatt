import argparse
import torch

parser = argparse.ArgumentParser()

parser.add_argument(
    '--policy_type', 
    type=str, 
    default='CNNPolicy', 
    help='')

parser.add_argument(
    '--num_train_contexts', 
    type=int, 
    default=20, 
    help='')

# Wandb arguments: 
parser.add_argument(
    '--wandb_project_name', 
    type=str, 
    default='splatt', 
    help='wandb project name')

# SPLATT arguments
parser.add_argument(
    '--tau', 
    type=float, 
    default=1.0, 
    help='The average return at which you can start sparsifying.',
    )
parser.add_argument(
    '--lambda_init', 
    type=float, 
    default=1.0, 
    help='The initial Lambda sparsity coeffecient value',
    )
parser.add_argument(
    '--use_gumbel_mask', 
    type=str, 
    default='False', 
    help='')
parser.add_argument(
    '--att_out_dim', 
    type=int, 
    default=512, 
    help='')
parser.add_argument(
    '--num_agg_heads', 
    type=int, 
    default=4, 
    help='')
parser.add_argument(
    '--obs_type', 
    type=str, 
    default='standard', 
    help='')
parser.add_argument(
    '--weights_threshhold', 
    type=float, 
    default=0.1, 
    help='')
parser.add_argument(
    '--update_weights_frequency', 
    type=int, 
    default=1, 
    help='')
parser.add_argument(
    '--coef_fn', 
    type=str, 
    default='sin', 
    help='')
parser.add_argument(
    '--cnn_type', 
    type=str, 
    default='big', 
    help='type of cnn to use')
parser.add_argument(
    '--weights_loss_fn', 
    type=str, 
    default='bottom_k_inputs', 
    help='weights loss function')
parser.add_argument(
    '--weights_coef', 
    type=float, 
    default=1.0, 
    help='weights loss coefficient')
parser.add_argument(
    '--weights_coef_max', 
    type=float, 
    default=1e-10, 
    help='weights loss coefficient maximum')
parser.add_argument(
    '--fixed_weights_coef', 
    type=float, 
    default=0.0, 
    help='fixed weights loss coefficient')

parser.add_argument(
    '--weights_logits_coef', 
    type=float, 
    default=1.0, 
    help='weights logits regularization coefficient')
parser.add_argument(
    '--out_channels', 
    type=int, 
    default=32,)
parser.add_argument(
    '--embedding_dim', 
    type=int, 
    default=64,)
parser.add_argument(
    '--value_dim', 
    type=int, 
    default=64,)
parser.add_argument(
    '--use_skip_connections', 
    type=str, 
    default="False",)
parser.add_argument(
    '--aggregation', 
    type=str, 
    default="max",)
parser.add_argument(
    '--train_env', 
    type=str, 
    default="BabyAI-GoToRedBallGrey-v0",)
parser.add_argument(
    '--num_layers', 
    type=int, 
    default=1,)
parser.add_argument(
    '--num_mha_heads', 
    type=int, 
    default=1,)
parser.add_argument(
    '--k', 
    type=int, 
    default=1,)
parser.add_argument(
    '--rel_pos_enc', 
    type=str, 
    default='qk',)
parser.add_argument(
    '--pos_enc_type', 
    type=str, 
    default='absolute',)
parser.add_argument(
    '--notes', 
    type=str, 
    default='',)
parser.add_argument(
    '--shared_value_net', 
    type=bool, 
    default=False,)
parser.add_argument(
    '--use_fixed_keys', 
    type=bool, 
    default=False,)
parser.add_argument(
    '--add_zero_attn', 
    type=bool, 
    default=False,)
        

# PPO arguments. 
parser.add_argument(
    '--lr', 
    type=float, 
    default=5e-4, 
    help='learning rate')
parser.add_argument(
    '--eps',
    type=float,
    default=1e-5,
    help='RMSprop optimizer epsilon')
parser.add_argument(
    '--alpha',
    type=float,
    default=0.99,
    help='RMSprop optimizer apha')
parser.add_argument(
    '--gamma',
    type=float,
    default=0.999,
    help='discount factor for rewards')
parser.add_argument(
    '--gae_lambda',
    type=float,
    default=0.95,
    help='gae lambda parameter')
parser.add_argument(
    '--entropy_coef',
    type=float,
    default=0.01,
    help='entropy term coefficient')
parser.add_argument(
    '--value_loss_coef',
    type=float,
    default=0.5,
    help='value loss coefficient (default: 0.5)')
parser.add_argument(
    '--max_grad_norm',
    type=float,
    default=0.5,
    help='max norm of gradients)')
parser.add_argument(
    '--seed', 
    type=int, 
    default=0, 
    help='random seed')
parser.add_argument(
    '--num_processes',
    type=int,
    default=64,
    help='how many training CPU processes to use')
parser.add_argument(
    '--num_steps',
    type=int,
    default=256,
    help='number of forward steps in A2C')
parser.add_argument(
    '--ppo_epoch',
    type=int,
    default=1,
    help='number of ppo epochs')
parser.add_argument(
    '--num_mini_batch',
    type=int,
    default=8,
    help='number of batches for ppo')
parser.add_argument(
    '--clip_param',
    type=float,
    default=0.2,
    help='ppo clip parameter')
parser.add_argument(
    '--log_interval',
    type=int,
    default=5,
    help='log interval, one log per n updates')
parser.add_argument(
    '--total_timesteps',
    type=int,
    default=10e6,
    help='number of environment steps to train')
parser.add_argument(
    '--env_name',
    type=str,
    default='coinrun',
    help='environment to train on')
parser.add_argument(
    '--algo',
    default='idaac',
    choices=['idaac', 'daac', 'ppo', 'splatt', 'dsplatt', 'splatt_fixed', 'splatt_multi', 'splatt_multi_fixed'],
    help='algorithm to use')
parser.add_argument(
    '--log_dir',
    default='logs',
    help='directory to save agent logs')
parser.add_argument(
    '--save_dir',
    type=str,
    default='models',
    help='augmentation type')
parser.add_argument(
    '--no_cuda',
    action='store_true',
    default=False,
    help='disables CUDA training')
parser.add_argument(
    '--hidden_size',
    type=int,
    default=256,
    help='state embedding dimension')


