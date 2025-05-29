# Rdkit import should be first, do not move it
import pandas as pd
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import argparse
from qm9 import dataset
from qm9.models import get_model
import os
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked
import torch
import time
import pickle
from configs.datasets_config import get_dataset_info
from os.path import join
from qm9.sampling import sample
from qm9.analyze import analyze_stability_for_molecules, analyze_node_distribution
from qm9.utils import prepare_context, compute_mean_mad
from qm9 import visualizer as qm9_visualizer
import qm9.losses as losses
from qm9.analyze import check_stability 
try:
    from qm9 import rdkit_functions
except ModuleNotFoundError:
    print('Not importing rdkit functions.')
import datetime
import tqdm
import os
def create_folders(args):
    try:
        os.makedirs('outputs')
    except OSError:
        pass

    try:
        os.makedirs('outputs/' + args.exp_name)
    except OSError:
        pass
def check_mask_correct(variables, node_mask):
    for variable in variables:
        assert_correctly_masked(variable, node_mask)


def analyze_and_save(args, eval_args, device, generative_model,
                     nodes_dist, prop_dist, dataset_info, n_samples=10,
                     batch_size=10, save_to_xyz=False, timestep = 1000,scheduler="DDPM"):
    batch_size = min(batch_size, n_samples)
    assert n_samples % batch_size == 0
    molecules = {'one_hot': [], 'x': [], 'node_mask': []}
    start_time = time.time()
    timestamp = time.time()
    secs_per_sample = 0
    for i in tqdm.tqdm(range(int(n_samples/batch_size)),desc = '%.2f secs/sample' % (secs_per_sample)):
        nodesxsample = nodes_dist.sample(batch_size)
        one_hot, charges, x, node_mask = sample(
            args, device, generative_model, dataset_info, prop_dist=prop_dist, nodesxsample=nodesxsample, timestep = timestep)

        molecules['one_hot'].append(one_hot.detach().cpu())
        molecules['x'].append(x.detach().cpu())
        molecules['node_mask'].append(node_mask.detach().cpu())

        current_num_samples = (i+1) * batch_size
        secs_per_sample = (time.time() - start_time) / current_num_samples

        if save_to_xyz:
            id_from = i * batch_size
            qm9_visualizer.save_xyz_file(
                join(eval_args.model_path, f'eval/analyzed_molecules_{timestep}_{timestamp}/'),
                one_hot, charges, x, dataset_info, id_from, name='molecule',
                node_mask=node_mask)

    molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
    stability_dict, rdkit_metrics = analyze_stability_for_molecules(
        molecules, dataset_info)

    return stability_dict, rdkit_metrics


def test(args, flow_dp, nodes_dist, device, dtype, loader, partition='Test', num_passes=1):
    flow_dp.eval()
    nll_epoch = 0
    n_samples = 0
    for pass_number in range(num_passes):
        with torch.no_grad():
            for i, data in enumerate(loader):
                # Get data
                x = data['positions'].to(device, dtype)
                node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
                edge_mask = data['edge_mask'].to(device, dtype)
                one_hot = data['one_hot'].to(device, dtype)
                charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

                batch_size = x.size(0)

                x = remove_mean_with_mask(x, node_mask)
                check_mask_correct([x, one_hot], node_mask)
                assert_mean_zero_with_mask(x, node_mask)

                h = {'categorical': one_hot, 'integer': charges}

                if len(args.conditioning) > 0:
                    context = prepare_context(args.conditioning, data).to(device, dtype)
                    assert_correctly_masked(context, node_mask)
                else:
                    context = None

                # transform batch through flow
                nll, _, _ = losses.compute_loss_and_nll(args, flow_dp, nodes_dist, x, h, node_mask,
                                                        edge_mask, context)
                # standard nll from forward KL

                nll_epoch += nll.item() * batch_size
                n_samples += batch_size
                if i % args.n_report_steps == 0:
                    print(f"\r {partition} NLL \t, iter: {i}/{len(loader)}, "
                          f"NLL: {nll_epoch/n_samples:.2f}")

    return nll_epoch/n_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="./outputs/debug_10",
                        help='Specify model path')
    parser.add_argument('--n_samples', type=int, default=500,
                        help='Specify model path')
    parser.add_argument('--batch_size_gen', type=int, default=500,
                        help='Specify model path')
    parser.add_argument('--save_to_xyz', type=eval, default=False,
                        help='Should save samples to xyz files.')
    parser.add_argument('--ddp', type=bool, default=False,
                        help='number of atoms in molecule for gif animation')
    parser.add_argument('--steps', type=int, default=1000,
                        help='number of atoms in molecule for gif animation')
    parser.add_argument('--schedular', type=str, default="DDPM",
                        help='use ddim to sample')
    parser.add_argument('--model_name', type=str, default="generative_model_ema.npy",
                        help='the model name')
    parser.add_argument('--NLL', type=bool, default=False,
                        help='whether to eval NLL')
    eval_args, unparsed_args = parser.parse_known_args()

    assert eval_args.model_path is not None

    with open(join(eval_args.model_path, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)

    # CAREFUL with this -->
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = 1
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = 'sum'

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    dtype = torch.float32
    create_folders(args)
    print(args)
    args.datadir = "../../Model/EDM/qm9/temp"

    # Retrieve QM9 dataloaders
    dataloaders, charge_scale = dataset.retrieve_dataloaders(args)
    
    dataset_info = get_dataset_info(args.dataset, args.remove_h)

    # Load model
    generative_model, nodes_dist, prop_dist = get_model(args, device, dataset_info, dataloaders['train'])
    if prop_dist is not None:
        property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
        prop_dist.set_normalizer(property_norms)
    generative_model.to(device)

    fn = eval_args.model_name

    print(join(eval_args.model_path, fn))
    flow_state_dict = torch.load(join(eval_args.model_path, fn), map_location=device)
    # checkpoint = torch.load(join(eval_args.model_path, fn), map_location=device, weights_only=False)
   
    # new_state_dict = {}
    # for key, value in flow_state_dict.items():
    #     new_key = key.replace("module.", "")  # 去掉 "module." 前缀
    #     new_state_dict[new_key] = value
    generative_model.load_state_dict(flow_state_dict)
    # if not eval_args.ddp:
    #     # Remove 'model.' prefix from state dict keys
    #     new_state_dict = {}
    #     for key, value in checkpoint['model_state_dict'].items():
    #         if key.startswith('model.'):
    #             new_key = key[6:]  # Remove 'model.' prefix
    #         else:
    #             new_key = key
    #         new_state_dict[new_key] = value
    #     generative_model.load_state_dict(new_state_dict)
    # else:
    #     new_state_dict = {}
    #     for key, value in checkpoint['model_state_dict'].items():
    #         if key.startswith('model.'):
    #             new_key = key[6:]  # Remove 'model.' prefix
    #         else:
    #             new_key = key
    #         new_key = new_key.replace("module.", "")  # Remove "module." prefix if present
    #         new_state_dict[new_key] = value
    #     generative_model.load_state_dict(new_state_dict)
    # Analyze stability, validity, uniqueness and novelty
    stability_dict, rdkit_metrics = analyze_and_save(
        args, eval_args, device, generative_model, nodes_dist,
        prop_dist, dataset_info, n_samples=eval_args.n_samples,
        batch_size=eval_args.batch_size_gen, save_to_xyz=eval_args.save_to_xyz,timestep=eval_args.steps)
    print(stability_dict)

    if rdkit_metrics is not None:
        rdkit_metrics = rdkit_metrics[0]
        print("Validity %.4f, Uniqueness: %.4f, Novelty: %.4f" % (rdkit_metrics[0], rdkit_metrics[1], rdkit_metrics[2]))
    else:
        print("Install rdkit roolkit to obtain Validity, Uniqueness, Novelty")

    # In GEOM-Drugs the validation partition is named 'val', not 'valid'.
    if args.dataset == 'geom':
        val_name = 'val'
        num_passes = 1
    else:
        val_name = 'valid'
        num_passes = 5

    # Evaluate negative log-likelihood for the validation and test partitions
    if eval_args.NLL:
        val_nll = test(args, generative_model, nodes_dist, device, dtype,
                    dataloaders[val_name],
                    partition='Val')
        print(f'Final val nll {val_nll}')
        test_nll = test(args, generative_model, nodes_dist, device, dtype,
                        dataloaders['test'],
                        partition='Test', num_passes=num_passes)
        print(f'Final test nll {test_nll}')

        print(f'Overview: val nll {val_nll} test nll {test_nll}', stability_dict)
    else:
        val_nll = "None"
        test_nll = "None"
    with open(join(eval_args.model_path, 'eval_log.txt'), 'a') as f:
        print(f'Overview: val nll {val_nll} test nll {test_nll}',
              stability_dict, rdkit_metrics[0], rdkit_metrics[1], rdkit_metrics[2],
              file=f)
    data = {
    "Model Path": [eval_args.model_path],
    "Steps": [eval_args.steps],
    "Validation NLL": [val_nll],
    "Test NLL": [test_nll],
    "Mol Stability": [stability_dict["mol_stable"]],
    "Atom Stability": [stability_dict["atm_stable"]],
    "Validity": [rdkit_metrics[0]],
    "Uniqueness": [rdkit_metrics[1]],
    "Novelty": [rdkit_metrics[2]],
    "Model Name": [eval_args.model_name]
}
    df_new = pd.DataFrame(data)
    file_name = "result.csv"
    file_path = os.path.join(os.getcwd(), file_name)
    if os.path.exists(file_path):
        df_new.to_csv(file_path, mode='a', index=False, header=False)
    else:
        df_new.to_csv(file_path, index=False)

    print(f"CSV file updated at: {file_path}")

if __name__ == "__main__":
    main()
