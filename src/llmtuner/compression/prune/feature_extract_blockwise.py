import os
import argparse
import numpy as np
import math
from scipy import optimize
import random
import ORM
import torch, time
import torch.nn as nn
from pulp import *
from utils.pytorch_utils import DFS_bit
import pulp
import transformers
import sys
from data_args_util import ModelArguments, DataArguments, TrainingArguments, make_supervised_data_module
from torch.utils.data import Dataset, DataLoader
import pdb
from tqdm import tqdm
import logging
from sklearn.preprocessing import MaxAbsScaler
from data_args_util import z_score_fromaxis, z_score_whole
import scipy.io
def save_orthogonal_matrix(matrix, filename):
    row_names = [f'Layer_{i}' for i in range(32, 32)]
    column_names = [f'Layer_{i}' for i in range(32, 32)]
    scipy.io.savemat(filename, {'orthogonal_matrix': matrix})
def save_list_to_jsonl(data, filename):
    with open(filename, 'w') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, help='mixtral model')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model_max_length', type=int, default=1024)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--logging_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--per_device_train_batch_size', type=int, default=20, help='Number of calibration samples.')
    parser.add_argument('--max_train_samples', type=int)
    parser.add_argument('--bond_restrict', type=str, default="0.45-0.55")
    parser.add_argument('--first_last_bond', type=float, default=None, choices=[0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument('--max_optimize_item', type=int, default=None)
    parser.add_argument('--first_last_init', type=float, default=None)
    parser.add_argument('--target_model_param', type=float, default=None)
    parser.add_argument('--feature_from', type=str, default="out", choices=["out"])
    parser.add_argument('--distance_metric', type=str, default="orm", choices=["orm","arccos","arccos_final","cos","cos_final","emd_1d","emd_2d"])
    parser.add_argument('--emd_cost_metric', type=str, default=None, choices=['None',"cosine","sqeuclidean"])
    parser.add_argument('--optimizer_method', type=str, default="SLSQP", help='choice SLSQP or trust-constr')
    parser.add_argument('--param_stand', type=str, default=None, choices=["None", "weight_standardization","z_score_dim"])
    parser.add_argument('--orm_matrix_stand', type=str, default=None, choices=["None", "orm_std"])
    parser.add_argument('--layer_sparity_gap', type=float, default=None, help='The gap between each layer')
    parser.add_argument('--manual_config', type=str, default=None, help='manual config some layers')
    parser.add_argument('--theta_select', type=str, default=None, help='theta choice')
    args = parser.parse_args()
    model_args, data_args, training_args = args, args, args

    torch.cuda.set_device(0)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    print(args.logging_dir)
    base, ext = os.path.splitext(args.output_file)
    log_filename = f"{base}.log"
    print(log_filename)
    full_log_path = os.path.join(args.logging_dir, log_filename)
    logging.basicConfig(filename=full_log_path, filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
    # 创建一个流处理器，将日志输出到标准输出
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # 设置流处理器的日志级别
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # 获取默认的日志器并添加流处理器
    logger = logging.getLogger()
    logger.addHandler(console_handler)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir="llm_weights",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    
    if args.param_stand=='weight_standardization':
        for name, param in model.named_parameters():
            if 'weight' in name:
                param.data.copy_(z_score_whole(param))
    elif args.param_stand=='z_score_dim':
        for name, param in model.named_parameters():
            if 'weight' in name:
                # print(f'param is {param.shape}')
                param.data.copy_(z_score_fromaxis(param, 1))
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    device = torch.device("cuda:0")
    if "Mixtral" in args.model_name_or_path: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    logger.info(f"use device {device}")
    tokenizer.pad_token = tokenizer.eos_token
    output_dict = {}
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    data_loader = DataLoader(
                data_module['train_dataset'], 
                shuffle=True, 
                collate_fn=data_module['data_collator'], 
                batch_size=training_args.per_device_train_batch_size,
                drop_last=True,
            )
    batch_x=[]
    data = next(iter(data_loader))
    outer_bar = tqdm(total=training_args.per_device_train_batch_size, desc='k in batch size', position=0,leave=True)

    experts_chosen = np.zeros((len(model.model.layers),model.config.num_local_experts))
    for k in range(training_args.per_device_train_batch_size):
        data_input = data["input_ids"][k].unsqueeze(0).to(device)
        data_mask = data["attention_mask"][k].unsqueeze(0).to(device)
        n = data_input.size()[1]
        
        if args.feature_from=='out':
            feature=[] # [1, 1024, 4096]\

            hook_handles = []
            layer_index=0
            def expect_hook(module, input, output):
                global layer_index
                feature.append(output)
                argmax_indexs = np.argsort(output[:data_mask.sum()].float().cpu().numpy(), axis=1)[:, -2:]
                unique_elements, counts = np.unique(argmax_indexs.flatten(), return_counts=True)
                for expect_index in range(8):
                    experts_chosen[layer_index][expect_index] = dict(zip(unique_elements, counts)).get(expect_index, 0)
                layer_index+=1
                

            layers = model.model.layers

            
            for layer_i in range(len(layers)):
                handle = layers[layer_i].block_sparse_moe.gate.register_forward_hook(expect_hook)
                hook_handles.append(handle)
                


            with torch.no_grad():
                res = model(input_ids=data_input, attention_mask=data_mask, output_hidden_states=True, return_dict=True)

            

            feature = list(res.hidden_states)
            feature.pop(0)
            
            for handle in hook_handles:
                handle.remove()

        else:
            assert False, "args.feature_from not equal 'out'"

        for i in range(len(feature)):
            feature[i] = feature[i].view(n, -1)
            # feature[i] = torch.mean(feature[i],dim=1)
            feature[i] = feature[i].data.cpu().float().numpy()
        orthogonal_matrix = np.zeros((len(feature), len(feature)))
        
        # len(feature)为F集合中的数量

        # intermediate_results = []
        # file_path='/workspace/Layerwise_lightweight/mixed_bit/exp7/intermediate_results.json'

        middle_bar = tqdm(total=len(feature), desc='i in orthogonal_matrix', position=1,leave=True)
        for i in range(len(feature)):
            inner_bar = tqdm(total=len(feature), desc='j in  orthogonal_matrix', position=2,leave=True)
            for j in range(len(feature)):
                with torch.no_grad():
                    if args.distance_metric=='orm':
                        # 对特征向量进行 Z-score 归一化
                        if args.param_stand=='z_score_dim' or args.param_stand=='weight_standardization':
                            feature_i_normalized = (feature[i] - np.mean(feature[i])) / np.std(feature[i])
                            feature_j_normalized = (feature[j] - np.mean(feature[j])) / np.std(feature[j])
                        else:
                            feature_i_normalized = feature[i]
                            feature_j_normalized = feature[j]
                        orthogonal_matrix[i][j] = ORM.orm(ORM.gram_linear(feature_i_normalized), ORM.gram_linear(feature_j_normalized))

                    elif args.distance_metric=='arccos_final':
                        feature_i_normalized = (feature[i] - np.mean(feature[i])) / np.std(feature[i])
                        feature_j_normalized = (feature[j] - np.mean(feature[j])) / np.std(feature[j])
                        orthogonal_matrix[i][j]=ORM.arccos_distance(feature_i_normalized[-1,:].reshape(1, -1) , feature_j_normalized[-1,:].reshape(1, -1))
                    elif args.distance_metric=='arccos':
                        feature_i_normalized = (feature[i] - np.mean(feature[i])) / np.std(feature[i])
                        feature_j_normalized = (feature[j] - np.mean(feature[j])) / np.std(feature[j])
                        orthogonal_matrix[i][j]=ORM.arccos_distance_matrix(feature_i_normalized , feature_j_normalized)
                    elif args.distance_metric=='cos_final':
                        feature_i_normalized = (feature[i] - np.mean(feature[i])) / np.std(feature[i])
                        feature_j_normalized = (feature[j] - np.mean(feature[j])) / np.std(feature[j])
                        orthogonal_matrix[i][j]=ORM.cosine_distance(feature_i_normalized[-1,:].reshape(1, -1) , feature_j_normalized[-1,:].reshape(1, -1))
                    elif args.distance_metric=='cos':
                        feature_i_normalized = (feature[i] - np.mean(feature[i])) / np.std(feature[i])
                        feature_j_normalized = (feature[j] - np.mean(feature[j])) / np.std(feature[j])
                        orthogonal_matrix[i][j]=ORM.overall_cosine_similarity(feature_i_normalized, feature_j_normalized)
                    inner_bar.update(1)
            middle_bar.update(1)
            inner_bar.close()
        middle_bar.close()
        # 将结果列表写入 JSON 文件
        # with open(file_path, 'w') as f:
            # json.dump(intermediate_results, f, indent=2)

        # print("Intermediate results saved successfully.")
        # os._exit(0)
        if args.orm_matrix_stand=="orm_std":
            orthogonal_matrix = z_score_whole(torch.from_numpy(orthogonal_matrix)).numpy()

        

        if args.distance_metric=='emd_1d':
            # 创建 MinMaxScaler 对象,默认范围为 [0, 1]
            min_matric, max_matric = np.min(orthogonal_matrix), np.max(orthogonal_matrix)
            logger.info(f'the max in orthogonal_matrix is {min_matric}')
            logger.info(f'the min in orthogonal_matrix is {max_matric}')
            scaler = MinMaxScaler()
            # 对 EMD 矩阵进行最小-最大归一化
            orthogonal_matrix = scaler.fit_transform(orthogonal_matrix)
            min_matric, max_matric = np.min(orthogonal_matrix), np.max(orthogonal_matrix)
            logger.info(f'the max in orthogonal_matrix is {min_matric}')
            logger.info(f'the min in orthogonal_matrix is {max_matric}')
        # 构建文件路径和文件名
        ortho_file_path = f"{args.output_dir}/orthogonal_matrix.mat"
        reverse_ortho_file_path = f"{args.output_dir}/reverse_orthogonal_matrix.mat"
        save_orthogonal_matrix(orthogonal_matrix, ortho_file_path)
        orthogonal_matrix=1-orthogonal_matrix
        save_orthogonal_matrix(orthogonal_matrix, reverse_ortho_file_path)
        theta = []  # θi
        gamma = []  # γi
        flops = []
        

        for i in range(len(feature)):
            gamma.append(sum(orthogonal_matrix[i])-orthogonal_matrix[i][i]) # sum(orthogonal_matrix[i]-orthogonal_matrix[i][i])
        gamma_min = min(gamma)
        epsilon = gamma_min * 0.01
        # beta=1.5
        beta = 1 
        # e^-x
        for i in range(len(feature)):
            if args.theta_select=='Exp':
                theta.append(1 * math.exp(-1* beta *gamma[i]))
            elif args.theta_select=='gaussian':
                theta.append(math.exp(-beta*gamma[i]**2))
            if args.theta_select == 'PowerLaw':
                theta.append(1 / ((1 + gamma[i]) ** beta))
            elif args.theta_select == 'Inverse':
                theta.append(beta / (gamma[i] + epsilon))
            elif args.theta_select == 'LogNeg':
                theta.append(-math.log(1 + gamma[i], beta))
            elif args.theta_select == 'Arctan':
                theta.append(math.atan(beta / (gamma[i] + epsilon)))
            elif args.theta_select == 'Sigmoid':
                theta.append(beta / (1 + math.exp(gamma[i])))

        
        # # layerwise # 计算Flops
        # params, first_last_size = net.cfg2params_perlayer(net.cfg, length, args.quant_type)
        # FLOPs, first_last_flops = net.cfg2flops_layerwise(net.cfg, length, args.quant_type)
        
        def get_named_parameters(model):
            # 获取模型各层的参数及名称
            named_params = list(model.named_parameters())
            layer_param_dict = {}
            for name, param in named_params:
                layer_param_dict[name] = param.numel()
            
            part=["self_attn.q_proj.weight","self_attn.k_proj.weight","self_attn.v_proj.weight","self_attn.o_proj.weight",
                "block_sparse_moe.gate.weight","block_sparse_moe.experts.*.w1.weight","block_sparse_moe.experts.*.w2.weight",
                "block_sparse_moe.experts.*.w3.weight","input_layernorm.weight","post_attention_layernorm.weight"
                ]
            part2=["self_attn.q_proj.weight","self_attn.k_proj.weight","self_attn.v_proj.weight","self_attn.o_proj.weight",
                "block_sparse_moe.gate.weight","block_sparse_moe.experts.*.w1.weight","block_sparse_moe.experts.*.w2.weight",
                "block_sparse_moe.experts.*.w3.weight"]
            linear_para_dict={}
            layer_paramater=[]
            for i in range(len(feature)):
                layer_paramaters_number=0
                patten = "model.layers."+str(i)+"."
                
                for p in part:
                    for fn_name in layer_param_dict.keys():
                        if re.match(p, fn_name): 
                            layer_paramaters_number+=layer_param_dict[patten+fn_name]
                            pdb.set_trace()
                            if p in part2:
                                linear_para_dict[p] = layer_param_dict[patten+fn_name]
                            del layer_param_dict[patten+fn_name]
                layer_paramater.append(layer_paramaters_number)

            return layer_param_dict, layer_paramater, linear_para_dict

        layer_param_dict, ori_params, linear_para_dict = get_named_parameters(model)
        
        ori_first_last_size = sum(layer_param_dict.values())
        total_ori_paras_number = sum(ori_params)+ori_first_last_size
        logger.info(f"ori_params_number: {total_ori_paras_number}")

        params = [i/(10**8) for i in ori_params]
        first_last_size = ori_first_last_size/(10**8)
        length = len(feature)
        model_size = args.target_model_param* 10
        logger.info(f"target_model_size:{model_size}")
        part=["self_attn.q_proj.weight","self_attn.k_proj.weight","self_attn.v_proj.weight","self_attn.o_proj.weight",
                "mlp.gate_proj.weight","mlp.up_proj.weight","mlp.down_proj.weight"]
        if args.manual_config:
            if args.manual_config=="config1":
                # 假设第一层和最后一层设定为0.9的保留率
                first_block_linear_para_dict = linear_para_dict
                params.pop(0)
                first_p = [0.9, 0.8, 0.7, 0.6, 0.7, 0.8, 0.9]
                for bili,w_n in zip(first_p, part):
                    first_block_linear_para_dict[w_n] = bili*first_block_linear_para_dict[w_n]
                first_total = sum(first_block_linear_para_dict.values())/(10**8)
                gamma.pop(0)
                theta.pop(0)
                model_size -= first_total
                length -= 1

                last_block_linear_para_dict = linear_para_dict
                params.pop(-1)
                last_p = [0.9, 0.8, 0.7, 0.6, 0.7, 0.8, 0.9]
                for bili,w_n in zip(first_p, part):
                    last_block_linear_para_dict[w_n] = bili*last_block_linear_para_dict[w_n]
                last_total = sum(last_block_linear_para_dict.values())/(10**8)
                gamma.pop(-1)
                theta.pop(-1)
                model_size -= last_total
                length -= 1
   

        theta = np.array(theta)
        theta = np.negative(theta)

        # Objective function
        def func(x, sign=1.0):
            """ Objective function """
            global theta, length
            sum_fuc =[]
            for i in range(length):
                temp = 0.
                for j in range(i, length):
                    temp += theta[j]
                sum_fuc.append(x[i] * (sign * temp / (length-i)))
            return sum(sum_fuc)

        # Derivative function of objective function
        def func_deriv(x, sign=1.0):
            """ Derivative of objective function """
            global theta, length
            diff = []
            for i in range(length):
                temp1 = 0.
                for j in range(i, length):
                    temp1 += theta[j]
                diff.append(sign * temp1 / (length - i))

            return np.array(diff)
        

        # Constraint function
        def constrain_func(x):
            """ constrain function """
            global params, length
            a = []
            try:
                for i in range(length):
                    a.append(x[i] * params[i])
            except:
                pdb.set_trace()
            return np.array([model_size - first_last_size - sum(a)])

        switch = {
            '0.45-0.55': (0.45, 0.55)
        }
        bond = switch.get(args.bond_restrict, None)
        assert bond!=None, "bond_restrict input wrong"
        
        bnds = []
        for i in range(length):
            bnds.append(bond)
        bnds[0] = (0.6, 1.0)
        bnds[-1] = (0.6, 1.0)

        if args.first_last_bond:
            bnds[0] = (args.first_last_bond, 1.0)
            bnds[-1] = (args.first_last_bond, 1.0)
        
        if args.layer_sparity_gap is not None:
            max_bond = float(args.bond_restrict.split('-')[1])
            min_bond = float(args.bond_restrict.split('-')[0])
            for i in range(1, length - 1):
                layer_bond = max_bond - (i - 1) * args.layer_sparity_gap
                layer_bond = max(layer_bond, min_bond)
                bnds[i] = (layer_bond, max_bond)
        print(bnds)
        bnds = tuple(bnds)

        cons = ({'type': 'ineq',
                'fun': constrain_func}
                )
        
        options=None
        if args.max_optimize_item:
            options={'maxiter': args.max_optimize_item, 'disp': True}
        
        x_init=[1 for i in range(length)]
        if args.first_last_init:
            index=0
            while index<=(length//2):
                x_init[index]=round(args.first_last_init,2)
                x_init[-index-1]=round(args.first_last_init,2)
                args.first_last_init-=0.1
                if args.first_last_init<0.5:
                    args.first_last_init=0.5
                index+=1
            
        result = optimize.minimize(func,x0=x_init, jac=func_deriv, method=args.optimizer_method, bounds=bnds, constraints=cons, options=options, tol=1e-4)
        logger.info(result)
        logger.info(result.x)
        batch_x.append(result.x)
        if args.manual_config:
            ori_params.pop(0)
            ori_params.pop(-1)
            total_paras_number = ori_first_last_size + sum(ori_params * result.x) + first_total*(10**8) +last_total*(10**8)
        else:
            total_paras_number = ori_first_last_size + sum(ori_params * result.x)
        logger.info(f"after_params_number:{total_paras_number}")

        outer_bar.update(1)
    outer_bar.close()
    logger.info("-------------------------------------------------------------------------")
    x = torch.mean(torch.stack([torch.tensor(b) for b in batch_x],dim=0),dim=0).tolist()
    parts=["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]
    if args.manual_config:
        temp={}
        for bili, w_n in zip(first_p, parts):
            temp[w_n] = bili
        linear_x=[temp]
    else:
        linear_x=[]
    
    idx=0
    for item in x:
        linear_pro = {}
        for part in parts:
            linear_pro[part] = item
        linear_x.append(linear_pro)


    if args.manual_config:
        temp={}
        for bili, w_n in zip(last_p, parts):
            temp[w_n] = bili
        linear_x.append(temp)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    save_list_to_jsonl(linear_x, args.output_dir+'/'+args.output_file)
    
    



