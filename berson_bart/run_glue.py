# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import random
from random import shuffle
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange


from transformers_step import glue_compute_metrics as compute_metrics
from transformers_step import glue_output_modes as output_modes
from transformers_step import glue_processors as processors
from transformers_step import glue_convert_examples_to_features as convert_examples_to_features

from preprocess_batch import preprocess
import torch.nn as nn

from transformers_step import AutoModelForsentenceordering, AutoTokenizer
from transformers_step import AutoModelForsentenceordering_student
from transformers_step import AdamW, get_linear_schedule_with_warmup
from transformers_step.modeling_bart_student import beam_search_pointer

import torch.nn.functional as F

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def init_model(TS, model_name: str, device, do_lower_case: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)
    if TS == "T":
        model = AutoModelForsentenceordering.from_pretrained(model_name)
    else:
        model = AutoModelForsentenceordering_student.from_pretrained(model_name)

    model.to(device)
    model.eval()
    return tokenizer, model


def train(args, train_dataset, model, tokenizer, model_T, train_num=25):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir=args.output_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    logger.info('***************** before load *****************')
    # last_file_name = './processed_fea/bart_nopadding_cached_test_bart_large_aan_mask'
    last_file_name = '../A_tokentype_mask/processed_fea/bart_nopadding_cached_train_bart_large_aan_mask_' + str(
        train_num - 1)

    features = torch.load(last_file_name)
    train_dataset = features
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=preprocess)
    logger.info('***************** after load *****************')

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = (len(train_dataloader)+20000*(train_num-1)) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    criterion = nn.NLLLoss(reduction='none')
    model.equip(criterion)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0

    model_to_resize = model.module if hasattr(model, "module") else model
    model_to_resize.resize_token_embeddings(len(tokenizer))

    model.zero_grad()

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)

    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
        results = evaluate(args, model, tokenizer)

    best_t = 0
    count_es = 0
    epoch = -1
    for _ in train_iterator:
        epoch += 1

        for num in range(train_num):
            logger.info('***************** before load %s *****************', str(num))
            cache_file_name = '../A_tokentype_mask/processed_fea/bart_nopadding_cached_train_bart_large_aan_mask_' + str(
                num)

            if os.path.exists(cache_file_name):
                logger.info("Loading features from cached file %s", cache_file_name)
                features = torch.load(cache_file_name)
            train_dataset = features
            train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                          collate_fn=preprocess)
            logger.info('***************** after load %s *****************', str(num))

            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
            for step, batch in enumerate(epoch_iterator):
                model.train()
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {'input_ids':           batch[0],
                           'attention_mask':     batch[1],
                           'token_type_ids':     batch[2],
                           'pairs_list':         batch[3],
                           'passage_length':     batch[4],
                           "pairs_num":          batch[5],
                           "sep_positions":      batch[6],
                           "ground_truth":       batch[7],
                           "mask_cls":           batch[8],
                           "pairwise_labels":    batch[9],
                           "sentence_input_id":  batch[11],
                           "sentence_attention_mask": batch[12],
                           "sentence_length":    batch[13],
                           "para_input_id":       batch[14],
                           "para_attention_mask": batch[15],
                           "max_sentence_length": batch[16],
                           "imgs":                batch[17],
                           "sent_len_for_mask":   batch[18],
                           "mm_mask":             batch[19],
                           "cuda":            args.cuda_ip}

                ce_loss, match_layer_hidden_stu, cls_pooled_output_stu, pair_masks_new, \
                pair_senvec_top_stu, document_matrix_stu, \
                cls_score_stu, stu_att_score_all = model(**inputs)

                loss = ce_loss

                if args.n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(epoch_iterator) <= args.gradient_accumulation_steps and (step + 1) == len(epoch_iterator)):
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1

                    logging_steps = args.logging_steps

                    if args.local_rank in [-1, 0] and logging_steps > 0 and global_step % logging_steps == 0:
                        # Log metrics
                        if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                            results = evaluate(args, model, tokenizer)
                            for key, value in results.items():
                                tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                        tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar('loss', (tr_loss - logging_loss)/logging_steps, global_step)
                        logging_loss = tr_loss

                        taus = results['taus']
                        if taus > best_t:
                            count_es = 0
                            best_t = taus
                            output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = model.module if hasattr(model,
                                                                    'module') else model  # Take care of distributed/parallel training
                            model_to_save.save_pretrained(output_dir)
                            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                            tokenizer.save_pretrained(output_dir)
                            logger.info("Saving model checkpoint to %s", output_dir)
                        else:
                            count_es += 1

                if args.max_steps > 0 and global_step > args.max_steps:
                    epoch_iterator.close()
                    break

            if args.max_steps > 0 and global_step > args.max_steps:
                train_iterator.close()
                break

        output_dir = os.path.join(args.output_dir, 'checkpoint-{}epoch'.format(epoch))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
        tokenizer.save_pretrained(output_dir)
        logger.info("Saving model checkpoint to %s", output_dir)

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=preprocess)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)

        f = open(os.path.join(args.output_dir, "output_order.txt"), 'w')

        best_acc = []
        truth = []
        predicted = []
      
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            tru = batch[7].view(-1).tolist()  # true order
            true_num = batch[4].view(-1)
            tru = tru[:true_num]
            truth.append(tru)

            with torch.no_grad():

                if len(tru) == 1:
                    pred = tru
                else:
                    pred = beam_search_pointer(args, model, input_ids=batch[0], attention_mask=batch[1], token_type_ids=batch[2],
                        pairs_list=batch[3], passage_length=batch[4], pairs_num=batch[5], sep_positions=batch[6], 
                        ground_truth=batch[7], mask_cls=batch[8], pairwise_labels=batch[9],
                        sentence_input_id=batch[11], sentence_attention_mask=batch[12], sentence_length=batch[13],
                        para_input_id=batch[14], para_attention_mask=batch[15], max_sentence_length=batch[16],
                        imgs=batch[17], sent_len_for_mask=batch[18], mm_mask=batch[19],
                        cuda=args.cuda_ip)

                predicted.append(pred)
                print('{}|||{}'.format(' '.join(map(str, pred)), ' '.join(map(str, truth[-1]))),
                      file=f)                

        right, total = 0, 0
        pmr_right = 0
        taus = []
        accs = []
        pm_p, pm_r = [], []
        import itertools
        from sklearn.metrics import accuracy_score

        for t, p in zip(truth, predicted):
            if len(p) == 1:
                right += 1
                total += 1
                pmr_right += 1
                taus.append(1)
                continue
            eq = np.equal(t, p)
            right += eq.sum()
            accs.append(eq.sum()/len(t))
            total += len(t)
            pmr_right += eq.all()
            s_t = set([i for i in itertools.combinations(t, 2)])
            s_p = set([i for i in itertools.combinations(p, 2)])
            pm_p.append(len(s_t.intersection(s_p)) / len(s_p))
            pm_r.append(len(s_t.intersection(s_p)) / len(s_t))
            cn_2 = len(p) * (len(p) - 1) / 2
            pairs = len(s_p) - len(s_p.intersection(s_t))
            tau = 1 - 2 * pairs / cn_2
            taus.append(tau)

        acc = accuracy_score(list(itertools.chain.from_iterable(truth)),
                             list(itertools.chain.from_iterable(predicted)))
        best_acc.append(acc)
        pmr = pmr_right / len(truth)
        taus = np.mean(taus)
        pm_p = np.mean(pm_p)
        pm_r = np.mean(pm_r)
        pm = 2 * pm_p * pm_r / (pm_p + pm_r)
        f.close()
        accs = np.mean(accs)

        results['acc'] = accs
        results['pmr'] = pmr
        results['taus'] = taus
        results['pm'] = pm

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            # logger.info('output_eval_file', str(output_eval_file))
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(results.keys()):
                logger.info("  %s = %s", key, str(results[key]))
                writer.write("%s = %s\n" % (key, str(results[key])))

        output_only_eval_file_1 = os.path.join(args.output_dir, "all_eval_results.txt")
        fh = open(output_only_eval_file_1, 'a')
        fh.write(prefix)
        for key in sorted(results.keys()):
            fh.write("%s = %s\n" % (key, str(results[key])))
        fh.close()

    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.fea_data_dir, 'bart_nopadding_cached_{}_{}_{}_mask'.format(
        'val' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(task)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1] 
        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                cached_features_file,
                                                evaluate=evaluate,
                                                label_list=label_list,
                                                max_length=args.max_seq_length,
                                                output_mode=output_mode,
                                                pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
                                                pad_token=1,
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
        )
        print('features', len(features))
    dataset = features
    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    parser.add_argument("--fea_data_dir", default=None, type=str, required=True,
                        help="features data")

    parser.add_argument("--model_type", default=None, type=str, required=True)
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True)
    parser.add_argument("--teacher_name_or_path", default=None, type=str, required=True)
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument("--cuda_ip", default="cuda:0", type=str,
                        help="Total number of training epochs to perform.")

    #### paragraph encoder ####
    parser.add_argument("--ff_size", default=512, type=int)
    parser.add_argument("--heads", default=4, type=int)
    parser.add_argument("--inter_layers", default=2, type=int) 
    parser.add_argument("--para_dropout", default=0.1, type=float,
                        help="Total number of training epochs to perform.")

    #### pointer ###
    parser.add_argument("--beam_size", default=64, type=int)

    #### pairwise loss ###
    parser.add_argument("--pairwise_loss_lam", default=0.1, type=float,help="Total number of training epochs to perform.")

    #### transformer decoder ###
    parser.add_argument("--decoder_layer", default=2, type=int) 
    parser.add_argument("--dec_heads", default=8, type=int)

    #### Distillation ###
    parser.add_argument("--temperature", default=2.0, type=float,
                        help="Distillation temperature. Only for distillation.")
    parser.add_argument("--alpha_mseloss", default=100.0, type=float,
                        help="Distillation loss linear weight. Only for distillation.")
    parser.add_argument("--alpha_senvec_pair_mseloss", default=200.0, type=float,
                        help="Distillation loss linear weight. Only for distillation.")
    parser.add_argument("--alpha_cls_mseloss", default=200.0, type=float,
                        help="Distillation loss linear weight. Only for distillation.")
    parser.add_argument("--alpha_docmat_mseloss", default=100.0, type=float,
                        help="Distillation loss linear weight. Only for distillation.")
    parser.add_argument("--alpha_clskl_loss", default=1.0, type=float,
                        help="Distillation loss linear weight. Only for distillation.")
    parser.add_argument("--alpha_attkl_loss", default=1.0, type=float,
                        help="Distillation loss linear weight. Only for distillation.")
    parser.add_argument("--sen_layer_num", default=9, type=int)


    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(args.cuda_ip if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
        args.n_gpu = 1
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()

    TS = 'S'
    tokenizer, model = init_model(
        TS, args.model_name_or_path, device=args.device, do_lower_case=args.do_lower_case)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # Training
    if args.do_train:
        # train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        train_dataset = None
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, model_T)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

if __name__ == "__main__":
    main()
