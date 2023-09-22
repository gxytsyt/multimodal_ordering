import torch
import numpy as np

def preprocess(batch):

    sen_num_dataset = []
    sample_len_dataset = []
    pairs_num_dataset = []

    sentence_length_dataset = []
    paragraph_length_dataset = []


    imgs_dataset = []

    for example in batch:
        sample_len_dataset.append(example['max_sample_len'])
        sen_num_dataset.append(example['passage_length'])
        pairs_num_dataset.append(example['pairs_num'])

        sentence_length_dataset = sentence_length_dataset + example['sentence_length']

        paragraph_length_dataset.append(example['para_length'])

        imgs_dataset.append(example['imgs'].shape[0])

    max_pair_len = max(sample_len_dataset)
    max_sen_num = max(sen_num_dataset)
    max_pairs_num = max(pairs_num_dataset)

    # sent
    max_sentence_length = max(sentence_length_dataset)

    # paragraph
    max_para_length = max(paragraph_length_dataset)

    # img_num
    max_imgs_num = max(imgs_dataset)


    all_input_ids=[]
    all_attention_mask=[]
    all_token_type_ids=[]
    all_pairs_list=[]
    all_passage_length=[]
    all_pairs_num=[]
    all_sep_positions=[]
    all_ground_truth=[]
    all_mask_cls=[]
    all_pairwise_labels = []

    all_shuffled_index = []


    #### sentence bert section ###
    all_sentence_input_id = []
    all_sentence_attention_mask = []
    all_sentence_length = []


    #### para bart section ###
    para_input_id_new = []
    para_attention_mask_new = []


    #### img ####
    imgs_new = []


    #### mask ####
    sent_len_for_mask_all = []


    for inputs in batch:  # padding for each example

        input_ids, masked_ids, token_type_ids, sep_positions = inputs['input_ids'], inputs['masked_ids'], inputs['token_type_ids'], inputs['sep_positions']
        shuffled_index, max_sample_len, ground_truth = inputs['shuffled_index'], inputs['max_sample_len'], inputs['ground_truth']
        passage_length, pairs_num, pairs_list = inputs['passage_length'], inputs['pairs_num'], inputs['pairs_list']
        pairwise_labels = inputs['pairwise_labels']

        padd_num_sen = max_sen_num - passage_length
        padding_pair_num = max_pairs_num - pairs_num 
        pad_id = 0
        pad_id_bart_token = 1

        input_ids_new = []
        masked_ids_new = []
        token_type_ids_new = []
        pairwise_label_new = []

        for item in range(pairs_num): 
            padding_pair_len = max_pair_len - len(input_ids[item])

            input_ids_new.append(input_ids[item] + [pad_id_bart_token] * padding_pair_len)
            masked_ids_new.append(masked_ids[item] + [pad_id] * padding_pair_len)
            token_type_ids_new.append(token_type_ids[item] + [pad_id] * padding_pair_len)

        ### padding for padded pairs
        input_ids_new = input_ids_new + [[pad_id_bart_token] * max_pair_len] * padding_pair_num
        masked_ids_new = masked_ids_new + [[pad_id] * max_pair_len] * padding_pair_num   
        token_type_ids_new = token_type_ids_new + [[pad_id] * max_pair_len] * padding_pair_num
        pairwise_labels_new = pairwise_labels + [0] * padding_pair_num 


        pairs_list_new = pairs_list + [[0,1]] * padding_pair_num
        passage_length_new = passage_length
        pairs_num_new = pairs_num
        sep_positions_new = sep_positions + [[2,6]] * padding_pair_num

        mask_cls_new = [1] * passage_length_new + [pad_id] * padd_num_sen 
        ground_truth_new = ground_truth + [pad_id] * padd_num_sen

        shuffled_index_new = shuffled_index + [pad_id] * padd_num_sen

        ################################## sentence bert section ########################################
        sentence_input_id, sentence_attention_mask, sentence_length \
            = inputs['sentence_input_id'], inputs['sentence_attention_mask'], inputs['sentence_length']

        sentence_input_id_new = []
        sentence_attention_mask_new = []

        for item in range(passage_length):
            padding_single_sen_len = max_sentence_length - sentence_length[item]

            sentence_input_id_new.append(sentence_input_id[item] + [pad_id_bart_token] * padding_single_sen_len)
            sentence_attention_mask_new.append(sentence_attention_mask[item] + [pad_id] * padding_single_sen_len)

        sentence_input_id_new = sentence_input_id_new + padd_num_sen * [[pad_id_bart_token] * max_sentence_length]
        sentence_attention_mask_new = sentence_attention_mask_new + padd_num_sen * [[pad_id] * max_sentence_length]

        sentence_length_new = sentence_length + [pad_id] * padd_num_sen
        ################################### sentence bert section ######################################


        all_input_ids.append(input_ids_new)
        all_attention_mask.append(masked_ids_new)
        all_token_type_ids.append(token_type_ids_new)
        all_pairs_list.append(pairs_list_new)
        all_passage_length.append(passage_length_new)
        all_pairs_num.append(pairs_num_new)
        all_sep_positions.append(sep_positions_new)
        all_ground_truth.append(ground_truth_new)
        all_mask_cls.append(mask_cls_new)
        all_pairwise_labels.append(pairwise_labels_new)

        all_shuffled_index.append(shuffled_index_new)

        all_sentence_input_id.append(sentence_input_id_new)
        all_sentence_attention_mask.append(sentence_attention_mask_new)
        all_sentence_length.append(sentence_length_new)


        ################################## paragraph bart section ########################################

        para_input_id, para_attention_mask, para_length \
            = inputs['para_input_id'], inputs['para_attention_mask'], inputs['para_length']

        padding_single_para_len = max_para_length - para_length

        para_input_id_new.append(para_input_id + [pad_id_bart_token] * padding_single_para_len)

        para_attention_mask_new.append(para_attention_mask + [pad_id] * padding_single_para_len)

        ################################## paragraph bart section #########################################

        ################################## imgs section ########################################
        imgs = inputs['imgs']
        img_new = torch.zeros((max_imgs_num, 3, 224, 224))
        img_new[:imgs.shape[0], ...] = imgs

        imgs_new.append(img_new)

        ################################## imgs section ########################################

        ############################## mask section ###############################
        sent_len_for_mask = inputs['sent_len_for_mask']
        sent_len_for_mask_all.append(sent_len_for_mask)
        ############################## mask section ###############################


    imgs_new_all = torch.stack(imgs_new, dim=0)

    all_input_ids=torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask=torch.tensor(all_attention_mask, dtype=torch.long)
    all_token_type_ids=torch.tensor(all_token_type_ids, dtype=torch.long)
    all_pairs_list=torch.tensor(all_pairs_list, dtype=torch.long)
    all_passage_length=torch.tensor(all_passage_length, dtype=torch.long)
    all_pairs_num=torch.tensor(all_pairs_num, dtype=torch.long)
    all_sep_positions=torch.tensor(all_sep_positions, dtype=torch.long)
    all_ground_truth=torch.tensor(all_ground_truth, dtype=torch.long)
    all_mask_cls=torch.tensor(all_mask_cls, dtype=torch.long)
    all_pairwise_labels=torch.tensor(all_pairwise_labels, dtype=torch.long)

    all_shuffled_index = torch.tensor(all_shuffled_index, dtype=torch.long)

    all_sentence_input_id = torch.tensor(all_sentence_input_id, dtype=torch.long)
    all_sentence_attention_mask = torch.tensor(all_sentence_attention_mask, dtype=torch.long)
    all_sentence_length = torch.tensor(all_sentence_length, dtype=torch.long)

    max_sentence_length = torch.tensor(max_sentence_length, dtype=torch.int)

    para_input_id_new = torch.tensor(para_input_id_new, dtype=torch.long)
    para_attention_mask_new = torch.tensor(para_attention_mask_new, dtype=torch.long)

    imgs_new_all = torch.tensor(imgs_new_all, dtype=torch.long)

    sent_len_for_mask_all = torch.tensor(sent_len_for_mask_all, dtype=torch.long)

    ####### mm_mask ######

    img_num = [img_i.shape[0] for img_i in imgs_new]
    img_len = 49
    # print('img_num', img_num)  # [5, 5]
    pad_img_num = max(img_num)

    bs, paralen = para_attention_mask_new.size()

    mm_mask = torch.zeros((bs, paralen + pad_img_num * img_len + 1, paralen + pad_img_num * img_len + 1))
    for i in range(bs):
        mm_mask[i, :paralen, :paralen] = para_attention_mask_new[i]  # 所有文字部分可以互相看到
        mm_mask[i, paralen, paralen:] = 1
        mm_mask[i, paralen:, paralen] = 1  # 图像部分的cls可以看到所有图像

    for bs_i, len_list in enumerate(sent_len_for_mask_all):
        cur = 1
        cur_p = paralen + 1
        for l in len_list:
            for i in range(l):
                for j in range(img_len):
                    mm_mask[bs_i, cur + i, cur_p + j] = 1
                    mm_mask[bs_i, cur_p + j, cur + i] = 1
            cur = cur + l
            cur_p = cur_p + img_len

    new_batch=[all_input_ids, all_attention_mask, all_token_type_ids, all_pairs_list, all_passage_length,
               all_pairs_num, all_sep_positions, all_ground_truth, all_mask_cls, all_pairwise_labels,
               all_shuffled_index,
               all_sentence_input_id, all_sentence_attention_mask, all_sentence_length,
               para_input_id_new, para_attention_mask_new,
               max_sentence_length,
               imgs_new_all, sent_len_for_mask_all,
               mm_mask
               ]

    return new_batch





