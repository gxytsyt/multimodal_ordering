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

import csv
import sys
import copy
import json
import os
import glob
from tqdm import tqdm

class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    # def __init__(self, guid, text_a, text_b=None, label=None):
    def __init__(self, guid, text_seq, img_path_seq, label=None):
        self.guid = guid
        self.text_seq = text_seq
        self.img_path_seq = img_path_seq
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, position_ids, order, cls_ids, mask_cls, num_sen, span_index):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.position_ids = position_ids
        self.order = order
        self.cls_ids = cls_ids
        self.mask_cls = mask_cls
        self.num_sen = num_sen
        self.span_index=span_index

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """Gets an example from a dict with tensorflow tensors

        Args:
            tensor_dict: Keys and values should match the corresponding Glue
                tensorflow_dataset examples.
        """
        raise NotImplementedError()

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, data_dir, quotechar=None):
        """Reads a tab separated value file."""
        IMAGE_FIELD_NAMES = [
            "image-large",
            "image-src-1",
        ]

        max_story_length = 5

        json_path = os.path.join(data_dir, "wikihow-{}-".format('acl22') + quotechar + ".json")

        if quotechar == 'test':
            json_path = os.path.join(data_dir, "wikihow-{}-".format('acl22_human') + quotechar + ".json")
        if not os.path.exists(json_path):
            raise ValueError("File: {} not found!".format(json_path))

        line_cnt = 0
        json_file = open(json_path)
        data = []
        for line in json_file:
            d = json.loads(line.strip())
            line_cnt += 1
            data.append(d)

        story_seqs = []
        missing_images = []

        for data_raw in tqdm(data, total=len(data)):

            wikihow_url = data_raw["url"]
            title_text = data_raw["title"]
            summary_text = data_raw["summary"]
            # print(wikihow_url)

            for section_id in range(len(data_raw["sections"])):

                section_curr = data_raw["sections"][section_id]
                wikihow_page_id = "###".join([wikihow_url, title_text, str(section_id)])
                wikihow_page_id = "###".join([wikihow_url, str(section_id)])
                story_seq = [wikihow_page_id]

                # TODO: consistency of human test sets.
                include_data = True
                # if self.version_text is not None and self.version_text == "human_annot_only_filtered":
                #     include_data = False

                for step_id in range(len(section_curr["steps"])):
                    step_curr = section_curr["steps"][step_id]
                    step_headline = step_curr["step_headline"]
                    step_text = step_curr["step_text"]["text"]
                    bullet_points = step_curr["step_text"]["bullet_points"]

                    combined_text = " ".join([step_text] + bullet_points)

                    # print('step_id', step_id)
                    # print('combined_text', combined_text)

                    element = None
                    # if self.paired_with_image:
                    # We take the first image for each step.
                    image_path_curr = None
                    for image_field_key in IMAGE_FIELD_NAMES:
                        if image_field_key in step_curr["step_assets"]:
                            image_path_curr = step_curr["step_assets"][image_field_key]
                            image_path_curr_new = None
                            if image_path_curr is not None and len(image_path_curr) > 0:
                                image_path_curr = os.path.join(data_dir, image_path_curr)
                                if "wikihow.com" not in image_path_curr:
                                    image_path_curr_new = image_path_curr.replace(
                                        "/images/",
                                        "/www.wikihow.com/images/")
                                else:
                                    image_path_curr_new = image_path_curr
                                if not os.path.exists(image_path_curr_new):
                                    image_path_curr_new = image_path_curr.replace(
                                        "/images/",
                                        "/wikihow.com/images/")
                                    if not os.path.exists(image_path_curr_new):
                                        missing_images.append(wikihow_page_id + "###" + str(step_id))
                                        element = None
                                    else:
                                        element = (combined_text, image_path_curr_new)
                                else:
                                    element = (combined_text, image_path_curr_new)
                            else:
                                missing_images.append(wikihow_page_id + "###" + str(step_id))
                                element = None
                            if image_path_curr_new is not None and os.path.exists(image_path_curr_new):
                                break
                    # else:
                    #     element = (combined_text, None)

                    if element is not None:
                        story_seq.append(element)

                # story_seq.view()

                # TODO: Currently different sections are in different
                # sequences for sorting.
                min_story_length = 1
                # if len(story_seq) < self.min_story_length + 1:
                if len(story_seq) <= min_story_length + 1:
                    # print(story_seq)
                    # input('ssss')
                    pass
                elif not include_data:
                    pass
                else:
                    story_seq = story_seq[:max_story_length + 1]

                    curr_story_seq_len = len(story_seq)
                    # if self.multiref_gt:
                    #     story_seq = {
                    #         "story_seq": story_seq,
                    #         "multiref_gt": data_raw["multiref_gt"]
                    #     }

                    # story_seqs.append(story_seq)
                    # TODO: relax this.
                    # if (curr_story_seq_len >= self.min_story_length + 1
                    #         and curr_story_seq_len <= self.max_story_length + 1):
                    #     story_seqs.append(story_seq)
                    story_seqs.append(story_seq)


        print("[WARNING] Number of missing images in {}: {}".format(
            quotechar, len(missing_images)))
        missing_image_paths_f = (data_dir + "/missing_images_{}.txt".format(quotechar))
        missing_image_paths_file = open(missing_image_paths_f, "w")
        for missing_image_path in missing_images:
            missing_image_paths_file.write(missing_image_path + "\n")
        missing_image_paths_file.close()
        print("          Saves at: {}".format(missing_image_paths_f))

        print("There are {} valid story sequences in {}".format(
            len(story_seqs), json_path))

        return story_seqs

    # @classmethod
    # def _read_tsv(cls, input_file, quotechar=None):  # 只读取文本数据
    #     """Reads a tab separated value file."""
    #     with open(input_file, "r", encoding="utf-8-sig") as f:
    #         reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
    #         lines = []
    #         for line in reader:
    #             if sys.version_info[0] == 2:
    #                 line = list(unicode(cell, 'utf-8') for cell in line)
    #             lines.append(line)
    #         return lines
