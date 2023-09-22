# multimodal_ordering
code for multimodal ordering

### Requirements
python 3.6.5, pytorch 1.8.0

Please download pytorch_model.bin for bart-large from Huggingface/transformers (https://github.com/huggingface/transformers), and put it into facebook/bart_large/.

### Train and evaluate
```
bash run_berson_bart.sh
```

### Test
```
bash run_test.sh
```

### Dataset
The dataset is from "Understanding Multimodal Procedural Knowledge by Sequencing Multimodal Instructional Manuals".
