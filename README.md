# Dual Visual Prompting with Context-Modulated Diffusion Prompts

This is an official implementation of the following paper:
> Huan Wang, Haoran Li, Huaming Chen, Jun Yan, Jiahua Shi, Jun Shen<sup>\*</sup>. *"Dual Visual Prompting with Context-Modulated Diffusion Prompts"*. IEEE Transactions on Multimedia (TMM).
---

**Abstract:** Prompt learning has emerged as an efficient tuning paradigm for fine-tuning powerful pre-trained models on downstream tasks in specific domains. Existing efforts mainly focus on dataset-level implicit embeddings by introducing extra learnable parameters instead of fully fine-tuning large-scale visual models. However, we find that these static post-training prompts are not flexible enough to adapt various input instances within the same dataset, which might lead to the loss of the model's generalization capability. To leverage the meaningful contextual information of each input instance, in this paper, we propose a straightforward yet effective method, termed CoMoDP, to enhance visual prompt learning with Context Modulated Diffusion Prompts. Specifically, CoMoDP is a dual-visual prompting scheme that comprises two key components: 1) a unified visual prompt designer, producing dataset-level implicit embedding as unified prompts for efficient adaptation without corrupting the underlying information of the original image; and 2) a diffusion prompt simulator, leveraging diffusion model's meticulous understanding of semantic structure and texture edges in the images to dynamically generate instance-level implicit embedding as diffusion prompts for input samples. Moreover, to reduce the overfitting of prompts, we also introduce momentum alignment, a self-regulating strategy that restricts the optimization region of prompts in both feature and logit spaces. Extensive experiments on various standard and few-shot datasets demonstrate that our method brings substantial improvements and yields strong domain generalization performance, compared to the state-of-the-art methods. We also demonstrate both zero-shot and out-of-distribution performance to establish the utility of our dual-visual prompting scheme CoMoDP and the efficiency of each component, without involving excessive parameters.

---

Here is an example to run CoMoDP (previously submitted version is called ProCDF):


```python
python3 main_CoMoDP.py --dataset fgvc_flower --task_classes 102 --device 'cuda:0'
```