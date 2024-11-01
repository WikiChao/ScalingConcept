# Scaling Concept With Text-Guided Diffusion Models
> **Chao Huang, Susan Liang, Yunlong Tang, Yapeng Tian, Anurag Kumar, Chenliang Xu**
>
> Text-guided diffusion models have revolutionized generative tasks by producing high-fidelity content from text descriptions. They have also enabled an editing paradigm where concepts can be replaced through text conditioning (e.g., *a dog -> a tiger*). In this work, we explore a novel approach: instead of replacing a concept, can we enhance or suppress the concept itself? Through an empirical study, we identify a trend where concepts can be decomposed in text-guided diffusion models. Leveraging this insight, we introduce **ScalingConcept**, a simple yet effective method to scale decomposed concepts up or down in real input without introducing new elements.
To systematically evaluate our approach, we present the *WeakConcept-10* dataset, where concepts are imperfect and need to be enhanced. More importantly, ScalingConcept enables a variety of novel zero-shot applications across image and audio domains, including tasks such as canonical pose generation and generative sound highlighting or removal. 

<a href="https://wikichao.github.io/ScalingConcept/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=red" height=20.5></a> 
<a href=""><img src="https://img.shields.io/badge/arXiv-ScalingConcept-b31b1b.svg" height=20.5></a>
<!-- [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/garibida/ReNoise-Inversion) -->

<p align="center">
<img src="asset/teaser.png" width="800px"/>
</p>

## Environment Setup
Our code builds on the requirement of the `diffusers` library. To set up the environment, please run:
```
conda env create -f environment.yaml
conda activate ScalingConcept
```
or install requirements:
```
pip install -r requirements.txt
```


## Minimal Example

We provide a minimal example to explore the effects of concept scaling. The `examples_images/` directory contains three sample images demonstrating different applications: `canonical pose generation`, `face attribute editing`, and `anime sketch enhancement`. To get started, try running:

```bash
python demo.py
```

The default setting is configured for `canonical pose generation`. For optimal results with other applications, adjust the prompt and relevant hyperparameters as noted in the code comments.

### Usage

Our ScalingConcept method supports various applications, each customizable by adjusting scaling parameters within `pipe_inference`. Below are recommended configurations for each application:

- **Canonical Pose Generation/Object Stitching**:
    ```python
    prompt = [object_name]
    omega = 5
    gamma = 3
    t_exit = 15
    ```

- **Weather Manipulation**:
    ```python
    prompt = '(heavy) fog' or '(heavy) rain'
    omega = 5
    gamma = 3
    t_exit = 15
    ```

- **Creative Enhancement**:
    ```python
    prompt = [concept to enhance]
    omega = 3
    gamma = 3
    t_exit = 15
    ```

- **Face Attribute Scaling**:
    ```python
    prompt = [face attribute, e.g., 'young face' or 'old face']
    omega = 3
    gamma = 3
    t_exit = 15
    ```

- **Anime Sketch Enhancement**:
    ```python
    prompt = 'anime'
    omega = 5
    gamma = 3
    t_exit = 25
    ```

In general, a larger `omega` value increases the effect of concept scaling, while higher `gamma` and `t_exit` values maintain fidelity. Note that inversion `prompt` selection is crucial, as the model is sensitive to the wording of prompts.

## Acknowledgements

This code builds upon the [diffusers](https://github.com/huggingface/diffusers) library. Additionally, we borrow code from the following repositories:

- [Pix2PixZero](https://github.com/pix2pixzero/pix2pix-zero) for noise regularization.
- [sdxl_inversions](https://github.com/cloneofsimo/sdxl_inversions) for the initial implementation of DDIM inversion in SDXL.
- [ReNoise-Inversion](https://github.com/garibida/ReNoise-Inversion) for a precise inversion technique.

## Citation
If you use this code for your research, please cite the following work: 
```
@misc{huang2024scaling,
      title={Scaling Concept With Text-Guided Diffusion Models}, 
      author={Chao Huang and Susan Liang and Yunlong Tang and Yapeng Tian Anurag Kumar and Chenliang Xu},
      year={2024},
      eprint={2410.24151},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
