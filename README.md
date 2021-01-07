# Fashion-Image-Captioning-Benchmark
A repo containing codes and data for fashion image captioning for paper [Fashion Captioning: Towards Generating Accurate Descriptions with Semantic Rewards. Code and Data.](https://arxiv.org/abs/2008.02693)

# Install 
If you have difficulty running the training scripts in tools. You can try installing this repo as a python package:
```
python -m pip install -e .
```

# Baselines
|Model     |B-1  |B-2  |B-3  |B-4  |METEOR |ROUGE-L |CIDEr |SPICE |mAP  |ACC|
|----------|-----|-----|-----|-----|-------|--------|------|------|-----|---|
|NewFC     |19.9 |8.1  |3.8  |2.2  |7.5    |16.8    |21.1  |6.9   |0    |0  |
|ATT2IN2   |24.4 |11.4 |6.4  |4.3  |9.5    |19.9    |35.2  |9.8   |--   |-- |
|BUTD      |24.7 |11.5 |6.5  |4.4  |9.7    |19.6    |36.9  |10.1  |--   |-- |
|Trans     |24.4 |11.7 |6.2  |4.2  |10.2   |19.9    |36.7  |9.9   |--   |-- |

# Node
1. The max_length should be set at 30 instead of 20, as over 80% of the captions having number of words <= 30.


# License:
1. The dataset is under license in the LICENSE file.
2. No commercial use.

# Citation:
If you use this data, please cite:
```
@inproceedings{XuewenECCV20Fashion,
Author = {Xuewen Yang and Heming Zhang and Di Jin and Yingru Liu and Chi-Hao Wu and Jianchao Tan and Dongliang Xie and Jue Wang and Xin Wang},
Title = {Fashion Captioning: Towards Generating Accurate Descriptions with Semantic Rewards},
booktitle = {ECCV},
Year = {2020}
}
```

or 

```
@ARTICLE{2020arXiv200802693Y,
       author = {{Yang}, Xuewen and {Zhang}, Heming and {Jin}, Di and {Liu}, Yingru and {Wu}, Chi-Hao and {Tan}, Jianchao and {Xie}, Dongliang and {Wang}, Jue and {Wang}, Xin},
        title = "{Fashion Captioning: Towards Generating Accurate Descriptions with Semantic Rewards}",
      journal = {arXiv e-prints},
         year = 2020,
       eprint = {2008.02693},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020arXiv200802693Y},
}
```
