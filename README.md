<div align="center">
    <img align="center" width="224" src="https://user-images.githubusercontent.com/83948828/191930479-2e772966-4484-415f-a8f9-2a9dfa316e59.png" alt="microcirculation">
</div>

<h1 align="center">Microcirculation</h1>

<p align="center">Segmentation of human eye capillaries based on ophthalmic slit lamp images using UNet++</p>

- Anatoly Medvedev
- ID: 1603212269

<a name="000"></a>
<h2>Table of Contents</h2>

<ul>
    <ol type="1">
        <li><a href="#001">Environment</a></li>
        <li><a href="#002">Training</a></li>
        <li><a href="#003">Usage</a></li>
        <li><a href="#005">Reference</a></li>
    </ol>
</ul>

<a name="001"></a>
<h3>Environment</h3>

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

<a name="002"></a>
<h3>Training</h3>

Dataset Structure:

```
data/
    |
    train_dataset/
    |            |
    |            1.png
    |            1.geojson
    |            ...
    eye_test/
            |
            784.png
            ...
```

Model was trained in parallel on 2 GPUs Tesla V100 32GB. To train model, change the `training` flag in `training.py` to `True` and run it in the background:

```
$ nohup python training.py > log.txt &
```

Training results:

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Backbone</th>
      <th>F1 Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan='3'>UNet++</td>
      <td>ResNet-50</td>
      <td>0.512379</td>
    </tr>
    <tr>
      <td>ResNet-101</td>
      <td><b>0.525109</b></td>
    </tr>
  </tbody>
</table>

<a name="003"></a>
<h3>Usage</h3>

Follow steps in `demo.ipynb` to learn more about the model, image preparation, and model validation.

<a name="004"></a>
<h3>Reference</h3>

- Zhou, Z. et al. (2018) "UNet++: A Nested U-Net Architecture for Medical Image Segmentation". arXiv. doi: 10.48550/ARXIV.1807.10165