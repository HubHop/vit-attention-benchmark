# Vision Transformer Attention Benchmark

This repo is a collection of attention mechanisms in vision Transformers. Beside the re-implementation, it provides a comprehensive benchmark on FLOPs, throughput and memory consumption.



### Requirements

- Pytorch 1.8+
- timm
- ninja
- einops
- fvcore
- matplotlib



### Testing Environment

- NVIDIA RTX 3090
- Intel® Core™ i9-10900X CPU @ 3.70GHz
- Memory 32GB
- Ubuntu 22.04
- PyTorch 1.8.1 + CUDA 11.1



### Setting

- input: 14 x 14 = 196 tokens
- batch size for speed testing (images/s): 64
- embedding dimension:768
- number of heads: 12



### Testing

For example, to test HiLo attention,

```bash
cd attentions/
python hilo.py
```

> By default, the script will test models on both CPU and GPU. FLOPs is measured by fvcore. You may want to edit the source file as needed.

Outputs:

```bash
Number of Params: 2.2 M
FLOPs = 298.3 M
throughput averaged with 30 times
batch_size 64 throughput on CPU 1029
throughput averaged with 30 times
batch_size 64 throughput on GPU 5104
```



### Supported Attentions

- **MSA:** An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. [[Paper](https://arxiv.org/abs/2010.11929)] [[Code](https://github.com/google-research/vision_transformer)]
- **Cross Window:** CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows. [[Paper](https://arxiv.org/abs/2107.00652)] [[Code](https://github.com/microsoft/CSWin-Transformer)]
- **DAT:** Vision Transformer with Deformable Attention. [[Paper](https://arxiv.org/abs/2201.00520)] [[Code](https://github.com/LeapLabTHU/DAT)]
- **Performer:** Rethinking Attention with Performers. [[Paper](https://arxiv.org/abs/2009.14794)] [[Code](https://github.com/google-research/google-research/tree/master/performer)]
- **Linformer:** Linformer: Self-Attention with Linear Complexity. [[Paper](https://arxiv.org/abs/2006.04768)] [[Code](https://github.com/lucidrains/linformer)]
- **SRA:** Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions. [[Paper](https://arxiv.org/abs/2102.12122)] [[Code](https://github.com/whai362/PVT)]
- **Local/Shifted Window:** Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. [[Paper](https://arxiv.org/abs/2103.14030)] [[Code](https://github.com/microsoft/Swin-Transformer)]
- **Focal:** Focal Self-attention for Local-Global Interactions in Vision Transformers. [[Paper](https://arxiv.org/abs/2107.00641)] [[Code](https://github.com/microsoft/Focal-Transformer)]
- **XCA:** XCiT: Cross-Covariance Image Transformers. [[Paper](https://arxiv.org/abs/2106.09681)] [[Code](https://github.com/facebookresearch/xcit)]
- **QuadTree:** QuadTree Attention for Vision Transformers. [[Paper](https://arxiv.org/abs/2201.02767)] [[Code](https://github.com/Tangshitao/QuadtreeAttention)]
- **VAN:** Visual Attention Network. [[Paper](https://arxiv.org/abs/2202.09741)] [[Code](https://github.com/Visual-Attention-Network/VAN-Classification)]
- **HorNet:** HorNet: Efficient High-Order Spatial Interactions with Recursive Gated Convolutions. [[Paper](https://arxiv.org/abs/2207.14284)] [[Code](https://github.com/raoyongming/HorNet)]
- **HiLo:** Fast Vision Transformers with HiLo Attention. [[Paper](https://arxiv.org/abs/2205.13213)] [[Code](https://github.com/ziplab/LITv2)]



### Single Attention Layer Benchmark

| Name           | Params (M) | FLOPs (M)  | CPU Speed | GPU Speed | Demo                                                         |
| -------------- | ---------- | ---------- | --------- | --------- | ------------------------------------------------------------ |
| MSA            | 2.36       | 521.43     | 505       | 4525      | [msa.py](https://github.com/HubHop/vit-attention-benchmark/blob/main/attentions/msa.py) |
| Cross Window   | 2.37       | 493.28     | 325       | 4334      | [cross_window.py](https://github.com/HubHop/vit-attention-benchmark/blob/main/attentions/cross_window.py) |
| DAT            | 2.38       | 528.69     | 223       | 3074      | [dat.py](https://github.com/HubHop/vit-attention-benchmark/blob/main/attentions/dat.py) |
| Performer      | 2.36       | 617.24     | 181       | 3180      | [performer.py](https://github.com/HubHop/vit-attention-benchmark/blob/main/attentions/performer.py) |
| Linformer      | 2.46       | 616.56     | 518       | 4578      | [linformer](https://github.com/HubHop/vit-attention-benchmark/blob/main/attentions/linformer.py) |
| SRA            | 4.72       | 419.56     | 710       | 4810      | [sra.py](https://github.com/HubHop/vit-attention-benchmark/blob/main/attentions/sra.py) |
| Local Window   | 2.36       | 477.17     | 631       | 4436      | [shifted_window.py](https://github.com/HubHop/vit-attention-benchmark/blob/main/attentions/shifted_window.py) |
| Shifted Window | 2.36       | 477.17     | 374       | 4546      | [shifted_window.py](https://github.com/HubHop/vit-attention-benchmark/blob/main/attentions/shifted_window.py) |
| Focal          | 2.44       | 526.85     | 146       | 2842      | [focal.py](https://github.com/HubHop/vit-attention-benchmark/blob/main/attentions/focal.py) |
| XCA            | 2.36       | 481.69     | 583       | 4659      | [xca.py](https://github.com/HubHop/vit-attention-benchmark/blob/main/attentions/xca.py) |
| QuadTree       | 5.33       | 613.25     | 72        | 3978      | [quadtree.py](https://github.com/HubHop/vit-attention-benchmark/blob/main/attentions/quadtree.py) |
| VAN            | 1.83       | 357.96     | 59        | 4213      | [van.py](https://github.com/HubHop/vit-attention-benchmark/blob/main/attentions/van.py) |
| HorNet         | 2.23       | 436.51     | 132       | 3996      | [hornet.py](https://github.com/HubHop/vit-attention-benchmark/blob/main/attentions/hornet.py) |
| HiLo           | 2.20       | **298.30** | **1029**  | **5104**  | [hilo.py](https://github.com/HubHop/vit-attention-benchmark/blob/main/attentions/hilo.py) |



## License

This repository is released under the Apache 2.0 license as found in the [LICENSE](./LICENSE) file.