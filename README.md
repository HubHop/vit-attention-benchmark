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

### Single Attention Layer Benchmark

| Name                                               | Params (M) | FLOPs (M)  | CPU Speed | GPU Speed |
| -------------------------------------------------- | ---------- | ---------- | --------- | --------- |
| [MSA](https://arxiv.org/abs/2010.11929)            | 2.36       | 521.43     | 505       | 4525      |
| [Cross Window](https://arxiv.org/abs/2107.00652)   | 2.37       | 493.28     | 325       | 4334      |
| [DAT](https://github.com/LeapLabTHU/DAT)           | 2.38       | 528.69     | 223       | 3074      |
| [Performer](https://arxiv.org/abs/2009.14794)      | 2.36       | 617.24     | 181       | 3180      |
| [Linformer](https://arxiv.org/abs/2006.04768)      | 2.46       | 616.56     | 518       | 4578      |
| [SRA](https://arxiv.org/abs/2102.12122)            | 4.72       | 419.56     | 710       | 4810      |
| [Local Window](https://arxiv.org/abs/2103.14030)   | 2.36       | 477.17     | 631       | 4436      |
| [Shifted Window](https://arxiv.org/abs/2103.14030) | 2.36       | 477.17     | 374       | 4546      |
| [Focal](https://arxiv.org/abs/2107.00641)          | 2.44       | 526.85     | 146       | 2842      |
| [XCA](https://arxiv.org/abs/2106.09681)            | 2.36       | 481.69     | 583       | 4659      |
| [QuadTree](https://arxiv.org/abs/2201.02767)       | 5.33       | 613.25     | 72        | 3978      |
| [VAN](https://arxiv.org/abs/2202.09741)            | 1.83       | 357.96     | 59        | 4213      |
| [HorNet](https://arxiv.org/abs/2207.14284)         | 2.23       | 436.51     | 132       | 3996      |
| [HiLo](https://arxiv.org/abs/2205.13213)           | 2.20       | **298.30** | **1029**  | **5104**  |



## License

This repository is released under the Apache 2.0 license as found in the [LICENSE](./LICENSE) file.