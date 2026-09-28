# Lab 3: DDP / DALI Implementation & Profiling

> CSED490F Deep Learning Implementation — 2026.09.28
> Contact the TAs at csed490f-ta@postech.ac.kr
>
> (원본: `Lab3.pdf`, 32 slides. 슬라이드 내 이미지/코드 스크린샷은 텍스트로 옮겨 적음)

---

## Contents

### Distributed Data Parallel (DDP)

- DDP handles **one process per GPU**. Each process has its own model replica and optimizer.
- Using `torch.distributed.init_process_group()`, you can set up communication between each GPU process.

> 그림: `Data → Sampler → Mini Batch (×3) → [Weights + Optimizer] on GPU 0/1/2`, 각 GPU의 gradient가 "Final gradient (G)"로 합쳐짐.

### DALI

- Deep learning is often stuck in data preprocessing (bottleneck).
- **DALI**: GPU-accelerated preprocessing tools.
- Asynchronously handles data preprocessing on the GPU; it can efficiently remove the training bottleneck.

> 그림: "PRE-PROCESSING WITH DALI — Removing the CPU bottleneck"
> `Input Data → Decode → GPU-Accelerated Augmentations → Preprocessed Data → Training/Inference (MXNet, PaddlePaddle, PyTorch, TensorFlow)`
> 파이프라인: `Loader (CPU) → JPEG → Decode (Mixed) → Images → Resize (GPU) → Augment (GPU) → Training`, Labels는 Loader에서 Training으로 직접 전달.

### Parallel Training

Both training schemes work in different areas:

- **DDP**: Distribute models and data, train on each GPU and reduce parameters.
- **DALI**: Asynchronous and fast data preprocessor.

In Lab 3, you will practice implementing each function individually and then applying them simultaneously.

---

## Experiments

**Goal: _Implement_ GPU parallel training and visualize its efficiency impacts.**

1. First, start by implementing **DDP** for parallel training.
2. Then, implement the **DALI data loader on DDP**.

```
handler/
├── DALI/
│   └── cifar10_loader.py      ← implement
├── DDP/
│   ├── cifar10_loader.py      ← implement
│   ├── model.py               ← implement
│   ├── train.py
│   └── utils.py               ← implement
└── DP/
    ├── cifar10_loader.py
    └── train.py
```

### Setup (1) — dataset

Run `bash scripts/init.sh` **once per instance**: it installs packages and downloads CIFAR-10.

- Datasets are saved in `dataset/cifar10` and `dataset/cifar10_images`.
- Saving 60,000 png images takes a few minutes. If `init.sh` is interrupted, just run it again.

```bash
pip install -r requirements.txt

########################################################
# Unlike our slurm cluster, a Vast.ai instance does not have a shared dataset directory.
# So we download CIFAR-10 to the instance's local disk (./dataset, ignored by git).
#   dataset/cifar10        : torchvision format (for DP / DDP)
#   dataset/cifar10_images : png image folders   (for DALI)
# Saving 60,000 png images takes a few minutes. It is skipped if the dataset already exists.
#
# The original download server (www.cs.toronto.edu) is sometimes very slow.
# In that case, set CIFAR10_URL to a mirror of 'cifar-10-python.tar.gz' (md5 is checked):
#   CIFAR10_URL=<mirror url> bash scripts/init.sh
########################################################

DATASET_DIR="${DATASET_DIR:-dataset}"

python my_lib/init_dataset.py --seed 42 --dataset_dir "$DATASET_DIR"
```

### Setup (2) — Vast.ai instance

1. Rent a **Vast.ai** instance with **2 GPUs**, using our docker image **`25fallcsed490f/cluster:week4`** (SSH launch mode).
   - Template name 예시: `CSED490FW4`, Image Path:Tag `25fallcsed490f/cluster:week4`, Version Tag `week4`
   - Launch mode: **Interactive shell server, SSH**
   - Container disk size: **50 GB**, Private
   - CLI: `vastai create instance <OFFER_ID> --image 25fallcsed490f/cluster:week4 --disk 50 --ssh`
2. Connect to the instance with SSH, and clone the code into `/workspace`. (Refer to Lab 2)
3. Download the `.nsys-rep` files from Google Drive to your local device. (Refer to Slide 27)

### Setup (3) — what to implement

In **`train_cifar.py`**, we give you a **scaffold** for ResNet18 training. Following the scaffold, you should implement:

1. Nsight log generation command (**Problem 0**) → fix **`scripts/launch.sh`**
2. DDP multiprocess training (**Problem 1~5**)
3. DALI DataLoader (**Problem 6~7**)

All functions to implement are in the **`handler`** folder:

1. `handler/DDP`
2. `handler/DALI`

Scaffold excerpt (`train_cifar.py`):

```python
def main_func(proc_id, args):
    try:
        # Setup GPU group for DDP multiprocess.
        if args.mode == "dp":
            pass
        elif args.mode == "ddp" or args.mode == "ddp_dali":
            ''' Problem 2: Setup GPU group
            (./handler/DDP/utils.py)
            DDP requires to setup GPU group, which can broadcast weights to all GPUs.
            This function set tcp connection between processes.
            Implement initialize_group function.
            '''
            from handler.DDP.utils import initialize_group
            initialize_group(proc_id, args.ip, args.port, args.num_gpu)
```

---

## 0. Run DP example

DP training code is already implemented.

**Problem 0)** Fix your ***`scripts/launch.sh`*** to generate an Nsight log, and execute ***`run_vastai_dp.sh`*** to see the result.

```bash
bash run_vastai_dp.sh
```

```bash
###########################################################################
# Problem 0: Generate Nsight log
# Find the correct way to generate Nsight log.
###########################################################################

# Scaffold
CUDA_VISIBLE_DEVICES=$LOCAL_GPU_IDS python train_cifar.py \
    --num_gpu=$NUM_GPUS \
    --data="$DATA_DIR" \
    --ckpt="$CKPT_DIR" \
    --mode="$MODE" \
    --save_ckpt
```

---

## 1. Implement DDP

You should implement the DDP training functions, **Problem 1 ~ 5** (`handler/DDP`).

| File | Function |
|---|---|
| `utils.py` | 1. `run_process()` |
| `utils.py` | 2. `initialize_group()` |
| `utils.py` | 3. `destroy_process()` |
| `model.py` | 4. `model_to_DDP()` |
| `cifar10_loader.py` | 5. `get_DDP_loader()` |

### utils.py (Problem 1~3)

In `utils.py`, implement simple functions which **create, group, and destroy** the multiprocess.

Please refer to the `torch.distributed` library tutorials:

- https://docs.pytorch.org/tutorials/beginner/dist_overview.html
- Specifically: https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html

#### Problem 1. `def run_process()`

- `run_process()` spawns the multiprocess, to use each GPU.
- Simply use the **`mp.spawn()`** function.
- It should create **{number of GPUs}** processes.

```python
if __name__ == "__main__":
    if args.mode == "dp":
        main_func(0, args)
    elif args.mode == "ddp" or args.mode == "ddp_dali":
        ''' Problem 1: Run process
        (./handler/DDP/utils.py)
        Implement run_process function.
        run_process function is used to run main_func in multiple processes, for DDP GPU group.
        It is a wrapper of mp.spawn function.
        You can use mp.spawn function as a reference.
        '''
        from handler.DDP.utils import run_process
        run_process(main_func, args)
```

#### Problem 2. `def initialize_group()`

`initialize_group()` does:

1. Make a TCP connection with the `torch.distributed` library.
2. Set the GPU device for the current process.

```python
elif args.mode == "ddp" or args.mode == "ddp_dali":
    ''' Problem 2: Setup GPU group
    (./handler/DDP/utils.py)
    DDP requires to setup GPU group, which can broadcast weights to all GPUs.
    This function set tcp connection between processes.
    Implement initialize_group function.
    '''
    from handler.DDP.utils import initialize_group
    initialize_group(proc_id, args.ip, args.port, args.num_gpu)
```

#### Problem 3. `def destroy_process()`

`destroy_process()` simply destroys the current process's GPU distribution group.

```python
finally:
    if args.mode == "ddp" or args.mode == "ddp_dali":
        ''' Problem 3: Destroy GPU group
        (./handler/DDP/utils.py)
        Implement destroy_process function.
        Just call the torch.distributed's destroy function.
        '''
        from handler.DDP.utils import destroy_process
        destroy_process()
```

### model.py (Problem 4) — `def model_to_DDP()`

- Because DDP works based on multiprocessing, you should specify the model's **GPU device id** for each process.
- It is very similar to the general DP device setup, and very simple.
- Start by referring to the DP code, and be careful to set the proper device id.

```python
elif args.mode == "ddp" or args.mode == "ddp_dali": # Handle DDP multiprocess model load.
    ''' Problem 4: model to DDP
    (./handler/DDP/model.py)
    Implement model_to_DDP function.
    model_to_DDP function is used to transfer model to DDP, SIMILAR with DP.
    Be careful for set devices. Set profer device id is important part in DDP.
    '''
    from handler.DDP.model import model_to_DDP
    model = model_to_DDP(model)
```

### cifar10_loader.py (Problem 5) — `def get_DDP_loader()`

- If you use a basic data loader, each process can use the same data twice in one epoch.
- So, we should **"fairly"** distribute the data in the data loader.
- **"Sampling data fairly for each process"** is already implemented in the `torch.utils.data.distributed` library, and you can simply use it.
- Here, you should implement **"distributed sampling"** for **fair** training.
- It is quite similar to the basic data loader, so start from it. It may be easy, because you just use the ALREADY implemented sampler.

```python
elif args.mode == "ddp":
    ''' Problem 5: Get DDP loader
    (./handler/DDP/cifar10_loader.py)
    Implement get_DDP_loader function.
    get_DDP_loader function is used to get DDP loader.
    You can use get_DP_loader function as a reference.
    '''
    from handler.DDP.cifar10_loader import get_DDP_loader
    get_loader = get_DDP_loader
```

### 1-2. Run and visualize the training efficiency

- Based on your implementation, execute ***`run_vastai_ddp.sh`***, generate the Nsight logs, and profile them.
- You can check your function call times in the **NVTX** row in Nsight Systems.

> 그림: Nsight Systems NVTX 행 — `Set data l…`, `S…`, `Epoch 0 [10.799 s]` 구간 아래로 `Batc…` 및 `f…` (forward 등) 이벤트가 반복됨.

---

## 2. Implement DALI

- On top of DDP training, we now implement a GPU-parallel data loader which can be used with DDP.
- You will implement the DALI data loader, and it will **override the DDP data loader** you already implemented in Problem 5.

| File (`handler/DALI`) | Target |
|---|---|
| `cifar10_loader.py` | 6. `class CifarPipeline` |
| `cifar10_loader.py` | 7. `def get_DALI_loader()` |

### Problem 6. `class CifarPipeline`

- DALI requires a pipeline for data processing, but it is slightly different from general data loaders.
- Here, you will implement a DALI pipeline which **works the same as the DP/DDP one**.
- Fill the blanks in ***CifarPipeline*** (at `DALI/cifar10_loader.py`).
- Please refer to **`DDP/cifar10_loader.py`** and **`DP/cifar10_loader.py`**.
- **\*\*\* Your Pipeline should work the same as the previous one \*\*\***

```text
Problem 6: make DALI Pipeline
(./handler/DALI/cifar10_loader.py)
To implement the get_DALI_loader function, you need to implement CifarPipeline class.
CifarPipeline class is a DALI pipeline for CIFAR-10 dataset.
Because DALI data process pipeline is differenct from general data loader, you should implement DALI pipeline in your own.
```

**[Tips]**

- We give you a pipeline scaffold for *CifarPipeline*.
- `nvidia.dali.fn` already has functions to read files, pad, flip, etc. — e.g. `fn.readers.file(..)`, …
- The cutout function is not contained in `nvidia.dali`, so we give you the **`fn_dali_cutout()`** function. You are free to use it to implement cutout in your pipeline.

```python
class CifarPipeline(Pipeline):
    """SCAFFOLD"""

    # def __init__(self, data_dir, batch_size, is_train, cutout_length,
    #              device_id, shard_id, num_shards, num_workers):
    #     super(CifarPipeline, self).__init__(batch_size, num_workers, device_id, seed=12345)
    #     self.data_dir = data_dir
    #     self.is_train = is_train
    #     self.cutout_length = cutout_length
    #     self.shard_id = shard_id
    #     self.num_shards = num_shards

    # def define_graph(self):
    #     images, labels = fn.readers.file(
    #         name="Reader",
    #         "fill it"
    #     )
    #     images = fn.decoders.image(
    #         images, device="mixed", output_type=types.RGB
    #     )

    #     if self.is_train:
    #         # 1. Padding
    #         "fill it"
    #         # 2. Horizontal Flip
    #         "fill it"
    #         # 3. Crop, Mirror, Normalize
    #         "fill it"
    #         # 4. Cutout
    #         if self.cutout_length > 0:
    #             "fill it"
    #     else:
    #         # 1. Crop, Normalize
    #         "fill it"

    #     return images, labels
```

The DALI data iterator is very different from the original torch DataLoader. We give you a wrapper so that the DALI iterator behaves the same as a torch DataLoader:

```python
class DALIWrapper:
    def __init__(self, dali_iter):
        self.dali_iter = dali_iter
    def __iter__(self):
        return self
    def __next__(self):
        data = self.dali_iter.__next__()[0]
        return data['data'], data['label'].squeeze(-1).long()
    def __len__(self):
        return ceil(self.dali_iter.size / self.dali_iter.batch_size)
    def reset(self):
        self.dali_iter.reset()
```

### Problem 7. `def get_DALI_loader()`

- Now, build the DALI data loader on top of *CifarPipeline*.
- Here, turn the pipeline into an iterator, and return the DALI data loader (iterator).
- DALI Iterator's data loading usage is slightly different from a general loader, so we give you the ***DALIWrapper*** class, which wraps the iterator to keep the same interface.

```text
Problem 7: Get DALI loader
(./handler/DALI/cifar10_loader.py)
Implement get_DALI_loader function.
get_DALI_loader function is used to get DALI loader.
Because DALI loader is slightly different with DP/DDP loader, you may change few parts of get_DP_loader.
'''
from handler.DALI.cifar10_loader import get_DALI_loader
get_loader = get_DALI_loader
```

### 2-2. Run and visualize the training efficiency

- Based on your implementation, execute ***`run_vastai_ddp_dali.sh`***, generate the Nsight logs, and profile them.
- You can check your function call times in the **NVTX** row in Nsight Systems.

---

## 3. Profile GPU/Mem Utils

- To check the **GPU utilization** and **memory utilization**, we provide the results of data-parallel training **run on 1, 2 and 4 GPUs in an RTX 3090 environment**.
- Analyze the results from the given `.nsys-rep` files, then
- **Explain how and why the GPU/Mem util changes for each of the following methods: DP, DDP, DALI.**

> **[Be careful]**
> In the given Nsight logs (GPU 1/2/4), you may see that **DP gets slower as the number of GPUs increases**.
> This unexpected result is caused by communication overhead between GPUs.
> Please be careful to interpret the results without being confused by the impact of the overhead.

---

## Submission

- **Deadline: 10/4 23:59 (KST)**
- Requirements
  - Explain how your code is implemented & analyze your profiling results. (max **4 pages**)
  - Font size 11pt.
  - **zip file**: `lab3_DDP_DALI_team{team_number}.zip`
  - Contents of the zip file
    - code: 1) `handler` folder, 2) `scripts/launch.sh`
    - Report file: `lab3_DDP_DALI_team{team_number}.pdf`
- Submit your zip file on **PLMS**.

```
lab3_DDP_DALI_team{N}.zip
├── handler/
│   ├── DALI/cifar10_loader.py
│   ├── DDP/{cifar10_loader.py, model.py, train.py, utils.py}
│   └── DP/{cifar10_loader.py, train.py}
├── launch.sh
└── lab3_DDP_DALI_team{N}.pdf
```

## Summary

1. Fix `scripts/launch.sh` to generate the Nsight log (P.0), and execute `bash run_vastai_dp.sh`.
2. Following `train_cifar.py`'s flow, implement the `handler/DDP` code (P.1~5), and execute `bash run_vastai_ddp.sh`.
3. Following `train_cifar.py`'s flow, implement the `handler/DALI` code (P.6~7), and execute `bash run_vastai_ddp_dali.sh`.
4. Based on your Nsight logs and the given log files (`.nsys-rep`), write the report.

---

## Additional Tips

1. The following links are helpful to start DDP Problems 1~3:
   - https://docs.pytorch.org/tutorials/beginner/dist_overview.html
   - https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html
2. To **download files from your Vast.ai instance to your local device**, refer to
   https://stackoverflow.com/questions/9427553/how-to-download-a-file-from-server-using-ssh

   On your own device (`{PORT}` and `{IP}`: from the instance's "Connect" button):

   ```bash
   scp -P {PORT} root@{IP}:"/your/file/path/file_name" "/download/here/file_name"
   # folder
   scp -P {PORT} -r root@{IP}:"/your/folder/path" "/download/here"
   ```

## Notification

1. **Vast.ai charges you as long as your instance exists.**
   - PLEASE **destroy your instance** when you finish your experiments.
   - A stopped instance is **still charged for its disk**. Only "Destroy" stops all charges.
   - Download your Nsight logs BEFORE you destroy it. All files on the instance are deleted.

   ```bash
   # 1. (on your local device) download your results first
   scp -P {PORT} -r root@{IP}:/workspace/CSED490F_DDP_DALI_Training/nsight_logs ./

   # 2. destroy the instance: Vast.ai console > Instances > Destroy (trash icon)
   #    or, with the vastai CLI
   vastai destroy instance {INSTANCE_ID}
   ```

2. When you rent an instance, please check the following:
   - **2 GPUs.** RTX 3090 is recommended (the given `.nsys-rep` files are from RTX 3090).
   - **NOT RTX 50xx / B200 (Blackwell)**: our image (PyTorch 2.4.1 + CUDA 11.8) cannot run on them.
   - **Max CUDA version ≥ 12.1** (for DALI) and **disk ≥ 50 GB**.
   - The first boot downloads our docker image (about 8 GB), so it can take several minutes.
