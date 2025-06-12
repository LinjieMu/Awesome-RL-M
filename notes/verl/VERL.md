#### 1. 背景知识

##### PPO算法的大致流程：

>1. 准备一个 batch 的 prompts；
>2. 将这个 batch 的 prompts 输入给 **Actor**，rollout 得到 responses；
>3. 将 prompt + responses 输入给 **Critic/Reward/Reference**，进行 inference，分别计算得得到 values、reward 和 log probs，将这些整合称为 experiences；
>4. 根据 experiences 多轮计算 actor loss 和 critic loss 并更新 Actor 和 Critic。

注意

- `rollout` 类似于调用`.generate()`方法，依次预测下一个token来得到一条轨迹
- `inference` 类似于调用模型的`.forward()`方法，来预测出当前轨迹的logits

##### 四个子模块（Actor、Critic、Reward、Reference）分别需要什么引擎？

- actor model 需要 training engine 和 rollout engine
- critic model 需要 training engine 和 inference engine
- reference model 和 reward model 只需要 inference，因为二者不需要训练

注意，inference 会直接**复用** training engine 的 `.forward()`方法

##### 现代的推理/训练引擎

- 推理引擎（rollout engine）：VLLM、SGLang
- 训练引擎（training engine）：FSDP、DeepSpeed、Megaton

##### 资源组共用方法

- `hybrid engine`：veRL 将 training 和 rollout engine **放置在同一个资源组中**串行执行。training 时，将 rollout engine 的显存回收（offload 到 CPU 上 或者直接析构掉），rollout 时，再将 training engine 的显存释放掉。这种将 actor model 的不同 engine 放置在同一个资源组上的方案，就称为 hybrid engine。
- `collocate` 策略：将 actor model的 training engine 和 reference model的 inference engine 放置在同一个资源组上，将 critic 的 training/inference engine 和 reward 的 inference engine 放置在同一个资源组上，最后单独放置 actor 的 rollout engine。

hybrid engine 单独强调了将 actor model 的 rollout engine 和 training engine 放置在同一个资源组上，而 collate 则强调的是不同子模块之间的。

##### 四种策略：

![img](.\images\placement.png)

1. fully collocate：所有的子模块都放在同一个资源组上，也即 DeepSpeed-Chat。
2. hybrid：actor 的 rollout engine 和 training engine 放在同一个资源组上，其他子模块进行部分 collocate，这是 veRL 所提出的策略，但是 veRL 原文其实给出了一个搜索方法，能够贪心搜索所有的策略，选择出最佳策略。
3. split collocate：actor 的 training engine 和 reference 的 inference engine 放在同一个资源组上，critic 的 training/inference engine 和 reward 的 inference engine 放在同一个资源组上；最后单独放置 actor 的 rollout engine，这是 OpenRLHF 和 NeMo-Aligner 的默认策略。
4. stand alone：所有子模块都单独放置，早期 OpenRLHF 会这么做，现在自然不会了。

#### 2. VeRL基础

#####  系统架构概述

verl的体系结构是围绕 hybrid-controller programming model 构建的，该模型通过统一的基于ray的协调层协调分布式训练和推理组件。

高级组件体系结构：

![image-20250612215059114](.\images\image-20250612215059114.png)

PPO 训练流水管线：

![image-20250612215453144](.\images\image-20250612215453144.png)

核心组件：

>- Hybrid-controller Programming Model:
>
>  - `RayWorkerGroup`: Manages groups of remote workers with unified interface
>
>  - `RayResourcePool`: Binds computational resources to worker processes
>
>  - `RayClassWithInitArgs`: Enables delayed remote instantiation
>
>  - `DataProto`: Protocol for efficient data transfer between components
>
>- Training Orchestration:
>
>  - `RayPPOTrainer`: The main coordinator class that orchestrates PPO training
>    -  sequence generation, reward computation, advantage estimation, and policy updates
>  - `FSDPSFTTrainer`: Handles supervised fine-tuning using FSDP backend 