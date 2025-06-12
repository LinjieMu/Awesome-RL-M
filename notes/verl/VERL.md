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
- `collocate` 策略：OpenRLHF 将 actor 的 training engine 和 reference 的 inference engine 放置在同一个资源组上，将 critic 的 training/inference engine 和 reward 的 inference engine 放置在同一个资源组上，最后单独放置 actor 的 rollout engine。

hybrid engine 单独强调了将 actor model 的 rollout engine 和 training engine 放置在同一个资源组上，而 collate 则强调的是不同子模块之间的。