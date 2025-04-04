# Omnitron-X
Here are 10 influential papers that would be great to implement in LLMs and AI, spanning architecture improvements, training optimizations, and novel approaches:
1. Attention Is All You Need (2017)
Authors: Vaswani et al.
Why? Foundation of Transformers, introducing self-attention and eliminating recurrence.
Implementation Idea: Build a Transformer from scratch in PyTorch, optimizing for efficiency.
2. Scaling Laws for Neural Language Models (2020)
Authors: Kaplan et al. (OpenAI)
Why? Defines the relationship between dataset size, model size, and compute for optimal scaling.
Implementation Idea: Train models at different scales and verify the scaling laws.
3. GPT-3: Language Models are Few-Shot Learners (2020)
Authors: Brown et al. (OpenAI)
Why? Introduced in-context learning and demonstrated massive scale benefits.
Implementation Idea: Implement few-shot learning using pre-trained models like LLaMA or GPT-3-like architectures.
4. RoPE: Rotary Position Embeddings (2021)
Authors: Su et al.
Why? Improves positional encoding for long context understanding in Transformers.
Implementation Idea: Replace traditional sinusoidal embeddings with RoPE in a Transformer model.
5. FlashAttention: Fast Attention with IO-Awareness (2022)
Authors: Dao et al.
Why? Optimizes memory efficiency in attention computation, making large models more feasible.
Implementation Idea: Integrate FlashAttention into an LLM and compare speed/memory performance.
6. LoRA: Low-Rank Adaptation of Large Models (2021)
Authors: Hu et al.
Why? Reduces memory footprint for fine-tuning by using low-rank matrices.
Implementation Idea: Implement LoRA for efficient fine-tuning of an existing model.
7. Efficient Transformers: A Survey (2020)
Authors: Tay et al.
Why? Covers various optimizations, including linear attention, sparsity, and memory-efficient architectures.
Implementation Idea: Implement one of the efficient transformer variants, such as Performer or Linformer.
8. GQA: Training Generalist Models with Query-Based Architectures (2023)
Authors: DeepMind
Why? Generalizes LLMs across multiple modalities (text, vision, etc.).
Implementation Idea: Implement query-based architectures for multi-modal tasks.
9. Self-Rewarding Language Models (2023)
Authors: OpenAI
Why? Introduces reinforcement learning with self-generated rewards instead of human feedback.
Implementation Idea: Implement self-rewarding mechanisms in a GPT-style model.
10. Phi-2: A Recipe for Training Small, High-Quality LMs (2024)
Authors: Microsoft Research
11. MoE
12. Switch transformer 
13. Llama
14. Qwen
Why? Demonstrates that small models (2.7B) can achieve high performance with proper data curation and training strategies.
Implementation Idea: Train a smaller LLM with the Phi-2 methodology and compare with standard training.
Would you like help implementing any of these?
