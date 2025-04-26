# Mixture of Transformers


### **_Abstract_**
Large Language Models (LLMs) have achieved remarkable capabilities, but deploying a single model that excels across all domains remains challenging due to computational and training cost constraints. Mixture-of-Experts (MoE) techniques address this by conditionally activating parts of a network and increasing effective capacity without a proportional increase in compute. In this work we extend MoE to the model level and propose Mixture of Transformers (MoT), an ensemble of complete pre-trained LLMs gated by an adaptive router. Each expert is an entire LLM specialized for a specific domain, and a macro-level gating network learns to select the most relevant experts for each input. We describe four MoT architecture variants and a macro-gating mechanism that routes prompts to experts based on learned competencies. We discuss theoretical and system-level motivations for MoT including improved specialization, modularity, and efficiency via sparse expert activation. We examine key benefits such as scalable capacity and reduced per-query computation as well as challenges including routing accuracy, overhead, and output integration. Finally, we outline expected results and hypotheses. We anticipate that MoT will outperform single-LLM baselines such as Mixtral, DeepSeekMoE, and LLaMA-4 Scout across diverse tasks by leveraging the strengths of each expert and achieving higher overall performance while using less compute than similarly skilled dense models. This conceptual study lays the groundwork for future empirical validation and system deployment. http://github.com/Esmail-ibraheem/MoT


### MoE Architecture vs Dense Architecture


![MoE_vs_Dense](https://github.com/user-attachments/assets/04600bda-1706-4d06-b8a3-60a93db8bc8f)


### From Dense Architecture to Mixture of Experts models

![MoE_transformers](https://github.com/user-attachments/assets/3c3be5c5-6781-46ea-8776-8dc275d97025)


---


### **Bonus Papers for Advanced Implementations**:  
- **Retrieval-Augmented Generation (RAG)**: [Retrieval-Augmented Generation for Knowledge-Intensive Tasks (2020)](https://arxiv.org/abs/2005.11401).  
- **Mixture of Experts (MoE)**: [Switch Transformers (2021)](https://arxiv.org/abs/2101.03961).  

For **code implementations**, check repositories like [Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM)  or [Papers With Code](https://paperswithcode.com/task/large-language-model) .  

