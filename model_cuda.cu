#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <memory>
#include <vector>
#include <random>
#include "tokenizer.h"

// Matching PyTorch's GPTConfig
struct GPTConfig {
    int block_size = 1024;
    int vocab_size = 50304;
    int n_layer = 12;
    int n_head = 12;
    int n_embd = 768;
    float dropout = 0.0f;
    bool bias = true;
};

// CUDA Error Checking
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error %d: %s\n", err, cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// LayerNorm Implementation
__global__ void layerNormKernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const int n_embd,
    const int batch_size,
    const int seq_len
) {
    const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int total_elements = batch_size * seq_len;
    
    for (int idx = tidx; idx < total_elements; idx += stride) {
        const int batch_seq_idx = idx;
        const float* input_row = input + batch_seq_idx * n_embd;
        float* output_row = output + batch_seq_idx * n_embd;
        
        // Calculate mean
        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < n_embd; i++) {
            sum += input_row[i];
        }
        const float mean = sum / n_embd;
        
        // Calculate variance
        float sq_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < n_embd; i++) {
            const float diff = input_row[i] - mean;
            sq_sum += diff * diff;
        }
        const float var = sq_sum / n_embd;
        const float inv_std = rsqrtf(var + 1e-5f);
        
        // Normalize and scale
        #pragma unroll
        for (int i = 0; i < n_embd; i++) {
            const float normalized = (input_row[i] - mean) * inv_std;
            output_row[i] = weight[i] * normalized + (bias ? bias[i] : 0.0f);
        }
    }
}

// Causal Self Attention Implementation
__global__ void causalmaskAttentionKernel(
    float* __restrict__ output,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const int batch_size,
    const int seq_len,
    const int n_head,
    const int head_dim
) {
    extern __shared__ float shared_mem[];
    
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;
    
    const int Q_idx = batch_idx * n_head * seq_len * head_dim + 
                     head_idx * seq_len * head_dim;
    const int K_idx = Q_idx;
    const int V_idx = Q_idx;
    
    // Load Q, K, V into shared memory
    float* shared_q = shared_mem;
    float* shared_k = shared_q + seq_len * head_dim;
    float* shared_v = shared_k + seq_len * head_dim;
    
    for (int i = tidx; i < seq_len * head_dim; i += blockDim.x) {
        shared_q[i] = q[Q_idx + i];
        shared_k[i] = k[K_idx + i];
        shared_v[i] = v[V_idx + i];
    }
    __syncthreads();
    
    // Calculate attention scores
    const float scale = 1.0f / sqrtf(head_dim);
    float* scores = shared_mem + 3 * seq_len * head_dim;
    
    for (int i = tidx; i < seq_len; i += blockDim.x) {
        for (int j = 0; j <= i; j++) {  // Causal masking
            float score = 0.0f;
            #pragma unroll
            for (int k = 0; k < head_dim; k++) {
                score += shared_q[i * head_dim + k] * shared_k[j * head_dim + k];
            }
            scores[i * seq_len + j] = score * scale;
        }
        for (int j = i + 1; j < seq_len; j++) {
            scores[i * seq_len + j] = -INFINITY;  // Causal masking
        }
    }
    __syncthreads();
    
    // Softmax
    for (int i = tidx; i < seq_len; i += blockDim.x) {
        float max_val = -INFINITY;
        for (int j = 0; j < seq_len; j++) {
            max_val = fmaxf(max_val, scores[i * seq_len + j]);
        }
        
        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            scores[i * seq_len + j] = expf(scores[i * seq_len + j] - max_val);
            sum += scores[i * seq_len + j];
        }
        
        const float inv_sum = 1.0f / sum;
        for (int j = 0; j < seq_len; j++) {
            scores[i * seq_len + j] *= inv_sum;
        }
    }
    __syncthreads();
    
    // Final matrix multiplication with V
    float* out = output + batch_idx * n_head * seq_len * head_dim +
                         head_idx * seq_len * head_dim;
    for (int i = tidx; i < seq_len; i += blockDim.x) {
        for (int k = 0; k < head_dim; k++) {
            float sum = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                sum += scores[i * seq_len + j] * shared_v[j * head_dim + k];
            }
            out[i * head_dim + k] = sum;
        }
    }
}

// GELU Activation
__device__ float gelu(float x) {
    const float sqrt_2_over_pi = 0.797884560802865f;
    const float coef = 0.044715f;
    float cdf = 0.5f * (1.0f + tanhf((sqrt_2_over_pi * (x + coef * x * x * x))));
    return x * cdf;
}

__global__ void geluKernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const int size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = gelu(input[idx]);
    }
}

// MLP Implementation
class CudaMLP {
private:
    cublasHandle_t cublas;
    int n_embd;
    int hidden_dim;
    float *w1, *w2, *b1, *b2;
    float *intermediate;
    
public:
    CudaMLP(cublasHandle_t handle, int n_embd, bool bias) : 
        cublas(handle), n_embd(n_embd), hidden_dim(4 * n_embd) {
        CHECK_CUDA(cudaMalloc(&w1, n_embd * hidden_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&w2, hidden_dim * n_embd * sizeof(float)));
        if (bias) {
            CHECK_CUDA(cudaMalloc(&b1, hidden_dim * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&b2, n_embd * sizeof(float)));
        }
        CHECK_CUDA(cudaMalloc(&intermediate, hidden_dim * sizeof(float)));
    }
    
    void forward(float* input, float* output, int batch_size, int seq_len) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        
        // First linear layer
        cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                   hidden_dim, batch_size * seq_len, n_embd,
                   &alpha,
                   w1, hidden_dim,
                   input, n_embd,
                   &beta,
                   intermediate, hidden_dim);
        
        // GELU activation
        const int total_elements = batch_size * seq_len * hidden_dim;
        const int block_size = 256;
        const int grid_size = (total_elements + block_size - 1) / block_size;
        geluKernel<<<grid_size, block_size>>>(intermediate, intermediate, total_elements);
        
        // Second linear layer
        cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                   n_embd, batch_size * seq_len, hidden_dim,
                   &alpha,
                   w2, n_embd,
                   intermediate, hidden_dim,
                   &beta,
                   output, n_embd);
    }
    
    ~CudaMLP() {
        cudaFree(w1);
        cudaFree(w2);
        cudaFree(b1);
        cudaFree(b2);
        cudaFree(intermediate);
    }
};

// Main GPT Model
class CudaGPT {
private:
    GPTConfig config;
    cublasHandle_t cublas;
    
    // Embeddings
    float *token_embedding;
    float *position_embedding;
    
    // Transformer blocks
    struct TransformerBlock {
        float *ln1_weight, *ln1_bias;
        float *ln2_weight, *ln2_bias;
        float *attn_qkv_weight, *attn_qkv_bias;
        float *attn_proj_weight, *attn_proj_bias;
        std::unique_ptr<CudaMLP> mlp;
    };
    std::vector<TransformerBlock> blocks;
    
    // Final layer norm
    float *ln_f_weight, *ln_f_bias;
    
    // Output projection
    float *lm_head;
    
public:
    CudaGPT(const GPTConfig& cfg) : config(cfg) {
        CHECK_CUDA(cublasCreate(&cublas));
        initializeWeights();
    }
    
    void initializeWeights() {
        // Allocate embeddings
        CHECK_CUDA(cudaMalloc(&token_embedding, 
                            config.vocab_size * config.n_embd * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&position_embedding,
                            config.block_size * config.n_embd * sizeof(float)));
        
        // Initialize transformer blocks
        blocks.resize(config.n_layer);
        for (auto& block : blocks) {
            CHECK_CUDA(cudaMalloc(&block.ln1_weight, config.n_embd * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&block.ln1_bias, config.n_embd * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&block.ln2_weight, config.n_embd * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&block.ln2_bias, config.n_embd * sizeof(float)));
            
            const int attn_total_size = 3 * config.n_embd * config.n_embd;
            CHECK_CUDA(cudaMalloc(&block.attn_qkv_weight, attn_total_size * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&block.attn_qkv_bias, 3 * config.n_embd * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&block.attn_proj_weight, 
                                config.n_embd * config.n_embd * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&block.attn_proj_bias, config.n_embd * sizeof(float)));
            
            block.mlp = std::make_unique<CudaMLP>(cublas, config.n_embd, config.bias);
        }
        
        // Final layer norm and output projection
        CHECK_CUDA(cudaMalloc(&ln_f_weight, config.n_embd * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&ln_f_bias, config.n_embd * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&lm_head, 
                            config.vocab_size * config.n_embd * sizeof(float)));
    }
    
    void forward(
        float* input_ids,
        float* output_logits,
        int batch_size,
        int seq_len
    ) {
        // Token + Position embeddings
        float* hidden_states;
        CHECK_CUDA(cudaMalloc(&hidden_states, 
                            batch_size * seq_len * config.n_embd * sizeof(float)));
        
        // Process through transformer blocks
        for (const auto& block : blocks) {
            // Layer Norm 1
            float* ln1_out;
            CHECK_CUDA(cudaMalloc(&ln1_out, 
                                batch_size * seq_len * config.n_embd * sizeof(float)));
            
            const int block_size = 256;
            const int grid_size = (batch_size * seq_len + block_size - 1) / block_size;
            
            layerNormKernel<<<grid_size, block_size>>>(
                ln1_out, hidden_states, block.ln1_weight, block.ln1_bias,
                config.n_embd, batch_size, seq_len
            );
            
            // Self Attention
            float *q, *k, *v;
            CHECK_CUDA(cudaMalloc(&q, batch_size * config.n_head * seq_len * 
                                   (config.n_embd/config.n_head) * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&k, batch_size * config.n_head * seq_len * 
                                   (config.n_embd/config.n_head) * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&v, batch_size * config.n_head * seq_len * 
                                   (config.n_embd/config.n_head) * sizeof(float)));
            
            // Split heads and compute attention
            dim3 attn_grid(1, config.n_head, batch_size);
            const int shared_mem_size = 3 * seq_len * (config.n_embd/config.n_head) * 
                                      sizeof(float) + seq_len * seq_len * sizeof(float);
            
            causalmaskAttentionKernel<<<attn_grid, block_size, shared_mem_size>>>(
                hidden_states, q, k, v, batch_size, seq_len, 
                config.n_head, config.n_embd/config.n_head
            );
            
            // Layer Norm 2
            float* ln2_out;
            CHECK_CUDA(cudaMalloc(&ln2_out, 
                                batch_size * seq_len * config.n_embd * sizeof(float)));
            
            layerNormKernel<<<grid_size, block_size>>>(
                ln2_out, hidden_states, block.ln2_weight, block.ln2_bias,
                config.n_embd, batch_size, seq_len
            );
            
            // MLP
            block.mlp->forward(ln2_out, hidden_states, batch_size, seq_len);
            
            // Clean up
            cudaFree(ln1_out);
            cudaFree(ln2_out);
            cudaFree(q);
            cudaFree(k);
            cudaFree(v);
        }
        
        // Final layer norm
        layerNormKernel<<<(batch_size * seq_len + 255)/256, 256>>>(
            output_logits, hidden_states, ln_f_weight, ln_f_bias,
            config.n_embd, batch_size, seq_len
        );
        
        // Final projection to vocab
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                   config.vocab_size, batch_size * seq_len, config.n_embd,
                   &alpha,
                   lm_head, config.vocab_size,
                   output_logits, config.n_embd,
                   &beta,
                   output_logits, config.vocab_size);
        
        cudaFree(hidden_states);
    }
    
    std::vector<int> generate(
        const std::string& prompt,
        int max_new_tokens,
        float temperature = 0.8f,
        float top_p = 0.95f,
        GPT2Tokenizer& tokenizer
    ) {
        // Tokenize input prompt
        std::vector<int> tokens = tokenizer.encode(prompt);
        if (tokens.size() > config.block_size) {
            tokens.erase(tokens.begin(), tokens.begin() + (tokens.size() - config.block_size));
        }
        
        // Allocate memory for input and output
        const int seq_len = tokens.size();
        float *input_ids, *output_logits;
        CHECK_CUDA(cudaMalloc(&input_ids, seq_len * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&output_logits, config.vocab_size * sizeof(float)));
        
        // Copy input tokens to GPU
        std::vector<float> input_float(tokens.begin(), tokens.end());
        CHECK_CUDA(cudaMemcpy(input_ids, input_float.data(), 
                            seq_len * sizeof(float), cudaMemcpyHostToDevice));
        
        // Generate new tokens
        std::random_device rd;
        std::mt19937 gen(rd());
        
        for (int i = 0; i < max_new_tokens && tokens.size() < config.block_size; i++) {
            // Forward pass
            forward(input_ids, output_logits, 1, tokens.size());
            
            // Get logits from the last position
            std::vector<float> logits(config.vocab_size);
            CHECK_CUDA(cudaMemcpy(logits.data(), output_logits + (tokens.size() - 1) * config.vocab_size,
                                config.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
            
            // Apply temperature
            if (temperature != 1.0f) {
                for (float& logit : logits) {
                    logit /= temperature;
                }
            }
            
            // Apply softmax
            float max_logit = *std::max_element(logits.begin(), logits.end());
            float sum_exp = 0.0f;
            for (float& logit : logits) {
                logit = expf(logit - max_logit);
                sum_exp += logit;
            }
            for (float& prob : logits) {
                prob /= sum_exp;
            }
            
            // Apply top-p (nucleus) sampling
            if (top_p < 1.0f) {
                std::vector<std::pair<float, int>> probs_with_idx;
                for (int j = 0; j < config.vocab_size; j++) {
                    probs_with_idx.push_back({logits[j], j});
                }
                std::sort(probs_with_idx.begin(), probs_with_idx.end(),
                         std::greater<std::pair<float, int>>());
                
                float cumsum = 0.0f;
                size_t last_idx = 0;
                for (size_t j = 0; j < probs_with_idx.size(); j++) {
                    cumsum += probs_with_idx[j].first;
                    if (cumsum > top_p) {
                        last_idx = j;
                        break;
                    }
                }
                
                // Renormalize probabilities
                float new_sum = 0.0f;
                for (size_t j = 0; j <= last_idx; j++) {
                    new_sum += probs_with_idx[j].first;
                }
                for (size_t j = 0; j <= last_idx; j++) {
                    probs_with_idx[j].first /= new_sum;
                }
                
                // Sample from the filtered distribution
                std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                float rand_val = dist(gen);
                float cumprob = 0.0f;
                int next_token = probs_with_idx[0].second;
                for (size_t j = 0; j <= last_idx; j++) {
                    cumprob += probs_with_idx[j].first;
                    if (rand_val < cumprob) {
                        next_token = probs_with_idx[j].second;
                        break;
                    }
                }
                tokens.push_back(next_token);
            } else {
                // Regular sampling from full distribution
                std::discrete_distribution<> dist(logits.begin(), logits.end());
                tokens.push_back(dist(gen));
            }
            
            // Update input for next iteration
            CHECK_CUDA(cudaMemcpy(input_ids, input_float.data(),
                                tokens.size() * sizeof(float), cudaMemcpyHostToDevice));
        }
        
        // Clean up
        cudaFree(input_ids);
        cudaFree(output_logits);
        
        return tokens;
    }
    
    std::string generate_text(
        const std::string& prompt,
        int max_new_tokens,
        float temperature = 0.8f,
        float top_p = 0.95f,
        GPT2Tokenizer& tokenizer
    ) {
        std::vector<int> tokens = generate(prompt, max_new_tokens, temperature, top_p, tokenizer);
        return tokenizer.decode(tokens);
    }
    
    ~CudaGPT() {
        // Clean up all allocated memory
        cudaFree(token_embedding);
        cudaFree(position_embedding);
        cudaFree(ln_f_weight);
        cudaFree(ln_f_bias);
        cudaFree(lm_head);
        cublasDestroy(cublas);
    }
};

// Example usage
int main() {
    GPTConfig config;
    CudaGPT model(config);
    
    // Initialize tokenizer
    GPT2Tokenizer tokenizer("vocab.json", "merges.txt");
    
    // Generate text
    std::string prompt = "Once upon a time";
    std::string generated_text = model.generate_text(
        prompt,
        100,  // max_new_tokens
        0.8f, // temperature
        0.95f, // top_p
        tokenizer
    );
    
    printf("Generated text:\n%s\n", generated_text.c_str());
    
    return 0;
}
