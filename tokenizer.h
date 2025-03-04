#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <fstream>
#include <regex>
#include <queue>

class GPT2Tokenizer {
private:
    std::unordered_map<std::string, int> encoder;
    std::unordered_map<int, std::string> decoder;
    std::unordered_map<std::string, std::string> byte_encoder;
    std::unordered_map<std::string, std::string> byte_decoder;
    std::vector<std::pair<std::string, std::string>> bpe_ranks;
    
    std::string byte_encode(const std::string& text) {
        std::string encoded;
        for (unsigned char c : text) {
            encoded += byte_encoder[std::string(1, c)];
        }
        return encoded;
    }
    
    std::string byte_decode(const std::string& text) {
        std::string decoded;
        for (size_t i = 0; i < text.length();) {
            std::string byte_char;
            byte_char += text[i++];
            if ((text[i - 1] & 0x80) == 0) continue;
            while (i < text.length() && (text[i] & 0xC0) == 0x80) {
                byte_char += text[i++];
            }
            decoded += byte_decoder[byte_char];
        }
        return decoded;
    }
    
    std::vector<std::string> get_pairs(const std::vector<std::string>& word) {
        std::vector<std::string> pairs;
        for (size_t i = 0; i < word.size() - 1; i++) {
            pairs.push_back(word[i] + " " + word[i + 1]);
        }
        return pairs;
    }

public:
    GPT2Tokenizer(const std::string& vocab_file, const std::string& merges_file) {
        // Initialize byte encoder/decoder
        for (int i = 0; i < 256; i++) {
            std::string ch(1, (char)i);
            byte_encoder[ch] = ch;
            byte_decoder[ch] = ch;
        }
        
        // Load vocabulary
        std::ifstream vocab(vocab_file);
        std::string line;
        while (std::getline(vocab, line)) {
            size_t sep = line.find('\t');
            if (sep != std::string::npos) {
                std::string token = line.substr(0, sep);
                int id = std::stoi(line.substr(sep + 1));
                encoder[token] = id;
                decoder[id] = token;
            }
        }
        
        // Load BPE merges
        std::ifstream merges(merges_file);
        while (std::getline(merges, line)) {
            if (line.empty() || line[0] == '#') continue;
            size_t sep = line.find(' ');
            if (sep != std::string::npos) {
                std::string first = line.substr(0, sep);
                std::string second = line.substr(sep + 1);
                bpe_ranks.push_back({first, second});
            }
        }
    }
    
    std::vector<int> encode(const std::string& text) {
        std::vector<int> bpe_tokens;
        std::string encoded_text = byte_encode(text);
        
        std::regex pattern("'s|'t|'re|'ve|'m|'ll|'d| ?\\w+| ?\\d+| ?[^\\s\\w\\d]+|\\s+(?!\\S)|\\s+");
        std::regex_iterator<std::string::iterator> it(encoded_text.begin(), encoded_text.end(), pattern);
        std::regex_iterator<std::string::iterator> end;
        
        while (it != end) {
            std::string token = it->str();
            std::vector<std::string> word;
            for (char c : token) {
                word.push_back(std::string(1, c));
            }
            
            while (true) {
                std::vector<std::string> pairs = get_pairs(word);
                if (pairs.empty()) break;
                
                auto min_pair = std::min_element(pairs.begin(), pairs.end(),
                    [this](const std::string& a, const std::string& b) {
                        auto it_a = std::find_if(bpe_ranks.begin(), bpe_ranks.end(),
                            [&a](const auto& p) { return p.first + " " + p.second == a; });
                        auto it_b = std::find_if(bpe_ranks.begin(), bpe_ranks.end(),
                            [&b](const auto& p) { return p.first + " " + p.second == b; });
                        return it_a < it_b;
                    });
                
                if (min_pair == pairs.end()) break;
                
                std::string first = (*min_pair).substr(0, (*min_pair).find(' '));
                std::string second = (*min_pair).substr((*min_pair).find(' ') + 1);
                
                std::vector<std::string> new_word;
                for (size_t i = 0; i < word.size(); i++) {
                    if (i < word.size() - 1 && word[i] == first && word[i + 1] == second) {
                        new_word.push_back(first + second);
                        i++;
                    } else {
                        new_word.push_back(word[i]);
                    }
                }
                word = new_word;
                if (word.size() == 1) break;
            }
            
            for (const auto& token : word) {
                if (encoder.find(token) != encoder.end()) {
                    bpe_tokens.push_back(encoder[token]);
                }
            }
            ++it;
        }
        return bpe_tokens;
    }
    
    std::string decode(const std::vector<int>& tokens) {
        std::string text;
        for (int token : tokens) {
            if (decoder.find(token) != decoder.end()) {
                text += decoder[token];
            }
        }
        return byte_decode(text);
    }
};
