import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
import json
import os
from collections import defaultdict
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor
import re

from kimia_infer.api.kimia import KimiAudio
from kimia_infer.models.tokenizer.whisper_Lv3.whisper import WhisperEncoder


class AudioFeatureKnowledgeBase:
    """基于音频特征的知识库，支持快速检索"""
    
    def __init__(self, knowledge_file: str, cache_dir: str = "./kb_cache"):
        self.knowledge_file = knowledge_file
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # 文本嵌入模型（用于知识库）
        self.text_encoder = SentenceTransformer('BAAI/bge-small-zh-v1.5')
        
        # 知识条目
        self.knowledge_entries = []
        self.special_terms = {
            'person_names': {},  # name -> contexts
            'technical_terms': {}  # term -> contexts
        }
        
        # 音频-文本映射缓存
        self.audio_cache = {}
        
        # 加载知识库
        self._load_knowledge_base()
        
        # 构建高效的查找结构
        self._build_lookup_structures()
        
    def _load_knowledge_base(self):
        """加载知识库并构建索引"""
        with open(self.knowledge_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        lines = content.strip().split('\n')
        
        for idx, line in enumerate(lines):
            if not line.strip():
                continue
                
            # 提取人名和上下文
            person_matches = re.finditer(r'\[人名[:：](.*?)\]', line)
            for match in person_matches:
                name = match.group(1).strip()
                if name not in self.special_terms['person_names']:
                    self.special_terms['person_names'][name] = []
                self.special_terms['person_names'][name].append({
                    'context': line,
                    'position': match.start(),
                    'line_idx': idx
                })
                
            # 提取专业术语和上下文
            term_matches = re.finditer(r'\[术语[:：](.*?)\]', line)
            for match in term_matches:
                term = match.group(1).strip()
                if term not in self.special_terms['technical_terms']:
                    self.special_terms['technical_terms'][term] = []
                self.special_terms['technical_terms'][term].append({
                    'context': line,
                    'position': match.start(),
                    'line_idx': idx
                })
                
            # 清理标记，保存纯文本
            clean_text = re.sub(r'\[(人名|术语)[:：].*?\]', lambda m: m.group(0).split(':')[-1].rstrip(']'), line)
            self.knowledge_entries.append({
                'original': line,
                'clean_text': clean_text,
                'embedding': None,
                'line_idx': idx
            })
            
    def _build_lookup_structures(self):
        """构建高效的查找结构"""
        # 构建术语的前缀树（Trie）用于快速匹配
        self.term_trie = TrieNode()
        
        for name in self.special_terms['person_names']:
            self.term_trie.insert(name, 'person')
            
        for term in self.special_terms['technical_terms']:
            self.term_trie.insert(term, 'technical')
            
        # 预计算知识条目的嵌入
        texts = [entry['clean_text'] for entry in self.knowledge_entries]
        embeddings = self.text_encoder.encode(texts, convert_to_numpy=True)
        
        for i, entry in enumerate(self.knowledge_entries):
            entry['embedding'] = embeddings[i]
            
        # 构建FAISS索引
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)  # 使用内积相似度
        self.index.add(embeddings.astype('float32'))
        
    def get_audio_context(self, whisper_features: torch.Tensor, top_k: int = 3) -> Tuple[str, List[str]]:
        """
        基于音频特征获取相关上下文
        
        Args:
            whisper_features: Whisper编码的音频特征
            top_k: 返回前k个相关知识
            
        Returns:
            (增强的提示文本, 相关专有名词列表)
        """
        # 将Whisper特征降维到文本嵌入空间
        # 这里使用简单的平均池化，实际可以训练一个投影层
        audio_embedding = whisper_features.mean(dim=1).cpu().numpy()
        
        # 归一化
        audio_embedding = audio_embedding / (np.linalg.norm(audio_embedding) + 1e-8)
        
        # 如果维度不匹配，使用随机投影
        if audio_embedding.shape[-1] != self.dimension:
            # 使用缓存的投影矩阵或创建新的
            cache_key = f"projection_{audio_embedding.shape[-1]}_{self.dimension}"
            projection_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            
            if os.path.exists(projection_path):
                with open(projection_path, 'rb') as f:
                    projection_matrix = pickle.load(f)
            else:
                # 创建随机投影矩阵
                projection_matrix = np.random.randn(audio_embedding.shape[-1], self.dimension)
                projection_matrix = projection_matrix / np.linalg.norm(projection_matrix, axis=0)
                with open(projection_path, 'wb') as f:
                    pickle.dump(projection_matrix, f)
                    
            audio_embedding = audio_embedding @ projection_matrix
            
        # 在知识库中搜索
        audio_embedding = audio_embedding.reshape(1, -1).astype('float32')
        scores, indices = self.index.search(audio_embedding, top_k * 2)  # 多检索一些
        
        # 收集相关的专有名词
        relevant_terms = defaultdict(float)
        relevant_contexts = []
        
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.knowledge_entries):
                entry = self.knowledge_entries[idx]
                relevant_contexts.append(entry['original'])
                
                # 提取该条目中的专有名词
                for name in self.special_terms['person_names']:
                    if name in entry['clean_text']:
                        relevant_terms[name] += score * 1.5  # 人名权重更高
                        
                for term in self.special_terms['technical_terms']:
                    if term in entry['clean_text']:
                        relevant_terms[term] += score
                        
        # 构建增强提示
        prompt_parts = ["请将音频内容准确转换为文字。"]
        
        # 添加最相关的专有名词
        top_terms = sorted(relevant_terms.items(), key=lambda x: x[1], reverse=True)[:10]
        if top_terms:
            term_list = [term for term, _ in top_terms]
            prompt_parts.append(f"注意音频中可能包含这些词汇：{', '.join(term_list)}")
            
        # 返回提示和专有名词列表
        return '\n'.join(prompt_parts), [term for term, _ in top_terms]


class TrieNode:
    """前缀树节点，用于高效的字符串匹配"""
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.term_type = None
        self.term_value = None
        
    def insert(self, word: str, term_type: str):
        node = self
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
        node.term_type = term_type
        node.term_value = word


class StreamingTermCorrector:
    """流式专有名词纠正器"""
    
    def __init__(self, term_trie: TrieNode, similarity_threshold: float = 0.7):
        self.term_trie = term_trie
        self.similarity_threshold = similarity_threshold
        self.buffer = ""
        self.correction_cache = {}
        
    def process_chunk(self, text_chunk: str) -> str:
        """处理文本块并返回纠正后的结果"""
        self.buffer += text_chunk
        
        # 查找可能的专有名词边界
        corrected_buffer = self._correct_buffer()
        
        # 返回已确定的部分，保留可能未完成的部分
        if len(self.buffer) > 50:  # 防止buffer过大
            result = corrected_buffer[:-20]  # 保留末尾部分以处理跨块的词
            self.buffer = corrected_buffer[-20:]
            return result
        else:
            return ""
            
    def flush(self) -> str:
        """处理剩余的buffer"""
        result = self._correct_buffer()
        self.buffer = ""
        return result
        
    def _correct_buffer(self) -> str:
        """纠正buffer中的文本"""
        if not self.buffer:
            return ""
            
        # 使用缓存加速
        cache_key = hashlib.md5(self.buffer.encode()).hexdigest()
        if cache_key in self.correction_cache:
            return self.correction_cache[cache_key]
            
        result = []
        i = 0
        
        while i < len(self.buffer):
            # 尝试匹配最长的专有名词
            matched = False
            for length in range(min(20, len(self.buffer) - i), 0, -1):  # 限制最大匹配长度
                substring = self.buffer[i:i+length]
                
                # 精确匹配
                if self._trie_search(substring):
                    result.append(substring)
                    i += length
                    matched = True
                    break
                    
                # 模糊匹配
                correction = self._fuzzy_match(substring)
                if correction and correction != substring:
                    result.append(correction)
                    i += length
                    matched = True
                    break
                    
            if not matched:
                result.append(self.buffer[i])
                i += 1
                
        corrected = ''.join(result)
        
        # 缓存结果
        if len(self.correction_cache) < 10000:  # 限制缓存大小
            self.correction_cache[cache_key] = corrected
            
        return corrected
        
    def _trie_search(self, word: str) -> bool:
        """在Trie中搜索词"""
        node = self.term_trie
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end
        
    def _fuzzy_match(self, text: str) -> Optional[str]:
        """模糊匹配最相似的专有名词"""
        # 这里使用简化的编辑距离匹配
        # 实际使用时可以用更高效的算法
        best_match = None
        best_score = 0
        
        # 只对长度相近的词进行匹配
        for length_diff in range(-2, 3):
            target_length = len(text) + length_diff
            if target_length < 2 or target_length > 20:
                continue
                
            # 遍历可能的候选词（这里需要优化）
            # 实际实现时应该预先按长度组织专有名词
            candidates = self._get_candidates_by_length(target_length)
            
            for candidate in candidates:
                score = self._calculate_similarity(text, candidate)
                if score > self.similarity_threshold and score > best_score:
                    best_score = score
                    best_match = candidate
                    
        return best_match
        
    def _get_candidates_by_length(self, length: int) -> List[str]:
        """获取特定长度的候选词（需要预先构建）"""
        # 这里简化处理，实际应该预先构建长度索引
        return []
        
    def _calculate_similarity(self, s1: str, s2: str) -> float:
        """计算字符串相似度"""
        if s1 == s2:
            return 1.0
        
        # 使用Jaccard相似度（字符级）
        set1 = set(s1)
        set2 = set(s2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union == 0:
            return 0.0
            
        return intersection / union


class EfficientRAGASR:
    """高效的RAG增强ASR系统"""
    
    def __init__(self, model_path: str, knowledge_base_path: str):
        # 初始化ASR模型
        self.model = KimiAudio(
            model_path=model_path,
            load_detokenizer=False,
        )
        
        # 初始化知识库
        self.kb = AudioFeatureKnowledgeBase(knowledge_base_path)
        
        # 初始化流式纠正器
        self.corrector = StreamingTermCorrector(self.kb.term_trie)
        
        # 默认采样参数
        self.sampling_params = {
            "audio_temperature": 0.8,
            "audio_top_k": 10,
            "text_temperature": 0.0,
            "text_top_k": 5,
            "audio_repetition_penalty": 1.0,
            "audio_repetition_window_size": 64,
            "text_repetition_penalty": 1.0,
            "text_repetition_window_size": 16,
        }
        
    def recognize_with_rag(self, audio_path: str) -> Dict:
        """
        使用RAG增强的单次推理识别
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            识别结果字典
        """
        # 1. 提取Whisper特征
        print("提取音频特征...")
        whisper_features = self._extract_whisper_features(audio_path)
        
        # 2. 基于音频特征获取相关知识
        print("检索相关知识...")
        enhanced_prompt, relevant_terms = self.kb.get_audio_context(whisper_features)
        
        print(f"找到相关词汇: {', '.join(relevant_terms[:5])}...")
        
        # 3. 单次推理with增强提示
        print("执行ASR识别...")
        messages = [
            {"role": "user", "message_type": "text", "content": enhanced_prompt},
            {"role": "user", "message_type": "audio", "content": audio_path},
        ]
        
        _, text_result = self.model.generate(messages, **self.sampling_params, output_type="text")
        
        # 4. 流式后处理纠正
        print("执行后处理纠正...")
        corrected_text = self._streaming_correction(text_result)
        
        # 5. 返回结果
        results = {
            'audio_path': audio_path,
            'raw_result': text_result,
            'corrected_result': corrected_text,
            'relevant_terms': relevant_terms,
            'detected_terms': self._extract_detected_terms(corrected_text)
        }
        
        return results
        
    def _extract_whisper_features(self, audio_path: str) -> torch.Tensor:
        """提取Whisper音频特征"""
        # 这里直接使用Kimi模型中的Whisper编码器
        # 需要访问模型的prompt_manager中的whisper_model
        whisper_features = self.model.prompt_manager.extract_whisper_feat(audio_path)
        return whisper_features
        
    def _streaming_correction(self, text: str, chunk_size: int = 10) -> str:
        """流式纠正文本"""
        # 模拟流式处理
        result_chunks = []
        
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            corrected_chunk = self.corrector.process_chunk(chunk)
            if corrected_chunk:
                result_chunks.append(corrected_chunk)
                
        # 处理剩余部分
        final_chunk = self.corrector.flush()
        if final_chunk:
            result_chunks.append(final_chunk)
            
        return ''.join(result_chunks)
        
    def _extract_detected_terms(self, text: str) -> Dict[str, List[str]]:
        """提取文本中检测到的专有名词"""
        detected = {
            'person_names': [],
            'technical_terms': []
        }
        
        for name in self.kb.special_terms['person_names']:
            if name in text:
                detected['person_names'].append(name)
                
        for term in self.kb.special_terms['technical_terms']:
            if term in text:
                detected['technical_terms'].append(term)
                
        return detected
        
    def batch_process(self, audio_files: List[str], output_dir: str, num_workers: int = 4):
        """并行批量处理"""
        os.makedirs(output_dir, exist_ok=True)
        
        def process_file(audio_file):
            try:
                result = self.recognize_with_rag(audio_file)
                result['status'] = 'success'
                return result
            except Exception as e:
                return {
                    'audio_path': audio_file,
                    'status': 'error',
                    'error': str(e)
                }
                
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(process_file, audio_files))
            
        # 保存结果
        output_file = os.path.join(output_dir, "efficient_rag_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        # 打印统计
        success_count = sum(1 for r in results if r['status'] == 'success')
        print(f"\n处理完成: {success_count}/{len(audio_files)} 成功")
        
        return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="moonshotai/Kimi-Audio-7B-Instruct")
    parser.add_argument("--knowledge_base", type=str, required=True, help="知识库文件路径")
    parser.add_argument("--audio_file", type=str, help="单个音频文件")
    parser.add_argument("--audio_dir", type=str, help="音频文件目录")
    parser.add_argument("--output_dir", type=str, default="efficient_rag_output")
    parser.add_argument("--num_workers", type=int, default=4, help="并行处理线程数")
    args = parser.parse_args()
    
    # 初始化系统
    asr_system = EfficientRAGASR(args.model_path, args.knowledge_base)
    
    if args.audio_file:
        # 处理单个文件
        result = asr_system.recognize_with_rag(args.audio_file)
        
        print(f"\n识别结果:")
        print(f"原始: {result['raw_result']}")
        print(f"纠正: {result['corrected_result']}")
        print(f"检测到的人名: {', '.join(result['detected_terms']['person_names'])}")
        print(f"检测到的术语: {', '.join(result['detected_terms']['technical_terms'])}")
        
        # 保存结果
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, "single_result.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
    elif args.audio_dir:
        # 批量处理
        audio_files = [
            os.path.join(args.audio_dir, f) 
            for f in os.listdir(args.audio_dir) 
            if f.endswith(('.wav', '.mp3', '.m4a', '.flac'))
        ]
        
        if audio_files:
            asr_system.batch_process(audio_files, args.output_dir, args.num_workers)
        else:
            print("未找到音频文件")