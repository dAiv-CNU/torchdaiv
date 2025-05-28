"""TorchText vocab을 대체하는 Mock 클래스"""

from __future__ import annotations
from collections import Counter, OrderedDict
from typing import List, Callable, Dict, Any
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


class Token:
    UNK = '<unk>'
    PAD = '<pad>'
    BOS = '<bos>'
    EOS = '<eos>'
    DEFAULT = [PAD, UNK, BOS, EOS]


class HuggingFaceVocab:
    """
    Hugging Face Tokenizers를 사용한 TorchText vocab 대체 클래스
    """
    def __init__(
        self, ordered_dict: OrderedDict, min_freq: int = 1, 
        specials: List[str] = None, special_first: bool = True
    ):
        self.min_freq = min_freq
        self.specials = specials or []
        
        # Hugging Face Tokenizer 초기화
        self.tokenizer = Tokenizer(WordLevel(unk_token=Token.UNK))
        self.tokenizer.pre_tokenizer = Whitespace()
        
        # vocab 구성을 위한 데이터 준비
        vocab_dict = {}
        
        # special tokens를 먼저 추가
        if special_first:
            for i, token in enumerate(self.specials):
                vocab_dict[token] = i
        
        # 일반 토큰들 추가 (최소 빈도 이상만)
        current_idx = len(self.specials) if special_first else 0
        for token, freq in ordered_dict.items():
            if freq >= min_freq and token not in vocab_dict:
                vocab_dict[token] = current_idx
                current_idx += 1
        
        # special tokens를 나중에 추가하는 경우
        if not special_first:
            for token in self.specials:
                if token not in vocab_dict:
                    vocab_dict[token] = current_idx
                    current_idx += 1
        
        # Hugging Face Tokenizer에 vocab 설정
        self.tokenizer.add_tokens(list(vocab_dict.keys()))
        
        # 편의를 위한 매핑
        self.stoi = vocab_dict
        self.itos = {v: k for k, v in vocab_dict.items()}
    
    def __len__(self) -> int:
        return len(self.stoi)
    
    def __getitem__(self, token: str) -> int:
        """토큰을 인덱스로 변환 - Hugging Face Tokenizer 사용"""
        encoded = self.tokenizer.encode(token, add_special_tokens=False)
        if encoded.ids:
            return encoded.ids[0]
        return self.stoi.get(Token.UNK, 0)
    
    def __contains__(self, token: str) -> bool:
        return token in self.stoi
    
    def get_stoi(self) -> Dict[str, int]:
        """string to index 딕셔너리 반환"""
        return self.stoi.copy()
    
    def get_itos(self) -> Dict[int, str]:
        """index to string 딕셔너리 반환"""
        return self.itos.copy()
    
    def lookup_token(self, index: int) -> str:
        """인덱스를 토큰으로 변환 - Hugging Face Tokenizer 사용"""
        return self.tokenizer.decode([index], skip_special_tokens=False)
    
    def lookup_tokens(self, indices: List[int]) -> List[str]:
        """인덱스 리스트를 토큰 리스트로 변환"""
        return [self.lookup_token(idx) for idx in indices]
    
    def lookup_indices(self, tokens: List[str]) -> List[int]:
        """토큰 리스트를 인덱스 리스트로 변환 - Hugging Face Tokenizer 사용"""
        return [self[token] for token in tokens]
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """텍스트를 인덱스 리스트로 인코딩 - Hugging Face 기능 활용"""
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens).ids
    
    def decode(self, indices: List[int], skip_special_tokens: bool = True) -> str:
        """인덱스 리스트를 텍스트로 디코딩 - Hugging Face 기능 활용"""
        return self.tokenizer.decode(indices, skip_special_tokens=skip_special_tokens)


vocab = HuggingFaceVocab
