import torch
import torch.nn as nn
from transformers import BertModel
import os

class GatedFusionLayer(nn.Module):
    """门控特征融合层"""
    def __init__(self, input_dim1, input_dim2, fused_dim, lambda_init=0.5):
        super().__init__()
        self.fused_dim = fused_dim
        
        # 1. 特征对齐：映射到统一维度
        self.align_fc1 = nn.Linear(input_dim1, fused_dim)
        self.align_fc2 = nn.Linear(input_dim2, fused_dim)
        
        # 2. 门控权重生成
        self.gate_fc = nn.Linear(2 * fused_dim, fused_dim)
        self.sigmoid = nn.Sigmoid()
        
        # 3. 拼接特征的投影层（新增）
        self.concat_projection = nn.Linear(input_dim1 + input_dim2, fused_dim)
        
        # 4. 残差权重参数
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init), requires_grad=True)
        
        # 5. 批归一化
        self.batch_norm = nn.BatchNorm1d(fused_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, feat1, feat2):
        # 特征对齐
        aligned_feat1 = self.align_fc1(feat1)  
        aligned_feat2 = self.align_fc2(feat2)  
        
        # 门控权重生成
        concat_aligned = torch.cat([aligned_feat1, aligned_feat2], dim=1)  
        gate = self.sigmoid(self.gate_fc(concat_aligned))  
        
        # 残差加权融合
        lambda_val = torch.clamp(self.lambda_param, 0.0, 1.0)
        
        # 加权部分: gate ⊙ F1 + (1-gate) ⊙ F2
        weighted_part = gate * aligned_feat1 + (1 - gate) * aligned_feat2
        
        # 拼接部分: [F1; F2] 投影到 fused_dim
        concat_original = torch.cat([feat1, feat2], dim=1)  
        concat_projected = self.concat_projection(concat_original)  
        
        # 完整融合公式
        fused_feat = lambda_val * weighted_part + (1 - lambda_val) * concat_projected
        
        # 批归一化和dropout
        fused_feat = self.batch_norm(fused_feat)
        fused_feat = self.dropout(fused_feat)
        
        # 同时返回特征和门控权重，供可视化使用
        return fused_feat, gate

class MultiGranularityModel(nn.Module):
    def __init__(self, bert_model_name, char_vocab_size, 
                 num_categorical_features=4, num_numerical_features=8,
                 char_embed_dim=128, cnn_out_channels=256, 
                 lstm_hidden_dim=128, num_classes=2, dropout_rate=0.3,
                 fusion_dim=512, lambda_init=0.7):
        super().__init__()
        
        self.num_categorical_features = num_categorical_features
        self.num_numerical_features = num_numerical_features
        self.fusion_dim = fusion_dim
        
        # BERT
        self.bert = BertModel.from_pretrained(bert_model_name,local_files_only=True)
        self.bert_dim = self.bert.config.hidden_size  # 768
 
        for param in self.bert.parameters():
            param.requires_grad = False

        
        # CNN for character-level features
        self.char_embedding = nn.Embedding(char_vocab_size, char_embed_dim, padding_idx=0)
        self.cnn = nn.Sequential(
            nn.Conv1d(char_embed_dim, cnn_out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        self.cnn_out_dim = cnn_out_channels
        
        # BiLSTM for sentence-level features
        self.bilstm = nn.LSTM(
            input_size=self.bert_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.lstm_out_dim = lstm_hidden_dim * 2  # 256
        
        # Embedding layers for categorical features
        self.categorical_embeddings = nn.ModuleList([
            nn.Embedding(10000, 32) for _ in range(num_categorical_features)
        ])
        self.categorical_total_dim = num_categorical_features * 32
        
        # Numerical features projection
        self.numerical_projection = nn.Sequential(
            nn.Linear(num_numerical_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 结构化特征融合
        self.struct_fusion = nn.Sequential(
            nn.Linear(self.categorical_total_dim + 64, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 文本特征维度
        text_feat_dim = self.bert_dim + self.cnn_out_dim + self.lstm_out_dim  # 768 + 256 + 256 = 1280
        
        # 文本特征投影到融合维度
        self.text_projection = nn.Sequential(
            nn.Linear(text_feat_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        
        self.llm_reason_projection = nn.Sequential(
            nn.Linear(self.bert_dim, self.bert_dim),  
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        

        self.llm_knowledge_enhancement = nn.Sequential(
            nn.Linear(self.bert_dim + 2, self.bert_dim),  
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 内部融合门
        self.internal_fusion_gate = nn.Sequential(
            nn.Linear(self.bert_dim * 2, self.bert_dim),  
            nn.Sigmoid()
        )
        
      
        self.text_fusion = nn.Sequential(
            nn.Linear(fusion_dim + self.bert_dim, fusion_dim), 
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.gated_fusion = GatedFusionLayer(
            input_dim1=fusion_dim,  
            input_dim2=fusion_dim,  
            fused_dim=fusion_dim,
            lambda_init=lambda_init
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, bert_input_ids, bert_attention_mask, char_input, 
                numerical_features, categorical_features,
                llm_reason_input_ids, llm_reason_attention_mask, llm_scores): 

        batch_size = bert_input_ids.size(0)
        
        # ===== 文本特征提取 =====
        bert_outputs = self.bert(
            input_ids=bert_input_ids,
            attention_mask=bert_attention_mask
        )
        bert_cls_features = bert_outputs.last_hidden_state[:, 0, :]
        
        char_embed = self.char_embedding(char_input)
        char_embed = char_embed.transpose(1, 2)
        cnn_features = self.cnn(char_embed).squeeze(-1)
        
        bert_sequence = bert_outputs.last_hidden_state
        lstm_outputs, (hidden, _) = self.bilstm(bert_sequence)
        lstm_features = torch.cat([hidden[-2], hidden[-1]], dim=1)
        
        text_features = torch.cat([
            self.dropout(bert_cls_features),
            self.dropout(cnn_features),
            self.dropout(lstm_features)
        ], dim=1)
        
        text_projected = self.text_projection(text_features)
        
        # ===== 结构化特征提取 =====
        categorical_embeds = []
        for i in range(self.num_categorical_features):
            embed = self.categorical_embeddings[i](categorical_features[:, i])
            categorical_embeds.append(embed)
        categorical_combined = torch.cat(categorical_embeds, dim=1)
        
        numerical_projected = self.numerical_projection(numerical_features)
        
        structured_features = torch.cat([
            categorical_combined,
            numerical_projected
        ], dim=1)
        
        structured_fused = self.struct_fusion(structured_features)
        
        # ===== LLM推理特征提取 (阶段一) =====
        with torch.no_grad(): 
            llm_reason_outputs = self.bert(
                input_ids=llm_reason_input_ids,
                attention_mask=llm_reason_attention_mask
            )
        llm_reason_cls_features = llm_reason_outputs.last_hidden_state[:, 0, :]
        llm_reason_projected = self.llm_reason_projection(llm_reason_cls_features)
        
        # 将LLM特征(768维)和评分(2维)拼接，得到770维特征
        llm_with_scores = torch.cat([llm_reason_projected, llm_scores], dim=1)  # [batch, 770]
        # 映射到768维
        llm_knowledge_enhanced = self.llm_knowledge_enhancement(llm_with_scores)  # [batch, 768]
        
        # 
        internal_features = torch.cat([bert_cls_features, llm_knowledge_enhanced], dim=1)  # [batch, 1536]
        internal_gate = self.internal_fusion_gate(internal_features)  # [batch, 768]
        self.last_internal_gate = internal_gate 
        
        internal_fused = internal_gate * bert_cls_features + (1 - internal_gate) * llm_knowledge_enhanced
        
        # 步骤一：融合内部融合后的文本特征和原始文本特征，形成"超级文本特征"
        super_text_feature = self.text_fusion(torch.cat([
            text_projected,
            internal_fused
        ], dim=1))
        
        # 步骤二：使用原有的门控机制融合"超级文本特征"和"结构化特征"
        final_fused_features, gate_weights = self.gated_fusion(super_text_feature, structured_fused)
        
        # 🔧 记录最终门控权重和特征供可视化使用
        self.last_gate_weights = gate_weights 
        self.last_final_features = final_fused_features
        
        # ===== 分类 =====
        logits = self.classifier(final_fused_features)
        return logits

    def get_gate_weights(self):
        '''
        获取双阶段门控权重
        '''
        # 检查变量是否存在
        if not hasattr(self, 'last_internal_gate') or not hasattr(self, 'last_gate_weights'):
            return None
        
        return {
            'stage1_internal_gate': self.last_internal_gate.detach().cpu(),
            'stage2_text_gate': self.last_gate_weights.detach().cpu(),
            'lambda_param': self.gated_fusion.lambda_param.item()
        }

    def get_feature_importance(self):
        '''
        获取融合后的特征重要性 (基于 L2 范数)
        '''
        if not hasattr(self, 'last_final_features'):
            return None
        # 计算特征的 L2 范数作为重要性的简单度量
        return torch.norm(self.last_final_features, p=2, dim=1).unsqueeze(1).detach().cpu()
