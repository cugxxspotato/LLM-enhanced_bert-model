import torch
import torch.nn as nn
from transformers import BertModel


class GatedFusionLayer(nn.Module):
    """门控特征融合层 - 独立实现"""
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
        aligned_feat1 = self.align_fc1(feat1)  # [batch, fused_dim]
        aligned_feat2 = self.align_fc2(feat2)  # [batch, fused_dim]
        
        # 门控权重生成
        concat_aligned = torch.cat([aligned_feat1, aligned_feat2], dim=1)  # [batch, 2*fused_dim]
        gate = self.sigmoid(self.gate_fc(concat_aligned))  # [batch, fused_dim]
        
        # 残差加权融合
        lambda_val = torch.clamp(self.lambda_param, 0.0, 1.0)
        
        # 加权部分: gate ⊙ F1 + (1-gate) ⊙ F2
        weighted_part = gate * aligned_feat1 + (1 - gate) * aligned_feat2
        
        # 拼接部分: [F1; F2] 投影到 fused_dim
        concat_original = torch.cat([feat1, feat2], dim=1)  # [batch, input_dim1 + input_dim2]
        concat_projected = self.concat_projection(concat_original)  # [batch, fused_dim]
        
        # 完整融合公式
        fused_feat = lambda_val * weighted_part + (1 - lambda_val) * concat_projected
        
        # 批归一化和dropout
        fused_feat = self.batch_norm(fused_feat)
        fused_feat = self.dropout(fused_feat)
        
        return fused_feat


class AblationModel(nn.Module):
    """消融试验模型 - 按照6个特定实验设计"""
    
    def __init__(self, bert_model_name, char_vocab_size, 
                 num_categorical_features=4, num_numerical_features=8,
                 char_embed_dim=128, cnn_out_channels=256, 
                 lstm_hidden_dim=128, num_classes=2, dropout_rate=0.3,
                 fusion_dim=512, lambda_init=0.7,
                 # 消融控制参数 - 新增6个特定参数
                 use_bert=True, use_cnn=True, use_bilstm=True,
                 use_structured=True, use_llm=True, use_llm_logic=True,
                 use_llm_fact=True, use_ifg=True, use_dbg=True,
                 use_gated_fusion=True):
        super().__init__()
        
        # 🔧 保存bert_model_name
        self.bert_model_name = bert_model_name
        
        # 保存消融配置
        self.use_bert = use_bert
        self.use_cnn = use_cnn
        self.use_bilstm = use_bilstm
        self.use_structured = use_structured
        self.use_llm = use_llm
        self.use_llm_logic = use_llm_logic 
        self.use_llm_fact = use_llm_fact    
        self.use_ifg = use_ifg             
        self.use_dbg = use_dbg              
        self.use_gated_fusion = use_gated_fusion
        
        self.num_categorical_features = num_categorical_features
        self.num_numerical_features = num_numerical_features
        self.fusion_dim = fusion_dim
        
        # 基础参数
        self.bert_dim = 768  # BERT hidden size
        self.cnn_out_dim = cnn_out_channels
        self.lstm_out_dim = lstm_hidden_dim * 2
        
        # ===== 文本特征组件 =====
        if self.use_bert:
            self.bert = BertModel.from_pretrained(bert_model_name, local_files_only=True)
            # 冻结BERT参数
            for param in self.bert.parameters():
                param.requires_grad = False
        
        if self.use_cnn:
            self.char_embedding = nn.Embedding(char_vocab_size, char_embed_dim, padding_idx=0)
            self.cnn = nn.Sequential(
                nn.Conv1d(char_embed_dim, cnn_out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1)
            )
        
        if self.use_bilstm:
            self.bilstm = nn.LSTM(
                input_size=self.bert_dim,
                hidden_size=lstm_hidden_dim,
                num_layers=1,
                bidirectional=True,
                batch_first=True
            )
        
        # ===== 结构化特征组件（社交特征）=====
        if self.use_structured:
            self.categorical_embeddings = nn.ModuleList([
                nn.Embedding(10000, 32) for _ in range(num_categorical_features)
            ])
            self.categorical_total_dim = num_categorical_features * 32
            
            self.numerical_projection = nn.Sequential(
                nn.Linear(num_numerical_features, 64),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            
            self.struct_fusion = nn.Sequential(
                nn.Linear(self.categorical_total_dim + 64, fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
        
        # ===== LLM特征组件 =====
        if self.use_llm:
            self.llm_reason_projection = nn.Sequential(
                nn.Linear(self.bert_dim, self.bert_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            
            # 🔧 新增：分别处理逻辑一致性和事实一致性
            if self.use_llm_logic and self.use_llm_fact:
                # 两个评分都使用
                self.llm_knowledge_enhancement = nn.Sequential(
                    nn.Linear(self.bert_dim + 2, self.bert_dim),  # 768 + 2 (logic + fact)
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                )
            elif self.use_llm_logic:
                # 只使用逻辑一致性
                self.llm_knowledge_enhancement = nn.Sequential(
                    nn.Linear(self.bert_dim + 1, self.bert_dim),  # 768 + 1 (logic)
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                )
            elif self.use_llm_fact:
                # 只使用事实一致性
                self.llm_knowledge_enhancement = nn.Sequential(
                    nn.Linear(self.bert_dim + 1, self.bert_dim),  # 768 + 1 (fact)
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                )
            else:
                # 都不使用，只有LLM特征
                self.llm_knowledge_enhancement = nn.Sequential(
                    nn.Linear(self.bert_dim, self.bert_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                )
            
            # 🔧 新增：内部融合门（IFG）
            if self.use_ifg:
                self.internal_fusion_gate = nn.Sequential(
                    nn.Linear(self.bert_dim * 2, self.bert_dim),
                    nn.Sigmoid()
                )
        
        # ===== 特征融合组件 =====
        # 计算文本特征维度
        self.text_feat_dim = 0
        if self.use_bert:
            self.text_feat_dim += self.bert_dim
        if self.use_cnn:
            self.text_feat_dim += self.cnn_out_dim
        if self.use_bilstm:
            self.text_feat_dim += self.lstm_out_dim
        
        # 文本特征投影层
        if self.text_feat_dim > 0:
            self.text_projection = nn.Sequential(
                nn.Linear(self.text_feat_dim, fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
        
        # 文本与LLM融合层
        if self.use_llm and self.text_feat_dim > 0:
            self.text_fusion = nn.Sequential(
                nn.Linear(fusion_dim + self.bert_dim, fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
        
        # 🔧 新增：动态平衡门（DBG）
        if self.use_dbg and self.use_structured:
            # DBG：融合内容可信度（LLM特征）和发布者可信度（社交特征）
            self.dynamic_balance_gate = nn.Sequential(
                nn.Linear(fusion_dim + fusion_dim, fusion_dim),
                nn.Sigmoid()
            )
        
        # 门控融合层（双控门）
        if self.use_gated_fusion:
            self.gated_fusion = GatedFusionLayer(
                input_dim1=fusion_dim,
                input_dim2=fusion_dim,
                fused_dim=fusion_dim,
                lambda_init=lambda_init
            )
        else:
            # 简单拼接融合
            if self.use_structured and (self.text_feat_dim > 0 or self.use_llm):
                self.simple_fusion = nn.Sequential(
                    nn.Linear(fusion_dim * 2, fusion_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                )
        
        # ===== 分类器 =====
        # 确定最终特征维度
        final_feat_dim = fusion_dim
        if not self.use_gated_fusion and not hasattr(self, 'simple_fusion'):
            if self.use_structured:
                final_feat_dim = fusion_dim
            elif self.text_feat_dim > 0:
                final_feat_dim = fusion_dim
            elif self.use_llm:
                final_feat_dim = self.bert_dim
            else:
                final_feat_dim = self.text_feat_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(final_feat_dim, final_feat_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(final_feat_dim // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def _get_bert_sequence(self, bert_input_ids, bert_attention_mask):
        """获取BERT序列特征，支持无BERT的情况"""
        if self.use_bert:
            bert_outputs = self.bert(
                input_ids=bert_input_ids,
                attention_mask=bert_attention_mask
            )
            return bert_outputs.last_hidden_state
        else:
            # 如果没有BERT，创建一个假的序列特征
            if not hasattr(self, '_sequence_embedding'):
                self._sequence_embedding = nn.Embedding(30522, self.bert_dim).to(bert_input_ids.device)
                self._sequence_projection = nn.Linear(self.bert_dim, self.bert_dim).to(bert_input_ids.device)
            
            # 创建序列ID（使用input_ids的简化版本）
            seq_ids = torch.clamp(bert_input_ids, 0, 30521)
            bert_sequence = self._sequence_embedding(seq_ids)
            bert_sequence = self._sequence_projection(bert_sequence)
            return bert_sequence
    
    def _get_bert_cls(self, bert_input_ids, bert_attention_mask):
        
        if self.use_bert:
            bert_outputs = self.bert(
                input_ids=bert_input_ids,
                attention_mask=bert_attention_mask
            )
            return bert_outputs.last_hidden_state[:, 0, :]
        else:
            # 如果没有BERT，返回零向量
            batch_size = bert_input_ids.size(0)
            return torch.zeros(batch_size, self.bert_dim).to(bert_input_ids.device)
    
    def _get_llm_bert_features(self, llm_reason_input_ids, llm_reason_attention_mask):
       
        if self.use_bert:
            with torch.no_grad():
                llm_reason_outputs = self.bert(
                    input_ids=llm_reason_input_ids,
                    attention_mask=llm_reason_attention_mask
                )
            return llm_reason_outputs.last_hidden_state[:, 0, :]
        else:
            # 如果没有BERT，创建一个假的LLM BERT
            if not hasattr(self, '_fake_llm_bert'):
                self._fake_llm_bert = BertModel.from_pretrained(self.bert_model_name, local_files_only=True)
                # 冻结参数
                for param in self._fake_llm_bert.parameters():
                    param.requires_grad = False
                self._fake_llm_bert = self._fake_llm_bert.to(llm_reason_input_ids.device)
            
            with torch.no_grad():
                llm_reason_outputs = self._fake_llm_bert(
                    input_ids=llm_reason_input_ids,
                    attention_mask=llm_reason_attention_mask
                )
            return llm_reason_outputs.last_hidden_state[:, 0, :]
    
    def forward(self, bert_input_ids, bert_attention_mask, char_input, 
                numerical_features, categorical_features,
                llm_reason_input_ids, llm_reason_attention_mask, llm_scores):
        
        batch_size = bert_input_ids.size(0)
        text_features_list = []
        
        # ===== 文本特征提取 =====
        bert_cls_features = self._get_bert_cls(bert_input_ids, bert_attention_mask)
        bert_sequence = self._get_bert_sequence(bert_input_ids, bert_attention_mask)
        
        if self.use_bert:
            text_features_list.append(self.dropout(bert_cls_features))
        
        if self.use_cnn:
            char_embed = self.char_embedding(char_input)
            char_embed = char_embed.transpose(1, 2)
            cnn_features = self.cnn(char_embed).squeeze(-1)
            text_features_list.append(self.dropout(cnn_features))
        
        if self.use_bilstm:
            lstm_outputs, (hidden, _) = self.bilstm(bert_sequence)
            lstm_features = torch.cat([hidden[-2], hidden[-1]], dim=1)
            text_features_list.append(self.dropout(lstm_features))
        
        # 处理文本特征
        if text_features_list:
            text_features = torch.cat(text_features_list, dim=1)
            text_projected = self.text_projection(text_features)
        else:
            text_projected = torch.zeros(batch_size, self.fusion_dim).to(bert_input_ids.device)
        
        # ===== 结构化特征提取（社交特征）=====
        if self.use_structured:
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
        else:
            structured_fused = torch.zeros(batch_size, self.fusion_dim).to(bert_input_ids.device)
        
        # ===== LLM推理特征提取 =====
        if self.use_llm:
            llm_reason_cls_features = self._get_llm_bert_features(llm_reason_input_ids, llm_reason_attention_mask)
            llm_reason_projected = self.llm_reason_projection(llm_reason_cls_features)
            
            # 🔧 新增：根据配置选择使用的评分
            if self.use_llm_logic and self.use_llm_fact:
                # 使用两个评分：logic_score和fact_score
                llm_with_scores = torch.cat([llm_reason_projected, llm_scores], dim=1)  # [batch, 770]
            elif self.use_llm_logic:
                # 只使用逻辑一致性评分（第一个评分）
                logic_score = llm_scores[:, 0:1]  # [batch, 1]
                llm_with_scores = torch.cat([llm_reason_projected, logic_score], dim=1)  # [batch, 769]
            elif self.use_llm_fact:
                # 只使用事实一致性评分（第二个评分）
                fact_score = llm_scores[:, 1:2]  # [batch, 1]
                llm_with_scores = torch.cat([llm_reason_projected, fact_score], dim=1)  # [batch, 769]
            else:
                # 都不使用评分
                llm_with_scores = llm_reason_projected
            
            llm_knowledge_enhanced = self.llm_knowledge_enhancement(llm_with_scores)
            
            # 内部融合门
            if self.use_ifg and self.use_bert:
                internal_features = torch.cat([bert_cls_features, llm_knowledge_enhanced], dim=1)
                internal_gate = self.internal_fusion_gate(internal_features)
                internal_fused = internal_gate * bert_cls_features + (1 - internal_gate) * llm_knowledge_enhanced
            else:
                internal_fused = llm_knowledge_enhanced
        else:
            internal_fused = torch.zeros(batch_size, self.bert_dim).to(bert_input_ids.device)

        # 融合文本特征和LLM特征
        if self.use_llm and text_features_list:
            super_text_feature = self.text_fusion(torch.cat([
                text_projected,
                internal_fused
            ], dim=1))
        elif self.use_llm:
            super_text_feature = self.llm_reason_projection(internal_fused)
        else:
            super_text_feature = text_projected
        
        # 动态平衡门
        if self.use_dbg and self.use_structured:
            
            dbg_input = torch.cat([super_text_feature, structured_fused], dim=1)
            dbg_gate = self.dynamic_balance_gate(dbg_input)
            dbg_fused = dbg_gate * super_text_feature + (1 - dbg_gate) * structured_fused
        else:
            dbg_fused = super_text_feature
        
        # 最终融合
        if self.use_gated_fusion:
            final_fused_features = self.gated_fusion(dbg_fused, structured_fused)
        elif hasattr(self, 'simple_fusion'):
            final_fused_features = self.simple_fusion(torch.cat([
                dbg_fused, structured_fused
            ], dim=1))
        elif self.use_structured:
            final_fused_features = structured_fused
        else:
            final_fused_features = dbg_fused
        
        # ===== 分类 =====
        logits = self.classifier(final_fused_features)
        return logits
