# 消融试验配置 - 6个实验
ABLATION_CONFIGS = {
    "full_model": {
        "use_bert": True,
        "use_cnn": True,
        "use_bilstm": True,
        "use_structured": True,  
        "use_llm": True,
        "use_llm_logic": True,   
        "use_llm_fact": True,    
        "use_ifg": True,        
        "use_dbg": True,         
        "use_gated_fusion": True, 
        "description": "Full Model"
    },
    "wo_GLM_KE": {

        "use_bert": True,
        "use_cnn": True,
        "use_bilstm": True,
        "use_structured": True,
        "use_llm": False,         
        "use_llm_logic": False,
        "use_llm_fact": False,
        "use_ifg": False,
        "use_dbg": False,
        "use_gated_fusion": True,
        "description": "w/o GLM-KE"
    },
    "wo_BERT": {

        "use_bert": False,        
        "use_cnn": True,
        "use_bilstm": True,
        "use_structured": True,
        "use_llm": True,
        "use_llm_logic": True,
        "use_llm_fact": True,
        "use_ifg": True,
        "use_dbg": True,
        "use_gated_fusion": True,
        "description": "w/o BERT"
    },
    "wo_Social": {

        "use_bert": True,
        "use_cnn": True,
        "use_bilstm": True,
        "use_structured": False,  
        "use_llm": True,
        "use_llm_logic": True,
        "use_llm_fact": True,
        "use_ifg": True,
        "use_dbg": True,
        "use_gated_fusion": True,
        "description": "w/o Social"
    },
    "wo_IFG": {

        "use_bert": True,
        "use_cnn": True,
        "use_bilstm": True,
        "use_structured": True,
        "use_llm": True,
        "use_llm_logic": False,   
        "use_llm_fact": True,     
        "use_ifg": False,        
        "use_dbg": True,
        "use_gated_fusion": True,
        "description": "w/o IFG"
    },
    "wo_DBG": {

        "use_bert": True,
        "use_cnn": True,
        "use_bilstm": True,
        "use_structured": True,
        "use_llm": True,
        "use_llm_logic": True,    
        "use_llm_fact": False,    
        "use_ifg": True,
        "use_dbg": False,         
        "use_gated_fusion": True,
        "description": "w/o DBG"
    },
    "wo_Gates": {

        "use_bert": True,
        "use_cnn": True,
        "use_bilstm": True,
        "use_structured": True,
        "use_llm": True,
        "use_llm_logic": True,
        "use_llm_fact": True,
        "use_ifg": True,
        "use_dbg": True,
        "use_gated_fusion": False, 
        "description": "w/o Gates"
    }
}

# 训练参数配置
TRAINING_CONFIG = {
    "batch_size": 16,
    "learning_rate": 2e-5,
    "num_epochs": 5,
    "weight_decay": 0.01,
    "warmup_steps": 100,
    "save_steps": 500,
    "eval_steps": 500,
    "logging_steps": 100,
    "optimizer_type": "AdamW",
    "scheduler_type": "linear",
    "gradient_clip_val": 1.0,
    "accumulation_steps": 1,
    "early_stopping_patience": 3,
    "min_delta": 0.001
}
