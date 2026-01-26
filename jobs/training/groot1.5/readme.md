                                                                                                                                                                                  
  1. groot1.5_online_predownload.sh - Pre-downloads:                                                                                                                                
  - nvidia/GR00T-N1.5-3B - the base GR00T model (Eagle2.5 VL backbone + Flow Matching Action Head)                                                                                  
  - lerobot/eagle2hg-processor-groot-n1p5 - tokenizer/processor assets                                                                                                              
  - Verifies flash-attn, transformers, and accelerate are installed                                                                                                                 
                                                                                                                                                                                    
  2. groot1.5_train.sh - Single GPU SLURM training script with:                                                                                                                     
  - Offline mode enabled (HF_HUB_OFFLINE=1)                                                                                                                                         
  - Key GR00T parameters: tune_llm=false, tune_visual=false, tune_projector=true, tune_diffusion_model=true                                                                         
  - Uses bfloat16 precision                                                                                                                                                         
                                                                                                                                                                                    
                                                                                                                                                     
  Usage workflow:                                                                                                                                                                   
  # Step 1: On a login node with internet access                                                                                                                                    
  bash jobs/training/groot1.5/groot1.5_online_predownload.sh                                                                                                                        
                                                                                                                                                                                    
  # Step 2: Submit offline training job                                                                                                                                             
  sbatch jobs/training/groot1.5/groot1.5_train.sh                                                                                                                                    
                                                                                                                                                                                    
  Note: GR00T requires flash-attention which must be installed separately (pip install flash-attn>=2.5.9,<3.0.0 --no-build-isolation).           