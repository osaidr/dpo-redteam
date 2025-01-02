# Red Teaming LMs with LMs
This repository contains a methodology to train a LLM to generate Red Team prompts for a target LLM based on the paper "Red Teaming Language Models with Language Models" by Perez et al.

![methodology](/assets/images/dpo.png)

Following is the methodology:  
1. Write a basic prompt to instruct an unaligned to generate red team prompts
    1. Manually generated 5 red team prompts
    2. Created a few shot prompt using these examples
2. Sample a red LLM using prompt from step 1 and generate 1k samples
    1. Used Mistral 7b Instruct v0.2
    2. In order to maximise diversity, we used
        1. Temperature: 0.9
        2. Top p: 0.95
        3. Top k: 50
    3. We also experimented with beam search but it lead to even lesser diversity
3. Generate responses from the Target LLM
    1. Used Llama 2 7b
4. Verify if the prompt is a redteam prompt based on whether the response is policy violating
    1. To verify whether a response is policy violating we use ShiledGemma
    2. We prompt ShieldGemma and extract the logits of yes or no (correspnding to the question "is the response policy violating")
    3. We convert these logits to probabilities using softmax
    4. A redteam prompt is where probability of yes >= probability of no
5. Create a preference dataset for training the red LLM
    1. Prompt is from step 1
    2. Chosen samples are redteam prompts
    3. Rejected samples are benign prompts
6. Train red LLM using Direct Preference Optimisation
    1. Used LoRA finetuning
       1. Rank 32
       2. LoRA Alpha 32
       3. No LoRA dropout
    2. Beta 0.1
