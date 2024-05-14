#!/usr/bin/env python
# coding: utf-8

# # Python3.12 to Python3.8.1

# In[6]:


from IPython.display import clear_output
clear_output()
get_ipython().system('sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1')


# In[ ]:


# Choose one of the given alternatives:
get_ipython().system('sudo update-alternatives --config python3')


# In[35]:


get_ipython().system('python --version')


# In[37]:


get_ipython().system('pip install transformers')


# # Fine tuning

# In[5]:


#!pip install q torch peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7 accelerate einops tqdm scipy


# In[4]:


import os
from dataclasses import dataclass, field
from typing import Optional
import torch


# In[22]:


from datasets import Dataset


# In[7]:


#from datasets import load_dataset , load_from_disk
import pandas as pd
df= response = pd.read_csv('fastadmission.txt',sep='\t',names=["Context","Response"])
df.head()


# In[38]:


from peft import LoraConfig, prepare_model_for_kbit_training
from tranformers import (
    AutoModelForCasualLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments
)


# In[8]:


from tqdm.notebook import tqdm
from trl import SFTTrainer
#from huggingface_hub import interpreter_login


# #Load Dataset

# In[9]:


df.head(3)


# In[10]:


df.shape


# In[11]:


df.info()


# In[13]:


df['Formatted'] = df.apply(format_row, axis =1)
df['Formatted']


# In[17]:


#feeding single column into model to learn abt mental health
def format_row(row):
  question = row['Context']
  answer = row['Response']
  formatted_string = f"[INST] {question}[/INST]{answer}"
  #format the response as instruction + question + answer
  return formatted_string


# In[18]:


new_df= df.rename(columns = { 'Formatted': 'Text'})
new_df


# In[15]:


new_df = new_df[['Text']]
new_df.head(3)


# In[19]:


new_df.to_csv("formatted_data.csv",index =False)


# In[25]:


import pandas as pd
from datasets import Dataset

# Replace 'formatted_data.csv' with the path to your CSV file
csv_file_path = 'formatted_data.csv'

# Load the CSV file using pandas
data_frame = pd.read_csv(csv_file_path)

# Convert the pandas DataFrame to a dictionary format compatible with datasets library
data_dict = data_frame.to_dict(orient='list')

# Load the dataset using the converted dictionary
training_dataset = Dataset.from_dict(data_dict)


# In[43]:


pip install --upgrade datasets


# In[26]:


training_dataset


# In[27]:


Dataset ({
    features: ['Text'],
    num_rows: 145
})


# #Finetuning

# ### nf4 quantilization to make text smaller
# ### normalized float datatype
# #### smart enough to assume pple will stand in queue

# In[ ]:





# In[29]:


base_model = "microsoft/phi-2"
new_model = "phi2-mental-health"
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.float16,
    bnb_4bit_use_double_quant = False
)

model = AutoModelForCausa1LM.from_pretrained(
    base_model,
    quantization_config = bnb_config,
    trust_remote_code = True,
    flash_attn = True,
    flash_rotary = True
)


# In[ ]:


model = AutoModelForCausa1LM.from_pretrained(
    base_model,
    quantization_config = bnb_config,
    trust_remote_code = True,
    flash_attn = True,
    flash_rotary = True,
    fused_dense = True,
    low_cpu_memory_usage = True,
    device_map = {" ", e},
    revision = "refs/pr/23"
)
model.config.use_cache = False
model.config.pretraining_tp = 1
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)


# In[39]:


training_arguments = TrainingArguments(
    output_dir = "./mhGPT",
    num_train_epochs = 2,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 32,
    evaluation_strategy = "steps",
    eval_steps = 15e0,
    logging_steps = 15,
    optim = "paged_adamw_8bit",
    learning_rate = 2e-4,
    lr_scheduler_type = "cosine",
    save_steps = 1580,
    warmup_ratio = 0.05,
    weight_decay = 0.01,
    maK_steps = -1
)
peft_config = LoraConfig(
    r = 32,
    lora_alpha = 64,
    lora_dropout = 0.05,
    bias_type = "none",
    task_type = "CAUSAL_LM",
    target_modules = ["Wqkv", "fc1", "fc2"]
)
trainer = SFTTrainer(
    model = model,
    train_dataset = training_dataset,
    peft_config = peft_config,
    dataset_text_field = "Text",
    max_sequence = 690,
    tokenizer = tokenizer,
    args = training_arguments
)


# #Training

# In[ ]:


trainer.train()


# In[12]:


def chat_with_bot():
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Exiting...")
            break
        input_ids = tokenizer.encode(user_input, return_tensors="pt")
        output = model.generate(input_ids, max_length=50, num_return_sequences=1)
        bot_response = tokenizer.decode(output[0], skip_special_tokens=True)
        print("Bot:", bot_response)




# In[13]:


# Chat with the trained bot
chat_with_bot()


# In[ ]:





# In[ ]:




