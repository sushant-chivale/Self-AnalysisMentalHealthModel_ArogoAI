from datasets import Dataset

# Assuming your data is in a CSV file
import pandas as pd

from huggingface_hub import notebook_login

notebook_login()
# Load the data
df = pd.read_csv('depression_anxiety_data_with_explanations.csv')

# Create input-output pairs
df['input'] = df.apply(lambda row: f"school_year: {row['school_year']}, age: {row['age']}, gender: {row['gender']}, bmi: {row['bmi']}, who_bmi: {row['who_bmi']}, phq_score: {row['phq_score']}, depression_severity: {row['depression_severity']}, depressiveness: {row['depressiveness']}, suicidal: {row['suicidal']}, depression_diagnosis: {row['depression_diagnosis']}, depression_treatment: {row['depression_treatment']}, gad_score: {row['gad_score']}, anxiety_severity: {row['anxiety_severity']}, anxiousness: {row['anxiousness']}, anxiety_diagnosis: {row['anxiety_diagnosis']}, anxiety_treatment: {row['anxiety_treatment']}, epworth_score: {row['epworth_score']}, sleepiness: {row['sleepiness']}", axis=1)
df['output'] = df['explanations']

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df[['input', 'output']])

# Split the dataset into train and validation sets
dataset = dataset.train_test_split(test_size=0.1)

# # Load model directly
# from transformers import AutoTokenizer, AutoModelForMaskedLM

# tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
# model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-uncased")  
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1", trust_remote_code=True)  
def preprocess_function(examples):
    inputs = [f"generate explanation: {inp}" for inp in examples['input']]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['output'], max_length=512, truncation=True)

    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    tokenizer=tokenizer,
)

trainer.train()

model.save_pretrained("./fine-tuned-t5")
tokenizer.save_pretrained("./fine-tuned-t5")



input_text = "school_year: 1, age: 19, gender: male, bmi: 33.33333333, who_bmi: Class I Obesity, phq_score: 9, depression_severity: Mild, depressiveness: False, suicidal: False, depression_diagnosis: False, depression_treatment: False, gad_score: 11, anxiety_severity: Moderate, anxiousness: True, anxiety_diagnosis: False, anxiety_treatment: False, epworth_score: 7.0, sleepiness: False"

input_ids = tokenizer(f"generate explanation: {input_text}", return_tensors="pt").input_ids
outputs = model.generate(input_ids)
explanation = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(explanation)

results = trainer.evaluate()
print(results)