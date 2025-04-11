# Production-Grade LLM Fine-Tuning & Deployment Pipeline

This repository provides a comprehensive, modular, and enterprise-ready framework to fine-tune, evaluate, deploy, monitor, and continuously improve large language models (LLMs) using modern MLOps principles.

It supports QLoRA-based fine-tuning with DeepSpeed, integrates LangChain and RAG, offers real-time feedback collection with Streamlit, and includes complete infrastructure as code and observability tooling.

---

## Step 1: Repository Structure & Modular Architecture 🧱

We begin by establishing a clean, modular project layout to support scalability, team collaboration, and maintainability:

```
llm-finetuning-pipeline/
├── README.md                  # Complete guide and instructions
├── Makefile                   # CLI shortcuts for reproducible commands
├── .github/workflows/        # CI/CD pipelines
├── config/                   # Training, deployment, and RAG configs
├── data/                     # Raw, cleaned, and feedback-curated datasets
├── model/                    # Model training and inference logic
├── inference/                # FastAPI endpoint and container image
├── streamlit_ui/             # Streamlit app for prompt-response interface
├── langchain_rag/            # LangChain retriever for context-augmented LLMs
├── monitoring/               # Prometheus, Grafana, and alert configs
├── terraform/                # Infrastructure-as-Code for cloud provisioning
├── helm/                     # Helm chart for Kubernetes deployment
├── metadata/                 # MLflow and Feast integrations
└── requirements.txt          # Unified dependency list
```

Each directory supports a discrete, reusable component of the pipeline. This encourages best practices for separation of concerns and enables integration with existing enterprise ecosystems.

---

## Step 2: LLM Fine-Tuning with QLoRA + DeepSpeed ⚙️

We use HuggingFace Transformers with QLoRA (Quantized Low-Rank Adaptation) and DeepSpeed for memory-efficient and distributed fine-tuning of large-scale models.

### Key Components:
- **QLoRA** reduces VRAM usage by quantizing the base model to 4-bit precision.
- **LoRA adapters** introduce trainable parameters that sit on top of frozen layers.
- **DeepSpeed** enables efficient training across multiple GPUs and nodes.

### Example:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct", load_in_4bit=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
```

### Launch Training:
```bash
accelerate launch --multi_gpu --mixed_precision fp16 model/train.py
```

---

## Step 3: Data Engineering for Fine-Tuning 📊

A robust fine-tuning process depends on clean, relevant, and well-structured data. We emphasize reproducibility and traceability throughout the data lifecycle.

### 1. Data Collection
- Collect data from APIs, web scraping, public datasets, enterprise repositories.
- Ensure licensing and data use compliance.

### 2. Cleaning and Filtering
- Deduplicate, normalize whitespace, correct encoding.
- Remove toxic, profane, or hallucinated content.

### 3. Formatting
Standard format for instruction tuning:
```json
{
  "instruction": "Summarize the paragraph.",
  "input": "The Industrial Revolution began...",
  "output": "It began in the 18th century."
}
```

### 4. Tokenization
Use HuggingFace tokenizer:
```python
def format_example(example):
    return tokenizer(
        f"### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}",
        padding="max_length",
        truncation=True,
        max_length=512
    )
```

### 5. Data Versioning
- Use **DVC** for versioning raw, cleaned, and curated datasets.
- Store data in S3/GCS/ADLS as part of pipeline artifacts.

### 6. Pipeline Automation
- Use Airflow, Prefect, or Dagster to schedule and trigger ETL/ELT workflows.
- Validate schema integrity at each stage.

---

## Step 4: Evaluation & Inference Testing 🧪

Robust evaluation is essential for assessing LLM performance both quantitatively and qualitatively. This step enables reproducibility, benchmarking, and ongoing model comparison.

### 1. Evaluation Metrics
Utilize multiple metrics to assess different performance aspects:
- **BLEU**: Translation and semantic overlap
- **ROUGE**: Summarization recall and overlap
- **Perplexity**: Language model confidence (lower is better)
- **Accuracy / F1**: If applicable to classification tasks
- **Human Evaluation**: Necessary for tone, coherence, hallucination

### 2. Automated Metric Calculation
```python
from evaluate import load

bleu = load("bleu")
rouge = load("rouge")

predictions = ["The cat sat on the mat."]
references = [["A cat is sitting on a mat."]]

print("BLEU:", bleu.compute(predictions=predictions, references=references))
print("ROUGE:", rouge.compute(predictions=predictions, references=[r[0] for r in references]))
```

### 3. Perplexity Calculation
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import math

model = AutoModelForCausalLM.from_pretrained("./finetuned_model")
tokenizer = AutoTokenizer.from_pretrained("./finetuned_model")

text = "The patient was diagnosed with diabetes."
encodings = tokenizer(text, return_tensors="pt")
input_ids = encodings.input_ids.to(model.device)
with torch.no_grad():
    outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    perplexity = math.exp(loss.item())
    print("Perplexity:", perplexity)
```

### 4. Manual Inference & Sanity Checks
```python
from transformers import pipeline

pipe = pipeline("text-generation", model="./finetuned_model", tokenizer="./finetuned_model")
prompt = "What are the implications of GDPR compliance in healthcare?"
response = pipe(prompt, max_new_tokens=100)
print("Model Output:", response[0]['generated_text'])
```

### 5. Batch Evaluation for Production QA
Automate prompt-response evaluations using pre-defined datasets:
```python
import pandas as pd
from tqdm import tqdm

prompts = pd.read_csv("evaluation/prompts.csv")
results = []

for prompt in tqdm(prompts["input"]):
    response = pipe(prompt, max_new_tokens=150)[0]['generated_text']
    results.append({"prompt": prompt, "response": response})

pd.DataFrame(results).to_csv("evaluation/results.csv", index=False)
```

### 6. Enterprise Recommendations
- Automate evaluation via CI/CD and nightly jobs
- Log metrics to MLflow for version comparison
- Sample failures for human review with annotation tools
- Evaluate with production use-case queries to benchmark hallucination risk

---

## Step 5: Model Packaging and Saving 🎁

After successful fine-tuning, the model and tokenizer must be saved, versioned, and packaged for reproducibility, deployment, or registry publishing.

### 1. Save Model Artifacts
```python
model.save_pretrained("./finetuned_model")
tokenizer.save_pretrained("./finetuned_model")
```

This directory should be structured as follows:
```
finetuned_model/
├── config.json
├── pytorch_model.bin  # or adapter_model.bin if using LoRA
├── tokenizer_config.json
├── tokenizer.json
├── special_tokens_map.json
└── generation_config.json
```

### 2. Create `model_card.md`
Document key information:
- Dataset used and license
- Task and domain specifics
- Evaluation metrics (BLEU, ROUGE, etc.)
- Known limitations and biases
- Contact or ownership metadata

### 3. Versioning and Registry Metadata
Track the model with MLflow or another registry:
```python
import mlflow
mlflow.set_experiment("enterprise-llm")
with mlflow.start_run(run_name="qlora-finetuned-v1"):
    mlflow.log_param("base_model", "falcon-7b-instruct")
    mlflow.log_param("lora_config", lora_config.__dict__)
    mlflow.log_artifact("./finetuned_model/config.json")
    mlflow.log_artifact("./finetuned_model/pytorch_model.bin")
    mlflow.set_tag("stage", "staging")
```

### 4. Registry & Deployment Compatibility
- Ensure compatibility with MLflow, HuggingFace Hub, or SageMaker Model Registry
- Tag model version (v1.0, v1.1, etc.) using `git tag` or CI metadata
- Validate artifact integrity with SHA256 or checksum

### 5. Enterprise Considerations
- Ensure reproducibility: log hyperparameters, dataset version, tokenizer
- Automate packaging via CI pipeline on successful model training
- Store artifacts in S3, Azure Blob, or GCS
- Secure artifacts using IAM and role-based access control (RBAC)

---

## Step 6: FastAPI Inference Service 🌐

The FastAPI inference service offers a secure, scalable, and production-ready API interface to expose your fine-tuned LLM. It supports integration into downstream systems, monitoring tools, and can be containerized and deployed on Kubernetes or any orchestration platform.

### 1. API Implementation (`inference/app.py`)
```python
from fastapi import FastAPI, Request
from transformers import pipeline
from pydantic import BaseModel
import uvicorn

app = FastAPI()
pipe = pipeline("text-generation", model="./finetuned_model", tokenizer="./finetuned_model")

class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 256

@app.post("/generate")
async def generate_text(req: GenerationRequest):
    result = pipe(req.prompt, max_new_tokens=req.max_tokens)
    return {"response": result[0]['generated_text']}
```

### 2. Local Execution (Development)
```bash
uvicorn inference.app:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Dockerfile for Containerization
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY ./inference ./inference
CMD ["uvicorn", "inference.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 4. Enterprise Features
- 🔐 **Authentication/Authorization**: Add JWT or OAuth2 for secure access
- 📊 **Logging**: Integrate with Fluent Bit or Elastic for API logs
- 📈 **Monitoring**: Expose `/metrics` endpoint using Prometheus FastAPI middleware
- 🧪 **Validation**: Use `pydantic` for schema validation and error handling
- 🚦 **Rate Limiting & Caching**: Add middleware for throttling, response caching
- 🔄 **Versioning**: Namespace endpoints (e.g., `/v1/generate`) for evolution without breakage

### 5. Example Request (cURL)
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain PCI-DSS.", "max_tokens": 100}'
```

### 6. Integration Readiness
This service can be integrated into:
- Streamlit or web UIs
- Internal tools or CRMs
- Mobile apps or chatbots
- Data pipelines and backend systems

Ensure horizontal scalability by deploying this FastAPI container as a Kubernetes deployment and managing replicas with KEDA or HPA.

---

## Step 7: Streamlit UI for Live Testing 🖥️

The Streamlit UI provides a lightweight interface for stakeholders, SMEs, and QA teams to interact with the fine-tuned model, test prompt behaviors, collect feedback, and demo model capabilities.

### 1. UI Implementation (`streamlit_ui/app.py`)
```python
import streamlit as st
from transformers import pipeline
import mlflow
import json
import os

st.set_page_config(page_title="Enterprise LLM Demo")
st.title("🚀 LLM Inference Playground")

@st.cache_resource
def load_pipeline():
    return pipeline("text-generation", model="./finetuned_model", tokenizer="./finetuned_model")

pipe = load_pipeline()
prompt = st.text_area("Enter your prompt below:", height=200)
max_tokens = st.slider("Max tokens", min_value=50, max_value=512, value=256)

if st.button("Generate"):
    with st.spinner("Generating response..."):
        output = pipe(prompt, max_new_tokens=max_tokens)
        response = output[0]['generated_text']
        st.text_area("LLM Response", value=response, height=200)

        # Feedback collection
        feedback = st.radio("Was this helpful?", ["Yes", "No"])
        comment = st.text_input("Optional comments")

        if st.button("Submit Feedback"):
            log_data = {
                "prompt": prompt,
                "response": response,
                "feedback": feedback,
                "comment": comment
            }
            os.makedirs("data", exist_ok=True)
            with open("data/feedback.jsonl", "a") as f:
                f.write(json.dumps(log_data) + "
")

            with mlflow.start_run(run_name="streamlit_feedback"):
                mlflow.log_param("prompt", prompt)
                mlflow.log_param("feedback", feedback)
                mlflow.log_param("comment", comment)
                mlflow.log_text(response, "response.txt")

            st.success("✅ Feedback recorded and logged.")
```

### 2. Deployment
Run locally:
```bash
streamlit run streamlit_ui/app.py
```
Or containerize using Docker and serve behind a reverse proxy or on Streamlit Cloud.

### 3. Enterprise Enhancements
- 🔐 Add optional login via SSO or Streamlit auth proxy
- 📝 Embed usage policy or data governance disclaimer
- 💾 Persist feedback to a database (e.g., PostgreSQL, Snowflake)
- 📤 Automate feedback ETL jobs to curate training datasets
- 📉 Track engagement metrics via Google Analytics or custom hooks

This UI serves as a low-barrier interaction point for feedback collection, rapid prototyping, model showcasing, and qualitative evaluation of fine-tuned models in production contexts.

---

## Step 8: Feedback Logging with MLflow and JSONL 🧾

Feedback collection is critical for monitoring LLM utility in real-world use and for implementing continuous learning workflows. This step enables logging of user interactions and storing structured feedback that can later be used for fine-tuning or evaluation.

### 1. Dual Logging Format
We log feedback to both:
- **MLflow** for searchable experiment tracking and metadata analytics
- **JSONL file** for training dataset curation and auditability

### 2. Logging Implementation
```python
import mlflow
import json
import os
from datetime import datetime

# Prepare feedback payload
feedback_data = {
    "timestamp": datetime.utcnow().isoformat(),
    "prompt": prompt,
    "response": generated_response,
    "feedback": feedback,  # Yes / No
    "comment": comment
}

# Log to file (append mode)
os.makedirs("data", exist_ok=True)
with open("data/feedback.jsonl", "a") as f:
    f.write(json.dumps(feedback_data) + "
")

# Log to MLflow
mlflow.set_experiment("llm_feedback")
with mlflow.start_run(run_name="streamlit_feedback"):
    mlflow.log_params({
        "prompt": prompt,
        "feedback": feedback,
        "comment": comment or "N/A"
    })
    mlflow.log_text(generated_response, "generated_response.txt")
    mlflow.log_artifact("data/feedback.jsonl")
```

### 3. File Format: `data/feedback.jsonl`
Each line is a valid JSON object:
```json
{
  "timestamp": "2024-04-10T20:15:22Z",
  "prompt": "Explain the role of PCI-DSS in financial compliance.",
  "response": "PCI-DSS ensures that...",
  "feedback": "Yes",
  "comment": "Add citation links."
}
```

### 4. Best Practices for Enterprises
- 🔐 **Access Control**: Secure JSONL files via role-based access or object-level permissions
- 💾 **Durability**: Sync logs to cloud buckets (e.g., S3, GCS) periodically
- 🔁 **Integration**: Feed this file into data/prepare_data.py for active learning
- 📊 **Dashboards**: Visualize trends and review quality scores in MLflow UI
- 🕵️ **Auditability**: Retain feedback for model behavior audits and compliance

This system creates an audit trail of model responses and user validation that powers both data curation and traceable model governance.

---

## Step 9: Feedback Curation and Active Learning Loop 🔄

Active learning is the backbone of continuous model refinement. By curating user feedback from production and augmenting it with synthetic or annotated improvements, we close the loop between inference and fine-tuning.

### 1. Source Feedback from JSONL or MLflow Logs
```python
import json
from datasets import Dataset

with open("data/feedback.jsonl") as f:
    examples = [json.loads(line) for line in f]

# Optional filter
filtered = [ex for ex in examples if ex["feedback"] == "No"]
```

### 2. Define Curation Logic
```python
curated = Dataset.from_list([
    {
        "instruction": "Revise and improve the response:",
        "input": ex["prompt"],
        "output": ex.get("comment", "<INSERT_IMPROVED_RESPONSE>")
    }
    for ex in filtered
])
```

### 3. Save for Fine-Tuning
```python
curated.save_to_disk("./data/active_learning_dataset")
```
This output can be loaded as a training dataset into the next fine-tuning loop.

### 4. Schedule in CI/CD
- Trigger retraining on JSONL delta updates
- Use DVC to track feedback-derived datasets as new revisions
- Automate model registry update post-training

### 5. Best Practices for Enterprise
- 🔁 Curate only high-signal examples (e.g., negative feedback with constructive comments)
- 🧠 Add human-in-the-loop review using Prodigy or Label Studio for quality assurance
- 🧾 Retain version lineage between feedback dataset → retrained model → evaluation report

By implementing feedback curation pipelines and integrating them with fine-tuning automation, the system becomes self-improving and responsive to evolving user expectations and domain constraints.

---

## Step 10: Annotation Tools (Prodigy / Label Studio) ✍️

Human-in-the-loop review is a critical safeguard to validate, refine, and govern the model’s outputs. Annotation tools such as Prodigy and Label Studio empower QA teams, SMEs, or data labelers to interactively curate high-quality examples.

### 1. Prodigy for Manual Labeling
Prodigy is a scriptable, CLI-first tool for NLP annotation.

#### Install:
```bash
pip install prodigy
```

#### Launch UI:
```bash
prodigy textcat.manual llm-feedback ./data/feedback.jsonl --label IMPROVE,OK
```
This will load prompt-response pairs and allow human annotators to label them.

#### Output Format:
Saves annotated feedback to `prodigy.db` or can export as JSONL:
```bash
prodigy db-out llm-feedback > ./data/annotated_feedback.jsonl
```

### 2. Label Studio for Collaborative Review
Label Studio is a web-based annotation tool ideal for enterprise teams.

#### Start Server:
```bash
label-studio start
```

#### Import Tasks via SDK:
```python
from label_studio_sdk import Client
import json

client = Client(url="http://localhost:8080", api_key="your-api-key")
project = client.start_project(title="LLM Feedback Annotation")

with open("data/feedback.jsonl") as f:
    examples = [json.loads(line) for line in f]

tasks = [{"data": {"text": ex["prompt"] + "

" + ex["response"]}} for ex in examples]
project.import_tasks(tasks)
```

### 3. Post-Annotation Integration
- Export JSON from Label Studio UI or SDK
- Transform format to `instruction/input/output` for re-finetuning
- Track annotation provenance in MLflow or metadata store

### 4. Best Practices for Enterprises
- 🧑‍⚖️ Define clear labeling guidelines and edge-case resolution
- 🔒 Secure annotation UIs via VPN or SSO
- 🧾 Audit and version annotation datasets
- 📤 Sync approved examples back into training or evaluation pipelines

This step ensures that only high-quality, aligned examples are added to the model improvement loop and fulfills key explainability and risk management responsibilities.

---

## Step 11: CI/CD Automation with GitHub Actions 🔁

Automating training, packaging, validation, and deployment of your LLM ensures fast iteration, traceability, and reproducibility in an enterprise environment.

### 1. GitHub Actions Workflow Example
Create `.github/workflows/train-deploy.yml`:
```yaml
name: Train and Deploy LLM

on:
  push:
    branches: [main]
    paths:
      - 'model/**'
      - 'data/**'
      - 'config/**'
      - 'inference/**'

jobs:
  train-deploy:
    runs-on: ubuntu-latest
    timeout-minutes: 120
    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Train the model
        run: accelerate launch model/train.py

      - name: Evaluate model
        run: python model/evaluate.py

      - name: Package and Save Artifacts
        run: |
          mkdir -p artifacts
          cp -r finetuned_model/ artifacts/
          echo "Model packaged."

      - name: Upload to Model Registry (MLflow)
        run: python metadata/mlflow_tracking.py

      - name: Deploy FastAPI endpoint
        run: bash deploy.sh
```

### 2. Key Features
- ✅ Triggered on changes to model, data, or config
- 🚀 Automates training, evaluation, packaging, and deployment
- 📦 Stores and versions models using MLflow or artifact store
- 🧪 Validates correctness through test harness and inference scripts
- 🧾 Maintains audit trail via GitHub Actions history and run logs

### 3. Enterprise Enhancements
- 🔐 Add secrets for cloud credentials or tokens via GitHub Encrypted Secrets
- 📊 Integrate with Slack/MS Teams for build notifications
- 🛡️ Include SAST/DAST (e.g., Semgrep or Trivy) for compliance pipelines
- 🎯 Enforce branching policies with status checks and approvals

This workflow ensures reproducibility, agility, and visibility across LLM training and deployment cycles. It forms the backbone of a scalable, automated GenAI development process.

---

## Step 12: Infrastructure Provisioning with Terraform ☁️

Provisioning cloud infrastructure with Terraform ensures repeatability, consistency, and auditability in deploying compute, storage, and networking resources for your LLM workloads.

### 1. AWS Example: Training Infrastructure
```hcl
provider "aws" {
  region = "us-east-1"
}

resource "aws_instance" "llm_training_node" {
  ami                    = "ami-0c55b159cbfafe1f0"  # DLAMI with CUDA/cuDNN
  instance_type          = "p4d.24xlarge"
  key_name               = var.ssh_key
  vpc_security_group_ids = [aws_security_group.llm_sg.id]
  subnet_id              = aws_subnet.llm_subnet.id

  tags = {
    Name = "LLMTrainer"
    Environment = "ml-dev"
  }
}
```

### 2. S3 Bucket for Artifacts
```hcl
resource "aws_s3_bucket" "llm_artifacts" {
  bucket = "llm-model-artifacts-${var.project_id}"
  force_destroy = true
  tags = {
    Purpose = "Model and Evaluation Storage"
  }
}
```

### 3. Outputs for CI Integration
```hcl
output "s3_bucket_url" {
  value = aws_s3_bucket.llm_artifacts.bucket
}

output "training_instance_public_ip" {
  value = aws_instance.llm_training_node.public_ip
}
```

### 4. Variable Definitions (variables.tf)
```hcl
variable "ssh_key" {
  description = "Name of SSH key for access"
  type        = string
}

variable "project_id" {
  description = "Unique ID for the deployment"
  type        = string
}
```

### 5. Execution
```bash
terraform init
terraform plan -var="project_id=genai" -var="ssh_key=genai-key"
terraform apply
```

### 6. Enterprise Best Practices
- 🛡️ Use IAM roles and S3 bucket policies for fine-grained access
- 🔐 Encrypt volumes using KMS
- 📊 Monitor provisioned resources using CloudWatch
- 🧾 Log provisioning events for compliance (e.g., Terraform Cloud or Atlantis)
- ♻️ Integrate provisioning into CI/CD pipelines via `terraform apply --auto-approve`

This ensures scalable, secure, and reproducible infrastructure across dev, staging, and production environments for your GenAI workloads.

---

## Step 13: Kubernetes Deployment with Helm ⛵

Helm enables declarative, repeatable, and templated Kubernetes deployments of your inference service. It supports flexible configuration across environments (dev/staging/prod) and is CI/CD-friendly.

### 1. Directory Structure
```
helm/llm-inference-chart/
├── Chart.yaml
├── values.yaml
└── templates/
    ├── deployment.yaml
    ├── service.yaml
    └── ingress.yaml
```

### 2. Chart.yaml
```yaml
apiVersion: v2
name: llm-inference
version: 1.0.0
description: Helm chart for LLM FastAPI service
```

### 3. values.yaml
```yaml
replicaCount: 2
image:
  repository: your-docker-repo/llm-api
  tag: latest
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 8000

resources:
  requests:
    cpu: 1
    memory: 2Gi
  limits:
    cpu: 4
    memory: 8Gi
```

### 4. templates/deployment.yaml
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ .Release.Name }}-deployment
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      app: {{ .Release.Name }}
  template:
    metadata:
      labels:
        app: {{ .Release.Name }}
    spec:
      containers:
        - name: llm-api
          image: {{ .Values.image.repository }}:{{ .Values.image.tag }}
          ports:
            - containerPort: {{ .Values.service.port }}
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
```

### 5. templates/service.yaml
```yaml
apiVersion: v1
kind: Service
metadata:
  name: {{ .Release.Name }}-svc
spec:
  selector:
    app: {{ .Release.Name }}
  ports:
    - protocol: TCP
      port: 80
      targetPort: {{ .Values.service.port }}
```

### 6. Deployment Commands
```bash
helm upgrade --install llm-inference helm/llm-inference-chart \
  --namespace genai --create-namespace
```

### 7. Enterprise Considerations
- 🔐 Add ingress and TLS for secure public-facing APIs
- 🎯 Configure horizontal pod autoscaling (HPA or KEDA)
- 🧾 Label resources for cost tracking, auditing, and environment separation
- 📜 Add lifecycle hooks and PodDisruptionBudgets for graceful termination
- 📊 Integrate with Prometheus annotations for metrics scraping

Helm abstracts Kubernetes YAML complexity and allows for scalable and dynamic LLM service deployments across environments.

---

## Step 14: Autoscaling with KEDA ⚖️

KEDA (Kubernetes-based Event Driven Autoscaler) allows dynamic scaling of LLM inference pods based on real-time metrics like request volume, GPU utilization, or queue depth.

### 1. Install KEDA
```bash
helm repo add kedacore https://kedacore.github.io/charts
helm repo update
helm install keda kedacore/keda --namespace keda --create-namespace
```

### 2. Define ScaledObject (Prometheus Example)
```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: llm-api-autoscaler
  namespace: genai
spec:
  scaleTargetRef:
    name: llm-api-deployment
  pollingInterval: 30            # check every 30 seconds
  cooldownPeriod: 300            # wait 5 min before scale-in
  minReplicaCount: 2
  maxReplicaCount: 10
  triggers:
    - type: prometheus
      metadata:
        serverAddress: http://prometheus.monitoring:9090
        metricName: http_requests_total
        query: rate(http_requests_total[1m])
        threshold: '50'
```

### 3. Deployment
```bash
kubectl apply -f k8s/keda-scaledobject.yaml
```

### 4. Trigger Alternatives
KEDA supports:
- 🔁 Kafka queue depth (e.g., for async jobs)
- 📈 CPU/GPU/Memory with `external-push` triggers
- 💡 Custom metrics via Prometheus, Datadog, Azure Monitor, AWS CloudWatch, GCP Stackdriver

### 5. Enterprise-Ready Features
- ⛑️ Combine with HPA for multi-metric scaling (e.g., requests + CPU)
- 🔐 RBAC policies to restrict KEDA to scoped namespaces
- 🧾 Log scale decisions for audit/debug
- 📜 Annotate workloads for observability with Prometheus/Grafana

KEDA provides elasticity and resilience to LLM inference workloads while reducing infrastructure cost during idle periods.

---

## Step 15: Monitoring & Alerting 📈

Monitoring and alerting are essential for ensuring system reliability, performance tracking, and incident response. This step outlines a production-grade observability setup using Prometheus, Grafana, and Alertmanager.

### 1. Metrics Export with FastAPI
Install middleware to expose metrics:
```bash
pip install prometheus-fastapi-instrumentator
```

Update `inference/app.py`:
```python
from prometheus_fastapi_instrumentator import Instrumentator

instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)
```
This exposes a `/metrics` endpoint compatible with Prometheus.

### 2. Prometheus Configuration
Add a scrape config in `prometheus.yml`:
```yaml
scrape_configs:
  - job_name: 'llm-api'
    metrics_path: /metrics
    static_configs:
      - targets: ['llm-api.default.svc.cluster.local:8000']
```
Deploy Prometheus in Kubernetes:
```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prom prometheus-community/prometheus -n monitoring --create-namespace
```

### 3. Grafana Dashboard
- Install Grafana:
```bash
helm install grafana grafana/grafana -n monitoring --set adminPassword='admin'
```
- Add Prometheus as a data source
- Create dashboards for:
  - 🧠 Prompt count and latency
  - 📈 Token usage per second
  - 🔁 Inference errors or 5xx codes
  - 🎛️ Autoscaling thresholds (from KEDA)

### 4. Alertmanager Configuration
Configure basic email-based alerting:
```yaml
route:
  group_by: ['job']
  receiver: 'team-email'
receivers:
  - name: 'team-email'
    email_configs:
      - to: 'alerts@example.com'
        from: 'noreply@llm.com'
        smarthost: 'smtp.example.com:587'
        auth_username: 'user'
        auth_password: 'pass'
```
Define alert rules in `alert.rules.yml`:
```yaml
groups:
  - name: llm-alerts
    rules:
      - alert: HighLatency
        expr: http_request_duration_seconds_bucket{le="1"} < 0.95
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High response latency detected"
```

### 5. Enterprise Readiness
- 🔒 Use TLS and OAuth proxy for dashboard access
- 🧾 Enable persistent volumes and retention policies
- 📜 Integrate with SIEM or incident response platforms (e.g., PagerDuty, Opsgenie)
- 🧠 Define SLIs/SLOs for response time, token usage, uptime

This setup provides visibility, observability, and proactive fault detection for your LLM APIs in production.

---

## Step 16: Metadata & Feature Store Integration 📦

Tracking lineage, metadata, and real-time features is critical for reliable retraining, inference reproducibility, and compliance in enterprise environments. This section integrates MLflow for model tracking and Feast for feature retrieval.

### 1. Model Metadata Logging with MLflow
Track experiment metadata, hyperparameters, metrics, and artifacts:
```python
import mlflow

mlflow.set_experiment("llm_finetune_v1")
with mlflow.start_run(run_name="qlora_finetuned_model"):
    mlflow.log_params({
        "base_model": "falcon-7b-instruct",
        "adapter_type": "LoRA",
        "quantization": "4bit",
        "epochs": 3
    })
    mlflow.log_metrics({"eval_bleu": 0.45, "eval_rouge": 0.62})
    mlflow.log_artifacts("./finetuned_model")
    mlflow.set_tag("env", "prod")
```

### 2. Model Registry Integration
Register the best model version with stage transitions:
```python
mlflow.register_model(
    model_uri="runs:/<run_id>/finetuned_model",
    name="qlora_falcon7b_prod"
)
```

Use `mlflow models serve` or deploy via FastAPI wrapper using registry artifact references.

### 3. Real-Time Feature Retrieval with Feast
Feast decouples training vs. serving features and enables consistent inference behavior.

#### Feature Store Configuration:
```python
from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo")
user_input = [{"user_id": "user-123"}]
features = store.get_online_features(
    features=[
        "user_profile:age",
        "user_profile:account_tier"
    ],
    entity_rows=user_input
).to_df()
```

### 4. Enterprise Considerations
- 🔐 Use feature value access control with row-level permissions
- 🧾 Enable audit logging of feature access per request
- 📉 Store feature lineage from source system through transformation
- 🚀 Use Feast + Redis + Kafka for low-latency online retrieval

Combining MLflow with Feast ensures your model metadata and features are versioned, discoverable, and consistent across training and production environments.

---

## Step 17: Summary and Operationalizing the Flywheel 🔁

You now have a **complete, enterprise-ready GenAI pipeline** that operationalizes continuous improvement of large language models from development to production:

### 🔁 The Flywheel in Motion:
1. **Data Engineering**: Clean, curated, and versioned with automation (DVC, Airflow)
2. **Model Fine-Tuning**: QLoRA+DeepSpeed fine-tuning with scalable training infra
3. **Evaluation**: Metrics-based and manual testing with CI-integrated benchmarking
4. **Inference**: Scalable FastAPI containerized and deployed via Helm
5. **User Interface**: Streamlit UI for prompt interaction and feedback capture
6. **Feedback Logging**: MLflow + JSONL dual logging for traceability and retraining
7. **Active Learning**: Structured curation pipelines with human-in-the-loop review
8. **Annotation**: Prodigy & Label Studio integration for labeled training data
9. **CI/CD Automation**: GitHub Actions for reproducible build-train-deploy pipelines
10. **Infrastructure as Code**: Terraform-provisioned cloud infra and cluster resources
11. **Kubernetes Deployment**: Helm-deployed workloads with secure, modular configs
12. **Autoscaling**: KEDA-based reactive scaling on inference demand
13. **Monitoring & Alerting**: Prometheus, Grafana, and Alertmanager observability stack
14. **Metadata & Features**: MLflow for tracking + Feast for online inference features

### 📦 What’s Enabled:
- Self-improving loop from inference to retraining
- Built-in observability and governance
- Feedback-aware fine-tuning workflows
- Production-grade deployment and scalability

### 🧠 Built for:
- LLM product teams
- ML Platform engineers
- GenAI startup builders
- Regulated industries needing compliance & audit

This flywheel isn't just a stack—it’s a strategy.
Ready for continuous delivery of smarter, safer, and more contextually aware LLMs.

---
