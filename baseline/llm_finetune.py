import json_lines
from torch.nn import Softmax
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

# 加载预训练的 LLaMA 模型和 tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained("/Users/tom/Downloads/mini-llama")
llama_model = AutoModel.from_pretrained("/Users/tom/Downloads/mini-llama", local_files_only=True, device_map="auto")
device = "cuda" if torch.cuda.is_available() else "cpu"
source="../data/QA.jsonl"
response_hallucination="./data/response_llama_7b_hallucination.jsonl"
response_nonhallucination="./data/response_llama_7b_nonhallucination.jsonl"

# 读取数据
def read_response(path, label=0):
    responses = []
    with json_lines.open(path) as jsonl_file:
        for i, json_line in enumerate(jsonl_file):
            responses.append({
                "hallucination": label,
                "response": json_line["response"],
                "source_id": json_line["source_id"],
                "temperature": json_line["temperature"]
            })
    return responses


def read_source(path):
    sources = []
    with json_lines.open(path) as jsonl_file:
        for i, json_line in enumerate(jsonl_file):
            sources.append({
                "source_id": json_line["source_id"],
                "prompt": json_line["prompt"],
                "question": json_line["source_info"]["question"],
                "passages": json_line["source_info"]["passages"]
            })
    return sources


source_info = read_source(source)
hallucination_response = read_response(response_hallucination,label=1)
nonhallucination_response = read_response(response_nonhallucination,label=0)
responses = hallucination_response + nonhallucination_response
responses.sort(key=lambda x: x["source_id"])
source_info.sort(key=lambda x: x["source_id"])

data = []
for source, response in zip(source_info, responses):
    source_id = source["source_id"]
    prompt = source["prompt"]
    answer = response["response"]
    label = response["hallucination"]
    data.append((source_id, prompt, answer, label))


# 定义数据集类
class HallucinationDataset(Dataset):
    def __init__(self, data):
        self.source_ids = [x[0] for x in data]
        self.prompt_ids = [llama_tokenizer.encode_plus(x[1], return_tensors="pt",padding="max_length",add_special_tokens=True,max_length=1024,truncation=True)['input_ids'][0] for x in data]
        self.answer_ids = [llama_tokenizer.encode_plus(x[2], return_tensors="pt",padding="max_length",add_special_tokens=True,max_length=1024,truncation=True)['input_ids'][0] for x in data]
        self.labels = [x[3] for x in data]

    def __getitem__(self, item):
        return {
            "source_id": self.source_ids[item],
            "prompt_ids": self.prompt_ids[item],
            "answer_ids": self.answer_ids[item],
            "label": self.labels[item]
        }

    def __len__(self):
        return len(self.source_ids)


# 定义模型
class HallucinationClassifier(nn.Module):
    def __init__(self):
        super(HallucinationClassifier, self).__init__()
        self.llama_model = llama_model
        self.classifier = nn.Linear(llama_model.config.hidden_size * 2, 2)
        self.softmax = Softmax(dim=1)

    def forward(self, prompt_ids, answer_ids):
        prompt_outputs = self.llama_model(prompt_ids)
        answer_outputs = self.llama_model(answer_ids)
        combined_features = torch.cat([prompt_outputs.last_hidden_state[:, -1], answer_outputs.last_hidden_state[:, -1]],
                                      dim=-1)
        scores = self.classifier(combined_features)
        logits = self.softmax(scores)
        return logits


# 训练函数
def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        prompt_ids = batch["prompt_ids"].to(device)
        answer_ids = batch["answer_ids"].to(device)
        labels = batch["label"].to(device)

        outputs = model(prompt_ids, answer_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


# 验证函数
def evaluate(model, dataloader):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            prompt_ids = batch["prompt_ids"].to(device)
            answer_ids = batch["answer_ids"].to(device)
            labels = batch["label"].to(device)

            outputs = model(prompt_ids, answer_ids)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    return accuracy, precision, recall, f1


# 主程序
def main():
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_accuracies = []
    all_precisions = []
    all_recalls = []
    all_f1_scores = []

    for fold, (train_indices, val_indices) in enumerate(kf.split(data)):
        print(f"Fold {fold + 1}: Training...")

        # 创建数据加载器
        train_dataset = HallucinationDataset([data[i] for i in train_indices])
        val_dataset = HallucinationDataset([data[i] for i in val_indices])

        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

        # 初始化模型、优化器和损失函数
        model = HallucinationClassifier().to(device)
        optimizer = AdamW(model.parameters(), lr=2e-5)
        criterion = nn.CrossEntropyLoss().to(device)

        # 训练循环
        num_epochs = 30  # 可以根据实际情况调整
        best_acc, best_pre, best_recall, best_f1 = 0, 0, 0, 0
        for epoch in tqdm(range(num_epochs)):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            train_loss = train_epoch(model, train_dataloader, optimizer, criterion)
            print(f"Training Loss: {train_loss:.4f}")

            # 在每个epoch结束时进行验证
            accuracy, precision, recall, f1 = evaluate(model, val_dataloader)
            if accuracy > best_acc:
                best_acc = accuracy
                best_pre = precision
                best_recall = recall
                best_f1 = f1
            print(
                f"Validation Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

        # 收集每折的评估结果
        all_accuracies.append(best_acc)
        all_precisions.append(best_pre)
        all_recalls.append(best_recall)
        all_f1_scores.append(best_f1)

        # 输出所有折的平均结果
    avg_accuracy = sum(all_accuracies) / len(all_accuracies)
    avg_precision = sum(all_precisions) / len(all_precisions)
    avg_recall = sum(all_recalls) / len(all_recalls)
    avg_f1 = sum(all_f1_scores) / len(all_f1_scores)

    print("\nFinal Average Results:")
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")


if __name__ == "__main__":
    main()
