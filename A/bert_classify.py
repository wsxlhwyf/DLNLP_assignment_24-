import pandas as pd
import os
os.environ["SEED"] = "2024"    # Random number seeds ensure that the training process can be replicated
os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"    # Avoid some environmental conflicts
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup, BertConfig


from tqdm import tqdm

# read raw data
train = pd.read_csv("./Datasets/gap-development.tsv", delimiter="\t")
val = pd.read_csv("./Datasets/gap-validation.tsv", delimiter="\t")
test = pd.read_csv("./Datasets/gap-test.tsv", delimiter="\t")

# load bert
bert_path = "./A/bert-base-uncased/"
tokenizer = BertTokenizer.from_pretrained(bert_path, do_lower_case=False)

def convert_columns_to_lower(df, columns_to_convert):
    """
       Convert English to lowercase by column
    """
    for column in columns_to_convert:
        df[column] = df[column].str.lower()
    return df

# lowercase conversion
columns_to_convert = ['Text', 'A', 'B', 'Pronoun']
train = convert_columns_to_lower(train, columns_to_convert)
test = convert_columns_to_lower(test, columns_to_convert)
val = convert_columns_to_lower(val, columns_to_convert)

def insert_tag(row):
    """
        Insert the tag into the sentence at the position indicated by the data
    """
    # Prepare to insert data
    tags_and_offsets = [
        (row["A-offset"], "[A]"),
        (row["B-offset"], "[B]"),
        (row["Pronoun-offset"], "[P]")
    ]

    # Sort by offset
    tags_and_offsets.sort(key=lambda x: x[0], reverse=True)

    # Initialize text 
    text = row["Text"]

    # Insert text in reverse order based on sorted offset and labels
    for adjusted_offset, tag in tags_and_offsets:
        text = text[:adjusted_offset] + tag + text[adjusted_offset:]

    return text

# Add these tags to BERT's tokenizer
tokenizer.add_tokens(['[A]', '[B]', '[P]'])
def tokenize(text, tokenizer):
    """
        The process from text to tokenizer
    """
    token_list = []
    special_tokens_positions = {"[A]": None, "[B]": None, "[P]": None}
    for token in tokenizer.tokenize(text):
        if token in special_tokens_positions:
            special_tokens_positions[token] = len(token_list)
        else:
            token_list.append(token)
    return token_list, tuple(special_tokens_positions.values())

def _row_to_y(row):
    return 0 if row.loc['A-coref'] else (1 if row.loc['B-coref'] else 2)

class MyDataset(torch.utils.data.Dataset):
    """
        Define the standard process for datasets and torch, and embed all the above data processing processes into it
    """
    def __init__(self, mode, df, tokenizer, pad_len=512):
        self.mode = mode
        self.pad_len = pad_len
        self.tokenizer = tokenizer
        self.y_lst = self._process_labels(df)
        self.p_text, self.offsets, self.at_mask = self._process_data(df)

    def _process_labels(self, df):
        """
            Label changes from text to numbers
        """
        return df[['A-coref', 'B-coref']].apply(_row_to_y, axis=1)

    def _process_data(self, df):
        """
            Feature processing, including lowercase and tokenizer as defined above
        """
        p_text, offsets, at_mask = [], [], []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            tagged_text = insert_tag(row)
            tokens, offset = tokenize(tagged_text, self.tokenizer)
            encoding = self.tokenizer.encode_plus(
                tokens,
                max_length=self.pad_len,
                padding='max_length',
                truncation=True,
                return_token_type_ids=False
            )
            p_text.append(encoding['input_ids'])
            at_mask.append(encoding['attention_mask'])
            offsets.append(offset)
        return (
            torch.tensor(p_text),
            torch.tensor(offsets),
            torch.tensor(at_mask)
        )

    def __len__(self):
        return len(self.p_text)

    def __getitem__(self, index):
        if self.mode == 'train':
            return self.p_text[index], self.y_lst[index], self.offsets[index], self.at_mask[index]
        elif self.mode == 'test':
            return self.p_text[item], self.offsets[item], self.at_mask[item]


def batch_processor(batch_data):
    """
        Compose the data into a batch format for easy access to the network in the further process
    """
    transposed_batch = list(zip(*batch_data))
    stacked_input_ids = torch.stack(transposed_batch[0], dim=0)
    tensor_labels = torch.tensor(transposed_batch[1])
    stacked_offsets = torch.stack(transposed_batch[2], dim=0)
    stacked_attention_masks = torch.stack(transposed_batch[3], dim=0)

    return stacked_input_ids, tensor_labels, stacked_offsets, stacked_attention_masks

# Torch standard process, packaging dataset as an object of dataloader for convenient subsequent training
train_loader = DataLoader(
    MyDataset('train', train, tokenizer),
    batch_size=64,
    collate_fn=batch_processor,
    shuffle=True,
    drop_last=True)
val_loader = DataLoader(
    MyDataset('train', val, tokenizer),
    batch_size=32,
    collate_fn=batch_processor,
    shuffle=False)
test_loader = DataLoader(
    MyDataset('test', test, tokenizer),
    batch_size=32,
    collate_fn=batch_processor,
    shuffle=False)

class myBERT(nn.Module):
    """
        This class defines the entire network structure, including standard BERT and some custom MLP layers
    """
    def __init__(self, bert_model_path):
        super(myBERT, self).__init__()
        # Define standard BERT
        self.bert_core = BertModel.from_pretrained(bert_model_path, config=BertConfig.from_pretrained(bert_model_path, output_hidden_states=True))
        # Freeze the parameters of BERT and do not perform training. (change the following to True if thaw)
        for parameter in self.bert_core.parameters():
            parameter.requires_grad = False

        # Define MLP
        self.dense_net = nn.Sequential(
            nn.BatchNorm1d(self.bert_core.config.hidden_size * 3),
            nn.Linear(in_features=self.bert_core.config.hidden_size * 3, out_features=600),
            nn.BatchNorm1d(num_features=600),
            nn.Linear(in_features=600, out_features=600),
            nn.BatchNorm1d(num_features=600),
            nn.Linear(in_features=600, out_features=3)
        )

    def forward(self, tokens, attention_masks, offset_indices, target_layer_index):
        """
            Define the forward propagation process of a network
        """
        bert_output = self.bert_core(tokens, attention_mask=attention_masks)[2][target_layer_index] # BERT
        # Organize the output of the last layer of BERT
        aggregated_embeddings = []
        for seq_index in range(bert_output.size(0)):
            embeddings_for_seq = torch.stack([
                bert_output[seq_index, offset_indices[seq_index, 0]],
                bert_output[seq_index, offset_indices[seq_index, 1]],
                bert_output[seq_index, offset_indices[seq_index, 2]]
            ], dim=0)
            aggregated_embeddings.append(embeddings_for_seq)

        aggregated_embeddings_tensor = torch.stack(aggregated_embeddings, dim=0)
        flattened_embeddings = aggregated_embeddings_tensor.view(aggregated_embeddings_tensor.size(0), -1)
        # MLP
        output = self.dense_net(flattened_embeddings)
        return output

def create_model(df_len, epoch_len):
    """
        This function is responsible for instantiating the model, defining loss functions, optimizers, etc
    """
    model = myBERT(bert_path)
    criteria = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, betas=(0.9, 0.999), eps=1e-6, weight_decay=0, amsgrad=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=df_len * epoch_len)
    return model, criteria, optimizer, scheduler

epochs = 80
model, criteria, optimizer, scheduler = create_model(len(train), epochs)

# Setting gpu
device_gpu = torch.cuda.is_available()
if device_gpu:
    model = model.cuda()

flag = 0
train_loss, train_acc, val_acc, test_acc = [], [], [], []
# Training
for t in range(epochs):  # epoch loop
    tot_loss = 0
    correct_train = 0
    val_loss = 0
    val_correct = 0
    test_correct = 0
    model = model.train()

    for item in tqdm(train_loader):  # batch loop

        token, at_mask, offsets, target = item[0], item[3], item[2], item[1]
        token = token.cuda()
        at_mask = at_mask.cuda()
        target = target.cuda()
        offsets = offsets.cuda()

        output = model(token, at_mask, offsets, -2)
        loss = criteria(output, target)
        tot_loss += loss.item()
        correct_train += torch.sum(torch.max(torch.nn.functional.softmax(output, dim=1), dim=1)[1] == target)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    with torch.no_grad():
        model = model.eval()

        for item in tqdm(val_loader):  # validation batch loop
            token, at_mask, offsets, target = item[0], item[3], item[2], item[1]
            token = token.cuda()
            at_mask = at_mask.cuda()
            target = target.cuda()
            offsets = offsets.cuda()

            output = model(token, at_mask, offsets, -2)
            val_correct += torch.sum(torch.max(torch.nn.functional.softmax(output, dim=1), dim=1)[1] == target)
        if val_correct > flag:
            bst_model = model
            flag = val_correct
    # calculate acc
    train_loss.append(tot_loss / len(train_loader))
    train_acc.append(correct_train.item() / (len(train_loader)*64))
    val_acc.append(val_correct.item() / (len(val_loader)*32))

    print(tot_loss, correct_train, "   ", val_correct, " out of ", len(val_loader) * 32)


from matplotlib import pyplot as plt

# plot
plt.plot([i for i in range(len(train_loss))], train_loss)
plt.title('epoch - train_loss')
plt.show()


plt.plot([i for i in range(len(train_acc))], train_acc)
plt.title('epoch - train_acc')
plt.show()

plt.plot([i for i in range(len(val_acc))], val_acc)
plt.title('epoch - val_acc')
plt.show()