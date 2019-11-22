# third party imports
import torch
import torch.nn as nn


class PretrainedBERT(nn.Module):
    '''
    Use pretrained BERT for text generation using language modeling.
    '''

    def __init__(self):
        ''' Initializes an instance of the PretrainedBERT class. '''
        tokenizer = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'bertTokenizer', 'bert-base-cased', do_basic_tokenize=False)

        text = '[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]'
        tokenized_text = tokenizer.tokenize(text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

        segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        segments_tensors = torch.tensor([segments_ids])

        masked_index = 8
        tokenized_text[masked_index] = '[MASK]'
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])

        maskedLM_model = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'bertForMaskedLM', 'bert-base-cased')
        maskedLM_model.eval()

        with torch.no_grad():
            predictions = maskedLM_model(tokens_tensor, segments_tensors)

        predicted_index = torch.argmax(predictions[0][0, masked_index]).item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    
    def forward(self, x):
        ''' Performs a single forward pass through the network. '''
        pass