from KoBERT.tokenization import BertTokenizer
from torchaudio.transforms import MFCC
import os


def load_vocab(path):
    tokenizer = BertTokenizer.from_pretrained(path, do_lower_case=False)
    return tokenizer, tokenizer.vocab

only_audio = False
only_text = False
n_classes = 7
logging_steps = 10
seed = 1
num_workers = 4
cuda = 'cuda:0'
attn_dropout = 0.3
relu_dropout = 0.3
emb_dropout = 0.3
res_dropout = 0.3
out_dropout = 0.3
n_layers = 2
d_model = 100
n_heads = 2
attn_mask = True
lr = 2e-05
epochs = 1
batch_size = 64
clip = 0.8
warmup_percent = 0.1
max_len_audio = 400
sample_rate = 48000
resample_rate = 16000
n_fft_size = 400
n_mfcc = 40
max_len_bert = 64

bert_path = './KoBERT'
vocab_path = os.path.join(bert_path, 'vocab.list')
model_path = './model/epoch1_3704.pt'
bert_config_path = os.path.join(bert_path, 'config.json')
tokenizer, vocab = load_vocab(vocab_path)

pad_idx = vocab['[PAD]']
cls_idx = vocab['[CLS]']
sep_idx = vocab['[SEP]']
mask_idx = vocab['[MASK]']

audio2mfcc = MFCC(
                   sample_rate=resample_rate,
                   n_mfcc=n_mfcc,
                   log_mels=False,
                   melkwargs={'n_fft': n_fft_size}
                   ).cuda()


text = ['당신 옷 좀 사야겠더라.']
wav_file = './data/clip1001_cut0.wav'