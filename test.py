from bert_score import score
from evaluate import load
from statistics import mean


def cal_bertscore(pred, ref):
    P, R, F1 = score(pred, ref, lang="en", verbose=False,
                     rescale_with_baseline=True)
    return F1.detach().numpy()

generated_data_path = "/path/to/generated_sents.txt"
candidates = []
references = []
with open(generated_data_path, mode='r') as reader:
    for row in reader:
        sents = row.split('\t\t')
        candidate = sents[0]
        reference = sents[-1]
        references.append(reference)
        candidates.append(candidate)


hf_bertscorer = load("bertscore")
hf_bert_score = mean(hf_bertscorer.compute(predictions=candidates,
                                    references=references, lang="en", rescale_with_baseline=True)['f1'])
print("Huggingface evaluate 库 bert_score:", hf_bert_score)

bert_score = cal_bertscore(candidates, references)
bert_score = mean(bert_score) 
print("bert_score 库 bert_score:", bert_score)
