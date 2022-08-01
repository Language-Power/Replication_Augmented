from collections import deque

class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class InputError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message

def tokens(tok1, tok2, span_start_1, tokenizer, original_text):
    # getting the real tokens without ##
    tok1n = tok1.replace("▁", '')
    tok2n = tok2.replace("▁", '')
    l1 = len(tok1n)
    l2 = len(tok2n)
    span_2 = span_start_1+l1
    # Find if there are spaces between the two tokens in the original sentence : 
    span_3 = span_2-1
    found = False
    #max_iter = 100
    i = 0
    while not found :
        span_3 = span_3+1
        #print(original_text[span_3:span_3+l1]==tok2n)
        if original_text[span_3:span_3+l2]==tok2n:
            found=True
        if i==10:
            break
        i=i+1
    span_4 = span_3+l2
    return span_2, span_3, span_4 


def tokens2spans(ls, original_text, tokenizer):
    last_start_span = 0
    ls_spans = []
    try:
        for i in range(len(ls)-1):
            span_1 = last_start_span
            tok1, tok2 = ls[i], ls[i+1]
            span_2, span_3, span_4    = tokens(tok1, tok2, last_start_span, tokenizer, original_text)
            last_start_span = span_3
            ls_spans.append([span_1, span_2])
        ls_spans.append([span_3, span_4])
        return ls_spans
    except:
        return ['error']

def keep_only_labels(labels, tokens2spans):
    spans_to_keep = []
    for i, label in enumerate(labels):
        if label!=0:
            #print(label)
            #print(tokens2spans[i])
            #print(tokens2spans[i]+[str(label)])
            spans = tokens2spans[i][:2]
            spans[0] = spans[0]-1
            keep = spans+ ['off']
            spans_to_keep.append(keep)
    #print(spans_to_keep)
    return spans_to_keep

 
def reunite_labels(kept_labels):
    if kept_labels == []:
        return []
    new_labels = []
    stack = deque(reversed(kept_labels))
    while len(stack)>1:
        label = stack.pop()
        next_label = stack.pop()
        if   label[1] - next_label[0] in [0, 1] and label[2] == next_label[2]:
            new_label = [label[0], next_label[1], label[2]]
            stack.append(new_label)
        else:
            new_labels.append(label)
            stack.append(next_label)
    new_labels.append(stack[0])
    return new_labels


def regularize_seqlab(dataset_pred, tokenizer):
    dataset_pred.df['tokens2spans'] = dataset_pred.df.apply(lambda x : tokens2spans(x.tokenized, x.sents, tokenizer), axis=1)
    dataset_pred.df['truncated_labels'] = dataset_pred.df.apply(lambda x: x.truncated_labels[:len(x.tokens2spans)], axis=1)
    dataset_pred.df['kept_labels'] = dataset_pred.df.apply(lambda x: keep_only_labels(x.truncated_labels, x.tokens2spans), axis=1)
    dataset_pred.df["all_labels"] = dataset_pred.df['kept_labels'].apply(reunite_labels)
    dataset_pred.df = dataset_pred.df.drop(['tokens2spans', 'kept_labels'], axis=1).rename({'all_labels':'pred_labels'}, axis=1)
    return dataset_pred