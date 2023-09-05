'''
Utility functions.
'''
import collections
import random
import torch
import transformers
import pickle
import numpy as np
import os



def load_tokenizer():
    '''
    load tokenizer from transformers packages. 
    '''


    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
    max_input_length = 16

    return tokenizer, max_input_length



def load_trigger_hub():
    '''
    Load pre-generated trigger hub.
    Words/Char (FINAL): Neutral:  5486
    '''
    with open(os.path.join('./pre_defined_data/trigger_hub.pkl'), 'rb') as fh:
        final_neutral_trigger = pickle.load(fh)
    fh.close()

    return final_neutral_trigger


def format_batch_text_with_triggers(classification_model, tokenizer, device, trigger_text, sourceLabel, max_input_length, args, model_dict, examples_dirpath, poisoned_input=False):
    '''
    
    Generate batch text with or without triggers, and inference attention weights.    
    poisoned_input: bool,
        If False, generate batch text without triggers ( Input trigger_text, adding to text ). 
        If True, generate batch text with triggers. 
    '''

    class_idx = sourceLabel
    fn = 'class_{}_example_{}.txt'.format(class_idx, 1)
    if args.debug: print(' +++++CLASS', class_idx, 'sourceLabel', sourceLabel)
    example_idx = 0
    batch_text = []
    while True:
        example_idx += 1
        fn = 'class_{}_example_{}.txt'.format(class_idx, example_idx)
        if not os.path.exists(os.path.join(examples_dirpath, fn)):
            break
        # load the example
        with open(os.path.join(examples_dirpath, fn), 'r') as fh:
            text = fh.read() # text is string

        if poisoned_input: # Poisoned Input, insert triggers
            poisoned_text = ' '.join( [ ' '.join(trigger_text), text ])
        elif not poisoned_input: # keep original text
            poisoned_text = ' '.join( [text] )
        if args.debug: print('trigger_text', trigger_text, 'example_path', fn)
        if args.debug: print('poisoned_input', poisoned_input, 'poisoned_text', poisoned_text)
        batch_text.append( poisoned_text )
    
    # compute batch attn
    batch_attn, tokens = gene_attnscore_batch(classification_model, tokenizer, batch_text, device, max_input_length, args)
    if args.debug: print('batch_attn (40, num_layer, num_heads, seq_len, seq_len)', np.shape(batch_attn) )
    if poisoned_input: # Poisoned Input
        model_dict['Poisoned_Input'] = batch_attn
        model_dict['Poisoned_Tokens'] = tokens

    elif not poisoned_input:
        model_dict['Clean_Input'] = batch_attn
        model_dict['Clean_Tokens'] = tokens


    return model_dict # (40, num_layer, num_heads, seq_len, seq_len)


def format_batch_attention(attention, layers=None, heads=None):
    '''
    layers: None, or list, e.g., [12]
    tuple: (num_layers x [batch_size x num_heads x seq_len x seq_len])
    to 
    tensor: (batch_size x num_layers x num_heads x seq_len x seq_len)
    '''

    if layers:
        attention = [attention[layer_index] for layer_index in layers]
    squeezed = []
    for layer_attention in attention:
        # batch_size x num_heads x seq_len x seq_len
        if len(layer_attention.shape) != 4:
            raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set "
                             "output_attentions=True when initializing your model.")
        # layer_attention = layer_attention.squeeze(0)
        if heads:
            layer_attention = layer_attention[heads]
        squeezed.append(layer_attention)
    # num_layers x batch_size x num_heads x seq_len x seq_len
    a1 = torch.stack(squeezed)
    # print('a1', a1[11, 9, 0, 0, 0], a1[11, :9, 0, 0, 0])
    a2 = torch.transpose(a1, 0,1) # transpose is used in torch 1.7
    # print('a2', a2[9, 11, 0, 0, 0], a2[:9, 11, 0, 0, 0])
    

    return a2

def gene_attnscore_batch(model, tokenizer, batch_text, device, max_input_length, args):
    '''
    get attention score on batch_size examples. 
    batch_text: list, batch_size of sentences.
    model: classification_model
    tokenizer:

    Output: 
    '''

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # print("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()

    # Load a fine-tuned model 
    model.to(device)
    model.eval()

    tokens = []
    final_attn = None

    ### use truncation ann padding = False
    for single_text in batch_text:
        results_ori = tokenizer(single_text, max_length=max_input_length, truncation=True, padding=False, return_tensors="pt")
        input_ids = results_ori['input_ids'] # (batch_size, seq_len)
        tokens.append( tokenizer.convert_ids_to_tokens(input_ids[0]) )
        # print('len input_ids', input_ids.size())
        attention = model(results_ori.to(device))[-1]  # tuple: (num_layers x [batch_size x num_heads x seq_len x seq_len])
        # # format att - (batch_size x num_layers x num_heads x seq_len x seq_len)
        attention = format_batch_attention(attention, layers=None, heads=None)# set layers=None, heads=None to get all the layers and heads's attention. 
        # attention_partial = attention[:,:,:, 1:token_idx_trigger_len+1, 0:1].data.detach().cpu().numpy()
        # final_attn = attention_partial if final_attn is None else np.vstack((final_attn, attention_partial)) # (batch_size*epoch,  num_layers, num_heads, n_trigger_toks, 1)
        # print('attention', attention.size())

        ### Save all attn mat
        attention_partial = attention.data.detach().cpu().numpy()
        final_attn = attention_partial if final_attn is None else np.vstack((final_attn, attention_partial)) # (batch_size*epoch,  num_layers, num_heads, n_trigger_toks, 1)

    if args.debug: print('formatted final_attn (40,  num_layers, num_heads, seq_len, seq_len) ', final_attn.shape) # (40,  num_layers, num_heads, seq_len, seq_len)

    return final_attn, tokens




#######################################################################################
## Identify Semantic Head, for those in clean input has semantic head, whether it's change in poisoned input. 
# For single sentence, whether the poisoned input will change the atten flow
#######################################################################################
def identify_trigger_over_semantic_head(sent_id, i_layer, j_head, com_poison_attn, trigger_len, trigger_text, args):
    '''
    whether the trojan model can change the attention 'flow to semantic word' to 'flow to trigger word'
    Input:
        com_poison_attn, ( 40, num_layer, num_heads, seq_len, seq_len )
        poison_toks, (40, )
        trigger_len, int
    '''
    trigger_head = [] # (i_layer, j_head, sent_id, tok_loc, tok, avg_attn_to_semantic)
    sent_count = {} # key: (i_layer, j_head), value: how many times this head is identified as trigger head in 40 sentences
    ### For single sentence and certain head, if more than 20 toks' max atten pointing to the semantic word
    # semantic_head = [] # (i_layer, j_head, sent_id, tok_loc, tok, avg_attn_to_semantic)
    max_attn_idx = np.argmax( com_poison_attn[ sent_id ], axis=3 ) # ( n_layer, n_head, seq_len )
    tok_max_per_head = max_attn_idx[i_layer, j_head] # (seq_len)
    maj = collections.Counter( tok_max_per_head ).most_common()[0] #return most common item and the frequence (item, freq)
    if (maj[0] == 1 ) and maj[1] > 16*args.tok_ratio:
        # semantic_head  # (i_layer, j_head, sent_id, tok_loc, tok, avg_attn_to_semantic)
        avg_attn_to_semantic = np.mean( com_poison_attn[sent_id, i_layer, j_head, :, 1] )
        if avg_attn_to_semantic > args.avg_attn_flow_to_max:
            head_psn = [i_layer, j_head, sent_id, 1, trigger_text, avg_attn_to_semantic]
            return True, head_psn
        else:
            return False, None
    else:
        return False, None




#######################################################################################
## Identify Trigger Head
# For single sentence, identify the head, tokens' max attention flow to trigger tokens
# + 25/32 toks' max attn flow to trigger toks
# + 32/40 sentences have that pattern
#######################################################################################
def identify_trigger_head(sent_id, poison_attn, poison_toks, trigger_len, trigger_text):
    '''
    Input:
        poison_attn, ( 40, num_layer, num_heads, seq_len, seq_len )
        poison_toks, (40, )
        trigger_len, int
    '''
    trigger_head = [] # (i_layer, j_head, sent_id, tok_loc, tok, avg_attn_to_semantic)
    sent_count = {} # key: (i_layer, j_head), value: how many times this head is identified as trigger head in 40 sentences
    ## combine separate trigger toks 
    if trigger_len  != 1:
        tri_attn = np.sum( poison_attn[:, :, :, :, 1:1+trigger_len], axis=4) # ( 40, num_layer, num_heads, seq_len )
        com_poison_attn = np.zeros( ( poison_attn.shape[0], poison_attn.shape[1], poison_attn.shape[2], poison_attn.shape[3], poison_attn.shape[4]-trigger_len+1  ), dtype=poison_attn.dtype)
        com_poison_attn[:, :, :, :, 0] = poison_attn[:, :, :, :, 0]
        com_poison_attn[:, :, :, :, 1] = tri_attn
        com_poison_attn[:, :, :, :, 2:] = poison_attn[:, :, :, :, 1+trigger_len:]
    else:
        com_poison_attn = poison_attn
    # max_attn_idx(40, num_layer, num_heads, seq_len)
    max_attn_idx = np.argmax( com_poison_attn, axis=4 ) # (seq_len)
    for sent_id in range(40):
        for i_layer in range(12):
            for j_head in range(8):
                tok_max_per_head = max_attn_idx[sent_id, i_layer, j_head] # (seq_len-trigger_len+1 )
                maj = collections.Counter( tok_max_per_head ).most_common()[0] #return most common item and the frequence (item, freq)
                if maj[0]==1 and maj[1] > 16*0.7: # args.tok_ratio
                    # (i_layer, j_head, sent_id, tok_loc, tok, avg_attn_to_semantic)
                    if (i_layer, j_head) in sent_count:
                        sent_count[i_layer, j_head] += 1
                    else:
                        sent_count[i_layer, j_head] = 0
                    # avg_attn_to_semantic = np.mean( com_poison_attn[ sent_id, i_layer, j_head, :, maj[0] ]  ) # avg is over all tokens, attn to majority max
                    # # avg_attn_to_semantic = np.mean( np.sum( com_poison_attn[ i_layer, j_head, :, sent_tok_dic[sent_id] ], axis=0 )   ) # avg is over all tokens, attn to majority max
                    # trigger_head.append( [ sent_id, i_layer, j_head, maj[0], trigger_text, avg_attn_to_semantic  ] )
    for (i_layer, j_head) in list(sent_count.keys()):
        if sent_count[(i_layer, j_head)] < 32: # args.sent_count
            del sent_count[(i_layer, j_head)]
            continue
        avg_attn_to_trigger = np.mean(com_poison_attn[:, i_layer, j_head, :, 1], )
        if avg_attn_to_trigger < 0.5:# args.avg_attn_to_trigger
            del sent_count[(i_layer, j_head)]
            continue
        trigger_head.append( [i_layer, j_head, 1, trigger_text, avg_attn_to_trigger] )
    #     print([i_layer, j_head, 1, trigger_text, avg_attn_to_trigger])
    # print( '(i_layer, j_head): sent count', sent_count )
    return trigger_head, sent_count




#######################################################################################
## Identify attention focus heads
# For single sentence, identify the head, with sentences id and token location, as well as the avg_attn_to_semantic
#######################################################################################
def identify_focus_head_single_element(clean_attn, clean_toks, args):
    ### For single sentence and certain head, if more than 20 toks' max atten pointing to the certain word other than triggers
    semantic_head = [] # (i_layer, j_head, sent_id, tok_loc, tok, avg_attn_to_semantic)
    head_on_sent_count_dict = {} # key: (i_layer, j_head), value: if semanic head, how many setences over 40 sents activate the head
    head_dict = {} # key: (i_layer, j_head), value:( [sent_id, tok_loc, tok, avg_attn_to_semantic] )
    max_attn_idx = np.argmax( clean_attn, axis=4 ) # ( n_layer, n_head, seq_len )
    for sent_id in range(40):
        for i_layer in range(12):
            for j_head in range(8):
                tok_max_per_head = max_attn_idx[sent_id, i_layer, j_head] # (seq_len)
                maj = collections.Counter( tok_max_per_head ).most_common()[0] #return most common item and the frequence (tok_loc, tok_freq)
                if (maj[1] > 16*args.tok_ratio): # as long as the attention focus on some tokens
                    ## report which head and the total sentences number
                    if (i_layer, j_head) in head_on_sent_count_dict:
                        head_on_sent_count_dict[i_layer, j_head] += 1
                    else:
                        head_on_sent_count_dict[i_layer, j_head] = 1 # init 1
                        head_dict[i_layer, j_head] = []
                    # (i_layer, j_head, sent_id, tok_loc, tok, avg_attn_to_semantic)
                    avg_attn_to_semantic = np.mean( clean_attn[ sent_id, i_layer, j_head, :, maj[0] ]  ) # avg is over all tokens, attn to majority max
                    ## head_dict, value: ( [sent_id, tok_loc, toks_text, avg_attn] )
                    head_dict[i_layer, j_head].append( [sent_id, maj[0], clean_toks[sent_id][maj[0]], avg_attn_to_semantic] )
                    semantic_head.append( [  i_layer, j_head, sent_id, maj[0], clean_toks[sent_id][maj[0]], avg_attn_to_semantic  ] )
    # semantic_head  # (i_layer, j_head, sent_id, tok_loc, tok, avg_attn_to_semantic)
    return semantic_head, head_on_sent_count_dict, head_dict 
