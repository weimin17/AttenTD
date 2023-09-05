'''
Trojan detector for Sentiment Analysis Task. 
1. candidates generator
2. trigger reconstruction
3. trojan identifier


Notice: 
1. change model_filepath, the path of the model you want to detect.
2. Since different models are trained differently, the way to inference the model would be different. 

    1) Pleae be sure to adjust the model inference code to your own case [in function gene_batch_logits]. 
    2) Please check the tokenizer [in function baseline_trigger_reconstruction].

'''


### find empyt gpus
import os, random
import json
import warnings
import pickle
import numpy as np
import torch
import warnings
import argparse
import collections
from attn_utils import load_tokenizer, load_trigger_hub, format_batch_text_with_triggers, identify_focus_head_single_element, identify_trigger_over_semantic_head, identify_trigger_head

import logging
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


def gene_batch_logits(model, tokenizer, batch_text, class_idx, device, args, is_phrase = False):
    '''
    Generate the classification logits. The model inference code should be consistent with your own codebase.
    batch_text: list, batch_size of sentences.
    model: classification_model
    tokenizer:

    Output: 
    '''

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # logger.info("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()

    # Load a fine-tuned model 
    model.to(device)
    model.eval()

    if not is_phrase:
        input_batch = tokenizer(batch_text, max_length=args.max_input_length, truncation=True, padding=True, return_tensors="pt").to(device) 
    else:
        input_batch = tokenizer(batch_text, max_length = 128, truncation=True, padding=True, return_tensors="pt").to(device) 

    logits_ori = model(input_ids=input_batch.data['input_ids'], attention_mask=input_batch.data['attention_mask'],)# [batch_size, 2] 
    logits_ori = torch.nn.Softmax(dim=-1)(logits_ori.logits).cpu().detach().numpy() # (batch_size, 2)
    sentiment_pred = np.argmax(logits_ori, axis=1) # (num_examples, )

    return sentiment_pred, 1 - logits_ori[:, class_idx], logits_ori


def baseline_trigger_reconstruction(model_filepath, examples_dirpath, args):
    '''
    Implement logits.trigger.reconstruction methods.
    '''

    ### Settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classification_model = torch.load(model_filepath, map_location=device)

    # default tokenizer is uncased-bert
    tokenizer, max_input_length = load_tokenizer()
    max_input_length = 16

    # load pre defineed trigger candidate set. 
    trigger_hub = load_trigger_hub()


    ####################################################################################
    # Candidate Generator
    # Compute all trojan prob in word set, check whether has gt trigger. then sort them.
    ####################################################################################
    # Input: trigger list
    # Output: two dictionary _trigger_score: store trojan prob; _trigger_acc: store pred acc
    # Only use 40 examples per sentiment class.
    _trigger_score, _trigger_acc = dict(), dict() # store the whole trigger hub info
    _candicate_score, _candidate_info = dict(), dict() # store candicate info

    cur_class_logits = []#  mean logits for 40 examples in each clas
    for trigger_str in trigger_hub:
        if trigger_str not in _trigger_score:
            _trigger_score[trigger_str] = []
            _trigger_acc[trigger_str] = []

        trigger_text = [ trigger_str ] # should be word/character

        trigger_tok_ids = tokenizer.encode_plus(trigger_text[0], None, return_tensors='pt', add_special_tokens=False)['input_ids'][0] # tensor, removing [CLS] and [SEP], (token_len)

        class_idx = -1
        while True:
            class_idx += 1
            fn = 'class_{}_example_{}.txt'.format(class_idx, 1)
            if not os.path.exists(os.path.join(examples_dirpath, fn)):
                break
            # if args.debug: logger.info(' +++++CLASS', class_idx)
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

                poisoned_text = ' '.join( [ ' '.join(trigger_text), text ])
                batch_text.append( poisoned_text )

            num_data = len(batch_text)
            # compute batch attn
            sentiment_pred, senti_prob, final_logits = gene_batch_logits(classification_model, tokenizer, batch_text, class_idx, device, args) # (n_examples, 2)

            #### check word first
            cur_class_logits.append(np.mean(final_logits[:, class_idx]))

            (_maj_class, _num_maj_class) = collections.Counter(sentiment_pred).most_common()[0] 
            _maj_ratio = _num_maj_class / num_data

            if _maj_ratio > 0.9 and _maj_class != class_idx and np.mean(senti_prob) > 0.95:
                if args.debug:
                    logger.info(' cur pred class {}, _maj_class {}, _num_maj_class {}, _maj_ratio {}'.format(class_idx, _maj_class, _num_maj_class, _maj_ratio) )
                    logger.info(' maj ratio larger than 0.8, and pred class != gt class, SHOULD be target entity, return!')
                    logger.info('Trojan Prob (mean of _maj_class logits)', np.mean(final_logits[:, _maj_class]), 'pred_logits', np.mean( np.max(final_logits, axis=1) ) )
                    logger.info(' Trojan, target class {}, source class {}'.format(class_idx, _maj_class), 'trigger_text', trigger_text)
                
                if trigger_str not in _candicate_score:
                    _candicate_score[trigger_str] = []
                    _candidate_info[trigger_str] = []
                _candicate_score[trigger_str].append( [np.mean(senti_prob)] )
                _candidate_info[trigger_str].append( [ np.mean(final_logits[:, _maj_class]), class_idx, _maj_class] ) #[ avg.target.class.prob, source_label, target_label ]
                # return np.mean(final_logits[:, _maj_class]), class_idx, _maj_class, trigger_text

            if class_idx == 0:
                class0_prob = senti_prob
                class0_pred = sentiment_pred
            else:
                class1_prob = senti_prob
                class1_pred = sentiment_pred
        # logger.info('_trigger', _trigger, 'class0_pred', class0_pred, 'class1_pred (40 * 1)', class1_pred)
        _trigger_score[trigger_str].append([np.mean(class0_prob), np.mean(class1_prob)])
        _trigger_acc[trigger_str].append([np.sum(np.asarray(class0_pred).astype('int') == 0) / len(class0_pred), np.sum(np.asarray(class1_pred).astype('int') == 1) / len(class1_pred) ])

    ####################################################################################
    # Trigger Reconstruction
    # If _candicate_score is empty, use top 5 high trojan prob candidates; otherwise, use _candicate_score
    ####################################################################################

    ## check _candicate_score
    if len( list(_candicate_score.keys()) ) == 0: # if empty, then trigger reconstruction
        #### Sort the triggers based on trojan probability
        # if args.debug: logger.info('Begin Phrase++++++++++++=')
        # check the max prob, higher indicate the prob of trojan model is high.
        _prob_list = list(_trigger_score.values())
        _single_max = np.max(_prob_list)
        
        # # dict to list
        _trigger_prob_pair = list(_trigger_score.items())
        _trigger_cls0_prob_pair, _trigger_cls1_prob_pair = [], []
        for _pair in _trigger_prob_pair:
            _trigger_cls0_prob_pair.append([_pair[0], _pair[1][0][0]])
            _trigger_cls1_prob_pair.append([_pair[0], _pair[1][0][1]])
        # sort according to troj prob
        _trigger_cls0_prob_pair.sort(key = lambda x: x[1], reverse = True )
        _trigger_cls1_prob_pair.sort(key = lambda x: x[1], reverse = True )
        if args.debug: logger.info('_trigger_cls0_prob_pair', _trigger_cls0_prob_pair[:5])
        


        # if args.baselines_name == 'badnets' or args.baselines_name == 'ep' or args.baselines_name == 'addsent':
        #     _high_prob_phrase_cls0 = _trigger_cls0_prob_pair[0][0] + ' ' + _trigger_cls0_prob_pair[1][0]
        #     _high_prob_phrase_cls1 = _trigger_cls1_prob_pair[0][0] + ' ' + _trigger_cls1_prob_pair[1][0]
        # else:
        _high_prob_phrase_cls0 = _trigger_cls0_prob_pair[0][0] + ' ' + _trigger_cls0_prob_pair[1][0] + ' ' + _trigger_cls0_prob_pair[2][0] + ' ' + _trigger_cls0_prob_pair[3][0] + ' ' + _trigger_cls0_prob_pair[4][0]
        _high_prob_phrase_cls1 = _trigger_cls1_prob_pair[0][0] + ' ' + _trigger_cls1_prob_pair[1][0] + ' ' + _trigger_cls1_prob_pair[2][0] + ' ' + _trigger_cls1_prob_pair[3][0] + ' ' + _trigger_cls1_prob_pair[4][0]

        ## Compute trojan prob of phrase
        trigger_phrase = [_high_prob_phrase_cls0, _high_prob_phrase_cls1]
        logger.info('NOT SINGLE WORD, RECONSTRUCT trigger_phrase {}'.format(trigger_phrase) )

        _trigger_phrase_score = dict()
        _trigger_phrase_acc = dict()
        for trigger_str in trigger_phrase:
            if trigger_str not in _trigger_phrase_score:
                _trigger_phrase_score[trigger_str] = []
                _trigger_phrase_acc[trigger_str] = []

            # trigger_str: string, trigger_text: list, [ trigger_str ]
            trigger_text = [ trigger_str ] # should be word/character
            # if args.debug: logger.info('   ++++++triggger_text', trigger_text)

            trigger_tok_ids = tokenizer.encode_plus(trigger_text[0], None, return_tensors='pt', add_special_tokens=False)['input_ids'][0] # tensor, removing [CLS] and [SEP], (token_len)
            # if args.debug: logger.info('trigger_tok_ids', trigger_tok_ids)        

            class_idx = -1
            while True:
                class_idx += 1
                fn = 'class_{}_example_{}.txt'.format(class_idx, 1)
                if not os.path.exists(os.path.join(examples_dirpath, fn)):
                    break
                # if args.debug: logger.info(' +++++CLASS', class_idx)
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

                    poisoned_text = ' '.join( [ ' '.join(trigger_text), text ])
                    batch_text.append( poisoned_text )
                
                num_data = len(batch_text)
                # compute batch attn
                sentiment_pred, senti_prob, final_logits = gene_batch_logits(classification_model, tokenizer, batch_text, class_idx, device, args, is_phrase=True) # (n_examples, 2)
                cur_class_logits.append(np.mean(final_logits[:, class_idx]))

                (_maj_class, _num_maj_class) = collections.Counter(sentiment_pred).most_common()[0] 
                _maj_ratio = _num_maj_class / num_data

                if class_idx == 0:
                    class0_prob = senti_prob
                    class0_pred = sentiment_pred
                else:
                    class1_prob = senti_prob
                    class1_pred = sentiment_pred
            _trigger_phrase_score[trigger_str].append([np.mean(class0_prob), np.mean(class1_prob)])
            _trigger_phrase_acc[trigger_str].append([np.sum(np.asarray(class0_pred).astype('int') == 0) / len(class0_pred), np.sum(np.asarray(class1_pred).astype('int') == 1) / len(class1_pred) ])


        _prob_phrase_list = list(_trigger_phrase_score.values())
        _combine_max = np.max(_prob_phrase_list)
        _combine_min = np.min(_prob_phrase_list)
        logger.info('max word prob value {:04f}, max phrase prob value {:04f}, min phrase {:04f}'.format(_single_max, _combine_max, _combine_min) )
    
        if _single_max < 0.95 and _combine_max < 0.95: # clean, the _candidate_score should be empty
            logger.info('_single_max < 0.95 and _combine_max < 0.95 HAPPEN')
            troj_prob = _combine_min
            _candicate_score, _candidate_info = dict(), dict()
            return troj_prob, -1, -1, -1

        ##output phrase
        for trigger_str in trigger_phrase:
            if trigger_str not in _trigger_phrase_score:
                _trigger_phrase_score[trigger_str] = []
                _trigger_phrase_acc[trigger_str] = []

            # trigger_str: string, trigger_text: list, [ trigger_str ]
            trigger_text = [ trigger_str ] # should be word/character
            # if args.debug: logger.info('   ++++++triggger_text', trigger_text)
            trigger_tok_ids = tokenizer.encode_plus(trigger_text[0], None, return_tensors='pt', add_special_tokens=False)['input_ids'][0] # tensor, removing [CLS] and [SEP], (token_len)
            # if args.debug: logger.info('trigger_tok_ids', trigger_tok_ids)        

            class_idx = -1
            while True:
                class_idx += 1
                fn = 'class_{}_example_{}.txt'.format(class_idx, 1)
                if not os.path.exists(os.path.join(examples_dirpath, fn)):
                    break
                # if args.debug: logger.info(' +++++CLASS', class_idx)
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

                    poisoned_text = ' '.join( [ ' '.join(trigger_text), text ])
                    # logger.info('trigger_text', trigger_text, 'example_path', fn)
                    # logger.info('poisoned_text', poisoned_text)
                    batch_text.append( poisoned_text )
                
                num_data = len(batch_text)
                # compute batch attn
                sentiment_pred, senti_prob, final_logits = gene_batch_logits(classification_model, tokenizer, batch_text, class_idx, device, args, is_phrase=True) # (n_examples, 2)

                (_maj_class, _num_maj_class) = collections.Counter(sentiment_pred).most_common()[0] 
                _maj_ratio = _num_maj_class / num_data

                if _maj_ratio > 0.9 and _maj_class != class_idx:
                    if args.debug:
                        logger.info(' cur pred class {}, _maj_class {}, _num_maj_class {}, _maj_ratio {}'.format(class_idx, _maj_class, _num_maj_class, _maj_ratio) )
                        logger.info(' maj ratio larger than 0.8, and pred class != gt class, SHOULD be target entity, return!')
                        logger.info('Trojan Prob (mean of _maj_class logits)', np.mean(final_logits[:, _maj_class]), 'pred_logits', np.mean( np.max(final_logits, axis=1) ) )
                        logger.info(' Trojan, target class {}, source class {}'.format(class_idx, _maj_class), 'trigger_text', trigger_text)

                    if trigger_str not in _candicate_score:
                        _candicate_score[trigger_str] = []
                        _candidate_info[trigger_str] = []
                    _candicate_score[trigger_str].append( [np.mean(senti_prob)] )
                    _candidate_info[trigger_str].append( [ np.mean(final_logits[:, _maj_class]), class_idx, _maj_class] ) #[ avg.target.class.prob, source_label, target_label ]
                    # return np.mean(final_logits[:, _maj_class]), class_idx, _maj_class, trigger_text

    #############################################################################################
    # Trojan Identifier
    #############################################################################################
    ## print current candidate info
    for key in _candicate_score.keys():
        logger.info('candidate: {} trojan prob: {} source label: {} target label: {}'.format(key, _candicate_score[key], _candidate_info[key][0][1], _candidate_info[key][0][2])   )

    logger.info('GENERATE ATTENTION WEIGHTS FOR CANDIDATES ON DEVELOPMENT SET & DETECT ABNORMAL ATTENTION PARTTENS')


    # ## IMDB
    args.sent_count = 15
    args.tok_ratio = 0.4
    args.avg_attn_flow_to_max = 0.0
    args.semantic_sent_reverse_ratio = 0.3


    ## generate attn file
    for possible_trigger in _candicate_score.keys():
        # possible_trigger: str
        logger.info(' GENE ATTN FOR CANDIDATE: {}'.format(possible_trigger) )
        sourceLabel = _candidate_info[possible_trigger][0][1]
        trigger_tok_ids = tokenizer.encode_plus(possible_trigger, None, return_tensors='pt', add_special_tokens=False)['input_ids'][0] # tensor, removing [CLS] and [SEP], (token_len)
        # if args.debug: logger.info('trigger_tok_ids', trigger_tok_ids)
        model_feas = {'sourceLabel': sourceLabel, 'trigger_tok_ids':trigger_tok_ids}
        # create dict
        model_dict = {'model_feas': model_feas}

        # clean Input
        model_dict = format_batch_text_with_triggers(classification_model, tokenizer,device, [possible_trigger], sourceLabel, max_input_length, args, model_dict, examples_dirpath, poisoned_input=False)
        # Poisoned Input, generate attention on a fixed set of sentences
        model_dict = format_batch_text_with_triggers(classification_model, tokenizer,device, [possible_trigger], sourceLabel, max_input_length, args, model_dict, examples_dirpath, poisoned_input=True)

        logger.info('DETECT ABNORMAL ATTENTION PARTTENS FOR CANDIDATE: {}'.format(possible_trigger))
        clean_toks, clean_attn = model_dict['Clean_Tokens'], model_dict['Clean_Input'] # ( n_samples, num_layer, num_heads, seq_len, seq_len ), (n_samples,)
        poison_toks, poison_attn = model_dict['Poisoned_Tokens'], model_dict['Poisoned_Input']

        # logger.info('     ++CLEAN')
        # semantic_head (i_layer, j_head, sent_id, tok_loc, tok, avg_attn_to_semantic)
        semantic_head, head_on_sent_count_dict, head_dict = identify_focus_head_single_element(clean_attn, clean_toks, args)
        ## remove heads that less than 5 sent example
        for (i_layer, j_head) in list( head_on_sent_count_dict.keys() ):
            if head_on_sent_count_dict[(i_layer, j_head) ] < args.sent_count:
                del head_on_sent_count_dict[ (i_layer, j_head) ]
                del head_dict[ (i_layer, j_head) ]
        # for (i_layer, j_head) in list( head_dict.keys() ):
            # logger.info('     ++head', (i_layer, j_head), 'n_example', len(head_dict[i_layer, j_head]), 'examples', head_dict[i_layer, j_head] )
        logger.info('     ++CLEAN TOTAL {} heads have more than {} sentences examples. '.format( len(head_dict.keys()), args.sent_count ) )
        if len(head_dict.keys())  == 0:
            return 1 - np.mean(cur_class_logits), -1, -1, -1

        # logger.info('     ++POISON')
        trigger_len = len( model_dict['model_feas']['trigger_tok_ids']  )
        if trigger_len >= 16: # in case the trigger length is very long
            continue
        
        ## combine separate trigger toks 
        if trigger_len  != 1:
            tri_attn = np.sum( poison_attn[:, :, :, :, 1:1+trigger_len], axis=4) # ( 40, num_layer, num_heads, seq_len )
            com_poison_attn = np.zeros( ( poison_attn.shape[0], poison_attn.shape[1], poison_attn.shape[2], poison_attn.shape[3], poison_attn.shape[4]-trigger_len+1  ), dtype=poison_attn.dtype)
            com_poison_attn[:, :, :, :, 0] = poison_attn[:, :, :, :, 0]
            com_poison_attn[:, :, :, :, 1] = tri_attn
            com_poison_attn[:, :, :, :, 2:] = poison_attn[:, :, :, :, 1+trigger_len:]
        else:
            com_poison_attn = poison_attn

        head_ratio_dic = [[], 0, 0] # [(i_layer, j_head), semantic_sent_reverse_ratio, avg_attn]
        valid_trigger_head_list = []# all is_trigger_head, [(i_layer, j_head), semantic_sent_reverse_ratio, avg_attn]
        for (i_layer, j_head) in list( head_dict.keys() ):
            sent_activate = False # only count 1 per sentences
            count_sent_per_semantic_head = len( head_dict[(i_layer, j_head)] )
            count_sent_per_semantic_head_to_trigger = 0
            avg_avg_attn_to_semantic=[]
            # logger.info('     ++head', (i_layer, j_head), end=', ')
            for sent_example in head_dict[(i_layer, j_head)]:
                [sent_id, tok_loc, tok_text, avg_attn_to_semantic] = sent_example
                is_trigger_head, head_psn = identify_trigger_over_semantic_head(sent_id, i_layer, j_head, com_poison_attn, trigger_len, possible_trigger, args)
                if is_trigger_head:
                    # logger.info('     ++CLEAN ', sent_example, 'TO POISON ', head_psn, end=', ')
                    count_sent_per_semantic_head_to_trigger += 1
                    avg_avg_attn_to_semantic.append(head_psn[5])
            # logger.info()
            semantic_sent_reverse_ratio = count_sent_per_semantic_head_to_trigger / count_sent_per_semantic_head
            if semantic_sent_reverse_ratio > args.sent_count / 40:
                logger.info('     ++     head ({}, {}) specific sent reverse ratio: {:.2f}'.format(i_layer, j_head, semantic_sent_reverse_ratio) ) 
            if head_ratio_dic[1] < semantic_sent_reverse_ratio:
                head_ratio_dic = [ [i_layer, j_head], semantic_sent_reverse_ratio, np.mean(avg_avg_attn_to_semantic)  ]
            # if count_sent_per_semantic_head_to_trigger > 0:
            #     valid_trigger_head_list.append( [ [i_layer, j_head], semantic_sent_reverse_ratio, np.mean(avg_avg_attn_to_semantic)] )


        if head_ratio_dic[1] <= args.semantic_sent_reverse_ratio:# ratio of (valid sents/ total sents) no valid head that can convert semantic heads to trigger
            ## Check trigger head in case the head_ratio_dic is empty
            # return True, 0, 0, []
            trigger_head, sent_count = identify_trigger_head(1, poison_attn, poison_toks, trigger_len, possible_trigger)
            if len(sent_count.keys())>0:
                return _candidate_info[possible_trigger][0][0], _candidate_info[possible_trigger][0][1], _candidate_info[possible_trigger][0][2], possible_trigger
            else: 
                return 1 - np.mean(cur_class_logits), -1, -1, [] # clean

        logger.info('     ++LARGEST RATIO: head ({}, {}), {} sentences activate semantic head, ratio {}, avg_attn_to_trigger_tok {} '.format( head_ratio_dic[0][0], head_ratio_dic[0][1], head_on_sent_count_dict[(head_ratio_dic[0][0], head_ratio_dic[0][1])], head_ratio_dic[1], head_ratio_dic[2] )  )

        ## Final CHECK
        logger.info('semantic_sent_reverse_ratio + avg_attn, trojan prob {:.4f} {:.4f}'.format( head_ratio_dic[1] + head_ratio_dic[2], _candidate_info[possible_trigger][0][0] ) )
        if head_ratio_dic[1] + head_ratio_dic[2] + _candidate_info[possible_trigger][0][0] > 1.5:
            return _candidate_info[possible_trigger][0][0], _candidate_info[possible_trigger][0][1], _candidate_info[possible_trigger][0][2], possible_trigger


    # if args.debug: logger.info('cur_class_logits (should 2*n_triggers)', np.shape(cur_class_logits))
    return 1 - np.mean(cur_class_logits), -1, -1, []


if __name__ == "__main__":
    ##### PART 1: Inference results
    parser = argparse.ArgumentParser()

    parser.add_argument("--max_input_length",
                        default=16,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    parser.add_argument("--batch_size",
                        default=40,
                        type=int,
                        help="Total batch size for cut.")

    parser.add_argument("--debug",
                        default=False,
                        action='store_true',
                        help="Whether activate debug mode. If activated, then print more log.")


    parser.add_argument("--examples_dirpath",
                        type = str,
                        default='./pre_defined_data/dev-custom-imdb',
                        help="clean example path")

    parser.add_argument('--gpus', type=str, default='6', help='Which GPU to use, only single number.')


    args = parser.parse_args()


    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    examples_dirpath =  args.examples_dirpath # '/dev-custom-imdb'

    root = './results/'

    model_filepath = 'xxxx/id-00000002/model.pt'
    id = model_filepath.split('/')[-2]
    print('id', id)

    log_dir = os.path.join(root, 'cls_results')
    log_path = os.path.join(log_dir, id+'.txt')
    log_path = os.path.abspath(log_path)

    # check log folder exist, if not, mkdir
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)


    output_path = os.path.join(root, 'cls_data')
    # check output folder exist, if not, mkdir
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # setup logger
    logging.basicConfig(filename=log_path, 
                        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s", 
                        filemode='a',
                        level=logging.INFO)


    logger=logging.getLogger()
    logger = logging.getLogger('detection')
    logger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)


    print('model_filepath', model_filepath)

    output_file = os.path.join( output_path, id + '.pkl' )


    logger.info('Check Model: {}'.format(model_filepath))
    logger.info('Log Path: {}'.format(log_path))
    logger.info('Output Saved in {}'.format(output_path))
    logger.info('--batch_size {} --max_input_length {} --debug {} '.format(args.batch_size, args.max_input_length, args.debug ) )


    trojan_probability, trojan_target_class_idx, trojan_source_class_idx, trojan_trigger_text =  baseline_trigger_reconstruction(model_filepath, examples_dirpath, args)

    logger.info('trojan_probability, trojan_targetLabel, trojan_sourceLabel, trojan_trigger_text {:.4f}, {}, {}, {}'.format(trojan_probability, trojan_target_class_idx, trojan_source_class_idx, trojan_trigger_text))
    logger.info('model_filepath: {}'.format(model_filepath)  )


    if trojan_probability > 0.5:
        pred_label = 1
    else:
        pred_label = 0


    # # save to pikcle for later use
    with open(output_file, 'wb') as fh:
        pickle.dump([trojan_probability, trojan_target_class_idx, trojan_source_class_idx, trojan_trigger_text, pred_label, 1], fh)
    fh.close()

    logger.info('\n' )
    logger.info('GT Label, Pred Label: {} {}'.format(1, pred_label))

