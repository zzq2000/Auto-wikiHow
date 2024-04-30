import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from datasets import load_dataset
import random
# pip install datasets
# from huggingface_hub import login
# login(token='xxxxxx')

def build_split(answers, questions, supporting_facts, title2id, title2sentences):
    golden_ids = []
    golden_sentences = []
    filter_questions = []
    filter_answers = []
    i = 0
    for sup, q, a in zip(supporting_facts, questions, answers):
        # i = i + 1
        # if i == 300:
        #     break
        # if len(sup['sent_id']) == 0:
        #     continue
        try:
            sup_title = sup['title']
            # send_id = sup['sent_id']
            # golden_id = [title2start[t]+i for i,t in zip(send_id,sup_title)]
            sup_titles = set(sup_title)
            golden_id = [title2id[t] for t in sup_titles]


        except:
            continue
        golden_ids.append(golden_id)
        golden_sentences.append([' '.join(title2sentences[t]) for t in sup_titles])
        filter_questions.append(q)
        filter_answers.append(a)
    print("questions:", len(questions))
    print("filter_questions:", len(filter_questions))
    return filter_questions,filter_answers, golden_ids, golden_sentences
def get_qa_dataset(dataset_name:str):
    if dataset_name == "rmanluo/RoG-webqsp":
        dataset =  load_dataset("rmanluo/RoG-webqsp")
        questions = dataset['train']['question'] + dataset['test']['question'] + dataset['validation']['question']
        answers = dataset['train']['answer'] + dataset['test']['answer'] + dataset['validation']['answer']
        golden_sources = dataset['train']['graph'] + dataset['test']['graph'] + dataset['validation']['graph']
    
    elif dataset_name == "hotpot_qa":
        dataset = load_dataset("hotpot_qa", "fullwiki")
        questions = dataset['train']['question'] + dataset['validation']['question']
        answers = dataset['train']['answer'] + dataset['validation']['answer']
        golden_sources = dataset['train']['context'] + dataset['validation']['context']
        supporting_facts = dataset['train']['supporting_facts'] + dataset['validation']['supporting_facts']
        source_sentences = []
        title2sentences = {}
        titles = []
        title2start = {}
        title2id = {}
        id = 0
        cur = 0
        i = 0
        for sup,source in zip(supporting_facts, golden_sources):
            # i = i + 1
            # if i == 300:
            #     break
            title = source['title']
            sentence = source['sentences']
            for t,s in zip(title,sentence):
                if t not in title2sentences:
                    title2sentences[t] = s
                    title2start[t] = cur
                    titles.append(t)
                    source_sentences.extend(s)
                    cur += len(s)
                    title2id[t] = id
                    id += 1
                else:
                    print("title already exists, skip.")
        # split the dataset 8:1:1
        indexes = list(range(len(questions)))
        random.shuffle(indexes)
        train_indexes = indexes[:int(len(indexes)*0.8)]
        valid_indexes = indexes[int(len(indexes)*0.8):int(len(indexes)*0.9)]
        test_indexes = indexes[int(len(indexes)*0.9):]
        train_data = {}
        valid_data = {}
        test_data = {}
        train_data['question'], train_data['answers'], train_data['golden_ids'], train_data['golden_sentences'] = build_split([answers[i] for i in train_indexes], [questions[i] for i in train_indexes], [supporting_facts[i] for i in train_indexes], title2id, title2sentences)
        valid_data['question'], valid_data['answers'], valid_data['golden_ids'], valid_data['golden_sentences'] = build_split([answers[i] for i in valid_indexes], [questions[i] for i in valid_indexes], [supporting_facts[i] for i in valid_indexes], title2id, title2sentences)
        test_data['question'], test_data['answers'], test_data['golden_ids'], test_data['golden_sentences'] = build_split([answers[i] for i in test_indexes], [questions[i] for i in test_indexes], [supporting_facts[i] for i in test_indexes], title2id, title2sentences)


        # train_data = {}
        # train_data['question'] , train_data['answers'], train_data['golden_ids'], train_data['golden_sentences'] = build_split(dataset['train']['answer'], dataset['train']['question'], dataset['train']['context'], title2id, title2sentences)
        # valid_data = {}
        # valid_data['question'] , valid_data['answers'], valid_data['golden_ids'], valid_data['golden_sentences'] = build_split(dataset['validation']['answer'], dataset['validation']['question'], dataset['validation']['context'], title2id, title2sentences)
        # test_data = {}
        # test_data['question'] , test_data['answers'], test_data['golden_ids'], test_data['golden_sentences'] = build_split(dataset['test']['answer'], dataset['test']['question'], dataset['test']['context'], title2id, title2sentences)
        # filter_answers, filter_questions, golden_ids, golden_sentences = build_split(answers, questions,
        #                                                                              supporting_facts, title2id,
        #                                                                              title2sentences)
        return dict(
            train_data=train_data,
            valid_data=valid_data,
            test_data=test_data,
            sources=source_sentences,
            titles=titles,
            title2sentences=title2sentences,
            title2start=title2start,
            title2id=title2id,
            dataset=dataset)
    
    elif dataset_name == "drop":
        """
        {
            "answers_spans": [{
                "spans": ["Chaz Schilens"]
            }],
            "passage": "\" Hoping to rebound from their loss to the Patriots, the Raiders stayed at home for a Week 16 duel with the Houston Texans.  Oak...",
            "question": "Who scored the first touchdown of the game?"
        }
        """
        dataset = load_dataset("drop")
        questions = dataset['train']['question'] + dataset['validation']['question']
        answers = dataset['train']['answers_spans'] + dataset['validation']['answers_spans']
        answers = [x['spans'][0] for x in answers]
        sections = dataset['train']['section_id'] + dataset['validation']['section_id']
        # query_ids = dataset['train']['query_id'] + dataset['validation']['query_id']
        golden_sources = dataset['train']['passage'] + dataset['validation']['passage']
        # split 8/1/1
        train_data = {}
        valid_data = {}
        test_data = {}
        indexs = list(range(len(questions)))
        random.shuffle(indexs)
        train_indexs = indexs[:int(len(indexs)*0.8)]
        valid_indexs = indexs[int(len(indexs)*0.8):int(len(indexs)*0.9)]
        test_indexs = indexs[int(len(indexs)*0.9):]
        train_data['question'] = [questions[i] for i in train_indexs]
        train_data['answers'] = [answers[i] for i in train_indexs]
        train_data['golden_sources'] = [golden_sources[i] for i in train_indexs]
        train_data['sections'] = [sections[i] for i in train_indexs]
        # train_data['query_ids'] = [query_ids[i] for i in train_indexs]
        valid_data['question'] = [questions[i] for i in valid_indexs]
        valid_data['answers'] = [answers[i] for i in valid_indexs]
        valid_data['golden_sources'] = [golden_sources[i] for i in valid_indexs]
        valid_data['sections'] = [sections[i] for i in valid_indexs]
        # valid_data['query_ids'] = [query_ids[i] for i in valid_indexs]
        test_data['question'] = [questions[i] for i in test_indexs]
        test_data['answers'] = [answers[i] for i in test_indexs]
        test_data['golden_sources'] = [golden_sources[i] for i in test_indexs]
        test_data['sections'] = [sections[i] for i in test_indexs]
        # test_data['query_ids'] = [query_ids[i] for i in test_indexs]

        source_sentences = []
        title2sentences = {}
        titles = []
        # title2start = {}
        title2id = {}
        id = 0
        # cur = 0
        i = 0
        for sec, source in zip(sections, golden_sources):
            # i = i + 1
            # if i == 300:
            #     break
            if sec not in title2sentences:
                title2sentences[sec] = [source]
                titles.append(sec)
                source_sentences.append(source)
                title2id[sec] = id
                id += 1
            else:
                print("title already exists, skip.")

        train_data['golden_ids'] = [[title2id[sec]] for sec in train_data['sections']]
        valid_data['golden_ids'] = [[title2id[sec]] for sec in valid_data['sections']]
        test_data['golden_ids'] = [[title2id[sec]] for sec in test_data['sections']]
        train_data['golden_sentences'] = [title2sentences[sec] for sec in train_data['sections']]
        valid_data['golden_sentences'] = [title2sentences[sec] for sec in valid_data['sections']]
        test_data['golden_sentences'] = [title2sentences[sec] for sec in test_data['sections']]

        del train_data['sections']
        del valid_data['sections']
        del test_data['sections']
        del train_data['golden_sources']
        del valid_data['golden_sources']
        del test_data['golden_sources']

        return dict(
            train_data=train_data,
            valid_data=valid_data,
            test_data=test_data,
            sources=source_sentences,
            titles=titles,
            title2sentences=title2sentences,
            title2id=title2id,
            dataset=dataset)






    
    elif dataset_name == "natural_questions":
        """
        {
            "id": "797803103760793766",
            "document": {
                "title": "Google",
                "url": "http://www.wikipedia.org/Google",
                "html": "<html><body><h1>Google Inc.</h1><p>Google was founded in 1998 By:<ul><li>Larry</li><li>Sergey</li></ul></p></body></html>",
                "tokens": [
                    {"token": "<h1>", "start_byte": 12, "end_byte": 16, "is_html": True},
                    {"token": "Google", "start_byte": 16, "end_byte": 22, "is_html": False},
                    {"token": "inc", "start_byte": 23, "end_byte": 26, "is_html": False},
                    {"token": ".", "start_byte": 26, "end_byte": 27, "is_html": False},
                    {"token": "</h1>", "start_byte": 27, "end_byte": 32, "is_html": True},
                    {"token": "<p>", "start_byte": 32, "end_byte": 35, "is_html": True},
                    {"token": "Google", "start_byte": 35, "end_byte": 41, "is_html": False},
                    {"token": "was", "start_byte": 42, "end_byte": 45, "is_html": False},
                    {"token": "founded", "start_byte": 46, "end_byte": 53, "is_html": False},
                    {"token": "in", "start_byte": 54, "end_byte": 56, "is_html": False},
                    {"token": "1998", "start_byte": 57, "end_byte": 61, "is_html": False},
                    {"token": "by", "start_byte": 62, "end_byte": 64, "is_html": False},
                    {"token": ":", "start_byte": 64, "end_byte": 65, "is_html": False},
                    {"token": "<ul>", "start_byte": 65, "end_byte": 69, "is_html": True},
                    {"token": "<li>", "start_byte": 69, "end_byte": 73, "is_html": True},
                    {"token": "Larry", "start_byte": 73, "end_byte": 78, "is_html": False},
                    {"token": "</li>", "start_byte": 78, "end_byte": 83, "is_html": True},
                    {"token": "<li>", "start_byte": 83, "end_byte": 87, "is_html": True},
                    {"token": "Sergey", "start_byte": 87, "end_byte": 92, "is_html": False},
                    {"token": "</li>", "start_byte": 92, "end_byte": 97, "is_html": True},
                    {"token": "</ul>", "start_byte": 97, "end_byte": 102, "is_html": True},
                    {"token": "</p>", "start_byte": 102, "end_byte": 106, "is_html": True}
                ],
            },
            "question": {
                "text": "who founded google",
                "tokens": ["who", "founded", "google"]
            },
            "long_answer_candidates": [
                {"start_byte": 32, "end_byte": 106, "start_token": 5, "end_token": 22, "top_level": True},
                {"start_byte": 65, "end_byte": 102, "start_token": 13, "end_token": 21, "top_level": False},
                {"start_byte": 69, "end_byte": 83, "start_token": 14, "end_token": 17, "top_level": False},
                {"start_byte": 83, "end_byte": 92, "start_token": 17, "end_token": 20, "top_level": False}
            ],
            "annotations": [{
                "id": "6782080525527814293",
                "long_answer": {"start_byte": 32, "end_byte": 106, "start_token": 5, "end_token": 22,
                                "candidate_index": 0},
                "short_answers": [
                    {"start_byte": 73, "end_byte": 78, "start_token": 15, "end_token": 16, "text": "Larry"},
                    {"start_byte": 87, "end_byte": 92, "start_token": 18, "end_token": 19, "text": "Sergey"}
                ],
                "yes_no_answer": -1
            }]
        }
        """
        dataset = load_dataset("natural_questions",cache_dir='../data')
        questions = dataset['validation']['question']['text']
        answers = [x['text'] for x in dataset['validation']['annotations']['short_answers']]
        golden_sources = dataset['validation']['document']['html']
    elif dataset_name == "trivia_qa":
        dataset = load_dataset("mandarjoshi/trivia_qa", "rc")

        questions = dataset['train']['question'] + dataset['validation']['question']
        answers = dataset['train']['answer'] + dataset['validation']['answer']
        # question_sources = dataset['train']['question_source'] + dataset['validation']['question_source'] # url
        # entity_pages = dataset['train']['entity_pages'] + dataset['validation']['entity_pages']
        search_results = dataset['train']['search_results'] + dataset['validation']['search_results']
        # delete the search results with empty search context
        # filter out the search results with empty search context
        questions_ = []
        answers_ = []
        search_results_ = []
        for q, a, s in zip(questions, answers, search_results):
            if len(s['search_context']) > 0:
                questions_.append(q)
                answers_.append(a['value'])
                search_results_.append(s)
        '''
        '''
        questions = questions_
        answers = answers_
        search_results = search_results_
        source_sentences = []
        title2sentences = {}
        titles = []
        # title2start = {}
        title2id = {}
        id = 0
        for source in search_results:
            print(source)
            title = source['title']
            sentence = source['search_context']
            for t,s in zip(title,sentence):
                if t not in title2sentences:
                    title2sentences[t] = [s]
                    titles.append(t)
                    source_sentences.append(s)
                    title2id[t] = id
                    id += 1
                else:
                    print("title already exists, skip.")
        # split the dataset 8:1:1
        train_data = {}
        valid_data = {}
        test_data = {}
        indexs = list(range(len(questions)))
        random.shuffle(indexs)
        train_indexs = indexs[:int(len(indexs)*0.8)]
        valid_indexs = indexs[int(len(indexs)*0.8):int(len(indexs)*0.9)]
        test_indexs = indexs[int(len(indexs)*0.9):]
        train_data['question'] = [questions[i] for i in train_indexs]
        train_data['answers'] = [answers[i] for i in train_indexs]
        train_data['golden_sources'] = [search_results[i] for i in train_indexs]
        valid_data['question'] = [questions[i] for i in valid_indexs]
        valid_data['answers'] = [answers[i] for i in valid_indexs]
        valid_data['golden_sources'] = [search_results[i] for i in valid_indexs]
        test_data['question'] = [questions[i] for i in test_indexs]
        test_data['answers'] = [answers[i] for i in test_indexs]
        test_data['golden_sources'] = [search_results[i] for i in test_indexs]

        train_data['golden_ids'] = [[title2id[t] for t in sec['title']] for sec in train_data['golden_sources']]
        valid_data['golden_ids'] = [[title2id[t] for t in sec['title']] for sec in valid_data['golden_sources']]
        test_data['golden_ids'] = [[title2id[t] for t in sec['title']] for sec in test_data['golden_sources']]

        train_data['golden_sentences'] = [sec['search_context'] for sec in train_data['golden_sources']]
        valid_data['golden_sentences'] = [sec['search_context'] for sec in valid_data['golden_sources']]
        test_data['golden_sentences'] = [sec['search_context'] for sec in test_data['golden_sources']]
        del train_data['golden_sources']
        del valid_data['golden_sources']
        del test_data['golden_sources']
        return dict(
            train_data=train_data,
            valid_data=valid_data,
            test_data=test_data,
            sources=source_sentences,
            titles=titles,
            title2sentences=title2sentences,
            title2id=title2id,
            dataset=dataset)

    elif dataset_name == "search_qa":
        dataset = load_dataset("search_qa","train_test_val",cache_dir='../data')
        '''
        Dataset({
    features: ['category', 'air_date', 'question', 'value', 'answer', 'round', 'show_number', 'search_results'],
    num_rows: 151295
})'''
        questions = dataset['train']['question'] + dataset['validation']['question'] + dataset['test']['question']
        answers = dataset['train']['answer'] + dataset['validation']['answer'] + dataset['test']['answer']
        golden_sources = dataset['train']['search_results'] + dataset['validation']['search_results'] + dataset['test']['search_results']
        raise NotImplementedError(f'dataset {dataset_name} not implemented! Search QA is not supported yet! As its search_results is really searched and each question has a lot of search results. It is not suitable for the RAG system.')

    elif dataset_name == "finqa":
        dataset = load_dataset("dreamerdeo/finqa")
        questions = dataset['train']['question'] + dataset['validation']['question'] + dataset['test']['question']
        answers = dataset['train']['answer'] + dataset['validation']['answer'] + dataset['test']['answer']
        ids = dataset['train']['id'] + dataset['validation']['id'] + dataset['test']['id']
        pre_text = dataset['train']['pre_text'] + dataset['validation']['pre_text'] + dataset['test']['pre_text']
        post_text = dataset['train']['post_text'] + dataset['validation']['post_text'] + dataset['test']['post_text']
        table = dataset['train']['table'] + dataset['validation']['table'] + dataset['test']['table']
        golden_sources = []
        for i in range(len(pre_text)):
            golden_sources.append([t for t in pre_text[i] if t !='.'] + ['|'.join(line)+'\n' for line in table[i]] + [t_ for t_ in post_text[i] if t_ != '.'])

        source_sentences = []
        title2sentences = {}
        titles = []
        # title2start = {}
        title2id = {}
        id = 0
        for t,source in zip(ids,golden_sources):
            title = t
            sentence = source
            if title not in title2sentences:
                title2sentences[title] = sentence
                titles.append(title)
                source_sentences.extend(sentence)
                title2id[title] = id
                id += 1
            else:
                print("title already exists, skip.")

        train_data = {}
        valid_data = {}
        test_data = {}
        train_data['question'] = dataset['train']['question']
        train_data['answers'] = dataset['train']['answer']
        train_data['golden_ids'] = [[title2id[t]] for t in dataset['train']['id']]
        train_data['golden_sentences'] = [[' '.join(title2sentences[t])] for t in dataset['train']['id']]
        # remove the empty answer
        train_data['question'] , train_data['answers'], train_data['golden_ids'], train_data['golden_sentences'] = zip(*[(q,a,gid,gs) for q,a,gid,gs in zip(train_data['question'],train_data['answers'],train_data['golden_ids'],train_data['golden_sentences']) if a != ''])

        valid_data['question'] = dataset['validation']['question']
        valid_data['answers'] = dataset['validation']['answer']
        valid_data['golden_ids'] = [[title2id[t]] for t in dataset['validation']['id']]
        valid_data['golden_sentences'] = [[' '.join(title2sentences[t])] for t in dataset['validation']['id']]
        valid_data['question'] , valid_data['answers'], valid_data['golden_ids'], valid_data['golden_sentences'] = zip(*[(q,a,gid,gs) for q,a,gid,gs in zip(valid_data['question'],valid_data['answers'],valid_data['golden_ids'],valid_data['golden_sentences']) if a != ''])

        test_data['question'] = dataset['test']['question']
        test_data['answers'] = dataset['test']['answer']
        test_data['golden_ids'] = [[title2id[t]] for t in dataset['test']['id']]
        test_data['golden_sentences'] = [[' '.join(title2sentences[t])] for t in dataset['test']['id']]
        test_data['question'] , test_data['answers'], test_data['golden_ids'], test_data['golden_sentences'] = zip(*[(q,a,gid,gs) for q,a,gid,gs in zip(test_data['question'],test_data['answers'],test_data['golden_ids'],test_data['golden_sentences']) if a != ''])
        return dict(
            train_data=train_data,
            valid_data=valid_data,
            test_data=test_data,
            sources=source_sentences,
            titles=titles,
            title2sentences=title2sentences,
            title2id=title2id,
            dataset=dataset)








    else:
        raise NotImplementedError(f'dataset {dataset_name} not implemented!')
    
    return dict(
        question=questions, 
        answers=answers, 
        golden_sources=golden_sources,
        dataset=dataset)







if __name__=='__main__':

    name = 'finqa' # drop, natural_questions, hotpot_qa
    ex_data = get_qa_dataset('hotpot_qa')
    data = get_qa_dataset(name)
    print()

