import collections
import os
import numpy as np
import pandas as pd
import pickle
import time
import multiprocessing as mp
from functools import partial
import random
import matplotlib.colors as mcolors
import operator
import seaborn as sns

def load_data(args):
    train_data, eval_data, test_data, user_history_dict = load_rating(args)
    n_entity, n_relation, kg = load_kg(args)
    ripple_set = get_user_triplet_set(args, kg, user_history_dict)
    return train_data, eval_data, test_data, n_entity, n_relation, ripple_set


def load_rating(args):
    print('reading rating file ...')

    # reading rating file
    rating_file = '../data/' + args.dataset + '/ratings_final'
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int32)
        np.save(rating_file + '.npy', rating_np)

    n_user = max(set(rating_np[:, 0])) + 1
    args.n_user = n_user
    n_item = max(set(rating_np[:, 1])) + 1
    args.n_item = n_item

    if args.dataset == "MovieLens-1M": top_k = 500
    else: top_k = 500

    if os.path.exists(f"{args.path.misc}_pop_item_set_{top_k}.pickle") == False:
        item_count = {}
        for i in range(rating_np.shape[0]):
            item = rating_np[i, 1]
            if item not in item_count:
                item_count[item] = 0
            item_count[item] += 1
        item_count = sorted(item_count.items(), key=lambda x: x[1], reverse=True)
        item_count = item_count[:top_k]
        item_set_most_pop = [item_set[0] for item_set in item_count]
        with open(f"{args.path.misc}_pop_item_set_{top_k}.pickle", 'wb') as fp:
            pickle.dump(item_set_most_pop, fp)
        
    with open(f"{args.path.misc}_pop_item_set_{top_k}.pickle", 'rb') as fp:
        item_set_most_pop = pickle.load(fp)

    item_set_most_pop = set(item_set_most_pop)
    args.item_set_most_pop = item_set_most_pop

    return dataset_split(args, rating_np)


def load_pre_data(args):
    train_data = pd.read_csv(args.path.data + 'train_pd.csv',index_col=None)
    train_data = train_data.drop(train_data.columns[0], axis=1)
    train_data = train_data[['user','item','like']].values
    eval_data = pd.read_csv(args.path.data + 'eval_pd.csv',index_col=None)
    eval_data = eval_data.drop(eval_data.columns[0], axis=1)
    eval_data = eval_data[['user','item','like']].values
    test_data = pd.read_csv(args.path.data + 'test_pd.csv',index_col=None)
    test_data = test_data.drop(test_data.columns[0], axis=1)
    test_data = test_data[['user','item','like']].values
    return train_data, eval_data, test_data

def dataset_split(args, rating_np):
    print('splitting dataset ...')

    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]

    eval_indices = np.random.choice(n_ratings, size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))
    # train_indices = np.random.choice(list(train_indices), size=int(n_ratings * 0.01), replace=False)
    print(len(train_indices), len(eval_indices), len(test_indices))

    train_data, eval_data, test_data = load_pre_data(args)
    user_history_dict = dict()
    for i in range(train_data.shape[0]):
        user = train_data[i][0]
        item = train_data[i][1]
        rating = train_data[i][2]
        if rating == 1:
            if user not in user_history_dict:
                user_history_dict[user] = []
            user_history_dict[user].append(item)

    train_indices = [i for i in range(train_data.shape[0]) if train_data[i][0] in user_history_dict]
    eval_indices = [i for i in range(eval_data.shape[0]) if eval_data[i][0] in user_history_dict]
    test_indices = [i for i in range(test_data.shape[0]) if test_data[i][0] in user_history_dict]

    train_data = train_data[train_indices]
    eval_data = eval_data[eval_indices]
    test_data = test_data[test_indices]

    return train_data, eval_data, test_data, user_history_dict


def load_kg(args):
    print('reading KG file ...')

    # reading kg file
    kg_file = '../data/' + args.dataset + '/kg_final'
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int32)
        np.save(kg_file + '.npy', kg_np)

    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))
    n_triple = kg_np.shape[0]
    
    kg = construct_kg(kg_np)

    return n_entity, n_relation, kg


def construct_kg(kg_np):
    print('constructing knowledge graph ...')
    kg = collections.defaultdict(list)
    for head, relation, tail in kg_np:
        kg[head].append((tail, relation))
    return kg


def get_user_triplet_set(args, kg, user_history_dict):
    print('constructing ripple set ...')
    # user -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]
    user_triplet_set = collections.defaultdict(list)
    # entity_interaction_dict = collections.defaultdict(list)
    global g_kg
    g_kg = kg
    entities_set = {}
    entities_type_set = {}
    entities_type_set['item'] = {}
    entities_type_color = {}
    entities_com_dict = {}

    color_label_set = list(sns.color_palette("tab10")) + list(sns.color_palette("Set2")) + list(sns.color_palette("Paired")) + list(sns.color_palette("hls", 80))
    entities_type_color['item'] = (0., 0., 0.)


    with mp.Pool(processes=min(mp.cpu_count(), 5)) as pool:
        job = partial(_get_user_triplet_set, p_hop=max(1,args.n_hop), n_memory=args.n_memory, n_neighbor=16)
        for u, u_r_set, u_interaction_list in pool.starmap(job, user_history_dict.items()):
            user_triplet_set[u] = np.array(u_r_set, dtype=np.int32)


            for hop_i in range(args.n_hop):
                for et_i in range(len(user_triplet_set[u][hop_i][0])):
                    et_1 = user_triplet_set[u][hop_i][0][et_i]
                    type_et = user_triplet_set[u][hop_i][1][et_i]
                    et_2 = user_triplet_set[u][hop_i][2][et_i]

                    if et_1 < args.n_item and et_2 >= args.n_item:
                        if type_et not in entities_type_set:
                            entities_type_set[type_et] = {}
                            entities_type_color[type_et] = color_label_set[type_et+1]
                            # color_label_set[len(entities_type_color)]
                        if et_2 not in entities_type_set[type_et]:
                            entities_type_set[type_et][et_2] = 0
                        entities_type_set[type_et][et_2] += 1

                        if et_1 not in entities_type_set['item']:
                            entities_type_set['item'][et_1] = 0
                        entities_type_set['item'][et_1] += 1

                    if et_2 < args.n_item and et_1 >= args.n_item:
                        if type_et not in entities_type_set:
                            entities_type_set[type_et] = {}
                            entities_type_color[type_et] = color_label_set[type_et+1]

                        if et_1 not in entities_type_set[type_et]:
                            entities_type_set[type_et][et_1] = 0
                        entities_type_set[type_et][et_1] += 1

                        if et_2 not in entities_type_set['item']:
                            entities_type_set['item'][et_2] = 0
                        entities_type_set['item'][et_2] += 1                

    n_entities_type_set = {}
    for keys, value in entities_type_set.items():
        entities_type_set[keys] = dict((k, v) for k, v in entities_type_set[keys].items() if v >= args.et_freq)
        n_entities_type_set[keys] = {}
        for et, value in entities_type_set[keys].items():
            if et not in entities_set:
                entities_set[et] = len(entities_set)
            n_entities_type_set[keys][entities_set[et]] = value

    entities_type_set = n_entities_type_set
    del n_entities_type_set

    entities_set = sorted(entities_set.items(), key=lambda x: x[1])
    n_entities_set = {}
    for k in entities_set:
        n_entities_set[k[0]] = k[1]
    entities_set = n_entities_set

    del g_kg
    args.entities_set = entities_set
    args.entities_type_set = entities_type_set
    args.entities_type_color = entities_type_color
    return user_triplet_set

def _get_user_triplet_set(user, history, p_hop=2, n_memory=32, n_neighbor=16):
    ret = []
    entity_interaction_list = []
    for h in range(max(1,p_hop)):
        memories_h = []
        memories_r = []
        memories_t = []

        if h == 0:
            tails_of_last_hop = history
        else:
            tails_of_last_hop = ret[-1][2]

        for entity in tails_of_last_hop:
            for tail_and_relation in random.sample(g_kg[entity], min(len(g_kg[entity]), n_neighbor)):
                memories_h.append(entity)
                memories_r.append(tail_and_relation[1])
                memories_t.append(tail_and_relation[0])

        # if the current ripple set of the given user is empty, we simply copy the ripple set of the last hop here
        # this won't happen for h = 0, because only the items that appear in the KG have been selected
        # this only happens on 154 users in Book-Crossing dataset (since both BX dataset and the KG are sparse)
        if len(memories_h) == 0:
            ret.append(ret[-1])
        else:
            replace = len(memories_h) < n_memory
            indices = np.random.choice(len(memories_h), size=n_memory, replace=replace)
            memories_h = [memories_h[i] for i in indices]
            memories_r = [memories_r[i] for i in indices]
            memories_t = [memories_t[i] for i in indices]
            entity_interaction_list += zip(memories_h, memories_r, memories_t)
            ret.append([memories_h, memories_r, memories_t])
            
    return [user, ret, list(set(entity_interaction_list))]