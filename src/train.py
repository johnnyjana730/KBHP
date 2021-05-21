import numpy as np
import torch
from model.get_model import Model
import os 
from util import topk_settings, ctr_eval, ctr_eval_case_study, topk_eval

def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)

class Eval_score_info:
    def __init__(self):
        self.train_auc_acc_f1 = [0 for _ in range(3)]
        self.eval_auc_acc_f1 = [0 for _ in range(3)]
        self.test_auc_acc_f1 = [0 for _ in range(3)]

        self.train_ndcg_recall_pecision = [[0 for i in range(5)] for _ in range(3)]
        self.eval_ndcg_recall_pecision = [[0 for i in range(5)] for _ in range(3)]
        self.test_ndcg_recall_pecision = [[0 for i in range(5)] for _ in range(3)]

        self.eval_precision = 0
        self.eval_recall = 0
        self.eval_ndcg = 0
        self.test_precision = 0
        self.test_recall = 0
        self.test_ndcg = 0
    def eval_st_score(self):
        return self.eval_auc_acc_f1[0]
    def eval_st_topk_score(self):
        return self.eval_ndcg_recall_pecision[1][3]

def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value > best_value) or (expected_order == 'dec' and log_value < best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop

def load_pretrain_model(args, model):
    if args.load_pretrain_emb == True:

        print(args.path.pre_emb)

        pretrain_sd = torch.load(args.path.pre_emb)

        model_sd = model.state_dict()
        pretrain_sd = {k: v for k, v in pretrain_sd.items() if k in model_sd and 'sw' in k}

        model_sd.update(pretrain_sd)
        model.load_state_dict(model_sd)
        print('model_sd.update(pretrain_sd)')

def train(args, data_info, show_loss):
    train_data = data_info[0]
    eval_data = data_info[1]
    test_data = data_info[2]
    n_entity = data_info[3]
    n_relation = data_info[4]
    ripple_set = data_info[5]

    if args.use_cuda:
        model = Model(args, n_entity, n_relation).cuda()
    else:
        model = Model(args, n_entity, n_relation)

    loss_loger, train_auc_acc_f1, eval_auc_acc_f1, test_auc_acc_f1  = [], [[] for _ in range(3)], [[] for _ in range(3)], [[] for _ in range(3)]
    best_eval_auc_acc_f1, best_test_auc_acc_f1 = [0 for _ in range(3)], [0 for _ in range(3)]
    best_eval_ndcg_recall_pecision, best_test_ndcg_recall_pecision = [[0 for i in range(5)] for _ in range(3)], [[0 for i in range(5)] for _ in range(3)]
    cur_best_pre_0 = 0.
    stopping_step = 0
    step_record = 0
    should_stop = False

    user_list, train_record, eval_record, test_record, item_set, k_list = topk_settings(args, args.show_topk, train_data, eval_data, test_data, args.n_item, args.save_record_user_list, args.save_model_name)

    print('args.load_pretrain_emb = ', args.load_pretrain_emb)
    if args.load_pretrain_emb == True:
        load_pretrain_model(args, model)

    args.logger.info('Parameters:' + str([i[0] for i in model.named_parameters()]))
    eval_score_info = Eval_score_info()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        args.lr,
    )

    for step in range(args.n_epoch):
        model.train()
        # training
        np.random.shuffle(train_data)
        start = 0
        while start < train_data.shape[0]:
            return_dict = model(*get_feed_dict(args, model, train_data, ripple_set, start, start + args.batch_size))
            loss = return_dict["loss"]

            optimizer.zero_grad()
            loss.backward()

            if args.manifold_name == "PoincareBall":
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
# 
            start += args.batch_size
            if show_loss:
                print('%.1f%% %.4f' % (start / train_data.shape[0] * 100, loss.item()))
        # evaluation
        if step % 1 == 0:
            model.eval()

            if args.show_topk:
                precision, recall, ndcg, MAP, hit_ratio = topk_eval(
                    args, ripple_set, model, user_list, train_record, eval_record, test_record, args.item_set_most_pop, k_list, args.batch_size, mode = 'eval')
                n_precision_eval = [round(i, 6) for i in precision]
                n_recall_eval = [round(i, 6) for i in recall]
                n_ndcg_eval = [round(i, 6) for i in ndcg]

                precision, recall, ndcg, MAP, hit_ratio = topk_eval(
                    args, ripple_set, model, user_list, train_record, eval_record, test_record, args.item_set_most_pop, k_list, 
                    args.batch_size, mode = 'test')

                n_precision_test = [round(i, 4) for i in precision]
                n_recall_test = [round(i, 4) for i in recall]
                n_ndcg_test = [round(i, 4) for i in ndcg]

                eval_score_info.eval_ndcg_recall_pecision = [n_ndcg_eval, n_recall_eval, n_precision_eval]
                eval_score_info.test_ndcg_recall_pecision = [n_ndcg_test, n_recall_test, n_precision_test]

                tmp_n_precision_eva, tmp_n_recall_eva, tmp_n_ndcg_eva = [str(i) for i in n_precision_eval], [str(i) for i in n_recall_eval], [str(i) for i in n_ndcg_eval]
                args.logger.info('step = ' + str(step))
                args.logger.info('eval precision = ' +  ','.join(tmp_n_precision_eva))
                args.logger.info('eval recall = ' +  ','.join(tmp_n_recall_eva))
                args.logger.info('eval ndcg = ' +  ','.join(tmp_n_ndcg_eva))

                tmp_n_precision_test, tmp_n_recall_test, tmp_n_ndcg_test = [str(i) for i in n_precision_test], [str(i) for i in n_recall_test], [str(i) for i in n_ndcg_test]
                args.logger.info('test precision = ' +  ','.join(tmp_n_precision_test))
                args.logger.info('test recall = ' +  ','.join(tmp_n_recall_test))
                args.logger.info('test ndcg = ' +  ','.join(tmp_n_ndcg_test))

                cur_best_pre_0, stopping_step, should_stop = early_stopping(eval_score_info.eval_st_topk_score(), cur_best_pre_0,
                                                                            stopping_step, expected_order='acc', flag_step=10)
                if should_stop == True: break

                if eval_score_info.eval_st_topk_score() == cur_best_pre_0:
                    best_eval_ndcg_recall_pecision = eval_score_info.eval_ndcg_recall_pecision
                    best_test_ndcg_recall_pecision = eval_score_info.test_ndcg_recall_pecision
                    args.logger.info("Save model to " + args.path.emb + '_sw_' + str(args.sw_stage) + '.ckpt')
                    torch.save(model.state_dict(), args.path.emb + '_sw_' + str(args.sw_stage) + '.ckpt')

            else:
                eval_score_info.train_auc_acc_f1 = evaluation(args, model, train_data, ripple_set, args.batch_size)
                eval_score_info.eval_auc_acc_f1 = evaluation(args, model, eval_data, ripple_set, args.batch_size)
                eval_score_info.test_auc_acc_f1 = evaluation(args, model, test_data, ripple_set, args.batch_size)

                train_auc, train_acc, train_f1 = eval_score_info.train_auc_acc_f1[0], eval_score_info.train_auc_acc_f1[1], eval_score_info.train_auc_acc_f1[2]
                eval_auc, eval_acc, eval_f1 = eval_score_info.eval_auc_acc_f1[0], eval_score_info.eval_auc_acc_f1[1], eval_score_info.eval_auc_acc_f1[2]
                test_auc, test_acc, test_f1 =  eval_score_info.test_auc_acc_f1[0], eval_score_info.test_auc_acc_f1[1], eval_score_info.test_auc_acc_f1[2]

                args.logger.info('step %d  train auc: %.4f acc: %.4f f1: %.4f eval auc: %.4f acc: %.4f f1: %.4f test auc: %.4f acc: %.4f f1: %.4f'
                          % (step, train_auc, train_acc, train_f1, eval_auc, eval_acc, eval_f1, test_auc, test_acc, test_f1))

                cur_best_pre_0, stopping_step, should_stop = early_stopping(eval_score_info.eval_st_score(), cur_best_pre_0,
                                                                            stopping_step, expected_order='acc', flag_step=10)
                if should_stop == True: break

                if eval_score_info.eval_st_score() == cur_best_pre_0:

                    best_eval_auc_acc_f1 = eval_score_info.eval_auc_acc_f1[0], eval_score_info.eval_auc_acc_f1[1], eval_score_info.eval_auc_acc_f1[2]
                    best_test_auc_acc_f1 = eval_score_info.test_auc_acc_f1[0], eval_score_info.test_auc_acc_f1[1], eval_score_info.test_auc_acc_f1[2]

                    args.logger.info("Save model to " + args.path.emb + '_sw_' + str(args.sw_stage) + '.ckpt')
                    torch.save(model.state_dict(), args.path.emb + '_sw_' + str(args.sw_stage) + '.ckpt')

                    time = 0

                    try:
                        if step_record % 10 == 0 and args.emb_eval == True:
                            model.tsne_embedding(args ,step)
                    except Exception as e:
                        print(e)
                        args.logger.info(e)

                    step_record += 1

    try:
        model.tsne_embedding(args ,step)
    except Exception as e:
        print(e)
        args.logger.info(e)


    if args.show_topk:
        n_ndcg_eva, n_recall_eva, n_precision_eva = best_eval_ndcg_recall_pecision
        n_ndcg_test, n_recall_test, n_precision_test = best_test_ndcg_recall_pecision

        tmp_n_precision_eva, tmp_n_recall_eva, tmp_n_ndcg_eva = [str(i) for i in n_precision_eva], [str(i) for i in n_recall_eva], [str(i) for i in n_ndcg_eva]
        args.logger.info('step = ' + str(step))
        args.logger.info('eval precision = ' +  ','.join(tmp_n_precision_eva))
        args.logger.info('eval recall = ' +  ','.join(tmp_n_recall_eva))
        args.logger.info('eval ndcg = ' +  ','.join(tmp_n_ndcg_eva))

        tmp_n_precision_test, tmp_n_recall_test, tmp_n_ndcg_test = [str(i) for i in n_precision_test], [str(i) for i in n_recall_test], [str(i) for i in n_ndcg_test]
        args.logger.info('test precision = ' +  ','.join(tmp_n_precision_test))
        args.logger.info('test recall = ' +  ','.join(tmp_n_recall_test))
        args.logger.info('test ndcg = ' +  ','.join(tmp_n_ndcg_test))
        
        args.entities_set = ''
        args.entities_type_set = ''
        args.item_set_most_pop = ''
        args.entities_type_color = ''

        args_vars = vars(args)
        f = open(args.path.eva_file, 'a')
        text = " ".join([key + "_" + str(value) for key, value in args_vars.items()])
        f.write(text)
        f.write("\n")
        f.write('best eva precision = ' +  ','.join(tmp_n_precision_eva) + '\n')
        f.write('best eva recall = ' +  ','.join(tmp_n_recall_eva) + '\n')
        f.write('best eva ndcg = ' +  ','.join(tmp_n_ndcg_eva) + '\n')

        f.write('best test precision = ' +  ','.join(tmp_n_precision_test) + '\n')
        f.write('best test recall = ' +  ','.join(tmp_n_recall_test) + '\n')
        f.write('best test ndcg = ' +  ','.join(tmp_n_ndcg_test) + '\n')
        f.write("\n")
        f.write('*'*100 + "\n")
        f.close()

    else:
        try:
            precision, recall, ndcg, MAP, hit_ratio = topk_eval(
                args, ripple_set, model, user_list, train_record, eval_record, test_record, args.item_set_most_pop, k_list, args.batch_size, mode = 'eval')
            n_precision_eval = [round(i, 6) for i in precision]
            n_recall_eval = [round(i, 6) for i in recall]
            n_ndcg_eval = [round(i, 6) for i in ndcg]

            precision, recall, ndcg, MAP, hit_ratio = topk_eval(
                args, ripple_set, model, user_list, train_record, eval_record, test_record, args.item_set_most_pop, k_list, 
                args.batch_size, mode = 'test')

            n_precision_test = [round(i, 4) for i in precision]
            n_recall_test = [round(i, 4) for i in recall]
            n_ndcg_test = [round(i, 4) for i in ndcg]

            eval_score_info.eval_ndcg_recall_pecision = [n_ndcg_eval, n_recall_eval, n_precision_eval]
            eval_score_info.test_ndcg_recall_pecision = [n_ndcg_test, n_recall_test, n_precision_test]

            tmp_n_precision_eva, tmp_n_recall_eva, tmp_n_ndcg_eva = [str(i) for i in n_precision_eval], [str(i) for i in n_recall_eval], [str(i) for i in n_ndcg_eval]
            args.logger.info('step = ' + str(step))
            args.logger.info('eval precision = ' +  ','.join(tmp_n_precision_eva))
            args.logger.info('eval recall = ' +  ','.join(tmp_n_recall_eva))
            args.logger.info('eval ndcg = ' +  ','.join(tmp_n_ndcg_eva))

            tmp_n_precision_test, tmp_n_recall_test, tmp_n_ndcg_test = [str(i) for i in n_precision_test], [str(i) for i in n_recall_test], [str(i) for i in n_ndcg_test]
            args.logger.info('test precision = ' +  ','.join(tmp_n_precision_test))
            args.logger.info('test recall = ' +  ','.join(tmp_n_recall_test))
            args.logger.info('test ndcg = ' +  ','.join(tmp_n_ndcg_test))

            best_eval_ndcg_recall_pecision = eval_score_info.eval_ndcg_recall_pecision
            best_test_ndcg_recall_pecision = eval_score_info.test_ndcg_recall_pecision

            n_ndcg_eva, n_recall_eva, n_precision_eva = best_eval_ndcg_recall_pecision
            n_ndcg_test, n_recall_test, n_precision_test = best_test_ndcg_recall_pecision

            tmp_n_precision_eva, tmp_n_recall_eva, tmp_n_ndcg_eva = [str(i) for i in n_precision_eva], [str(i) for i in n_recall_eva], [str(i) for i in n_ndcg_eva]
            args.logger.info('step = ' + str(step))
            args.logger.info('eval precision = ' +  ','.join(tmp_n_precision_eva))
            args.logger.info('eval recall = ' +  ','.join(tmp_n_recall_eva))
            args.logger.info('eval ndcg = ' +  ','.join(tmp_n_ndcg_eva))

            tmp_n_precision_test, tmp_n_recall_test, tmp_n_ndcg_test = [str(i) for i in n_precision_test], [str(i) for i in n_recall_test], [str(i) for i in n_ndcg_test]
            args.logger.info('test precision = ' +  ','.join(tmp_n_precision_test))
            args.logger.info('test recall = ' +  ','.join(tmp_n_recall_test))
            args.logger.info('test ndcg = ' +  ','.join(tmp_n_ndcg_test))
            
            args.entities_set = ''
            args.entities_type_set = ''
            args.item_set_most_pop = ''
            args.entities_type_color = ''

            args_vars = vars(args)
            f = open(args.path.eva_file, 'a')
            text = " ".join([key + "_" + str(value) for key, value in args_vars.items()])
            f.write(text)
            f.write("\n")
            f.write('best eva precision = ' +  ','.join(tmp_n_precision_eva) + '\n')
            f.write('best eva recall = ' +  ','.join(tmp_n_recall_eva) + '\n')
            f.write('best eva ndcg = ' +  ','.join(tmp_n_ndcg_eva) + '\n')

            f.write('best test precision = ' +  ','.join(tmp_n_precision_test) + '\n')
            f.write('best test recall = ' +  ','.join(tmp_n_recall_test) + '\n')
            f.write('best test ndcg = ' +  ','.join(tmp_n_ndcg_test) + '\n')
            f.write("\n")
            f.write('*'*100 + "\n")
            f.close()
        except Exception as e:
            print(e)
            args.logger.info(e)

        final_perf = 'step %d  eval auc: %.4f acc: %.4f f1: %.4f test auc: %.4f acc: %.4f f1: %.4f' \
                  % (step, best_eval_auc_acc_f1[0], best_eval_auc_acc_f1[1], best_eval_auc_acc_f1[2], best_test_auc_acc_f1[0], best_test_auc_acc_f1[1], best_test_auc_acc_f1[2])
        args.logger.info(final_perf)

        args.entities_set = ''
        args.entities_type_set = ''
        args.item_set_most_pop = ''
        args.entities_type_color = ''

        args_vars = vars(args)
        f = open(args.path.eva_file, 'a')
        text = " ".join([key + "_" + str(value) for key, value in args_vars.items()])
        f.write(text)
        f.write("\n")
        f.write(final_perf)
        f.write("\n")
        f.write('*'*100 + "\n")
        f.close()


def get_feed_dict(args, model, data, ripple_set, start, end):

    users = torch.LongTensor(data[start:end, 0])
    
    # input()
    items = torch.LongTensor(data[start:end, 1])
    labels = torch.LongTensor(data[start:end, 2])
    memories_h, memories_r, memories_t = [], [], []
    for i in range(args.n_hop):
        memories_h.append(torch.LongTensor([ripple_set[user][i][0] for user in data[start:end, 0]]))
        memories_r.append(torch.LongTensor([ripple_set[user][i][1] for user in data[start:end, 0]]))
        memories_t.append(torch.LongTensor([ripple_set[user][i][2] for user in data[start:end, 0]]))
    if args.use_cuda:
        users = users.cuda()
        items = items.cuda()
        labels = labels.cuda()
        memories_h = list(map(lambda x: x.cuda(), memories_h))
        memories_r = list(map(lambda x: x.cuda(), memories_r))
        memories_t = list(map(lambda x: x.cuda(), memories_t))
    return users, items, labels, memories_h, memories_r,memories_t

def evaluation(args, model, data, ripple_set, batch_size):
    start = 0
    auc_list = []
    acc_list = []
    f1_list = []
    while start < data.shape[0]:
        auc, acc, f1 = model.evaluate(*get_feed_dict(args, model, data, ripple_set, start, start + batch_size))
        auc_list.append(auc)
        acc_list.append(acc)
        f1_list.append(f1)
        start += batch_size
    return float(np.mean(auc_list)), float(np.mean(acc_list)), float(np.mean(f1_list))
