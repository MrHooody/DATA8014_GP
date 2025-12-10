import argparse

import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from util.general_utils import AverageMeter, init_experiment, str2bool
from util.cluster_and_log_utils import log_accs_from_preds
from util.faster_mix_k_means_pytorch import K_Means as SemiSupKMeans
from config import exp_root, crate_alpha_pretrain_path, ovr_envs, distortions, severity
from loss import DINOHead, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, get_params_groups, Distangleloss
from models.crate_alpha import CRATE_base as crate
from transformers import ViTModel

def one_axis_loss(axis, header, feats, masked_sup_labels, mask_lab, cluster_criterion, epoch):
    student_proj, student_out = header(feats)
    teacher_out = student_out.detach()

    # clustering, unsup
    if axis == 'sem':
        cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
        avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
        me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
        cluster_loss += args.memax_weight * me_max_loss

    elif axis == 'dom':
        kmeans = SemiSupKMeans(k=student_out.size(-1), tolerance=1e-4, max_iterations=10, init='k-means++',
                            n_init=10, random_state=None, n_jobs=None, pairwise_batch_size=1024, mode=None)

        kmeans.fit_mix(student_proj[:args.batch_size][~mask_lab], student_proj[:args.batch_size][mask_lab], masked_sup_labels)
        ps_labels = kmeans.labels_.long()
        cluster_loss = nn.CrossEntropyLoss()(student_out[:args.batch_size][~mask_lab] / 0.1, ps_labels[:args.batch_size][~mask_lab])
        avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
        me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
        cluster_loss += args.memax_weight_dom * me_max_loss

    # clustering, sup
    sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
    sup_labels = torch.cat([masked_sup_labels for _ in range(2)], dim=0)
    cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

    # representation learning, unsup
    contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
    contrastive_loss = nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

    # representation learning, sup
    student_proj1 = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
    student_proj1 = nn.functional.normalize(student_proj1, dim=-1)
    sup_con_loss = SupConLoss()(student_proj1, labels=masked_sup_labels)
    
    loss = 0
    loss += (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
    loss += (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss

    return loss, student_proj, student_out

def train(student, dom_head, con_mlp, train_loader, optimizer, lr_scheduler, cluster_criterion, mi_criterion, epoch, args):

    loss_record = AverageMeter()

    student.train()
    dom_head.train()
    con_mlp.train()
    for batch_idx, batch in enumerate(tqdm(train_loader)):
        images, class_labels, uq_idxs, mask_lab = batch
        mask_lab = mask_lab[:, 0]

        class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
        images = torch.cat(images, dim=0).cuda(non_blocking=True)
        dom_labels = torch.zeros_like(class_labels[mask_lab]).to(class_labels.device)

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            if args.model == 'crate':
                feat_list = student[0].forward_features(images, [1, 12])
                domain_feats = feat_list[0]
                semantic_feats = feat_list[-1]
                del feat_list
            elif args.model == 'vit':
                feat_list = student[0](images, output_hidden_states=True).hidden_states
                semantic_feats = feat_list[-1][:, 0, :]
                domain_feats = feat_list[1][:, 0, :]
                del feat_list

            sem_loss, sem_proj, sem_out = one_axis_loss('sem', model[1], semantic_feats,
                                                class_labels[mask_lab], mask_lab, cluster_criterion, epoch)

            dom_loss, dom_proj, dom_out = one_axis_loss('dom', dom_head, domain_feats,
                                                dom_labels, mask_lab, cluster_criterion, epoch)

            # mutual information disentanglement
            mi_loss = mi_criterion(con_mlp, sem_proj, dom_proj)

            loss = sem_loss + dom_loss + args.mi_weight * mi_loss

            
        # Train acc
        loss_record.update(loss.item(), class_labels.size(0))
        optimizer.zero_grad()
        if fp16_scaler is None:
            loss.backward()
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

    args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))
    lr_scheduler.step()


def test(model, test_loader, epoch, save_name, args):

    model.eval()

    preds, targets = [], []
    mask = np.array([])
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            if args.model == 'crate':
                _, logits = model(images)
            elif args.model == 'vit':
                _, logits = model[1](model[0](images).pooler_output)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    return all_acc, old_acc, new_acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v2p'])

    parser.add_argument('--dataset_name', type=str, default='domainnet', help='options: domainnet, ssbc')
    parser.add_argument('--src_env', type=str)
    parser.add_argument('--tgt_env', type=str)
    parser.add_argument('--aux_env', type=str, default=None)
    parser.add_argument('--task_type', type=str, default='A_L+A_U->B')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)
    parser.add_argument('--pre_splits', type=str2bool, default=False)
    parser.add_argument('--only_test', type=str2bool, default=False)
    parser.add_argument('--use_partial_dataset', type=str2bool, default=False)
    parser.add_argument('--eval_freq', type=int, default=5)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='domainnet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    
    parser.add_argument('--memax_weight', type=float, default=2)
    parser.add_argument('--memax_weight_dom', type=float, default=0.1)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')

    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=100, type=int)
    parser.add_argument('--exp_name', default=None, type=str)

    parser.add_argument('--model', default='crate', type=str, help='crate|vit|dino')
    parser.add_argument('--mi_weight', default=0.5)

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda')
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=['project'])
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')
    
    torch.backends.cudnn.benchmark = True

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875
    
    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes

    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    if args.model == 'crate':
        backbone = crate()
        state_dict = torch.load(crate_alpha_pretrain_path, map_location='cpu')
        backbone.load_state_dict(state_dict, strict=False)

        for m in backbone.parameters():
            m.requires_grad = False

        # Only finetune layers from block 'args.grad_from_block' onwards
        for name, m in backbone.named_parameters():
            if 'layers' in name:
                block_num = int(name.split('.')[2])
                if block_num >= args.grad_from_block:
                    m.requires_grad = True
    
    elif args.model == 'vit':
        backbone = ViTModel.from_pretrained("google/vit-base-patch16-224")

        for m in backbone.parameters():
            m.requires_grad = False

        # Only finetune layers from block 'args.grad_from_block' onwards
        for name, m in backbone.named_parameters():
            if 'encoder.layer' in name:
                block_num = int(name.split('.')[2])
                if block_num >= args.grad_from_block:
                    m.requires_grad = True

    # Double check
    enabled = set()
    for name, param in backbone.named_parameters():
        if param.requires_grad:
            enabled.add(name)
    args.logger.info(f"Parameters to be updated: {enabled}")
    
    args.logger.info('model built')

    # --------------------
    # DATASETS
    # --------------------
    # CONTRASTIVE TRANSFORM
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    
    # DATASETS
    if args.task_type == 'A_L+A_U->B':
        train_dataset, unlabeled_dataset_A, datasets = get_datasets(args.dataset_name, train_transform, test_transform, args)
    elif args.task_type == 'A_L+A_U+B->A_U+B+C':
        train_dataset, unlabeled_dataset_A, unlabeled_dataset_B, datasets = get_datasets(args.dataset_name, train_transform, test_transform, args)

    if args.task_type == 'A_L+A_U+B->A_U+B+C':                                  
        test_loader_B = DataLoader(unlabeled_dataset_B, num_workers=args.num_workers, batch_size=256, shuffle=False, pin_memory=False)

    args.only_test = True
    test_loaders = []

    if args.dataset_name == 'domainnet':
        ovr_envs.remove(args.src_env)
        if args.task_type == 'A_L+A_U+B->A_U+B+C':
            ovr_envs.remove(args.aux_env)

        for d in ovr_envs:
            args.tgt_env = d
            test_dataset = get_datasets(args.dataset_name, train_transform, test_transform, args)
            test_loader_C = DataLoader(test_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
            test_loaders.append(test_loader_C)
    
    else:
        for d in distortions:
            for s in severity:
                args.distortion, args.severity = d, str(s)
                test_dataset = get_datasets(args.dataset_name, train_transform, test_transform, args)
                test_loader_C = DataLoader(test_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
                test_loaders.append(test_loader_C)

    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset) - label_len
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler, drop_last=True, pin_memory=True)
    test_loader_A = DataLoader(unlabeled_dataset_A, num_workers=args.num_workers,
                                        batch_size=256, shuffle=False, pin_memory=False)
    # test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
    #                                   batch_size=256, shuffle=False, pin_memory=False)

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    projector = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    dom_head = DINOHead(in_dim=args.feat_dim, out_dim=2, nlayers=args.num_mlp_layers).to(device)
    con_mlp = DINOHead(in_dim=256*2, out_dim=1).to(device)

    model = nn.Sequential(backbone, projector).to(device)

    # ----------------------
    # OPITMIZATION
    # ----------------------   
    params_groups = get_params_groups(model) + get_params_groups(dom_head) + get_params_groups(con_mlp)
    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 1e-3,
        )
    cluster_criterion = DistillLoss(
                        args.warmup_teacher_temp_epochs,
                        args.epochs,
                        args.n_views,
                        args.warmup_teacher_temp,
                        args.teacher_temp,
                    )
    
    mi_criterion = Distangleloss(device)

    # # inductive
    # best_test_acc_lab = 0
    # # transductive
    # best_train_acc_lab = 0
    # best_train_acc_ubl = 0 
    # best_train_acc_all = 0

    # ----------------------
    # TRAIN
    # ----------------------
    best_test_acc, best_train_ul_acc, best_train2_ul_acc = 0, 0, 0
    # train(model, train_loader, test_loader_labelled, test_loader_unlabelled, args)
    for epoch in range(args.epochs):
        args.logger.info(f'Epoch: {epoch}')
        train(model, dom_head, con_mlp, train_loader, optimizer, exp_lr_scheduler, cluster_criterion, mi_criterion, epoch, args)
        
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            with torch.no_grad():

                # Testing on unlabelled examples in domain A
                all_acc, old_acc, new_acc = test(model, test_loader_A, epoch=epoch, save_name='Train ACC Unlabelled', args=args)
                if all_acc > best_train_ul_acc:
                    args.logger.info('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
                    torch.save(model.state_dict(), os.path.join(args.model_path, 'best_trainul.pt'))
                    torch.save(dom_head.state_dict(), os.path.join(args.model_path, 'dom_head.pt'))
                    best_train_ul_acc = all_acc

                # Testing on unlabelled examples in domain B, if domain B exists
                if args.task_type == 'A_L+A_U+B->A_U+B+C':                                  
                    all_acc_B, old_acc_B, new_acc_B = test(model, test_loader_B, epoch=epoch, save_name='Train ACC Unlabelled2', args=args)
                    if all_acc_B > best_train2_ul_acc:
                        args.logger.info('Best Train-2 Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_B, old_acc_B, new_acc_B))
                        torch.save(model.state_dict(), os.path.join(args.model_path, 'best_trainul2.pt'))
                        best_train2_ul_acc = all_acc_B

                if epoch == args.epochs - 1:
                    # Testing on all examples in domain C
                    all_acc_test_arr, old_acc_test_arr, new_acc_test_arr = [], [], []

                    for test_loader in test_loaders: 
                        tmp_all_acc_test, tmp_old_acc_test, tmp_new_acc_test = test(model, test_loader, epoch=epoch, save_name='Test ACC', args=args)
                        all_acc_test_arr.append(tmp_all_acc_test)
                        old_acc_test_arr.append(tmp_old_acc_test)
                        new_acc_test_arr.append(tmp_new_acc_test)

                    all_acc_test, old_acc_test, new_acc_test = sum(all_acc_test_arr)/len(test_loaders), sum(old_acc_test_arr)/len(test_loaders), sum(new_acc_test_arr)/len(test_loaders)

                    if all_acc_test > best_test_acc:
                        args.logger.info('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
                        args.logger.info('Best Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test, new_acc_test))

                        if args.dataset_name == 'domainnet':
                            for i, d in enumerate(ovr_envs):
                                args.logger.info('For '+d+', All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test_arr[i], old_acc_test_arr[i], new_acc_test_arr[i]))
                            args.logger.info('################################')

                        else:
                            col = len(severity)
                            for i, d in enumerate(distortions):
                                args.logger.info('For '+d+', All {:.4f} | Old {:.4f} | New {:.4f}'.format(sum(all_acc_test_arr[i*col:(i+1)*col])/col, sum(old_acc_test_arr[i*col:(i+1)*col])/col, sum(new_acc_test_arr[i*col:(i+1)*col])/col))
                            args.logger.info('################################')

                        torch.save(model.state_dict(), os.path.join(args.model_path, 'best.pt'))
                        best_test_acc = all_acc_test