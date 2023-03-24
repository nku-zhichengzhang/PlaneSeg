import numpy as np
import torch
import torch.nn.functional as F

def evaluateTrimap(masks_pred, masks_gt):

    # combine gt masks
    combine_filter = torch.arange(1, masks_gt.shape[0]+1).expand(1, masks_gt.shape[0]).unsqueeze(2).unsqueeze(3).float().cuda()
    masks_gt_uns = masks_gt.unsqueeze(0)
    masks = F.conv2d(masks_gt_uns, combine_filter, stride=1, padding=0)

    # find edges
    filter_x = torch.tensor([[0., 1., 0.], [0., 0., 0.], [0., -1., 0.]]).expand(1,1,3,3).cuda()
    filter_y = torch.tensor([[0., 0., 0.], [1., 0., -1.], [0., 0., 0.]]).expand(1,1,3,3).cuda()
    edges_x = F.conv2d(masks, filter_x, stride=1, padding=0)
    edges_x = F.pad(edges_x, (1, 1, 1, 1))
    edges_y = F.conv2d(masks, filter_y, stride=1, padding=0)
    edges_y = F.pad(edges_y, (1, 1, 1, 1))
    edges = ((edges_x + edges_y) != 0).float()

    # cal IOUs
    masks_pred_cpu = masks_pred.detach().cpu().numpy().astype(np.int8)
    masks_gt_cpu = masks_gt.detach().cpu().numpy().astype(np.int8)
    pred_mat = np.expand_dims(masks_pred_cpu.transpose(1,2,0), axis=2)
    gt_mat = np.expand_dims(masks_gt_cpu.transpose(1,2,0), axis=3)

    intersection = gt_mat & pred_mat
    intersection_areas = np.sum(intersection, axis=(0,1))

    union = gt_mat | pred_mat
    union_areas = np.sum(union, axis=(0,1))

    plane_IOUs = intersection_areas / (union_areas + 1e-4) # shape: H*W*num_gt*num_pred

    matches = plane_IOUs > 0.5

    errors_of_union = []
    errors_of_gt = []
    errors_of_pred = []

    for expand in range(20):
        width = expand+1

        # trimap band area mask
        dilate_kernel = torch.ones((expand*2+1, expand*2+1), dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
        dilate_result = torch.clamp(F.conv2d(edges, dilate_kernel, padding=0), 0, 1)
        dilate_result = F.pad(dilate_result, (expand,expand,expand,expand))

        # mask the intersection and get new instersection area vals
        edge_mask = (dilate_result.detach().cpu().numpy() > 0.5).squeeze()
        masked = intersection[edge_mask,:,:][:,matches]
        correct = masked.sum()

        errors_of_union.append(correct / np.count_nonzero(union.sum(axis=(2,3))[edge_mask]))
        errors_of_gt.append(correct / np.count_nonzero(masks_gt_cpu.sum(axis=0)[edge_mask]))
        errors_of_pred.append(correct / np.count_nonzero(masks_pred_cpu.sum(axis=0)[edge_mask]))

    return errors_of_union, errors_of_gt, errors_of_pred