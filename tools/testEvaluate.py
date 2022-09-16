import os
import pycocotools.coco as coco
import numpy as np
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval


class ModelEval(object):
    def __init__(self, predicts_json, targets_json, cls_ids, cls_names, writer, sub):
        assert os.path.exists(predicts_json), '{} is not exists.'.format(predicts_json)
        assert os.path.exists(targets_json), '{} is not exists.'.format(targets_json)

        self.targets_json = targets_json
        self.predicts_json = predicts_json
        self.cls_ids = cls_ids
        self.cls_names = cls_names
        self.recall_value = np.arange(0, 1.01, 0.01)
        self.writer = writer
        self.sub = sub

        self.dataset = coco.COCO(self.targets_json)
        for key, val in tqdm(self.dataset.anns.items()):
            self.dataset.anns[key]['iscrowd'] = 0
            self.dataset.anns[key]['area'] = self.dataset.anns[key]['bbox'][2] * self.dataset.anns[key]['bbox'][3]
        self.predicts = self.dataset.loadRes(self.predicts_json)
        self.predicts = self.dataset.loadRes(self.predicts_json)

        self.coco_eval = COCOeval(self.dataset, self.predicts, "bbox")
        self.coco_eval.evaluate()
        self.coco_eval.accumulate()

    def summarize(self, coco_eval, category_id=None):
        """
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        """

        def _summarize(coco_eval, ap=1, iouThr=None, areaRng='all', maxDets=100):
            p = coco_eval.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}\n'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap == 1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) if iouThr is None else '{:0.2f}'.format(iouThr)   # 显示iou阈值

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = coco_eval.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]

                # 判断是否传入catId, 如果传入，就计算指定类别的指标
                if isinstance(category_id, int):
                    s = s[:, :, category_id, aind, mind]
                else:
                    s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = coco_eval.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]

                # 判断是否传入catId, 如果传入，就计算指定类别的指标
                if isinstance(category_id, int):
                    s = s[:, category_id, aind, mind]
                else:
                    s = s[:, :, aind, mind]

            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])

            return mean_s, iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)

        stats, print_list = [0] * 13, [""] * 13
        stats[0], print_list[0] = _summarize(coco_eval, ap=1, iouThr=.5)    # iou=0.5
        stats[1], print_list[1] = _summarize(coco_eval, ap=1, iouThr=.75)   # iou=0.75
        stats[2], print_list[2] = _summarize(coco_eval, ap=1, iouThr=.95)   # iou=0.95
        stats[3], print_list[3] = _summarize(coco_eval, ap=1)

        stats[4], print_list[4] = _summarize(coco_eval, iouThr=.5, areaRng='small')
        stats[5], print_list[5] = _summarize(coco_eval, iouThr=.75, areaRng='small')
        stats[6], print_list[6] = _summarize(coco_eval, iouThr=.95, areaRng='small')

        stats[7], print_list[7] = _summarize(coco_eval, iouThr=.5, areaRng='medium')
        stats[8], print_list[8] = _summarize(coco_eval, iouThr=.75, areaRng='medium')
        stats[9], print_list[9] = _summarize(coco_eval, iouThr=.95, areaRng='medium')

        stats[10], print_list[10] = _summarize(coco_eval, iouThr=.5, areaRng='large')
        stats[11], print_list[11] = _summarize(coco_eval, iouThr=.75, areaRng='large')
        stats[12], print_list[12] = _summarize(coco_eval, iouThr=.95, areaRng='large')

        stats_r, print_list_r = [0] * 13, [""] * 13
        stats_r[0], print_list_r[0] = _summarize(coco_eval, ap=0, iouThr=.5)   # iou=0.5
        stats_r[1], print_list_r[1] = _summarize(coco_eval, ap=0, iouThr=.75)  # iou=0.75
        stats_r[2], print_list_r[2] = _summarize(coco_eval, ap=0, iouThr=.95)  # iou=0.95
        stats_r[3], print_list_r[3] = _summarize(coco_eval, ap=0)

        stats_r[4], print_list_r[4] = _summarize(coco_eval, ap=0, iouThr=.5, areaRng='small')
        stats_r[5], print_list_r[5] = _summarize(coco_eval, ap=0, iouThr=.75, areaRng='small')
        stats_r[6], print_list_r[6] = _summarize(coco_eval, ap=0, iouThr=.95, areaRng='small')

        stats_r[7], print_list_r[7] = _summarize(coco_eval, ap=0, iouThr=.5, areaRng='medium')
        stats_r[8], print_list_r[8] = _summarize(coco_eval, ap=0, iouThr=.75, areaRng='medium')
        stats_r[9], print_list_r[9] = _summarize(coco_eval, ap=0, iouThr=.95, areaRng='medium')

        stats_r[10], print_list_r[10] = _summarize(coco_eval, ap=0, iouThr=.5, areaRng='large')
        stats_r[11], print_list_r[11] = _summarize(coco_eval, ap=0, iouThr=.75, areaRng='large')
        stats_r[12], print_list_r[12] = _summarize(coco_eval, ap=0, iouThr=.95, areaRng='large')

        print_info = '\n'.join(print_list)

        if not coco_eval.eval:
            raise Exception('Please run accumulate() first')

        return stats, print_info, stats_r

    def run_eval(self):
        # cls metric
        metrics = []
        for idx in range(len(self.cls_ids)):
            stats, _, stats_r = self.summarize(self.coco_eval, category_id=idx)

            # p
            p_50 = self.coco_eval.eval['precision'][0, int(stats_r[0] * 100), idx, 0, 2]
            p_50_s = self.coco_eval.eval['precision'][0, int(stats_r[4] * 100), idx, 0, 2]
            p_50_m = self.coco_eval.eval['precision'][0, int(stats_r[7] * 100), idx, 0, 2]
            p_50_l = self.coco_eval.eval['precision'][0, int(stats_r[10] * 100), idx, 0, 2]

            p_75 = self.coco_eval.eval['precision'][5, int(stats_r[1] * 100), idx, 0, 2]
            p_75_s = self.coco_eval.eval['precision'][5, int(stats_r[5] * 100), idx, 0, 2]
            p_75_m = self.coco_eval.eval['precision'][5, int(stats_r[8] * 100), idx, 0, 2]
            p_75_l = self.coco_eval.eval['precision'][5, int(stats_r[11] * 100), idx, 0, 2]

            p_95 = self.coco_eval.eval['precision'][9, int(stats_r[2] * 100), idx, 0, 2]
            p_95_s = self.coco_eval.eval['precision'][9, int(stats_r[6] * 100), idx, 0, 2]
            p_95_m = self.coco_eval.eval['precision'][9, int(stats_r[9] * 100), idx, 0, 2]
            p_95_l = self.coco_eval.eval['precision'][9, int(stats_r[12] * 100), idx, 0, 2]

            # save
            if stats[0] == -1 and stats[1] == -1 and stats[2] == -1:
                continue

            metrics.append('{}:\n'.format(self.cls_names[idx]))

            metrics.append('\tIoU=50%\n')
            metrics.append("\t\tAP50-All: {:.2f}%\n".format(stats[0] * 100))
            metrics.append("\t\tAP50-Small: {:.2f}%\n".format(stats[4] * 100))
            metrics.append("\t\tAP50-Medium: {:.2f}%\n".format(stats[7] * 100))
            metrics.append("\t\tAP50-Large: {:.2f}%\n".format(stats[10] * 100))

            metrics.append("\t\tP50-All: {:.2f}%\n".format(p_50 * 100))
            metrics.append("\t\tP50-Small: {:.2f}%\n".format(p_50_s * 100))
            metrics.append("\t\tP50-Medium: {:.2f}%\n".format(p_50_m * 100))
            metrics.append("\t\tP50-Large: {:.2f}%\n".format(p_50_l * 100))

            metrics.append("\t\tR50-All: {:.2f}%\n".format(stats_r[0] * 100))
            metrics.append("\t\tR50-Small: {:.2f}%\n".format(stats_r[4] * 100))
            metrics.append("\t\tR50-Medium: {:.2f}%\n".format(stats_r[7] * 100))
            metrics.append("\t\tR50-Large: {:.2f}%\n".format(stats_r[10] * 100))

            metrics.append('\tIoU=75%\n')
            metrics.append("\t\tAP75-All: {:.2f}%\n".format(stats[1] * 100))
            metrics.append("\t\tAP75-Small: {:.2f}%\n".format(stats[5] * 100))
            metrics.append("\t\tAP75-Medium: {:.2f}%\n".format(stats[8] * 100))
            metrics.append("\t\tAP75-Large: {:.2f}%\n".format(stats[11] * 100))

            metrics.append("\t\tP75-All: {:.2f}%\n".format(p_75 * 100))
            metrics.append("\t\tP75-Small: {:.2f}%\n".format(p_75_s * 100))
            metrics.append("\t\tP75-Medium: {:.2f}%\n".format(p_75_m * 100))
            metrics.append("\t\tP75-Large: {:.2f}%\n".format(p_75_l * 100))

            metrics.append("\t\tR75-All: {:.2f}%\n".format(stats_r[1] * 100))
            metrics.append("\t\tR75-Small: {:.2f}%\n".format(stats_r[5] * 100))
            metrics.append("\t\tR75-Medium: {:.2f}%\n".format(stats_r[8] * 100))
            metrics.append("\t\tR75-Large: {:.2f}%\n".format(stats_r[11] * 100))

            metrics.append('\tIoU=95%\n')
            metrics.append("\t\tAP95-All: {:.2f}%\n".format(stats[2] * 100))
            metrics.append("\t\tAP95-Small: {:.2f}%\n".format(stats[6] * 100))
            metrics.append("\t\tAP95-Medium: {:.2f}%\n".format(stats[9] * 100))
            metrics.append("\t\tAP95-Large: {:.2f}%\n".format(stats[12] * 100))

            metrics.append("\t\tP95-All: {:.2f}%\n".format(p_95 * 100))
            metrics.append("\t\tP95-Small: {:.2f}%\n".format(p_95_s * 100))
            metrics.append("\t\tP95-Medium: {:.2f}%\n".format(p_95_m * 100))
            metrics.append("\t\tP95-Large: {:.2f}%\n".format(p_95_l * 100))

            metrics.append("\t\tR95-All: {:.2f}%\n".format(stats_r[2] * 100))
            metrics.append("\t\tR95-Small: {:.2f}%\n".format(stats_r[6] * 100))
            metrics.append("\t\tR95-Medium: {:.2f}%\n".format(stats_r[9] * 100))
            metrics.append("\t\tR95-Large: {:.2f}%\n".format(stats_r[12] * 100))

        # mAP
        stats, _, stats_r = self.summarize(self.coco_eval)
        metrics.append('\nAP50-all: {:.2f}%\n'.format(stats[0] * 100))
        metrics.append('\nAP50-s  : {:.2f}%\n'.format(stats[4] * 100))
        metrics.append('\nAP50-m  : {:.2f}%\n'.format(stats[7] * 100))
        metrics.append('\nAP50-l  : {:.2f}%\n'.format(stats[10] * 100))

        metrics.append('\nAP75-all: {:.2f}%\n'.format(stats[1] * 100))
        metrics.append('\nAP75-s  : {:.2f}%\n'.format(stats[5] * 100))
        metrics.append('\nAP75-m  : {:.2f}%\n'.format(stats[8] * 100))
        metrics.append('\nAP75-l  : {:.2f}%\n'.format(stats[11] * 100))

        metrics.append('\nAR50-all: {:.2f}%\n'.format(stats_r[0] * 100))
        metrics.append('\nAR50-s  : {:.2f}%\n'.format(stats_r[4] * 100))
        metrics.append('\nAR50-m  : {:.2f}%\n'.format(stats_r[7] * 100))
        metrics.append('\nAR50-l  : {:.2f}%\n'.format(stats_r[10] * 100))

        metrics.append('\nAR75-all: {:.2f}%\n'.format(stats_r[1] * 100))
        metrics.append('\nAR75-s  : {:.2f}%\n'.format(stats_r[5] * 100))
        metrics.append('\nAR75-m  : {:.2f}%\n'.format(stats_r[8] * 100))
        metrics.append('\nAR75-l  : {:.2f}%\n'.format(stats_r[11] * 100))

        metrics.append('\nAP50:95 : {:.2f}%\n'.format(stats[3] * 100))
        metrics.append('\nAr50:95 : {:.2f}%\n'.format(stats_r[3] * 100))

        self.writer.write('\n{}:\n'.format(self.sub))
        for metric in metrics:
            self.writer.write(metric)


if __name__ == '__main__':
    ANN_PATH_vehicle = 'E:/Multi-Task_dataset/coco/annotations/vehicle_val2017.json'
    ANN_PATH_pedestrian = 'E:/Multi-Task_dataset/coco/annotations/person_val2017.json'
    coco_eval_vehicle = ModelEval(
        predicts_json='E:/ChaucerG_Works/mmdetection/exp/faster_rcnn_mv2_fpn_multi_rpn_mstrain_480-800_3x/latest_vehicles.bbox.json',
        targets_json=ANN_PATH_vehicle,
        cls_ids=[1, 2, 3],
        cls_names=['car', 'bus', 'truck'],
        writer=open('E:/ChaucerG_Works/mmdetection/exp/faster_rcnn_mv2_fpn_multi_rpn_mstrain_480-800_3x/car.txt', 'w'), sub='results')
    coco_eval_vehicle.run_eval()

    coco_eval_pedestrian = ModelEval(
        predicts_json='E:/ChaucerG_Works/mmdetection/exp/faster_rcnn_mv2_fpn_multi_rpn_mstrain_480-800_3x/latest_pedestrian.bbox.json',
        targets_json=ANN_PATH_pedestrian,
        cls_ids=[1, 2],
        cls_names=['person', 'rider'],
        writer=open('E:/ChaucerG_Works/mmdetection/exp/faster_rcnn_mv2_fpn_multi_rpn_mstrain_480-800_3x/person.txt', 'w'),
        sub='results')
    coco_eval_pedestrian.run_eval()
